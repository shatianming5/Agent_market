"""Between-panel routing controller for the colony.

PheroViz Sec. 4.7 formalises routing as a constrained local control problem
over the safe feasible set ``A_safe(s_t)``. wq_brain agents run as full
opencode/hermes subprocesses, so we cannot inject control between LLM turns;
we instead compute a *between-panel* routing advisory that ships into the
next panel's prompt as a controller note. The four routing actions mirror
PheroViz exactly:

  * ``stay`` — keep refining the current locus / altitude.
  * ``deeper`` — expand into a child node (drill into the same family /
    skeleton at a lower altitude).
  * ``bubble_up`` — escalate to a higher altitude (e.g. swap operator
    family) because the current locus is stalled or oscillating.
  * ``jump_root`` — restart from the HCT root (new region/universe family)
    because cross-panel constraints conflict or diagnosis says global.

The decision is purely a function of recent utility trajectory and the
diagnosis scope; no learned policy yet (PheroViz's μ_θ would belong in a
follow-up patch).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from .tried_log import (
    ALTITUDE_L1_REGION_UNIVERSE,
    ALTITUDE_L2_OP_FAMILY,
    ALTITUDE_L3_SLOT_PARAM,
    ALTITUDE_L4_NUMERIC_TWEAK,
    utility_score,
)


# Routing action constants — keep the literal strings stable across versions.
ACTION_STAY = "stay"
ACTION_DEEPER = "deeper"
ACTION_BUBBLE_UP = "bubble_up"
ACTION_JUMP_ROOT = "jump_root"

ACTIONS: tuple[str, ...] = (
    ACTION_STAY,
    ACTION_DEEPER,
    ACTION_BUBBLE_UP,
    ACTION_JUMP_ROOT,
)


@dataclass(frozen=True)
class RoutingThresholds:
    """Tunables for ``decide()``. Defaults match the PheroViz Sec. 4.7 rubric."""

    epsilon_stay: float = 0.05
    c_expand: float = 0.10
    stall_threshold: int = 3
    osc_threshold: int = 2
    tau_root: float = 0.5


@dataclass
class RoutingState:
    """Summary of recent panel behaviour fed into ``decide()``."""

    last_delta_U: float = 0.0
    mean_recent_delta_U: float = 0.0
    stall_count: int = 0
    osc_count: int = 0
    cross_panel_conflict: float = 0.0
    diagnosis_scope: str = "local"   # local | child | ancestor | global
    current_altitude: str = ALTITUDE_L3_SLOT_PARAM


@dataclass
class RoutingDecision:
    """Output of ``decide()`` — small enough to render into the prompt."""

    action: str
    target_altitude: str
    rationale: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_prompt_block(self) -> str:
        """Render as a minimal markdown block for prompt injection."""
        return (
            "### 🧭 COLONY ROUTING ADVISORY\n\n"
            f"_The colony controller analysed recent edits and is recommending_\n"
            f"_the **{self.action.upper()}** action at altitude **{self.target_altitude}**._\n\n"
            f"**Rationale**: {self.rationale}\n\n"
            f"_(Inputs: {self.inputs})_"
        )


def _next_altitude_after_bubble(current: str) -> str:
    """Map an altitude to its parent (one step higher)."""
    chain = (
        ALTITUDE_L4_NUMERIC_TWEAK,
        ALTITUDE_L3_SLOT_PARAM,
        ALTITUDE_L2_OP_FAMILY,
        ALTITUDE_L1_REGION_UNIVERSE,
    )
    try:
        idx = chain.index(current)
    except ValueError:
        return ALTITUDE_L1_REGION_UNIVERSE
    return chain[min(idx + 1, len(chain) - 1)]


def _next_altitude_after_deeper(current: str) -> str:
    """Map an altitude to its child (one step lower)."""
    chain = (
        ALTITUDE_L1_REGION_UNIVERSE,
        ALTITUDE_L2_OP_FAMILY,
        ALTITUDE_L3_SLOT_PARAM,
        ALTITUDE_L4_NUMERIC_TWEAK,
    )
    try:
        idx = chain.index(current)
    except ValueError:
        return ALTITUDE_L3_SLOT_PARAM
    return chain[min(idx + 1, len(chain) - 1)]


def decide(
    state: RoutingState,
    thresholds: RoutingThresholds = RoutingThresholds(),
) -> RoutingDecision:
    """Compute the routing decision from a ``RoutingState`` snapshot.

    Rule precedence follows PheroViz Sec. 4.7 — ``jump_root`` overrides
    ``bubble_up`` overrides ``stay``/``deeper``, and ``deeper`` only fires
    when the *predicted* gain is positive (we approximate predicted gain by
    the most recent ΔU).
    """
    inputs: dict[str, Any] = {
        "last_delta_U": round(state.last_delta_U, 4),
        "mean_recent_delta_U": round(state.mean_recent_delta_U, 4),
        "stall_count": state.stall_count,
        "osc_count": state.osc_count,
        "cross_panel_conflict": round(state.cross_panel_conflict, 4),
        "diagnosis_scope": state.diagnosis_scope,
        "current_altitude": state.current_altitude,
    }
    # 1. Global / conflict → restart from root.
    if (
        state.cross_panel_conflict > thresholds.tau_root
        or state.diagnosis_scope == "global"
    ):
        rationale = (
            "Cross-panel conflict above τ_root or diagnosis=global — abandon "
            "current family and restart from a different region/universe."
        )
        return RoutingDecision(
            action=ACTION_JUMP_ROOT,
            target_altitude=ALTITUDE_L1_REGION_UNIVERSE,
            rationale=rationale,
            inputs=inputs,
        )
    # 2. Stall or oscillation → bubble up one altitude.
    if (
        state.stall_count >= thresholds.stall_threshold
        or state.osc_count >= thresholds.osc_threshold
        or state.diagnosis_scope == "ancestor"
    ):
        target = _next_altitude_after_bubble(state.current_altitude)
        rationale = (
            f"Stalled (stall_count={state.stall_count}) or oscillating "
            f"(osc_count={state.osc_count}) at {state.current_altitude}; "
            f"escalating to {target}."
        )
        return RoutingDecision(
            action=ACTION_BUBBLE_UP,
            target_altitude=target,
            rationale=rationale,
            inputs=inputs,
        )
    # 3. Diagnosis points to a child + positive predicted gain → deeper.
    predicted_gain = state.last_delta_U
    if (
        predicted_gain > thresholds.c_expand
        and state.diagnosis_scope == "child"
    ):
        target = _next_altitude_after_deeper(state.current_altitude)
        rationale = (
            f"Predicted ΔU={predicted_gain:+.3f} exceeds c_expand and diagnosis "
            f"points to a child node — drilling deeper to {target}."
        )
        return RoutingDecision(
            action=ACTION_DEEPER,
            target_altitude=target,
            rationale=rationale,
            inputs=inputs,
        )
    # 4. Default: stay — recent gains are small but the locus is still useful.
    rationale = (
        f"ΔU last={state.last_delta_U:+.3f}, mean recent="
        f"{state.mean_recent_delta_U:+.3f}; staying at {state.current_altitude} "
        "until either ΔU drops below ε_stay or evidence of higher-altitude "
        "opportunity accumulates."
    )
    return RoutingDecision(
        action=ACTION_STAY,
        target_altitude=state.current_altitude,
        rationale=rationale,
        inputs=inputs,
    )


def state_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    cross_panel_conflict: float = 0.0,
    diagnosis_scope: str = "local",
    window: int = 5,
) -> RoutingState:
    """Build a ``RoutingState`` from the tail of a tried_log.

    * ``last_delta_U`` uses the most recent row's ``delta_U`` when present,
      else falls back to ``utility_score(row) − utility_score(prev_row)``.
    * ``stall_count`` counts consecutive low-gain steps (|ΔU| < 0.05).
    * ``osc_count`` counts adjacent sign flips on ΔU.
    * ``current_altitude`` is the altitude of the most recent row carrying
      one; defaults to L3 when none.
    """
    if not rows:
        return RoutingState(diagnosis_scope=diagnosis_scope,
                            cross_panel_conflict=cross_panel_conflict)

    ordered = sorted(rows, key=lambda r: float(r.get("ts") or 0.0))[-window:]
    deltas: list[float] = []
    prev: Optional[dict[str, Any]] = None
    for row in ordered:
        du = row.get("delta_U")
        if du is None and prev is not None:
            du = utility_score(
                sharpe=row.get("sharpe"),
                fitness=row.get("fitness"),
                turnover=row.get("turnover"),
            ) - utility_score(
                sharpe=prev.get("sharpe"),
                fitness=prev.get("fitness"),
                turnover=prev.get("turnover"),
            )
        if du is not None:
            deltas.append(float(du))
        prev = row

    last_delta_U = deltas[-1] if deltas else 0.0
    mean_recent = sum(deltas) / len(deltas) if deltas else 0.0
    stall_count = sum(1 for d in deltas[-3:] if abs(d) < 0.05) if deltas else 0
    osc_count = sum(
        1
        for a, b in zip(deltas, deltas[1:])
        if (a > 0 > b) or (a < 0 < b)
    )

    last_with_altitude = next(
        (r for r in reversed(ordered) if r.get("altitude")), None
    )
    current_altitude = (
        last_with_altitude.get("altitude")
        if last_with_altitude is not None
        else ALTITUDE_L3_SLOT_PARAM
    )

    return RoutingState(
        last_delta_U=last_delta_U,
        mean_recent_delta_U=mean_recent,
        stall_count=stall_count,
        osc_count=osc_count,
        cross_panel_conflict=cross_panel_conflict,
        diagnosis_scope=diagnosis_scope,
        current_altitude=current_altitude,
    )
