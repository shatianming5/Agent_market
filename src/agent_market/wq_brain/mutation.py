"""Directed mutation engine — port of QuantGPT mutation_engine.py adapted for WQ FASTEXPR.

Diagnoses failure modes from a SimulationResult and recommends a targeted
mutation strategy. Used to guide the agent (via injected prompt hints) when
prior loop iterations produced near-misses.

Upstream: github.com/Miasyster/QuantGPT (MIT). Adapted for our WQ FASTEXPR
operator set (subset of QuantGPT's 80+) and English-first agent prompts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from .operators import OPERATORS_TS, OPERATORS_CS, OPERATORS_TRANSFORM, FIELDS_PRICE_VOLUME

logger = logging.getLogger(__name__)


class MutationStrategy(str, Enum):
    MUTATE_WINDOW = "mutate_window"
    MUTATE_OPERATOR = "mutate_operator"
    MUTATE_NORMALIZATION = "mutate_normalization"
    MUTATE_SIGNAL_TYPE = "mutate_signal_type"
    MUTATE_NONLINEAR = "mutate_nonlinear"
    MUTATE_INTERACTION = "mutate_interaction"
    REDUCE_TURNOVER = "reduce_turnover"
    SIMPLIFY = "simplify"
    REGENERATE_FULL = "regenerate_full"


# Maps each diagnostic strategy onto the canonical ``evidence_type`` token the
# tried_log + pheromone cache expect when the LLM acts on this suggestion.
STRATEGY_TO_EVIDENCE_TYPE: dict["MutationStrategy", str] = {}  # filled below


def _populate_strategy_evidence_map() -> None:
    """Populated once the class is defined; cleared here for readability."""
    STRATEGY_TO_EVIDENCE_TYPE.update({
        MutationStrategy.MUTATE_WINDOW: "numeric_tweak",
        MutationStrategy.MUTATE_OPERATOR: "op_swap",
        MutationStrategy.MUTATE_NORMALIZATION: "param_shift",
        MutationStrategy.MUTATE_SIGNAL_TYPE: "param_shift",
        MutationStrategy.MUTATE_NONLINEAR: "op_swap",
        MutationStrategy.MUTATE_INTERACTION: "crossover",
        MutationStrategy.REDUCE_TURNOVER: "param_shift",
        MutationStrategy.SIMPLIFY: "param_shift",
        MutationStrategy.REGENERATE_FULL: "seed",
    })


_populate_strategy_evidence_map()


@dataclass
class Diagnosis:
    strategy: MutationStrategy
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        return d


# Operator replacement map — within our WQ-available set.
# Keys MUST be in OPERATORS_TS / OPERATORS_CS / OPERATORS_TRANSFORM.
# Values are alternative ops that change the alpha shape meaningfully.
_OPERATOR_REPLACEMENTS: dict[str, list[str]] = {
    "ts_mean":          ["ts_decay_linear", "ts_sum", "ts_zscore"],
    "ts_delta":         ["ts_delay", "ts_rank"],
    "ts_corr":          ["ts_zscore", "ts_rank"],
    "ts_rank":          ["rank", "ts_zscore", "ts_decay_linear"],
    "ts_decay_linear":  ["ts_mean", "ts_sum"],
    "rank":             ["scale", "signed_power", "group_rank"],
    "group_rank":       ["group_zscore", "group_neutralize", "rank"],
    "group_zscore":     ["group_neutralize", "group_rank"],
    "ts_max":           ["ts_rank", "ts_sum"],
}

_NORMALIZATION_OPS = set(OPERATORS_CS) | {"scale"}
# Nonlinear shape transforms (we don't have tanh/sigmoid; use signed_power/sign/log/abs as proxies)
_NONLINEAR_OPS = {"signed_power", "sign", "log", "abs", "exp", "sqrt", "power"}
_TURNOVER_REDUCERS = set(OPERATORS_TRANSFORM) | {"ts_decay_linear"}
_BASE_FIELDS = set(FIELDS_PRICE_VOLUME)


@dataclass
class FailureContext:
    """Distilled metrics from a SimulationResult or pool entry, plus expr."""
    expr: str
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    status: str = "COMPLETE"
    error: Optional[str] = None
    alpha_id: Optional[str] = None
    region: Optional[str] = None
    universe: Optional[str] = None
    decay: Optional[int] = None

    @property
    def passes(self) -> bool:
        return (
            self.sharpe is not None and self.sharpe >= 1.25
            and self.fitness is not None and self.fitness >= 1.0
        )

    def quick_score(self) -> float:
        """Synthetic 0-100 score from sh/fi/to. Used for diagnosis only."""
        if self.status != "COMPLETE":
            return 0.0
        sh = max(0.0, min(2.0, self.sharpe or 0.0))
        fi = max(0.0, min(1.5, self.fitness or 0.0))
        to_penalty = max(0.0, (self.turnover or 0.5) - 0.2) * 50
        return max(0.0, min(100.0, 30 * sh + 40 * fi - to_penalty + 10))


class MutationEngine:
    """Diagnose factor failure modes, recommend targeted mutations."""

    def __init__(self, ctx: FailureContext) -> None:
        self.ctx = ctx
        self.expr = ctx.expr
        self.score = ctx.quick_score()

    def diagnose(self) -> Diagnosis:
        sharpe = self.ctx.sharpe or 0.0
        fitness = self.ctx.fitness or 0.0
        turnover = self.ctx.turnover or 0.0
        nesting = self._count_nesting(self.expr)
        has_norm = self._has_any(_NORMALIZATION_OPS)
        has_nonlinear = self._has_any(_NONLINEAR_OPS)
        has_turnover_reducer = self._has_any(_TURNOVER_REDUCERS)

        # 1. Hard error / no result
        if self.ctx.status != "COMPLETE":
            return Diagnosis(
                MutationStrategy.REGENERATE_FULL,
                f"Status={self.ctx.status} ({self.ctx.error or 'unknown'}); regenerate from scratch",
                {"status": self.ctx.status, "error": self.ctx.error},
            )

        # 2. Negative sharpe → flip sign
        if sharpe < -0.3:
            return Diagnosis(
                MutationStrategy.MUTATE_SIGNAL_TYPE,
                f"sharpe={sharpe:.2f} is negative — invert the sign of the alpha",
                {"sharpe": sharpe, "fitness": fitness},
            )

        # 3. Near-zero signal
        if abs(sharpe) < 0.2 and abs(fitness) < 0.1:
            return Diagnosis(
                MutationStrategy.MUTATE_OPERATOR,
                f"weak signal (sh={sharpe:.2f}, fi={fitness:.2f}) — replace core operators",
                {"sharpe": sharpe, "fitness": fitness,
                 "suggested_replacements": self._suggest_replacements()},
            )

        # 4. High turnover blocking fitness — KEY for our case (top alpha sh=1.47 fi=0.77 to=0.46)
        if turnover > 0.30 and fitness < 1.0 and not has_turnover_reducer:
            return Diagnosis(
                MutationStrategy.REDUCE_TURNOVER,
                f"turnover={turnover:.2f} caps fitness — wrap with hump() or ts_decay_linear()",
                {"turnover": turnover, "fitness": fitness,
                 "candidates": ["hump(_)  # 1-arg only on free tier",
                                "ts_decay_linear(_, 10)", "ts_decay_linear(_, 20)",
                                "keep adv20; current WQ endpoint rejects adv60/120",
                                "wrap inner ts_delta(close,N) → use larger N (10→20)"]},
            )

        # 5. Deep nesting → simplify
        if nesting > 8:
            return Diagnosis(
                MutationStrategy.SIMPLIFY,
                f"nesting depth={nesting} (>8); simplify to ≤6 levels",
                {"nesting_depth": nesting},
            )

        # 6. Mid-tier without nonlinear — add tanh-like transform
        if 1.0 <= sharpe < 1.5 and fitness < 1.0 and not has_nonlinear:
            return Diagnosis(
                MutationStrategy.MUTATE_NONLINEAR,
                f"sh={sharpe:.2f} fi={fitness:.2f} mid-tier; introduce nonlinear transform",
                {"sharpe": sharpe, "fitness": fitness,
                 "suggestions": ["wrap inner with signed_power(_, 0.5)",
                                 "wrap with sign(_) * sqrt(abs(_))",
                                 "log(1+abs(_))*sign(_)"]},
            )

        # 7. No normalization on outer layer
        if not has_norm:
            return Diagnosis(
                MutationStrategy.MUTATE_NORMALIZATION,
                "no rank/group_zscore/scale on outer layer; add normalization",
                {"has_normalization": False},
            )

        # 8. Single base-field signal → cross-field interaction
        if self._is_single_signal():
            return Diagnosis(
                MutationStrategy.MUTATE_INTERACTION,
                "single base field used; combine multiple signals (e.g. price × volume)",
                {"signal_count": 1},
            )

        # 9. Default: window tuning
        return Diagnosis(
            MutationStrategy.MUTATE_WINDOW,
            "default: tune time-series window parameters",
            {"sharpe": sharpe, "fitness": fitness,
             "current_windows": self._extract_windows(),
             "candidate_windows": [3, 5, 10, 20, 60, 120, 240]},
        )

    def format_for_prompt(self) -> str:
        """Render the diagnosis as a markdown block for prompt injection."""
        diag = self.diagnose()
        lines = [
            "### Mutation Engine Diagnosis",
            "",
            f"**Source alpha**: `{self.expr[:120]}`",
            f"**Quick-score**: {self.score:.0f} / 100 "
            f"(sh={self.ctx.sharpe or '?'}, fi={self.ctx.fitness or '?'}, to={self.ctx.turnover or '?'})",
            "",
            f"**Recommended strategy**: `{diag.strategy.value}`",
            f"**Reason**: {diag.reason}",
            "",
            "**Strategy guidance**:",
        ]
        lines.extend(_STRATEGY_GUIDANCE.get(diag.strategy, ["(no specific guidance)"]))
        if diag.details:
            lines.append("")
            lines.append("**Details**:")
            for k, v in diag.details.items():
                if isinstance(v, list) and v and len(str(v)) < 400:
                    lines.append(f"- {k}: {', '.join(str(x) for x in v)}")
                elif isinstance(v, dict):
                    lines.append(f"- {k}:")
                    for sk, sv in v.items():
                        lines.append(f"  - {sk}: {sv}")
                elif len(str(v)) < 200:
                    lines.append(f"- {k}: {v}")
        cmd = self.simulate_command_template()
        if cmd:
            lines.append("")
            lines.append("**Recommended `simulate` call** (replace `<new_expr>` "
                         "with the mutated FASTEXPR and TAG with your run tag):")
            lines.append("")
            lines.append("```bash")
            lines.append(cmd)
            lines.append("```")
        return "\n".join(lines)

    def simulate_command_template(self) -> str:
        """Render a ready-to-paste ``simulate`` command for this diagnosis.

        The command includes ``--parent-alpha-id`` and ``--evidence-type``
        flags pre-filled from the failure context, so the LLM only has to
        supply the mutated expression and the run tag. Returns an empty
        string when the failure context lacks an ``alpha_id`` (we cannot
        link the new row back to a parent then).
        """
        if not self.ctx.alpha_id:
            return ""
        diag = self.diagnose()
        evidence = STRATEGY_TO_EVIDENCE_TYPE.get(diag.strategy, "mutation")
        flags = [
            "python {WQ_TOOLS} simulate \"<new_expr>\"",
            f"--parent-alpha-id {self.ctx.alpha_id}",
            f"--evidence-type {evidence}",
            "--tag {TAG}",
        ]
        if self.ctx.region:
            flags.append(f"--region {self.ctx.region}")
        if self.ctx.universe:
            flags.append(f"--universe {self.ctx.universe}")
        if self.ctx.decay is not None:
            flags.append(f"--decay {self.ctx.decay}")
        return " \\\n  ".join(flags)

    # -- helpers ---------------------------------------------------------
    def _count_nesting(self, expr: str) -> int:
        max_d = d = 0
        for ch in expr:
            if ch == "(":
                d += 1
                max_d = max(max_d, d)
            elif ch == ")":
                d -= 1
        return max_d

    def _has_any(self, ops: set[str]) -> bool:
        e = self.expr.lower()
        return any(op + "(" in e for op in ops)

    def _is_single_signal(self) -> bool:
        e = self.expr.lower()
        used = [v for v in _BASE_FIELDS if re.search(rf"\b{v}\b", e)]
        return len(used) <= 1

    def _extract_windows(self) -> list[int]:
        pattern = r"ts_\w+\([^,()]+(?:\([^()]*\))?[^,()]*,\s*(\d+)\s*\)"
        matches = re.findall(pattern, self.expr)
        return sorted(set(int(m) for m in matches))

    def _suggest_replacements(self) -> dict[str, list[str]]:
        e = self.expr.lower()
        return {op: alts for op, alts in _OPERATOR_REPLACEMENTS.items() if op + "(" in e}


_STRATEGY_GUIDANCE: dict[MutationStrategy, list[str]] = {
    MutationStrategy.MUTATE_WINDOW: [
        "- Try different window lengths: 3 / 5 / 10 / 20 / 60 / 120 / 240",
        "- Keep the operator skeleton; only swap the integer window arg",
        "- Longer windows usually trade sharpe for lower turnover",
    ],
    MutationStrategy.MUTATE_OPERATOR: [
        "- Replace the core operator while keeping fields and structure",
        "- See `details.suggested_replacements` for the per-op alternatives",
        "- Cross-family swap (ts_mean ↔ ts_decay_linear ↔ ts_rank) is often most informative",
    ],
    MutationStrategy.MUTATE_NORMALIZATION: [
        "- Wrap the outer layer with `rank(...)`, `group_rank(..., sector)`,",
        "  or `group_zscore(..., sector)`",
        "- For inner sub-expressions try `scale(...)` to bring to [0,1]",
    ],
    MutationStrategy.MUTATE_SIGNAL_TYPE: [
        "- Negative sharpe means the signal is informative but inverted",
        "- Add `-1 *` in front, or invert the inner numerator/denominator",
        "- Example: `rank(ts_corr(close,volume,20))` → `rank(-ts_corr(close,volume,20))`",
    ],
    MutationStrategy.MUTATE_NONLINEAR: [
        "- Introduce a nonlinear shape transform on the inner alpha:",
        "    - `signed_power(_, 0.5)` — square-root with sign preserved",
        "    - `sign(_) * sqrt(abs(_))` — clamps tails",
        "    - `log(1+abs(_))*sign(_)` — log-compress",
        "- Linear stacks of ts_/group_ ops often plateau at sh~1.5 fi~0.8;",
        "  nonlinearity unlocks the next regime",
    ],
    MutationStrategy.MUTATE_INTERACTION: [
        "- Combine ≥2 base fields/signals to capture interaction effects",
        "- Examples:",
        "    - `rank(price_signal) * rank(volume_signal)`",
        "    - `rank(momentum) - 0.5 * rank(volatility)`",
        "    - `if_else(volume_high, alpha_a, alpha_b)`",
    ],
    MutationStrategy.REDUCE_TURNOVER: [
        "- Turnover capping fitness — apply a turnover-reduction layer:",
        "    - Outer: `hump(rank(<alpha>))`  — 1-arg ONLY on free tier (no threshold)",
        "    - Inner smooth: `ts_decay_linear(<alpha>, 10)` or 20",
        "    - Liquidity: keep `adv20`; current WQ endpoint rejects `adv60` / `adv120`",
        "    - Larger windows: `ts_delta(close, 10)` instead of `ts_delta(close, 3)`",
        "- WARNING: `hump(<alpha>, 0.01)` triggers 'Invalid number of inputs: 2'",
        "  on free tier — costs budget on rejection. Use 1-arg form only.",
        "- BEWARE: humping a near-zero alpha kills the signal. Only hump alphas",
        "  with sh ≥ 1.0 already.",
    ],
    MutationStrategy.SIMPLIFY: [
        "- Nesting too deep (>8 levels)",
        "- Pull out intermediate computations using FASTEXPR multi-line bindings:",
        "    `x = ts_mean(...); y = ts_zscore(...); rank(x * y)`",
        "- Drop redundant wrappers (e.g. nested rank inside rank)",
    ],
    MutationStrategy.REGENERATE_FULL: [
        "- Current alpha is unrecoverable; start from a different family",
        "- Try: intraday range, VWAP momentum, sector-relative reversal,",
        "  decay-weighted volume rank, open-gap signals",
        "- Avoid replicating the failed expression's operator stack",
    ],
}


def diagnose_from_record(record: dict[str, Any]) -> Optional[Diagnosis]:
    """Convenience: build a Diagnosis from a tried_log row dict."""
    expr = record.get("expr") or ""
    if not expr:
        return None
    ctx = FailureContext(
        expr=expr,
        sharpe=_to_float(record.get("sharpe")),
        fitness=_to_float(record.get("fitness")),
        turnover=_to_float(record.get("turnover")),
        returns=_to_float(record.get("returns")),
        status=str(record.get("status") or "COMPLETE"),
        error=record.get("error"),
        alpha_id=record.get("alpha_id"),
        region=record.get("region") or None,
        universe=record.get("universe") or None,
        decay=record.get("decay") if record.get("decay") is not None else None,
    )
    return MutationEngine(ctx).diagnose()


def render_top_failures_block(records: list[dict[str, Any]], *, top_n: int = 3) -> str:
    """Pick the top-N most-instructive near-failures and render mutation hints.

    Selection rule: COMPLETE status, sharpe positive, fitness < 1.0, prefer
    those closest to passing (highest score by quick_score).
    """
    candidates: list[tuple[FailureContext, MutationEngine]] = []
    for r in records:
        if r.get("status") != "COMPLETE":
            continue
        sh = _to_float(r.get("sharpe"))
        fi = _to_float(r.get("fitness"))
        if sh is None or fi is None:
            continue
        if sh < 0.5 or fi >= 1.0:
            continue
        ctx = FailureContext(
            expr=r.get("expr") or "",
            sharpe=sh, fitness=fi,
            turnover=_to_float(r.get("turnover")),
            returns=_to_float(r.get("returns")),
            status="COMPLETE",
            alpha_id=r.get("alpha_id"),
            region=r.get("region") or None,
            universe=r.get("universe") or None,
            decay=r.get("decay") if r.get("decay") is not None else None,
        )
        candidates.append((ctx, MutationEngine(ctx)))

    if not candidates:
        return ""

    candidates.sort(key=lambda c: -c[1].score)
    candidates = candidates[:top_n]

    out = [
        "## Mutation Hints (MANDATORY — see binding constraints below)",
        "",
        "### 🚫 HARD CONSTRAINTS (violating these wastes WQ budget and pool slot)",
        "",
        "1. **NO PURE PARAMETER TUNING THIS SESSION.** Re-running the same operator",
        "   stack with a different window (`ts_delta(close, 3)` → `ts_delta(close, 4)`)",
        "   is BANNED. Prior runs already explored windows 2/3/4/5/7/10/20 — fitness",
        "   ceiling is 0.80 in this family. Stop.",
        "",
        "2. **MUST simulate ≥ 2 cross-family alphas BEFORE any same-family attempt.**",
        "   Cross-family = different `family` label in the Cross-Over Candidates",
        "   table. If you've only seen ts_rank_close in prior runs, your first",
        "   2 simulates MUST be from: `ts_corr_pv`, `intraday_range`, `vwap_dev`,",
        "   `volume_rank`, `decay_linear`, or a NOVEL family not yet listed.",
        "",
        "3. **PRIORITIZE the recommended `strategy` field below.** If the diagnosis",
        "   says `reduce_turnover`, your next alpha MUST wrap with `hump(_)` or",
        "   `ts_decay_linear(_, N)`. If it says `mutate_nonlinear`, MUST add",
        "   `signed_power` / `sign(_) * sqrt(abs(_))`. Don't skip to window tuning.",
        "",
        "4. **Multi-signal alphas are HIGHEST priority.** Combinations like",
        "   `rank(family_A) + 0.5 * rank(family_B)` cracked turnover to 0.18",
        "   (vs 0.46 single-signal). If you haven't tried multi-signal yet,",
        "   try one as your FIRST simulate of this session.",
        "",
        "### Top near-failure diagnoses",
        "",
    ]
    for ctx, eng in candidates:
        out.append(eng.format_for_prompt())
        out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out).rstrip()


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
