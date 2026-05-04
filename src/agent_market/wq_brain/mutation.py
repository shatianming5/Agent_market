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
                                "switch adv20 → adv60/120 in liquidity terms",
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
        return "\n".join(lines)

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
        "    - Liquidity: replace `adv20` with `adv60` / `adv120`",
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
        )
        candidates.append((ctx, MutationEngine(ctx)))

    if not candidates:
        return ""

    candidates.sort(key=lambda c: -c[1].score)
    candidates = candidates[:top_n]

    out = ["## Mutation Hints (from top near-failures)", ""]
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
