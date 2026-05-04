"""Multi-dimensional alpha scoring (0-100 + grade A/B/C/D + recommendation).

Adapted from QuantGPT's `score_factor`. Until local IC data is available
(Day 3+ once yfinance pipeline lands), the score is dominated by WQ-aligned
metrics returned by `WQSession.simulate_and_parse`. Each dimension is
0-100 then weighted; final score capped at 0-100.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .dtypes import SimulationResult


@dataclass
class ScoreBreakdown:
    wq_sharpe: float = 0.0
    wq_fitness: float = 0.0
    turnover: float = 0.0
    sub_universe: float = 0.0
    ic_mean: Optional[float] = None  # Day 3+
    ic_ir: Optional[float] = None    # Day 3+
    stability: Optional[float] = None  # Day 3+
    anti_overfit: Optional[float] = None  # Day 4+

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AlphaScore:
    score: float  # 0-100
    grade: str    # A / B / C / D
    recommendation: str
    passes_quality: bool  # the binary gate (sh ≥ 1.25 AND fi ≥ 1.0)
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "grade": self.grade,
            "recommendation": self.recommendation,
            "passes_quality": self.passes_quality,
            "breakdown": self.breakdown.to_dict(),
            "reasons": self.reasons,
        }


# ── Per-dimension scoring functions (each returns 0-100) ─────────────────

def score_wq_sharpe(sharpe: Optional[float]) -> float:
    """sh ≥ 1.5 = 100, sh = 1.0 = 50, sh ≤ 0 = 0. Linear interp."""
    if sharpe is None:
        return 0.0
    if sharpe <= 0:
        return 0.0
    if sharpe >= 1.5:
        return 100.0
    return min(100.0, max(0.0, (sharpe / 1.5) * 100))


def score_wq_fitness(fitness: Optional[float]) -> float:
    """fi ≥ 1.0 = 100, fi = 0.5 = 50, fi ≤ 0 = 0."""
    if fitness is None:
        return 0.0
    if fitness <= 0:
        return 0.0
    if fitness >= 1.0:
        return 100.0
    return min(100.0, max(0.0, fitness * 100))


def score_turnover(turnover: Optional[float]) -> float:
    """to ≤ 0.15 = 100, to = 0.30 = 60, to = 0.50 = 30, to ≥ 0.70 = 0.

    Turnover scoring is inverse: we *prefer* low turnover.
    """
    if turnover is None:
        return 50.0  # unknown — neutral
    t = max(0.0, min(1.0, turnover))
    if t <= 0.15:
        return 100.0
    if t >= 0.70:
        return 0.0
    # Piecewise linear: 0.15 → 100, 0.30 → 60, 0.50 → 30, 0.70 → 0
    breakpoints = [(0.15, 100.0), (0.30, 60.0), (0.50, 30.0), (0.70, 0.0)]
    for (t1, s1), (t2, s2) in zip(breakpoints, breakpoints[1:]):
        if t1 <= t <= t2:
            frac = (t - t1) / (t2 - t1) if t2 > t1 else 0.0
            return s1 + (s2 - s1) * frac
    return 0.0


def score_sub_universe(checks: list[dict[str, Any]]) -> float:
    """Score WQ's IS check `LOW_SUB_UNIVERSE_SHARPE`.

    PASS = 100; FAIL = 0; missing = 50 (neutral).
    """
    if not checks:
        return 50.0
    for check in checks:
        if check.get("name") == "LOW_SUB_UNIVERSE_SHARPE":
            result = check.get("result")
            if result == "PASS":
                return 100.0
            if result == "FAIL":
                return 0.0
            return 50.0  # PENDING / UNKNOWN
    return 50.0


# ── Composite ────────────────────────────────────────────────────────────

# Weights (sum = 1.0). Tuned for the no-local-data regime (Days 1-2).
# When Day 3+ adds IC/stability dims, weights will rebalance.
_WEIGHTS_NO_LOCAL_DATA = {
    "wq_sharpe": 0.35,
    "wq_fitness": 0.35,
    "turnover": 0.15,
    "sub_universe": 0.15,
}

# Weights once local data lands (Day 3+).
_WEIGHTS_WITH_LOCAL_DATA = {
    "wq_sharpe": 0.20,
    "wq_fitness": 0.20,
    "turnover": 0.10,
    "sub_universe": 0.10,
    "ic_mean": 0.15,
    "ic_ir": 0.10,
    "stability": 0.10,
    "anti_overfit": 0.05,
}


def _grade(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _recommendation(score: float, passes_quality: bool, breakdown: ScoreBreakdown) -> str:
    if passes_quality:
        return "submit (passes WQ gate)"
    if score >= 65:
        return "near-miss — try mutation (turnover-reduction or normalization)"
    if score >= 40:
        return "weak — needs operator change or sign flip"
    return "very weak — regenerate from a new family"


def score_simulation_result(
    result: SimulationResult,
    *,
    sharpe_min: float = 1.25,
    fitness_min: float = 1.0,
    local_ic_mean: Optional[float] = None,
    local_ic_ir: Optional[float] = None,
    local_stability: Optional[float] = None,
    local_anti_overfit: Optional[float] = None,
) -> AlphaScore:
    """Score a single SimulationResult (and optional local-backtest dims)."""
    breakdown = ScoreBreakdown(
        wq_sharpe=score_wq_sharpe(result.sharpe),
        wq_fitness=score_wq_fitness(result.fitness),
        turnover=score_turnover(result.turnover),
        sub_universe=score_sub_universe(result.checks or []),
        ic_mean=local_ic_mean,
        ic_ir=local_ic_ir,
        stability=local_stability,
        anti_overfit=local_anti_overfit,
    )

    has_local = any(v is not None for v in
                    (local_ic_mean, local_ic_ir, local_stability, local_anti_overfit))
    weights = _WEIGHTS_WITH_LOCAL_DATA if has_local else _WEIGHTS_NO_LOCAL_DATA

    total = 0.0
    weight_sum = 0.0
    for dim, w in weights.items():
        v = getattr(breakdown, dim)
        if v is None:
            continue
        total += v * w
        weight_sum += w
    score = (total / weight_sum) if weight_sum > 0 else 0.0

    passes = result.passes_quality(sharpe_min=sharpe_min, fitness_min=fitness_min)

    # Hard cap: only alphas that PASS the WQ gate can earn grade A (≥80).
    # Near-misses are capped at 79 so the grade-letter reflects gate status.
    if not passes:
        score = min(score, 79.0)

    reasons: list[str] = []
    if breakdown.wq_sharpe < 50:
        reasons.append(f"sharpe {result.sharpe} too low")
    if breakdown.wq_fitness < 50:
        reasons.append(f"fitness {result.fitness} below 0.5")
    if breakdown.turnover < 50:
        reasons.append(f"turnover {result.turnover} above 0.30")
    if breakdown.sub_universe < 50:
        reasons.append("sub-universe sharpe FAIL")
    if not reasons and not passes:
        reasons.append("borderline — no single failing dim, but composite below gate")

    return AlphaScore(
        score=score,
        grade=_grade(score),
        recommendation=_recommendation(score, passes, breakdown),
        passes_quality=passes,
        breakdown=breakdown,
        reasons=reasons,
    )


def score_record(record: dict[str, Any]) -> AlphaScore:
    """Score a tried_log JSONL row."""
    # Build a synthetic SimulationResult from the row
    sim = SimulationResult(
        sharpe=_to_float(record.get("sharpe")),
        fitness=_to_float(record.get("fitness")),
        turnover=_to_float(record.get("turnover")),
        returns=_to_float(record.get("returns")),
        alpha_id=record.get("alpha_id"),
        status=record.get("status", "COMPLETE"),
        error=record.get("error"),
        checks=record.get("checks", []) or [],
    )
    return score_simulation_result(sim)


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
