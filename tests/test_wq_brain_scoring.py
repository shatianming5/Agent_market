"""Multi-dimensional alpha scoring tests."""
from __future__ import annotations

from agent_market.wq_brain.dtypes import SimulationResult
from agent_market.wq_brain.scoring import (
    AlphaScore,
    score_record,
    score_simulation_result,
    score_sub_universe,
    score_turnover,
    score_wq_fitness,
    score_wq_sharpe,
)


# ── Per-dimension ────────────────────────────────────────────────────────

def test_score_wq_sharpe_boundaries():
    assert score_wq_sharpe(None) == 0.0
    assert score_wq_sharpe(-0.5) == 0.0
    assert score_wq_sharpe(0.0) == 0.0
    assert score_wq_sharpe(1.5) == 100.0
    assert score_wq_sharpe(2.0) == 100.0
    assert 49.0 <= score_wq_sharpe(0.75) <= 51.0


def test_score_wq_fitness_boundaries():
    assert score_wq_fitness(None) == 0.0
    assert score_wq_fitness(0.0) == 0.0
    assert score_wq_fitness(1.0) == 100.0
    assert score_wq_fitness(1.5) == 100.0
    assert 49.0 <= score_wq_fitness(0.5) <= 51.0


def test_score_turnover_inverse():
    assert score_turnover(None) == 50.0  # neutral
    assert score_turnover(0.10) == 100.0
    assert score_turnover(0.15) == 100.0
    assert 55.0 <= score_turnover(0.30) <= 65.0
    assert 25.0 <= score_turnover(0.50) <= 35.0
    assert score_turnover(0.70) == 0.0
    assert score_turnover(1.0) == 0.0


def test_score_sub_universe_pass_fail():
    assert score_sub_universe([]) == 50.0
    assert score_sub_universe([{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}]) == 100.0
    assert score_sub_universe([{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL"}]) == 0.0
    assert score_sub_universe([{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PENDING"}]) == 50.0
    # Other check names ignored
    assert score_sub_universe([{"name": "OTHER_CHECK", "result": "PASS"}]) == 50.0


# ── Composite ────────────────────────────────────────────────────────────

def test_passing_alpha_gets_grade_a():
    sim = SimulationResult(
        sharpe=1.50, fitness=1.10, turnover=0.18, returns=0.20,
        status="COMPLETE",
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
    )
    s = score_simulation_result(sim)
    assert s.passes_quality is True
    assert s.grade == "A"
    assert s.score >= 80
    assert "submit" in s.recommendation.lower()


def test_top_known_alpha_scores_b_or_c():
    # Top alpha so far: sh=1.47 fi=0.77 to=0.46 — fails fitness gate
    sim = SimulationResult(
        sharpe=1.47, fitness=0.77, turnover=0.46, returns=0.11,
        status="COMPLETE",
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
    )
    s = score_simulation_result(sim)
    assert s.passes_quality is False
    assert s.grade in ("B", "C")
    # Should mention turnover as failure reason
    assert any("turnover" in r.lower() for r in s.reasons)


def test_negative_sharpe_alpha_gets_grade_d():
    sim = SimulationResult(
        sharpe=-1.02, fitness=-0.56, turnover=0.17,
        status="COMPLETE",
    )
    s = score_simulation_result(sim)
    assert s.grade == "D"
    assert s.score < 50
    assert "sharpe" in str(s.reasons).lower()


def test_with_local_data_uses_extended_weights():
    sim = SimulationResult(
        sharpe=1.40, fitness=0.95, turnover=0.20,
        status="COMPLETE",
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
    )
    s = score_simulation_result(
        sim,
        local_ic_mean=80.0,  # high IC
        local_ic_ir=70.0,
        local_stability=60.0,
        local_anti_overfit=50.0,
    )
    # Should have IC dims included in breakdown
    assert s.breakdown.ic_mean == 80.0
    assert s.breakdown.ic_ir == 70.0


def test_recommendation_for_near_miss():
    sim = SimulationResult(
        sharpe=1.45, fitness=0.85, turnover=0.40,
        status="COMPLETE",
        checks=[{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS"}],
    )
    s = score_simulation_result(sim)
    assert "near-miss" in s.recommendation or "mutation" in s.recommendation


def test_score_record_works_on_jsonl_row():
    row = {
        "expr": "rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))",
        "sharpe": 1.47, "fitness": 0.77, "turnover": 0.46,
        "status": "COMPLETE",
    }
    s = score_record(row)
    assert isinstance(s, AlphaScore)
    assert s.score > 30


def test_to_dict_serializable():
    sim = SimulationResult(sharpe=1.0, fitness=0.5, turnover=0.20, status="COMPLETE")
    s = score_simulation_result(sim)
    d = s.to_dict()
    assert "score" in d and "grade" in d and "breakdown" in d
    # Round-trip via JSON should work
    import json
    json.dumps(d)


def test_family_diversity_omitted_without_pool():
    sim = SimulationResult(sharpe=1.4, fitness=1.05, turnover=0.30, status="COMPLETE")
    s = score_simulation_result(sim)
    assert s.breakdown.family_diversity is None
    # to_dict() must drop None fields
    assert "family_diversity" not in s.to_dict()["breakdown"]


def test_family_diversity_full_credit_when_pool_empty():
    sim = SimulationResult(sharpe=1.4, fitness=1.05, turnover=0.30, status="COMPLETE")
    s = score_simulation_result(
        sim,
        expr="rank(close - vwap)",
        active_pool_exprs=[],
    )
    assert s.breakdown.family_diversity == 100.0


def test_family_diversity_penalizes_skeleton_dup():
    """A 'field-swap' clone of an ACTIVE expr → low family_diversity."""
    sim = SimulationResult(sharpe=1.4, fitness=1.05, turnover=0.30, status="COMPLETE")
    s_dup = score_simulation_result(
        sim,
        expr="rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))",
        active_pool_exprs=["rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"],
    )
    s_new = score_simulation_result(
        sim,
        expr="rank(group_zscore(close, sector))",
        active_pool_exprs=["rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"],
    )
    # New family scores higher than skeleton duplicate
    assert s_new.breakdown.family_diversity > s_dup.breakdown.family_diversity
    # And the duplicate should be < 30 (i.e. > 0.7 sim) — the threshold zone
    assert s_dup.breakdown.family_diversity < 30


def test_family_diversity_changes_composite_score():
    """Two alphas with identical sh/fi/to/sub_universe but different
    family_diversity must produce different composite scores."""
    sim = SimulationResult(sharpe=1.4, fitness=1.05, turnover=0.30, status="COMPLETE")
    pool = ["rank(ts_rank(close, 252) * (-ts_delta(close, 3) / close))"]
    s_dup = score_simulation_result(
        sim,
        expr="rank(ts_rank(vwap, 252) * (-ts_delta(vwap, 3) / vwap))",
        active_pool_exprs=pool,
    )
    s_new = score_simulation_result(
        sim,
        expr="rank(group_zscore(close, sector))",
        active_pool_exprs=pool,
    )
    assert s_new.score > s_dup.score
