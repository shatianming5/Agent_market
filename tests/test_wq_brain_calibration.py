"""Calibration tests — stats helpers, threshold sweep, save/load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market.wq_brain.calibration import (
    _pearson,
    _rmse,
    _spearman,
    find_best_threshold,
    load_calibrated_threshold,
    save_calibrated_threshold,
    select_calibration_samples,
    threshold_path,
)
from agent_market.wq_brain.paths import tried_exprs_path
from agent_market.wq_brain.tried_log import append_tried


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path))
    return tmp_path


# ── Stats ────────────────────────────────────────────────────────────────


def test_pearson_perfect_positive():
    assert _pearson([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert _pearson([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_pearson_constant_returns_none():
    # zero variance on one side
    assert _pearson([1.0, 1.0, 1.0], [3.0, 4.0, 5.0]) is None


def test_pearson_too_few_points():
    assert _pearson([1.0, 2.0], [1.0, 2.0]) is None


def test_spearman_handles_ties():
    # Same rank-ordering ignoring scale → should be ~1.0
    assert _spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_spearman_monotonic_nonlinear():
    # x → exp(x) is monotonic so spearman should still be 1.0
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.71, 7.38, 20.08, 54.59]
    assert _spearman(xs, ys) == pytest.approx(1.0, abs=0.01)


def test_rmse_zero_for_identical():
    assert _rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_known_value():
    # sqrt(((0-3)^2 + (0-4)^2)/2) = sqrt(12.5) ≈ 3.5355
    assert _rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(3.5355, abs=0.001)


# ── find_best_threshold ────────────────────────────────────────────────


def test_find_best_threshold_perfect_separator():
    """All passes_remote=True samples have local_fitness above all False ones."""
    samples = [
        {"local_fitness": 0.2, "passes_remote": False},
        {"local_fitness": 0.3, "passes_remote": False},
        {"local_fitness": 0.4, "passes_remote": False},
        {"local_fitness": 0.7, "passes_remote": True},
        {"local_fitness": 0.8, "passes_remote": True},
        {"local_fitness": 0.9, "passes_remote": True},
    ]
    result = find_best_threshold(samples)
    assert 0.4 < result["threshold"] <= 0.7
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_find_best_threshold_no_positives():
    # ≥5 samples but every passes_remote=False → "no positives" path
    samples = [{"local_fitness": float(x), "passes_remote": False}
               for x in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)]
    result = find_best_threshold(samples)
    assert "no positives" in result["method"]


def test_find_best_threshold_too_few_samples():
    samples = [{"local_fitness": 0.5, "passes_remote": True}]
    result = find_best_threshold(samples)
    assert "insufficient" in result["method"]


# ── Threshold persistence ──────────────────────────────────────────────


def test_load_calibrated_threshold_default_when_missing(isolated_artifacts):
    assert load_calibrated_threshold("missing_tag") == 0.5
    assert load_calibrated_threshold("missing_tag", default=0.7) == 0.7


def test_save_and_load_roundtrip(isolated_artifacts):
    p = save_calibrated_threshold("tag1", 0.42)
    assert p == threshold_path("tag1")
    assert json.loads(p.read_text())["threshold"] == 0.42
    assert load_calibrated_threshold("tag1") == 0.42


def test_save_with_metadata(isolated_artifacts):
    save_calibrated_threshold(
        "tag2", 0.55,
        metadata={"n_samples": 30, "f1": 0.91, "pearson_fitness": 0.8},
    )
    data = json.loads(threshold_path("tag2").read_text())
    assert data["metadata"]["n_samples"] == 30
    assert data["metadata"]["f1"] == 0.91


# ── Sample selection ───────────────────────────────────────────────────


def test_select_calibration_samples_filters_incomplete(isolated_artifacts):
    p = tried_exprs_path("seltag")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"expr": "rank(close)", "sharpe": 1.4, "fitness": 1.1, "turnover": 0.3,
         "alpha_id": "a1", "status": "COMPLETE"},
        {"expr": "rank(volume)", "sharpe": None, "fitness": None, "turnover": None,
         "status": "ERROR"},  # filtered
        {"expr": "ts_rank(close, 20)", "sharpe": 1.2, "fitness": 0.8, "turnover": 0.4,
         "alpha_id": "a2", "status": "COMPLETE"},
    ]
    for r in rows:
        append_tried(
            p,
            expr=r["expr"], sharpe=r.get("sharpe"), fitness=r.get("fitness"),
            turnover=r.get("turnover"), alpha_id=r.get("alpha_id"),
            status=r.get("status", "COMPLETE"),
            region="USA", universe="TOP3000", decay=6,
        )
    selected = select_calibration_samples("seltag", top_n=10)
    exprs = [s["expr"] for s in selected]
    assert "rank(close)" in exprs
    assert "ts_rank(close, 20)" in exprs
    assert "rank(volume)" not in exprs


def test_select_calibration_samples_dedupes(isolated_artifacts):
    p = tried_exprs_path("dup_tag")
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        append_tried(
            p,
            expr="rank(close)", sharpe=1.4, fitness=1.1, turnover=0.3,
            alpha_id="a1", status="COMPLETE",
            region="USA", universe="TOP3000", decay=6,
        )
    selected = select_calibration_samples("dup_tag", top_n=10)
    assert len(selected) == 1


def test_select_calibration_samples_sorts_by_remote_fitness(isolated_artifacts):
    p = tried_exprs_path("sort_tag")
    p.parent.mkdir(parents=True, exist_ok=True)
    for expr, fi in [("a", 0.6), ("b", 1.2), ("c", 0.9)]:
        append_tried(
            p,
            expr=expr, sharpe=1.4, fitness=fi, turnover=0.3,
            alpha_id=expr, status="COMPLETE",
            region="USA", universe="TOP3000", decay=6,
        )
    selected = select_calibration_samples("sort_tag", top_n=10)
    assert [s["expr"] for s in selected] == ["b", "c", "a"]


# ── _wq_rating gate parameter ──────────────────────────────────────────


def test_wq_rating_default_gate_uses_05():
    from agent_market.wq_brain.local_sim import _wq_rating
    assert _wq_rating(0.49) == "Needs Improvement"
    assert _wq_rating(0.51) == "Average"


def test_wq_rating_custom_gate_shifts_boundary():
    from agent_market.wq_brain.local_sim import _wq_rating
    # gate=0.7 → fi=0.6 should be "Needs Improvement", fi=0.71 should be "Average"
    assert _wq_rating(0.6, gate=0.7) == "Needs Improvement"
    assert _wq_rating(0.71, gate=0.7) == "Average"
    # The absolute milestones (Good/Excellent) don't move
    assert _wq_rating(1.05, gate=0.7) == "Good"
    assert _wq_rating(1.55, gate=0.7) == "Excellent"


def test_load_calibrated_threshold_used_by_simulate(isolated_artifacts, monkeypatch):
    """When tag has a saved calibration, simulate_expression_locally should
    pass that gate down to wq_simulate (verified through patched wq_simulate)."""
    from agent_market.wq_brain import local_sim

    save_calibrated_threshold("calibrated_tag", 0.85)

    captured: dict = {}

    def fake_wq_simulate(work_df, rebalance_dates, *, fitness_gate=0.5, **kwargs):
        captured["fitness_gate"] = fitness_gate
        return {
            "wq_sharpe": 1.0, "wq_turnover": 0.3, "wq_returns": 0.5,
            "wq_fitness": 0.8, "wq_max_weight": 0.05,
            "wq_rating": "Average",
            "passes_local_gate": False, "fitness_gate": fitness_gate,
            "margin_bps": 0.0, "submittable": False,
            "sub_universe": {}, "wq_is_tests": {},
        }

    fake_ohlcv_evaluate_called = {}

    def fake_evaluate(expr, ohlcv):
        import pandas as pd
        fake_ohlcv_evaluate_called["yes"] = True
        return pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "stock_code": ["A"] * 20,
            "factor_value": [0.1] * 20,
            "daily_ret": [0.01] * 20,
        })

    # Stub out the heavy work
    monkeypatch.setattr(local_sim, "wq_simulate", fake_wq_simulate)
    monkeypatch.setattr(local_sim, "evaluate_expression", fake_evaluate)

    # OHLCV must be non-empty for the dispatch to proceed
    import pandas as pd
    fake_ohlcv = pd.DataFrame({"close": [1.0]})
    result = local_sim.simulate_expression_locally(
        "rank(close)", ohlcv=fake_ohlcv, tag="calibrated_tag",
    )
    assert captured["fitness_gate"] == 0.85
    assert result.raw["fitness_gate"] == 0.85


def test_simulate_falls_back_to_05_when_no_calibration(isolated_artifacts, monkeypatch):
    from agent_market.wq_brain import local_sim
    captured: dict = {}

    def fake_wq_simulate(work_df, rebalance_dates, *, fitness_gate=0.5, **kwargs):
        captured["fitness_gate"] = fitness_gate
        return {"wq_sharpe": 0, "wq_turnover": 0, "wq_returns": 0,
                "wq_fitness": 0, "wq_max_weight": 0,
                "wq_rating": "x", "passes_local_gate": False,
                "fitness_gate": fitness_gate, "margin_bps": 0,
                "submittable": False, "sub_universe": {}, "wq_is_tests": {}}

    def fake_evaluate(expr, ohlcv):
        import pandas as pd
        return pd.DataFrame({
            "trade_date": pd.date_range("2024-01-01", periods=20),
            "stock_code": ["A"] * 20,
            "factor_value": [0.1] * 20,
            "daily_ret": [0.01] * 20,
        })

    monkeypatch.setattr(local_sim, "wq_simulate", fake_wq_simulate)
    monkeypatch.setattr(local_sim, "evaluate_expression", fake_evaluate)

    import pandas as pd
    local_sim.simulate_expression_locally(
        "rank(close)", ohlcv=pd.DataFrame({"close": [1]}), tag="never_calibrated",
    )
    assert captured["fitness_gate"] == 0.5
