"""Tests for workspace modules: model_loader, evaluator."""
from __future__ import annotations


def test_model_loader_discovers_ridge():
    import sys
    sys.path.insert(0, ".")
    from workspace.model_loader import scan_and_register, list_available_models

    registered = scan_and_register()
    assert "ridge_regression" in registered or "ridge_regression" in list_available_models()
    all_models = list_available_models()
    assert "lightgbm" in all_models  # built-in
    assert "ridge_regression" in all_models  # workspace


def test_ridge_model_trains():
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from workspace.model_loader import scan_and_register
    from agent_market.freqai.model.base import ModelRegistry
    import agent_market.freqai.model  # noqa: F401

    scan_and_register()
    model = ModelRegistry.create("ridge_regression", {"alpha": 1.0, "model_dir": "/tmp/ridge_test"})
    X = np.random.randn(100, 5).astype(np.float32)
    y = X @ np.array([1, 2, 3, 4, 5], dtype=np.float32) + np.random.randn(100).astype(np.float32) * 0.1
    result = model.fit(X, y)
    assert result.model_path.exists()
    assert "rmse_train" in result.metrics
    preds = model.predict(X)
    assert preds.shape == (100,)


def test_evaluator_scores_result():
    import sys
    sys.path.insert(0, ".")
    from workspace.evaluator import evaluate

    good_result = {
        "ok": True,
        "sharpe": 1.5,
        "sortino": 2.0,
        "max_drawdown_pct": 3.0,
        "profit_pct": 5.0,
        "win_rate": 0.55,
        "profit_factor": 1.5,
        "trades": 100,
    }
    score = evaluate(good_result)
    assert score["total_score"] > 70
    assert score["grade"] in ("A", "B+", "B")
    assert all(d["met"] for d in score["details"].values())

    bad_result = {
        "ok": True,
        "sharpe": -1.0,
        "sortino": -2.0,
        "max_drawdown_pct": 15.0,
        "profit_pct": -5.0,
        "win_rate": 0.30,
        "profit_factor": 0.5,
        "trades": 10,
    }
    score_bad = evaluate(bad_result)
    assert score_bad["total_score"] < 40
    assert len(score_bad["suggestions"]) > 0
    assert len(score_bad["constraint_violations"]) > 0  # min_trades violated


def test_evaluator_handles_failed_result():
    import sys
    sys.path.insert(0, ".")
    from workspace.evaluator import evaluate

    result = {"ok": False, "error": "strategy crashed"}
    score = evaluate(result)
    assert score["total_score"] == 0
    assert score["grade"] == "F"
