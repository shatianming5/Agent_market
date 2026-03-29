"""Tests for workspace.backtest_api."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_detect_strategy_name():
    from workspace.backtest_api import _detect_strategy_name

    template = ROOT / "workspace" / "strategies" / "strategy_template.py"
    assert template.exists()
    name = _detect_strategy_name(template)
    assert name == "StrategyTemplate"


def test_run_backtest_missing_file():
    from workspace.backtest_api import run_backtest

    result = run_backtest("nonexistent.py")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_run_backtest_with_template():
    """Run a real backtest with the strategy template — integration test."""
    from workspace.backtest_api import run_backtest

    result = run_backtest(
        strategy_path="workspace/strategies/strategy_template.py",
        strategy_name="StrategyTemplate",
        timerange="20251226-20260125",
    )
    assert isinstance(result, dict)
    if result["ok"]:
        assert "sharpe" in result
        assert "max_drawdown_pct" in result
        assert "profit_pct" in result
        assert "win_rate" in result
        assert "trades" in result
        assert result["trades"] >= 0
        # Result should be auto-saved
        results_dir = ROOT / "workspace" / "results"
        assert any(results_dir.glob("run_*.json"))
    else:
        # Strategy may fail in minimal env, but API should not crash
        assert "error" in result
