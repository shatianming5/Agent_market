from __future__ import annotations

from pathlib import Path

from agent_market.strategy_miner._benchmark import load_benchmark_suite, run_benchmark_suite
from agent_market.strategy_miner.dtypes import StrategyCandidate


def _valid_strategy_code() -> str:
    return """
from freqtrade.strategy import IStrategy

class BenchmarkedStrategy(IStrategy):
    timeframe = "5m"

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
"""


def test_load_benchmark_suite_from_directory() -> None:
    suite = load_benchmark_suite("benchmark_pack/default")

    assert suite["suite_id"] == "strategy_miner_default"
    assert suite["gates"]["selection_holdout_delta_max_pct"] == 50.0
    assert suite["_manifest_path"].endswith("benchmark_pack/default/manifest.json")


def test_run_benchmark_suite_passes_for_clean_candidate() -> None:
    candidate = StrategyCandidate(
        name="BenchmarkedStrategy",
        code=_valid_strategy_code(),
        strategy_path=Path("/tmp/BenchmarkedStrategy.py"),
        iteration=3,
    )
    candidate.reward = 1.2
    candidate.constraints_ok = True
    candidate.backtest_summary = {
        "profit_total_pct": 12.0,
        "trades": 24,
        "winrate": 0.58,
        "realistic_sharpe": 1.4,
        "profit_factor": 1.5,
        "max_drawdown_pct": 9.0,
        "positive_days_ratio": 0.55,
        "return_over_drawdown": 1.3,
        "observation_days": 8,
        "metrics_trusted": True,
        "metric_flags": [],
        "results_per_pair": [
            {"key": "BTC/USDT", "profit_total_pct": 2.1},
            {"key": "ETH/USDT", "profit_total_pct": 1.4},
        ],
        "walkforward": {
            "folds_completed": 3,
            "folds_total": 3,
            "sharpe_mean": 1.1,
            "sharpe_std": 0.2,
        },
        "daily_profit": [
            ["2026-03-12", 10.0],
            ["2026-03-13", 4.0],
            ["2026-03-14", -2.0],
            ["2026-03-15", 5.0],
            ["2026-03-16", 1.0],
            ["2026-03-17", 2.0],
        ],
        "starting_balance": 1000.0,
        "final_balance": 1020.0,
    }

    verdict = run_benchmark_suite(
        candidate,
        suite_path="benchmark_pack/default",
        holdout_result={"delta_pct": 12.0, "overfitting_flag": False},
    )

    assert verdict["passed"] is True
    assert verdict["failed_ids"] == []


def test_run_benchmark_suite_fails_for_leakage_pattern() -> None:
    candidate = StrategyCandidate(
        name="LeakyStrategy",
        code=_valid_strategy_code() + "\n# leak\ndf['future'] = df['close'].shift(-1)\n",
        strategy_path=Path("/tmp/LeakyStrategy.py"),
        iteration=1,
    )
    candidate.reward = 0.9
    candidate.constraints_ok = True
    candidate.backtest_summary = {
        "profit_total_pct": 8.0,
        "trades": 20,
        "winrate": 0.55,
        "realistic_sharpe": 0.9,
        "profit_factor": 1.2,
        "max_drawdown_pct": 8.0,
        "positive_days_ratio": 0.5,
        "return_over_drawdown": 1.0,
        "observation_days": 6,
        "metrics_trusted": False,
        "metric_flags": ["native_sharpe_inflated"],
        "results_per_pair": [{"key": "BTC/USDT", "profit_total_pct": 1.0}],
        "walkforward": {"folds_completed": 3, "folds_total": 3},
        "daily_profit": [
            ["2026-03-12", 10.0],
            ["2026-03-13", 4.0],
            ["2026-03-14", -2.0],
            ["2026-03-15", 5.0],
        ],
        "starting_balance": 1000.0,
    }

    verdict = run_benchmark_suite(
        candidate,
        suite_path="benchmark_pack/default",
        holdout_result={"delta_pct": 12.0, "overfitting_flag": False},
    )

    assert verdict["passed"] is False
    assert "metric_integrity" in verdict["failed_ids"]
    assert "no_negative_shift" in verdict["failed_ids"]
