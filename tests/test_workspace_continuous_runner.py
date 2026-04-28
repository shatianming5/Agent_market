from __future__ import annotations

from pathlib import Path

import pytest


def test_phase_discovery_retries_discovered_and_tops_up_paper_pool(tmp_path, monkeypatch) -> None:
    import workspace.adaptive_params as adaptive_mod
    import workspace.continuous_runner as continuous_runner_mod
    import workspace.gate_pipeline as gate_pipeline_mod
    import workspace.pairs_engine as pairs_engine_mod
    import workspace.strategy_lifecycle as lifecycle_mod

    monkeypatch.setattr(continuous_runner_mod, "ROOT", tmp_path)

    lifecycle_db = tmp_path / "workspace" / "results" / "lifecycle.json"
    real_lm = lifecycle_mod.LifecycleManager(db_path=lifecycle_db)
    real_lm.register(
        "paper_seed",
        strategy_type="pairs",
        config={"pair_a": "BTC/USDT", "pair_b": "ETH/USDT", "exchange": "gate"},
    )
    real_lm.promote("paper_seed")
    real_lm.promote("paper_seed")
    real_lm.register(
        "pairs_VALIDATED_QUEUE",
        strategy_type="pairs",
        config={"pair_a": "ADA/USDT", "pair_b": "AVAX/USDT", "exchange": "gate"},
    )
    real_lm.promote("pairs_VALIDATED_QUEUE")
    real_lm.register(
        "pairs_DOGE_SOL",
        strategy_type="pairs",
        config={"pair_a": "DOGE/USDT", "pair_b": "SOL/USDT", "exchange": "gate"},
    )

    monkeypatch.setattr(lifecycle_mod, "LifecycleManager", lambda: real_lm)
    monkeypatch.setattr(
        pairs_engine_mod,
        "scan_pairs",
        lambda **kwargs: [
            {
                "pair_a": "DOGE/USDT",
                "pair_b": "SOL/USDT",
                "correlation": 0.91,
                "cointegrated": True,
                "half_life": 48.0,
                "hedge_ratio": 1.2,
            }
        ],
    )

    class _FakeAdaptiveEngine:
        def __init__(self, exchange: str = "gate") -> None:
            assert exchange == "gate"

        def optimize_pairs(self, pair_a: str, pair_b: str, *, cost_bps: float = 10.0):  # noqa: ANN001
            assert (pair_a, pair_b) == ("DOGE/USDT", "SOL/USDT")
            assert cost_bps == 10.0
            return {
                "best_params": {"lookback": 120, "entry_z": 3.0, "exit_z": 0.7},
                "train": {"sharpe": 1.2, "profit_pct": 9.0, "trades": 16},
                "overfit_ratio": 1.5,
            }

    class _FakeGatePipeline:
        def run_gates(self, strategy_file: Path, **kwargs):  # noqa: ANN003
            body = Path(strategy_file).read_text(encoding="utf-8")
            assert "LOOKBACK = 120" in body
            assert "ENTRY_Z = 3.0" in body
            assert "EXIT_Z = 0.7" in body
            return {
                "final_gate_passed": "gate_3",
                "recommendation": "pass",
                "gates": {
                    "gate_2": {
                        "metrics": {
                            "sharpe": 1.4,
                            "trades": 42,
                            "win_rate": 0.58,
                            "profit_factor": 1.3,
                        }
                    },
                    "gate_3": {
                        "metrics": {
                            "mean_sharpe": 0.9,
                            "pct_profitable": 0.75,
                            "mean_profit": 4.2,
                        }
                    },
                },
            }

    monkeypatch.setattr(adaptive_mod, "AdaptiveEngine", _FakeAdaptiveEngine)
    monkeypatch.setattr(gate_pipeline_mod, "GatePipeline", _FakeGatePipeline)

    runner = continuous_runner_mod.ContinuousRunner(exchange="gate", min_paper_pool=3)
    payload = runner._phase_discovery()

    assert payload["paper_pool_before"] == 1
    assert payload["promoted_existing"] == ["pairs_VALIDATED_QUEUE"]
    assert payload["attempted"] == 1
    assert payload["new_validated"] == 1
    assert payload["validated_names"] == ["pairs_DOGE_SOL"]
    assert payload["promoted_new"] == ["pairs_DOGE_SOL"]
    assert payload["paper_pool_after"] == 3
    assert real_lm.get("pairs_VALIDATED_QUEUE")["state"] == "paper"
    discovered = real_lm.get("pairs_DOGE_SOL")
    assert discovered["state"] == "paper"
    assert discovered["config"]["params"]["lookback"] == 120
    assert discovered["config"]["params_validation"]["trades"] == 42
    assert discovered["config"]["params_overfit_ratio"] == pytest.approx(1.5)
    strategy_file = tmp_path / "workspace" / "strategies" / "type_C_pairs" / "pairs_DOGE_SOL.py"
    assert strategy_file.exists()


def test_phase_report_generates_daily_report(tmp_path, monkeypatch) -> None:
    import workspace.continuous_runner as continuous_runner_mod
    import workspace.report_generator as report_generator_mod
    import workspace.strategy_lifecycle as lifecycle_mod

    monkeypatch.setattr(continuous_runner_mod, "ROOT", tmp_path)

    class _FakeLM:
        def summary(self):
            return {
                "total": 1,
                "by_state": {"paper": 1},
                "strategies": [
                    {
                        "name": "paper_demo",
                        "state": "paper",
                        "paper_days": 2,
                        "paper_dates": ["2026-04-08", "2026-04-09"],
                        "paper_pnl": [1.0, 0.5],
                        "paper_daily_equity": {"2026-04-08": 1010.0, "2026-04-09": 1015.0},
                        "paper_last_equity": 1015.0,
                        "cumulative_return_pct": 1.5,
                        "latest_day_return_pct": 0.5,
                        "paper_last_processed_at": "2026-04-09T23:00:00+00:00",
                        "paper_state_path": "/tmp/paper_demo.json",
                    }
                ],
            }

    called = {"count": 0}

    def _fake_generate_daily_report():
        called["count"] += 1
        return {"type": "daily"}

    monkeypatch.setattr(lifecycle_mod, "LifecycleManager", lambda: _FakeLM())
    monkeypatch.setattr(report_generator_mod, "generate_daily_report", _fake_generate_daily_report)

    runner = continuous_runner_mod.ContinuousRunner(exchange="gate")
    payload = runner._phase_report({"phases": {"paper_trading": {}, "review": {}}})

    assert called["count"] == 1
    assert payload["daily_report_generated"] is True
    assert payload["daily_report_error"] is None
    assert payload["daily_report_path"].endswith(".json")
    assert payload["paper_returns"][0]["name"] == "paper_demo"
