"""Core workspace module regression tests."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestCostModel:
    def test_maker_cheaper_than_taker(self):
        from workspace.cost_model import CostModel
        cm = CostModel(exchange="gate")
        est = cm.estimate_total_cost(trade_size_usd=500)
        # Maker fee < taker fee in the model
        assert cm.maker_bps <= cm.taker_bps
        assert est.fee_bps > 0

    def test_slippage_increases_with_size(self):
        from workspace.cost_model import CostModel
        cm = CostModel(exchange="gate")
        small = cm.estimate_total_cost(trade_size_usd=100)
        large = cm.estimate_total_cost(trade_size_usd=10000)
        assert large.slippage_bps > small.slippage_bps


class TestRiskManager:
    def test_circuit_breaker_fires(self):
        from workspace.risk_manager import RiskManager
        rm = RiskManager(initial_equity=1000, max_drawdown_pct=5.0)
        rm.current_equity = 940  # 6% DD
        d = rm.check_trade(signal_strength=1.0, win_rate=0.6, avg_win_pct=2.0, avg_loss_pct=1.0)
        assert not d.allowed
        assert "CIRCUIT BREAKER" in d.reason

    def test_kelly_caps_position(self):
        from workspace.risk_manager import RiskManager
        rm = RiskManager(kelly_fraction_cap=0.25)
        d = rm.check_trade(signal_strength=1.0, win_rate=0.99, avg_win_pct=10.0, avg_loss_pct=0.1)
        assert d.position_size_pct <= 25.0


class TestStrategyLifecycle:
    def test_valid_transitions(self):
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        lm = LifecycleManager(db_path=db)
        lm.register("test", strategy_type="pairs")
        assert lm.get("test")["state"] == "discovered"
        assert lm.promote("test")  # → validated
        assert lm.get("test")["state"] == "validated"
        assert lm.promote("test")  # → paper
        assert lm.get("test")["state"] == "paper"
        Path(db).unlink()

    def test_invalid_transition_rejected(self):
        from workspace.strategy_lifecycle import LifecycleManager, StrategyState
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        lm = LifecycleManager(db_path=db)
        lm.register("test", strategy_type="pairs")
        # Can't go discovered → active directly
        assert not lm.transition("test", StrategyState.ACTIVE)
        Path(db).unlink()

    def test_legacy_loop_count_paper_history_resets(self):
        from workspace.strategy_lifecycle import LifecycleManager

        payload = {
            "s1": {
                "name": "s1",
                "state": "paper",
                "type": "pairs",
                "config": {},
                "source": "manual",
                "created_at": "2026-04-07T00:00:00+00:00",
                "updated_at": "2026-04-08T00:00:00+00:00",
                "history": [
                    {"state": "discovered", "at": "2026-04-07T00:00:00+00:00", "reason": "registered"},
                    {"state": "validated", "at": "2026-04-07T00:01:00+00:00", "reason": "ok"},
                    {"state": "paper", "at": "2026-04-08T00:00:00+00:00", "reason": "paper"},
                ],
                "paper_days": 223,
                "paper_pnl": [0.0] * 223,
                "active_days": 0,
                "rolling_sharpe": [],
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(json.dumps(payload).encode("utf-8"))
            db = f.name
        lm = LifecycleManager(db_path=db)
        entry = lm.get("s1")
        assert entry["paper_days"] == 0
        assert entry["paper_pnl"] == []
        assert entry["paper_dates"] == []
        assert entry["paper_daily_equity"] == {}
        assert entry["legacy_paper_tracking_reset_reason"] == "legacy_loop_count_without_dates"
        Path(db).unlink()

    def test_sync_paper_tracking_uses_unique_days(self):
        from workspace.strategy_lifecycle import LifecycleManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        lm = LifecycleManager(db_path=db)
        lm.register("paper", strategy_type="pairs")
        lm.promote("paper")
        lm.promote("paper")
        lm.sync_paper_tracking(
            "paper",
            daily_equity={
                "2026-04-08": 1010.0,
                "2026-04-09": 1020.0,
            },
            last_processed_at="2026-04-09T23:00:00+00:00",
            current_equity=1020.0,
        )
        entry = lm.get("paper")
        assert entry["paper_days"] == 2
        assert entry["paper_dates"] == ["2026-04-08", "2026-04-09"]
        assert entry["paper_pnl"][0] == pytest.approx(1.0)
        assert entry["paper_pnl"][1] == pytest.approx(0.99009901)
        Path(db).unlink()

    def test_auto_review_retires_health_and_guardrail_failures(self):
        from workspace.strategy_lifecycle import LifecycleManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        lm = LifecycleManager(db_path=db)
        lm.register("bad_health", strategy_type="pairs", config={})
        lm.promote("bad_health")
        lm.promote("bad_health")
        lm.register("bad_guardrail", strategy_type="pairs", config={
            "params_validation": {"sharpe": 0.0, "trades": 0},
            "params_overfit_ratio": 100.0,
        })
        lm.promote("bad_guardrail")
        lm.promote("bad_guardrail")

        actions = lm.auto_review(health_issues=[{
            "strategy": "bad_health",
            "type": "pair_health",
            "details": ["cointegration lost"],
        }])
        states = {a["name"]: a["reason"] for a in actions if a["action"] == "retire"}
        assert "bad_health" in states
        assert "bad_guardrail" in states
        assert lm.get("bad_health")["state"] == "retired"
        assert lm.get("bad_guardrail")["state"] == "retired"
        Path(db).unlink()


class TestVersionManager:
    def test_dedup_same_params(self):
        from workspace.strategy_versioning import VersionManager
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        vm = VersionManager(db_path=db)
        v1 = vm.save_version("s1", params={"a": 1})
        v2 = vm.save_version("s1", params={"a": 1})  # same params
        assert v1["version"] == v2["version"]  # deduped
        Path(db).unlink()

    def test_diff_detects_changes(self):
        from workspace.strategy_versioning import VersionManager
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            db = f.name
        vm = VersionManager(db_path=db)
        vm.save_version("s1", params={"a": 1, "b": 2})
        vm.save_version("s1", params={"a": 1, "b": 3})
        diff = vm.diff("s1", 1, 2)
        assert diff["params_changed"]
        assert "b" in diff["param_changes"]
        Path(db).unlink()


class TestSignalValidator:
    def test_validates_pairs_signal(self):
        from workspace.signal_validator import validate_pairs_signal
        r = validate_pairs_signal("BTC/USDT", "ETH/USDT", exchange="gate")
        assert "ic" in r
        assert "hit_rate" in r
        assert "passed" in r

    def test_rejects_insufficient_data(self):
        from workspace.signal_validator import validate_signal
        import pandas as pd
        r = validate_signal(pd.Series([1, 2, 3]), pd.Series([0.1, 0.2, 0.3]), min_signals=100)
        assert not r["passed"]


class TestTypedModels:
    def test_backtest_result_round_trip(self):
        from workspace.types import BacktestResult
        br = BacktestResult(ok=True, sharpe=0.72, profit_pct=10.98, trades=34)
        d = br.to_dict()
        assert d["sharpe"] == 0.72
        br2 = BacktestResult.from_dict(d)
        assert br2.sharpe == 0.72

    def test_gate_pipeline_result(self):
        from workspace.types import GatePipelineResult
        r = GatePipelineResult(final_gate_passed="gate_3", recommendation="PASSED")
        assert r.approved
