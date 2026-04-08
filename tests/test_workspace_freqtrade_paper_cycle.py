from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def _write_backtest_zip(tmp_path: Path, *, strategy_name: str = "FreqtradeMLStrategy") -> tuple[Path, datetime]:
    zip_path = tmp_path / "backtest-result-test.zip"
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    day_one = now - timedelta(days=1)
    day_two = now
    payload = {
        "strategy": {
            strategy_name: {
                "starting_balance": 1000.0,
                "final_balance": 1030.0,
                "trades": [
                    {
                        "close_date": day_one.isoformat(sep=" "),
                        "profit_abs": 10.0,
                    },
                    {
                        "close_date": day_two.isoformat(sep=" "),
                        "profit_abs": 20.0,
                    },
                ],
            }
        }
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("backtest-result-test.json", json.dumps(payload))
    return zip_path, day_one


def test_freqtrade_paper_cycle_replays_daily_equity(tmp_path, monkeypatch) -> None:
    from workspace.freqtrade_paper_cycle import FreqtradePaperCycle

    strategy_path = tmp_path / "my_ml.py"
    strategy_path.write_text("class Demo: pass\n", encoding="utf-8")
    backtest_zip, start_day = _write_backtest_zip(tmp_path)

    def _fake_run_backtest(*args, **kwargs):
        return {
            "ok": True,
            "strategy": "FreqtradeMLStrategy",
            "trades": 2,
            "run_id": "bt_test",
            "backtest_zip": str(backtest_zip),
        }

    monkeypatch.setattr("workspace.freqtrade_paper_cycle.run_backtest", _fake_run_backtest)

    cycle = FreqtradePaperCycle(exchange="gate", timeframe="1h", paper_dir=tmp_path / "paper")
    result = cycle.run_strategy(
        "ml_demo",
        {
            "strategy_path": str(strategy_path),
            "paper_started_at": start_day.isoformat(),
            "timeframe": "1h",
            "exchange": "gate",
        },
    )

    assert result["ok"] is True
    assert result["paper_days"] >= 2
    assert result["current_equity"] >= 1030.0
    assert result["cumulative_return_pct"] >= 3.0
    assert result["state_path"].endswith("ml_demo.json")


def test_freqtrade_paper_cycle_uses_existing_started_at(tmp_path, monkeypatch) -> None:
    from workspace.freqtrade_paper_cycle import FreqtradePaperCycle

    strategy_path = tmp_path / "my_ml.py"
    strategy_path.write_text("class Demo: pass\n", encoding="utf-8")
    backtest_zip, _ = _write_backtest_zip(tmp_path)

    def _fake_run_backtest(*args, **kwargs):
        return {
            "ok": True,
            "strategy": "FreqtradeMLStrategy",
            "trades": 2,
            "run_id": "bt_test",
            "backtest_zip": str(backtest_zip),
        }

    monkeypatch.setattr("workspace.freqtrade_paper_cycle.run_backtest", _fake_run_backtest)

    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "ml_demo.json").write_text(
        json.dumps(
            {
                "started_at": "2026-04-07T00:00:00+00:00",
                "daily_equity": {"2026-04-07": 1000.0},
                "last_processed_at": "2026-04-07T23:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    cycle = FreqtradePaperCycle(exchange="gate", timeframe="1h", paper_dir=paper_dir)
    result = cycle.run_strategy(
        "ml_demo",
        {
            "strategy_path": str(strategy_path),
            "paper_started_at": "2026-04-08T00:00:00+00:00",
        },
    )

    assert result["ok"] is True
    assert result["new_days"] >= 1
    state = json.loads((paper_dir / "ml_demo.json").read_text(encoding="utf-8"))
    assert state["started_at"] == "2026-04-07T00:00:00+00:00"


def test_continuous_runner_routes_ml_to_freqtrade_cycle(monkeypatch) -> None:
    import workspace.strategy_lifecycle as lifecycle_mod
    import workspace.freqtrade_paper_cycle as ml_cycle_mod
    from workspace.continuous_runner import ContinuousRunner

    class _FakeLM:
        def __init__(self):
            self.synced = []

        def list_by_state(self, state):  # noqa: ANN001
            return [
                {
                    "name": "ml_demo",
                    "state": "paper",
                    "type": "ml",
                    "config": {"strategy_path": "workspace/strategies/type_F_ml/freqtrade_ml_strategy.py"},
                    "history": [{"state": "paper", "at": "2026-04-08T00:00:00+00:00"}],
                }
            ]

        def sync_paper_tracking(self, name, **kwargs):  # noqa: ANN001
            self.synced.append((name, kwargs))

    class _FakeMLCycle:
        def __init__(self, *args, **kwargs):  # noqa: ANN001
            pass

        def run_strategy(self, name, config):  # noqa: ANN001
            assert config["paper_started_at"] == "2026-04-08T00:00:00+00:00"
            return {
                "ok": True,
                "initialized": False,
                "backfilled": True,
                "new_days": 2,
                "orders_added": 3,
                "paper_days": 2,
                "current_equity": 1010.0,
                "cumulative_return_pct": 1.0,
                "latest_day_return_pct": 0.5,
                "last_processed_at": "2026-04-09T00:00:00+00:00",
                "state_path": "/tmp/ml_demo.json",
                "strategy_path": "workspace/strategies/type_F_ml/freqtrade_ml_strategy.py",
                "timerange": "20260408-20260409",
                "backtest_run_id": "bt_test",
                "backtest_zip": "user_data/backtest_results/test.zip",
                "daily_equity": {"2026-04-08": 1005.0, "2026-04-09": 1010.0},
            }

    fake_lm = _FakeLM()
    monkeypatch.setattr(lifecycle_mod, "LifecycleManager", lambda: fake_lm)
    monkeypatch.setattr(ml_cycle_mod, "FreqtradePaperCycle", _FakeMLCycle)

    runner = ContinuousRunner(exchange="gate")
    payload = runner._phase_paper_trading()

    assert payload["n_paper"] == 1
    assert payload["updated"] == 1
    assert payload["new_bars"] == 2
    assert payload["strategies"][0]["type"] == "ml"
    assert payload["strategies"][0]["backfilled"] is True
    assert fake_lm.synced[0][0] == "ml_demo"


def test_freqtrade_paper_cycle_caps_timerange_to_latest_available_data(tmp_path, monkeypatch) -> None:
    from workspace.freqtrade_paper_cycle import FreqtradePaperCycle

    strategy_path = tmp_path / "my_ml.py"
    strategy_path.write_text("class Demo: pass\n", encoding="utf-8")
    backtest_zip, _ = _write_backtest_zip(tmp_path)
    seen = {}

    def _fake_run_backtest(*args, **kwargs):
        seen["timerange"] = kwargs["timerange"]
        return {
            "ok": True,
            "strategy": "FreqtradeMLStrategy",
            "trades": 2,
            "run_id": "bt_test",
            "backtest_zip": str(backtest_zip),
        }

    monkeypatch.setattr("workspace.freqtrade_paper_cycle.run_backtest", _fake_run_backtest)
    monkeypatch.setattr(
        FreqtradePaperCycle,
        "_latest_available_candle",
        lambda self, config: pd.Timestamp("2025-04-14T09:00:00+00:00"),
    )

    cycle = FreqtradePaperCycle(exchange="gate", timeframe="1h", paper_dir=tmp_path / "paper")
    result = cycle.run_strategy(
        "ml_demo",
        {
            "strategy_path": str(strategy_path),
            "paper_started_at": "2025-03-15T00:00:00+00:00",
            "timeframe": "1h",
            "exchange": "gate",
            "pairs": ["BTC/USDT", "ETH/USDT"],
        },
    )

    assert result["ok"] is True
    assert seen["timerange"] == "20250315-20250414"
    assert result["last_processed_at"].startswith("2025-04-14T09:00:00")
