from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from agent_market.backtest_results import build_backtest_summary, read_backtest_trades


def _write_backtest_zip(path: Path) -> None:
    payload = {
        "strategy": {
            "TestModel": {
                "profit_total_pct": 16.43,
                "profit_total_abs": 164.33795843,
                "profit_mean_pct": 0.21,
                "profit_factor": 2.8652525388306396,
                "expectancy": 0.1565,
                "expectancy_ratio": 0.3609,
                "sqn": 14.4124,
                "cagr": 6.8215,
                "sharpe": 330.61301136060723,
                "sortino": 332.74650982848874,
                "calmar": 3495.9492965971313,
                "max_drawdown_account": 0.003326254156616564,
                "max_drawdown_abs": 3.8829077199999915,
                "backtest_days": 29,
                "starting_balance": 1000.0,
                "final_balance": 1164.33795843,
                "best_pair": {"key": "ETH/USDT"},
                "worst_pair": {"key": "XRP/USDT"},
                "backtest_start": "2026-03-01 06:25:00",
                "backtest_end": "2026-03-29 00:00:00",
                "daily_profit": [
                    ["2026-03-01", 20.0],
                    ["2026-03-02", 15.0],
                    ["2026-03-03", -1.0],
                    ["2026-03-04", 10.0],
                    ["2026-03-05", -0.5],
                ],
                "trades": [
                    {
                        "pair": "ETH/USDT",
                        "stake_amount": 75.0,
                        "profit_ratio": 0.01,
                        "profit_abs": 0.75,
                        "fee_open": 0.001,
                        "fee_close": 0.001,
                    },
                    {
                        "pair": "BTC/USDT",
                        "stake_amount": 75.0,
                        "profit_ratio": -0.002,
                        "profit_abs": -0.15,
                        "fee_open": 0.001,
                        "fee_close": 0.001,
                    },
                ],
                "results_per_pair": [
                    {"key": "ETH/USDT", "profit_total_pct": 3.0},
                    {"key": "BTC/USDT", "profit_total_pct": 1.0},
                ],
            }
        },
        "strategy_comparison": [
            {
                "key": "TestModel",
                "trades": 2,
                "profit_total_pct": 16.43,
                "winrate": 0.5,
                "profit_factor": 2.8652525388306396,
                "sharpe": 330.61301136060723,
                "sortino": 332.74650982848874,
                "calmar": 3495.9492965971313,
                "max_drawdown_account": 0.003326254156616564,
                "max_drawdown_abs": "3.883",
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("backtest-result-test.json", json.dumps(payload))
        zf.writestr("backtest-result-test_config.json", json.dumps({"ok": True}))


def test_build_backtest_summary_audits_native_metrics():
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "backtest-result-test.zip"
        _write_backtest_zip(zip_path)

        summary = build_backtest_summary(zip_path)

        assert summary["sharpe"] == 330.61301136060723
        assert summary["realistic_sharpe"] < 5.0
        assert summary["max_drawdown_pct"] > 0.3
        assert summary["max_drawdown_account"] == summary["max_drawdown_pct"]
        assert "native_sharpe_inflated" in (summary.get("metric_flags") or [])
        assert summary["metrics_trusted"] is False


def test_read_backtest_trades_falls_back_to_embedded_strategy_trades():
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "backtest-result-test.zip"
        _write_backtest_zip(zip_path)

        trades = read_backtest_trades(zip_path)

        assert isinstance(trades, list)
        assert len(trades) == 2
        assert trades[0]["pair"] == "ETH/USDT"
