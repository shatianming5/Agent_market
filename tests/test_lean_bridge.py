from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest

from agent_market.factor_lab import lean_bridge
from agent_market.factor_lab.rank_portfolio import RiskConfig


def _dates(periods: int = 4, timeframe: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=periods, freq=timeframe, tz="UTC")


def _write_okx_feather(root, pair: str, dates: pd.DatetimeIndex, *, duplicate: bool = False) -> None:
    token = lean_bridge.pair_file_token(pair)
    frame = pd.DataFrame(
        {
            "date": list(dates),
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1000.0 + i for i in range(len(dates))],
        }
    )
    if duplicate:
        frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_feather(root / f"{token}-1h-futures.feather")


def _signal_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for i, date in enumerate(dates):
        for pair, weight in (("BTC/USDT", 0.5), ("ETH/USDT", -0.5)):
            rows.append(
                {
                    "date": date,
                    "pair": pair,
                    "rp_target_weight": weight if i < len(dates) - 1 else 0.0,
                    "rp_rebalance": True,
                    "rp_exit_long": False,
                    "rp_exit_short": False,
                    "rp_liq_reject": False,
                    "rp_kill_mode": "normal",
                    "rp_leverage": 2.0,
                    "rp_stop_pct": 0.02,
                    "rp_liq_distance": 0.2,
                    "rp_side": 1 if weight > 0 else -1,
                    "rp_rank": 1,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1000.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _write_rank_artifact(tmp_path, signals: pd.DataFrame, *, venue: str = "okx") -> tuple:
    signal_dir = tmp_path / "rank" / "signals"
    signal_dir.mkdir(parents=True)
    signal_path = signal_dir / "all.feather"
    signals.to_feather(signal_path)
    risk = RiskConfig(
        profile="aggressive",
        gross_cap=1.0,
        net_cap=1.0,
        single_pair_cap=1.0,
        fee_rate=0.0005,
        slippage=0.0003,
        top_k=1,
        timeframe="1h",
    )
    artifact = {
        "tag": "unit",
        "venue": venue,
        "timeframe": "1h",
        "data_venue": venue,
        "risk_config": asdict(risk),
        "signals": {"all": str(signal_path)},
    }
    artifact_path = tmp_path / "rank" / "rank_export.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact_path, signal_path


def test_export_project_from_rank_artifact_writes_lean_files(tmp_path) -> None:
    dates = _dates()
    data_root = tmp_path / "okx"
    _write_okx_feather(data_root, "BTC/USDT", dates)
    _write_okx_feather(data_root, "ETH/USDT", dates)
    artifact_path, _ = _write_rank_artifact(tmp_path, _signal_frame(dates))

    manifest = lean_bridge.export_project(
        rank_artifact=artifact_path,
        output=tmp_path / "lean" / "unit",
        data_root=data_root,
    )

    project = tmp_path / "lean" / "unit"
    assert (project / "main.py").exists()
    assert (project / "config.json").exists()
    assert (project / "manifest.json").exists()
    assert (project / "data" / "signals.csv").exists()
    assert (project / "data" / "ohlcv" / "BTCUSDT.csv").exists()
    assert manifest["local_only"] is True
    assert manifest["quantconnect_cloud"] is False
    assert manifest["pairs"][0]["pair"] == "BTC/USDT"
    signal_csv = pd.read_csv(project / "data" / "signals.csv")
    assert set(signal_csv["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert "lean_target_weight" in signal_csv.columns
    assert "lean_action" in signal_csv.columns
    main_py = (project / "main.py").read_text(encoding="utf-8")
    assert "self.SetTimeZone(TimeZones.Utc)" in main_py
    assert "self.SetBenchmark(first_symbol)" in main_py
    assert "self.SetBenchmark(lambda _: 0)" in main_py
    assert "LocalFuturesOhlcv" in main_py
    assert "SetMarketPrice(point)" in main_py
    assert "def OnEndOfAlgorithm(self):\n        self.Debug" in main_py
    assert "Bridge Exposure" in main_py


def test_export_project_uses_bybit_artifact_venue_without_okx_fallback(tmp_path) -> None:
    dates = _dates()
    bybit_root = tmp_path / "bybit"
    okx_root = tmp_path / "okx"
    _write_okx_feather(bybit_root, "BTC/USDT", dates)
    _write_okx_feather(bybit_root, "ETH/USDT", dates)
    artifact_path, _ = _write_rank_artifact(tmp_path, _signal_frame(dates), venue="bybit")

    manifest = lean_bridge.export_project(
        rank_artifact=artifact_path,
        output=tmp_path / "lean" / "bybit_unit",
        data_root=bybit_root,
    )

    assert manifest["venue"] == "bybit"
    assert manifest["data_venue"] == "bybit"
    assert manifest["coverage"]["venue"] == "bybit"
    assert all(str(bybit_root) in item["path"] for item in manifest["coverage"]["files"].values())

    with pytest.raises(ValueError, match="missing OHLCV"):
        lean_bridge.export_project(
            rank_artifact=artifact_path,
            output=tmp_path / "lean" / "bybit_missing",
            data_root=okx_root,
        )


def test_timeframe_override_must_match_artifact(tmp_path) -> None:
    dates = _dates()
    artifact_path, _ = _write_rank_artifact(tmp_path, _signal_frame(dates))

    with pytest.raises(ValueError, match="does not match artifact timeframe"):
        lean_bridge.load_rank_artifact(artifact_path, timeframe="15m")


def test_coverage_preflight_rejects_duplicate_ohlcv_timestamp(tmp_path) -> None:
    dates = _dates()
    data_root = tmp_path / "okx"
    _write_okx_feather(data_root, "BTC/USDT", dates, duplicate=True)
    signals = _signal_frame(dates).loc[lambda df: df["pair"] == "BTC/USDT"].copy()
    signals = lean_bridge.load_signals(_write_rank_artifact(tmp_path, signals)[1])

    with pytest.raises(ValueError, match="duplicate OHLCV timestamps"):
        lean_bridge.preflight_coverage(signals, timeframe="1h", data_root=data_root)


def test_coverage_preflight_rejects_missing_file_and_gap(tmp_path) -> None:
    dates = _dates()
    signals = lean_bridge.load_signals(_write_rank_artifact(tmp_path, _signal_frame(dates))[1])

    # With skip_gap_pairs=False the old strict behaviour is preserved.
    with pytest.raises(ValueError, match="missing OHLCV"):
        lean_bridge.preflight_coverage(signals, timeframe="1h", data_root=tmp_path / "empty_okx", skip_gap_pairs=False)

    data_root = tmp_path / "okx_gap"
    _write_okx_feather(data_root, "BTC/USDT", dates.delete(1))
    _write_okx_feather(data_root, "ETH/USDT", dates)

    with pytest.raises(ValueError, match="missing signal timestamps|OHLCV gap|OHLCV coverage issue"):
        lean_bridge.preflight_coverage(signals, timeframe="1h", data_root=data_root, skip_gap_pairs=False)

    # Default (skip_gap_pairs=True): gap pair is excluded rather than raising.
    cov = lean_bridge.preflight_coverage(signals, timeframe="1h", data_root=data_root)
    assert "BTC/USDT" in cov.get("excluded_pairs", {})


def test_normalize_signal_targets_holds_until_exit_liq_or_kill(tmp_path) -> None:
    dates = _dates(periods=7)
    signals = pd.DataFrame(
        {
            "date": dates,
            "pair": ["BTC/USDT"] * len(dates),
            "rp_target_weight": [0.5, 0.0, 0.0, 0.4, 0.4, 0.3, 0.3],
            "rp_rebalance": [True, False, False, True, False, True, False],
            "rp_exit_long": [False, False, True, False, False, False, False],
            "rp_exit_short": [False] * len(dates),
            "rp_liq_reject": [False, False, False, False, True, False, False],
            "rp_kill_mode": ["normal", "normal", "normal", "normal", "normal", "normal", "daily_halt"],
        }
    )

    out = lean_bridge.normalize_signal_targets(lean_bridge.load_signals(_write_rank_artifact(tmp_path, signals)[1]))

    assert out["lean_target_weight"].tolist() == [0.5, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0]
    assert out["lean_force_flat"].tolist() == [False, False, True, False, True, False, True]
    assert out["lean_action"].tolist() == [True, False, True, True, True, True, True]


def test_normalize_signal_targets_does_not_trade_same_target_rebalance(tmp_path) -> None:
    dates = _dates(periods=3)
    signals = pd.DataFrame(
        {
            "date": dates,
            "pair": ["BTC/USDT"] * len(dates),
            "rp_target_weight": [0.5, 0.5, 0.0],
            "rp_rebalance": [True, True, True],
            "rp_exit_long": [False, False, False],
            "rp_exit_short": [False, False, False],
            "rp_liq_reject": [False, False, False],
            "rp_kill_mode": ["normal", "normal", "normal"],
        }
    )

    out = lean_bridge.normalize_signal_targets(lean_bridge.load_signals(_write_rank_artifact(tmp_path, signals)[1]))

    assert out["lean_target_weight"].tolist() == [0.5, 0.5, 0.0]
    assert out["lean_target_delta"].tolist() == [0.5, 0.0, -0.5]
    assert out["lean_action"].tolist() == [True, False, True]


def test_export_project_includes_next_execution_bar(tmp_path) -> None:
    dates = _dates(periods=4)
    data_root = tmp_path / "okx"
    _write_okx_feather(data_root, "BTC/USDT", dates)
    signals = _signal_frame(dates[:3]).loc[lambda df: df["pair"] == "BTC/USDT"].copy()
    artifact_path, _ = _write_rank_artifact(tmp_path, signals)

    lean_bridge.export_project(
        rank_artifact=artifact_path,
        output=tmp_path / "lean" / "next_bar_unit",
        data_root=data_root,
    )

    ohlcv_csv = pd.read_csv(tmp_path / "lean" / "next_bar_unit" / "data" / "ohlcv" / "BTCUSDT.csv")
    assert ohlcv_csv["time"].tolist() == [
        "2026-01-01 00:00:00",
        "2026-01-01 01:00:00",
        "2026-01-01 02:00:00",
        "2026-01-01 03:00:00",
    ]


def test_research_execution_stats_ignore_terminal_signal_without_next_bar(tmp_path) -> None:
    dates = _dates(periods=3)
    signals = pd.DataFrame(
        {
            "date": dates,
            "pair": ["BTC/USDT"] * len(dates),
            "rp_target_weight": [0.0, 0.5, 0.0],
            "rp_rebalance": [True, True, True],
            "rp_exit_long": [False, False, False],
            "rp_exit_short": [False, False, False],
            "rp_liq_reject": [False, False, False],
            "rp_kill_mode": ["normal", "normal", "normal"],
        }
    )
    loaded = lean_bridge.load_signals(_write_rank_artifact(tmp_path, signals)[1])

    stats = lean_bridge._signal_turnover_stats(loaded, RiskConfig(timeframe="1h"))  # noqa: SLF001

    assert stats["orders"] == 1.0
    assert stats["entries"] == 1.0
    assert stats["exits"] == 0.0
    assert stats["turnover"] == 0.5


def test_compare_results_writes_status_ok_for_matching_metrics(tmp_path) -> None:
    dates = _dates()
    data_root = tmp_path / "okx"
    _write_okx_feather(data_root, "BTC/USDT", dates)
    _write_okx_feather(data_root, "ETH/USDT", dates)
    artifact_path, _ = _write_rank_artifact(tmp_path, _signal_frame(dates))
    lean_bridge.export_project(rank_artifact=artifact_path, output=tmp_path / "lean" / "unit", data_root=data_root)
    research, _ = lean_bridge.research_metrics_from_artifact(artifact_path)
    lean_result = tmp_path / "lean" / "unit" / "results.json"
    lean_result.write_text(json.dumps({"statistics": research}), encoding="utf-8")

    report = lean_bridge.compare_results(rank_artifact=artifact_path, lean_result=lean_result)

    assert report["status"] == "ok"
    assert (tmp_path / "lean" / "unit" / "comparison.json").exists()


def test_run_lean_backtest_accepts_relative_binary_path(tmp_path, monkeypatch) -> None:
    project = tmp_path / "lean_project"
    project.mkdir()
    binary = tmp_path / "lean"
    binary.write_text("#!/bin/sh\nprintf '{\"statistics\":{}}' > \"$2/123-summary.json\"\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    monkeypatch.chdir(tmp_path)
    result = lean_bridge.run_lean_backtest(
        lean_project=project,
        lean_bin=str(binary),
        timeout=5,
    )

    assert result["returncode"] == 0
    assert result["command"][0] == str(binary)


def test_run_lean_backtest_uses_repo_lean_config_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    project = tmp_path / "lean_project"
    project.mkdir()
    lean_config = tmp_path / "artifacts" / "lean" / "lean.json"
    lean_config.parent.mkdir(parents=True)
    lean_config.write_text("{}", encoding="utf-8")
    binary = tmp_path / "lean"
    binary.write_text("#!/bin/sh\nprintf '{\"statistics\":{}}' > \"$2/123-summary.json\"\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    result = lean_bridge.run_lean_backtest(
        lean_project=project,
        lean_bin=str(binary),
        timeout=5,
    )

    assert "--lean-config" in result["command"]
    assert str(lean_config.resolve()) in result["command"]


def test_run_lean_backtest_fails_closed_on_nonzero_cli_exit(tmp_path, monkeypatch) -> None:
    project = tmp_path / "lean_project"
    project.mkdir()
    binary = tmp_path / "lean"
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run_no_result(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(lean_bridge.subprocess, "run", fake_run_no_result)

    # returncode=1 with no result file → must raise
    with pytest.raises(RuntimeError, match="returncode=1"):
        lean_bridge.run_lean_backtest(
            lean_project=project,
            lean_bin=str(binary),
            timeout=5,
        )

    # returncode=1 but a result file was produced → treat as success (post-processing warnings)
    result_json = project / "123-summary.json"
    result_json.write_text(json.dumps({"statistics": {"End Equity": "100000"}}), encoding="utf-8")

    def fake_run_with_result(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="Engine.Main(): Analysis Complete.", stderr="")

    monkeypatch.setattr(lean_bridge.subprocess, "run", fake_run_with_result)

    summary = lean_bridge.run_lean_backtest(
        lean_project=project,
        lean_bin=str(binary),
        timeout=5,
    )
    assert summary["ok"] is True
    assert summary["returncode"] == 1

    run = json.loads((project / "lean_backtest_run.json").read_text(encoding="utf-8"))
    assert run["returncode"] == 1
    assert run["ok"] is True
    assert run["result_path"] == str(result_json.resolve())


def test_parse_lean_summary_normalizes_cash_statistics(tmp_path) -> None:
    summary = tmp_path / "123-summary.json"
    summary.write_text(
        json.dumps(
            {
                "statistics": {
                    "Start Equity": "100000",
                    "End Equity": "161689.82",
                    "Net Profit": "61.690%",
                    "Drawdown": "4.500%",
                    "Total Orders": "446",
                    "Total Fees": "$9465.24",
                    "Portfolio Turnover": "109.61%",
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = lean_bridge.parse_lean_metrics(summary)

    assert metrics["final_equity"] == pytest.approx(1.6168982)
    assert metrics["total_return"] == pytest.approx(0.6169)
    assert metrics["max_drawdown"] == pytest.approx(0.045)
    assert metrics["orders"] == pytest.approx(446)
    assert metrics["trades"] is None
    assert metrics["fee_cost"] == pytest.approx(0.0946524)
    assert metrics["turnover"] == pytest.approx(1.0961)
    assert metrics["summary_portfolio_turnover"] == pytest.approx(1.0961)


def test_parse_lean_order_events_reconstructs_comparable_execution_metrics(tmp_path) -> None:
    prefix = tmp_path / "123"
    (tmp_path / "123-summary.json").write_text(
        json.dumps(
            {
                "statistics": {
                    "Start Equity": "100000",
                    "End Equity": "101000",
                    "Net Profit": "1.000%",
                    "Drawdown": "2.000%",
                    "Total Orders": "2",
                    "Total Fees": "$1.05",
                    "Portfolio Turnover": "123.00%",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "123.json").write_text(
        json.dumps(
            {
                "charts": {
                    "Strategy Equity": {
                        "series": {
                            "Equity": {
                                "values": [
                                    [1000, 100000.0],
                                    [2000, 105000.0],
                                ]
                            }
                        }
                    },
                    "Bridge Exposure": {
                        "series": {
                            "Gross": {
                                "values": [
                                    [1000, 0.5],
                                    [2000, 0.25],
                                ]
                            }
                        }
                    },
                },
                "orders": {
                    "1": {
                        "id": 1,
                        "status": 3,
                        "quantity": 10,
                        "price": 100,
                        "lastFillTime": "1970-01-01T00:16:40Z",
                        "symbol": {"value": "BTCUSDT"},
                        "orderSubmissionData": {"lastPrice": 99.9},
                    },
                    "2": {
                        "id": 2,
                        "status": 3,
                        "quantity": -10,
                        "price": 110,
                        "lastFillTime": "1970-01-01T00:33:20Z",
                        "symbol": {"value": "BTCUSDT"},
                        "orderSubmissionData": {"lastPrice": 110.1},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "123-order-events.json").write_text(
        json.dumps(
            [
                {
                    "orderId": 1,
                    "time": 1000,
                    "status": "filled",
                    "symbolValue": "BTCUSDT",
                    "fillQuantity": 10,
                    "fillPrice": 100,
                    "orderFeeAmount": 0.5,
                },
                {
                    "orderId": 2,
                    "time": 2000,
                    "status": "filled",
                    "symbolValue": "BTCUSDT",
                    "fillQuantity": -10,
                    "fillPrice": 110,
                    "orderFeeAmount": 0.55,
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = lean_bridge.parse_lean_metrics(prefix.with_name("123-summary.json"))

    assert metrics["orders"] == pytest.approx(2)
    assert metrics["trades"] == pytest.approx(1)
    assert metrics["entries"] == pytest.approx(1)
    assert metrics["exits"] == pytest.approx(1)
    assert metrics["turnover"] == pytest.approx((1000 / 100000) + (1100 / 105000))
    assert metrics["turnover_start_equity"] == pytest.approx(0.021)
    assert metrics["fee_cost"] == pytest.approx((0.5 / 100000) + (0.55 / 105000))
    assert metrics["slippage_cost"] == pytest.approx((1.0 / 100000) + (1.0 / 105000))
    assert metrics["summary_portfolio_turnover"] == pytest.approx(1.23)
    assert metrics["avg_gross"] == pytest.approx(0.375)
    assert metrics["max_gross"] == pytest.approx(0.5)
    assert metrics["execution_metric_source"] == "order_events"
