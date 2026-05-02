from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_market import paths as repo_paths
from agent_market.factor_lab import mine_lean_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_state(path: Path) -> Path:
    _write_json(
        path,
        {
            "survivors": [
                {
                    "expression": "close",
                    "origin": "unit",
                    "neutralized_ic": 0.02,
                    "oos_ic": 0.02,
                    "sign_agree": 8,
                    "residual_ic_ratio": 1.0,
                }
            ]
        },
    )
    return path


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, *, comparison_status: str = "ok") -> dict:
    captured: dict = {}

    def fake_rank_backtest(**kwargs):
        captured["rank_kwargs"] = kwargs
        root = repo_paths.artifacts_root() / "rank_portfolio" / str(kwargs["tag"])
        signal_dir = root / "signals"
        signal_dir.mkdir(parents=True, exist_ok=True)
        signal_path = signal_dir / "all.feather"
        signal_path.write_bytes(b"unit")
        _write_json(
            root / "backtest.json",
            {
                "tag": kwargs["tag"],
                "venue": kwargs["venue"],
                "timeframe": kwargs["timeframe"],
                "data_venue": kwargs["data_venue"],
                "signals": {"all": str(signal_path)},
                "risk_config": {"profile": "aggressive", "timeframe": kwargs["timeframe"]},
            },
        )
        return {"tag": kwargs["tag"], "signals": str(signal_path), "trades": 100}

    def fake_export_project(*, rank_artifact, output, timeframe=None, data_root=None):
        captured["export_output"] = Path(output)
        project = Path(output)
        project.mkdir(parents=True, exist_ok=True)
        signal_dir = project / "data"
        signal_dir.mkdir(parents=True, exist_ok=True)
        (signal_dir / "signals.csv").write_text(
            "time,symbol,lean_target_weight\n"
            "2026-01-01T00:00:00Z,BTCUSDT,0.5\n"
            "2026-01-01T04:00:00Z,BTCUSDT,0.0\n",
            encoding="utf-8",
        )
        _write_json(project / "manifest.json", {"rank_artifact": str(rank_artifact), "timeframe": timeframe, "data_root": str(data_root or "")})
        return {"rank_artifact": str(rank_artifact), "timeframe": timeframe, "data_root": str(data_root or "")}

    def fake_run_lean_backtest(*, lean_project, lean_bin="lean", timeout=None):
        result = Path(lean_project) / "results.json"
        _write_json(result, {"statistics": {"unit": True}})
        _write_json(Path(lean_project) / "lean_backtest_run.json", {"command": [lean_bin, "backtest", str(lean_project)], "timeout": timeout, "returncode": 0, "ok": True, "result_path": str(result)})
        return {"command": [lean_bin, "backtest", str(lean_project)], "returncode": 0, "ok": True, "result_path": str(result)}

    def fake_compare_results(*, rank_artifact, lean_result, output=None, timeframe=None):
        lean = {
            "final_equity": 1.15,
            "max_drawdown": 0.08,
            "trades": 120.0,
            "orders": 12.0,
            "turnover": 1.0,
            "max_gross": 1.0,
            "fee_cost": 0.01,
            "ending_open_positions": 0.0,
        }
        metrics = {
            field: {"status": "ok", "research": lean[field], "lean": lean[field], "threshold": 0.05}
            for field in ("final_equity", "max_drawdown", "trades", "orders", "turnover")
        }
        report = {
            "status": comparison_status,
            "metrics": metrics,
            "lean": lean,
            "research": dict(lean),
            "rank_artifact": str(rank_artifact),
            "lean_result": str(lean_result),
            "timeframe": timeframe,
        }
        if output:
            _write_json(Path(output), report)
        return report

    monkeypatch.setattr(mine_lean_gate.rank_portfolio, "rank_backtest", fake_rank_backtest)
    monkeypatch.setattr(mine_lean_gate.lean_bridge, "export_project", fake_export_project)
    monkeypatch.setattr(mine_lean_gate.lean_bridge, "run_lean_backtest", fake_run_lean_backtest)
    monkeypatch.setattr(mine_lean_gate.lean_bridge, "compare_results", fake_compare_results)
    return captured


def test_mine_lean_gate_passes_and_records_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    captured = _install_fake_pipeline(monkeypatch, comparison_status="ok")
    state = _candidate_state(tmp_path / "state.json")

    result = mine_lean_gate.run_mine_lean_gate(
        tag="unit_mine",
        candidate_state=state,
        run_id="unit_run",
        venue="binance",
        data_venue="binance",
        timeframe="4h",
        pairs="BTC/USDT,ETH/USDT",
        lean_bin="missing-lean-for-unit-test",
        min_trades=10,
        output=tmp_path / "gate",
    )

    assert result["status"] == mine_lean_gate.STATUS_PASSED
    assert result["comparison_status"] == "ok"
    assert result["candidate_state"]["candidate_count"] == 1
    assert result["artifacts"]["rank_artifact"].endswith("backtest.json")
    assert result["artifacts"]["comparison_json"].endswith("comparison.json")
    assert result["expected_ending_open_positions"]["expected"] == 0
    assert captured["rank_kwargs"]["pairs"] == "BTC/USDT,ETH/USDT"
    assert captured["rank_kwargs"]["candidate_state"] == tmp_path / "gate" / "candidate_state.json"
    assert captured["rank_kwargs"]["timeframe"] == "4h"
    assert (tmp_path / "gate" / "mine_lean_gate.json").exists()


def test_mine_lean_gate_uses_lean_workspace_project_when_config_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(artifacts))
    lean_config = artifacts / "lean" / "lean.json"
    lean_config.parent.mkdir(parents=True)
    lean_config.write_text("{}", encoding="utf-8")
    captured = _install_fake_pipeline(monkeypatch, comparison_status="ok")
    state = _candidate_state(tmp_path / "state.json")

    result = mine_lean_gate.run_mine_lean_gate(
        tag="unit_mine",
        candidate_state=state,
        run_id="unit_run",
        timeframe="4h",
        lean_bin="missing-lean-for-unit-test",
        min_trades=10,
        output=tmp_path / "gate",
        force=True,
    )

    assert result["status"] == mine_lean_gate.STATUS_PASSED
    assert captured["export_output"] == artifacts / "lean" / "bridge_projects" / "unit_mine" / "unit_run" / "lean_project"
    assert result["artifacts"]["lean_project"].endswith("artifacts/lean/bridge_projects/unit_mine/unit_run/lean_project")


def test_mine_lean_gate_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    _install_fake_pipeline(monkeypatch, comparison_status="drift")
    state = _candidate_state(tmp_path / "state.json")

    result = mine_lean_gate.run_mine_lean_gate(
        tag="unit_mine",
        candidate_state=state,
        run_id="unit_drift",
        output=tmp_path / "gate",
        lean_bin="missing-lean-for-unit-test",
        min_trades=10,
    )

    assert result["status"] == mine_lean_gate.STATUS_FAILED
    assert "status='drift'" in result["reason"]
    assert json.loads((tmp_path / "gate" / "mine_lean_gate.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_mine_lean_gate_fails_closed_when_backtest_run_not_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    _install_fake_pipeline(monkeypatch, comparison_status="ok")

    def fake_run_lean_backtest(*, lean_project, lean_bin="lean", timeout=None):
        result = Path(lean_project) / "results.json"
        _write_json(result, {"statistics": {"unit": True}})
        _write_json(Path(lean_project) / "lean_backtest_run.json", {"returncode": 1, "ok": False, "result_path": str(result)})
        return {"returncode": 1, "ok": False, "result_path": str(result)}

    monkeypatch.setattr(mine_lean_gate.lean_bridge, "run_lean_backtest", fake_run_lean_backtest)
    state = _candidate_state(tmp_path / "state.json")

    result = mine_lean_gate.run_mine_lean_gate(
        tag="unit_mine",
        candidate_state=state,
        run_id="unit_backtest_fail",
        output=tmp_path / "gate",
        lean_bin="missing-lean-for-unit-test",
        min_trades=10,
    )

    assert result["status"] == mine_lean_gate.STATUS_FAILED
    assert "LEAN backtest failed: returncode=1 ok=False" in result["reason"]
    assert "comparison_status" not in result


def test_assess_comparison_requires_core_metrics() -> None:
    result = mine_lean_gate.assess_comparison({"status": "ok", "metrics": {}, "lean": {}}, min_trades=1)

    assert result["status"] == mine_lean_gate.STATUS_FAILED
    assert "LEAN final_equity missing" in result["violations"]
    assert "LEAN comparison metric missing: trades" in result["violations"]


def test_assess_comparison_fails_on_unexpected_open_positions() -> None:
    lean = {
        "final_equity": 1.2,
        "max_drawdown": 0.05,
        "trades": 10.0,
        "orders": 12.0,
        "turnover": 1.0,
        "max_gross": 1.0,
        "fee_cost": 0.01,
        "ending_open_positions": 1.0,
    }
    report = {
        "status": "ok",
        "metrics": {
            field: {"status": "ok", "research": lean[field], "lean": lean[field], "threshold": 0.05}
            for field in ("final_equity", "max_drawdown", "trades", "orders", "turnover")
        },
        "research": {"max_gross": 1.0},
        "lean": lean,
    }

    result = mine_lean_gate.assess_comparison(
        report,
        min_trades=1,
        expected_positions={"expected": 0, "latest_time": "2026-01-01T04:00:00Z", "nonzero_symbols": []},
    )

    assert result["status"] == mine_lean_gate.STATUS_FAILED
    assert "LEAN ending_open_positions=1 > expected 0" in result["violations"]
