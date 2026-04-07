from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

from agent_market.agent_flow import AgentFlow, AgentFlowConfig


def _prepare_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT_MARKET_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("AGENT_MARKET_USER_DATA_ROOT", str(tmp_path / "user_data"))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_agent_flow_preflight_sets_dummy_key_for_expression_proxy(monkeypatch, tmp_path: Path) -> None:
    from agent_market import runtime_preflight

    _prepare_env(monkeypatch, tmp_path)
    ft_cfg = tmp_path / "freqtrade.json"
    ft_cfg.write_text(
        '{"datadir":"user_data/data","exchange":{"pair_whitelist":["BTC/USDT"]},"timeframe":"5m"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runtime_preflight,
        "check_opencli",
        lambda: {"name": "system.opencli", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "check_freqtrade_cli",
        lambda: {"name": "system.freqtrade", "ok": True, "severity": "info", "detail": "freqtrade ok"},
    )
    monkeypatch.setattr(runtime_preflight, "load_project_dotenv", lambda: None)

    calls: list[tuple[str, str]] = []

    def _fake_request_json(url: str, *, api_key: str, timeout: float = 5.0):  # noqa: ARG001
        calls.append((url, api_key))
        return True, {"data": [{"id": "gpt-5.2"}]}, ""

    monkeypatch.setattr(runtime_preflight, "_request_json", _fake_request_json)

    cfg = AgentFlowConfig(
        expression={
            "args": [
                "--llm-enabled",
                "--llm-provider",
                "openai_compatible",
                "--llm-base-url",
                "http://127.0.0.1:4141/v1",
                "--llm-model",
                "gpt-5.2",
            ]
        },
        backtest={"config": str(ft_cfg)},
    )
    report = runtime_preflight.run_agent_flow_preflight(
        cfg,
        run_dir=tmp_path / "artifacts" / "runs" / "flow-run",
        requested_steps=["expression", "backtest"],
        raise_on_error=False,
    )

    assert report["ok"] is True
    assert report["applied_env"] == {"LLM_API_KEY": "_"}
    assert calls == [("http://127.0.0.1:4141/v1/models", "_")]
    assert runtime_preflight.os.environ["LLM_API_KEY"] == "_"
    assert (tmp_path / "artifacts" / "runs" / "flow-run" / "preflight.json").exists()


def test_agent_flow_run_records_preflight_metadata(tmp_path: Path) -> None:
    cfg = AgentFlowConfig(report={})
    flow = AgentFlow(cfg)

    report_payload = {
        "ok": True,
        "warnings": 1,
        "errors": 0,
        "applied_env": {"LLM_API_KEY": "_"},
    }

    with patch("agent_market.agent_flow.paths") as mock_paths:
        mock_paths.REPO_ROOT = Path(tmp_path)
        mock_paths.run_meta_latest_path.return_value = Path(tmp_path) / "meta_latest.json"
        mock_paths.run_meta_path.return_value = Path(tmp_path) / "runs" / "flow" / "meta.json"
        mock_paths.models_root.return_value = Path(tmp_path) / "models"
        mock_paths.resolve_repo_path.side_effect = lambda x: Path(tmp_path) / str(x)
        mock_paths.relpath_under_repo.side_effect = lambda x: str(x)
        mock_paths.default_feedback_path.return_value = "feedback.json"
        with patch("agent_market.agent_flow.run_agent_flow_preflight", return_value=report_payload):
            with patch("agent_market.agent_flow.STEP_HANDLERS", {"report": lambda cfg, arts, ctx: None}):
                run_id = flow.run(steps=["report"])

    meta = json.loads((Path(tmp_path) / "runs" / "flow" / "meta.json").read_text(encoding="utf-8"))
    assert run_id
    assert meta["preflight"]["ok"] is True
    assert meta["preflight"]["warnings"] == 1
    assert meta["preflight"]["applied_env"] == {"LLM_API_KEY": "_"}


def test_tca_report_script_writes_preflight(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "tca_report.py"
    zip_path = tmp_path / "backtest-result.zip"
    payload = {
        "strategy": {
            "DemoStrategy": {
                "timeframe": "1h",
                "trades": [
                    {
                        "pair": "BTC/USDT",
                        "profit_abs": 1.0,
                        "profit_ratio": 0.01,
                        "fee_open": 0.1,
                        "fee_close": 0.1,
                        "funding_fees": 0.0,
                        "trade_duration": 60,
                    }
                ],
            }
        },
        "strategy_comparison": [],
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("backtest-result.json", json.dumps(payload))

    out_path = tmp_path / "out" / "tca_report.json"
    subprocess.run(  # noqa: S603,S607
        [
            sys.executable,
            str(script),
            "--run-id",
            "abcd1234",
            "--backtest-zip",
            str(zip_path),
            "--out",
            str(out_path),
        ],
        cwd=str(root),
        check=True,
    )

    preflight = json.loads((out_path.parent / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["kind"] == "tca"
    assert preflight["ok"] is True


def test_ws_production_preflight_reports_ready(monkeypatch, tmp_path: Path) -> None:
    from agent_market import runtime_preflight

    workspace = tmp_path / "ws_production"
    for rel in ("strategies", "results", "reports", "validation", "backtests", "paper", "signals"):
        (workspace / rel).mkdir(parents=True, exist_ok=True)
    (workspace / "GUIDE.md").write_text("# guide\n", encoding="utf-8")
    (workspace / "meta.json").write_text("{}\n", encoding="utf-8")
    (workspace / "sop.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_preflight,
        "check_opencli",
        lambda: {"name": "system.opencli", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "check_freqtrade_cli",
        lambda: {"name": "system.freqtrade", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "check_opencode_cli",
        lambda **_kwargs: {"name": "system.opencode", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "check_python_imports",
        lambda *_args, **_kwargs: {"name": "ws.python_imports", "ok": True, "severity": "info", "detail": "ok"},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "check_openai_compatible",
        lambda **_kwargs: {"name": "llm.openai_compatible", "ok": True, "severity": "info", "detail": "ok"},
    )

    report = runtime_preflight.run_ws_production_preflight(
        workspace=workspace,
        model="openai/gpt-5.2",
        raise_on_error=False,
    )

    assert report["ok"] is True
    assert (workspace / "results" / "preflight.json").exists()


def test_runtime_preflight_normalizes_ws_model_and_env(monkeypatch) -> None:
    from agent_market.runtime_preflight import normalize_opencode_model, resolve_openai_compatible_settings

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy.internal:4141/v1")
    monkeypatch.setenv("LLM_API_KEY", "_")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert normalize_opencode_model("gpt-5.2") == "openai/gpt-5.2"
    settings = resolve_openai_compatible_settings(model="gpt-5.2")
    assert settings["opencode_model"] == "openai/gpt-5.2"
    assert settings["model_id"] == "gpt-5.2"
    assert settings["base_url"] == "http://proxy.internal:4141/v1"
    assert settings["api_key"] == "_"
