from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


pytest.importorskip("pypfopt")


def test_flow_portfolio_step_produces_report_and_api_can_read_latest() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_kucoin_cpu_nollm_portfolio.json"
    script = root / "scripts" / "agent_flow.py"

    from agent_market.demo_data import ensure_demo_ohlcv_from_flow_config

    ensure_demo_ohlcv_from_flow_config(root, cfg)

    cmd = [sys.executable, str(script), "--config", str(cfg), "--steps", "portfolio"]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    meta_path = root / "artifacts" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = meta.get("artifacts") or {}

    report_rel = artifacts.get("portfolio_report")
    weights_rel = artifacts.get("portfolio_weights")
    assert isinstance(report_rel, str) and report_rel
    assert isinstance(weights_rel, str) and weights_rel
    assert (root / report_rel).exists()
    assert (root / weights_rel).exists()

    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    resp = client.get("/flow/portfolio/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("method") == "hrp"
    assert isinstance(body.get("weights"), dict)
    assert isinstance(body.get("stats"), dict)
    assert isinstance(body.get("inputs"), dict)

