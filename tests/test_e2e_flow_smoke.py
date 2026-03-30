from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_MISSING_DEPS: list[str] = []
for _mod in ("lightgbm",):
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        _MISSING_DEPS.append(_mod)


@pytest.mark.skipif(bool(_MISSING_DEPS), reason=f"missing optional deps: {_MISSING_DEPS}")
def test_e2e_flow_produces_required_artifacts():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_kucoin_cpu_nollm_smoke.json"
    script = root / "scripts" / "e2e_smoke_flow.py"

    cmd = [sys.executable, str(script), "--config", str(cfg)]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    # UI "golden path" defaults: ensure /web points to the golden config.
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    resp = client.get("/web/index.html")
    assert resp.status_code == 200
    html = resp.text
    assert "configs/agent_flow_kucoin_cpu_nollm.json" in html
    assert "cdn.jsdelivr.net" not in html
    assert "自动布局" in html
    assert 'id="btnTrainBt"' in html
    assert 'id="flowArtifacts"' in html

    # Artifact-check endpoint used by the web UI.
    meta = client.get("/flow/run-meta/latest").json()
    assert meta.get("status") != "error", meta
    assert isinstance(meta.get("run_id"), str) and meta.get("run_id")
    checks = meta.get("checks") or {}
    assert checks.get("feature_output", {}).get("exists") is True
    assert checks.get("expression_output", {}).get("exists") is True
    assert checks.get("feedback_summary", {}).get("exists") is True

    runs = client.get("/flow/runs/list?limit=10").json()
    assert runs.get("items") and isinstance(runs["items"], list)
    assert any(it.get("run_id") == meta["run_id"] for it in runs["items"])
