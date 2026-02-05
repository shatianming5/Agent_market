from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_flow_capture_and_lob_rebuild_write_run_meta_artifacts():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_capture_lob_smoke.json"
    script = root / "scripts" / "agent_flow.py"

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(cfg),
        "--steps",
        "capture",
        "lob_rebuild",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    meta = client.get("/flow/run-meta/latest").json()
    assert meta.get("status") != "error", meta

    checks = meta.get("checks") or {}
    assert checks.get("capture_manifest", {}).get("exists") is True
    assert checks.get("lob_state_parquet", {}).get("exists") is True
    assert checks.get("rebuild_report", {}).get("exists") is True

