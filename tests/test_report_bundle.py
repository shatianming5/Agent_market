from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_report_step_writes_bundle_zip_and_results_endpoints_expose_it():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_bundle_smoke.json"
    script = root / "scripts" / "agent_flow.py"

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(cfg),
        "--steps",
        "factor_compile",
        "factor_eval",
        "report",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    meta = json.loads((root / "artifacts" / "run_meta.json").read_text(encoding="utf-8"))
    run_id = meta.get("run_id")
    assert isinstance(run_id, str) and run_id

    bundle = root / "artifacts" / "runs" / run_id / "bundle" / "bundle.zip"
    assert bundle.exists()

    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    rows = client.get("/results/bundles/list?limit=20").json().get("items") or []
    assert any(it.get("run_id") == run_id for it in rows)

    resp = client.get(f"/results/bundles/download/{run_id}")
    assert resp.status_code == 200
    assert resp.content

