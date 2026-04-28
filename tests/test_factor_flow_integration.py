from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def test_flow_factor_compile_and_eval_write_run_meta_checks():
    root = Path(__file__).resolve().parents[1]
    cfg = root / "configs" / "agent_flow_factor_smoke.json"
    script = root / "scripts" / "agent_flow.py"

    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(cfg),
        "--steps",
        "factor_compile",
        "factor_eval",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    meta = client.get("/flow/run-meta/latest").json()
    assert meta.get("status") != "error", meta
    checks = meta.get("checks") or {}
    run_id = meta.get("run_id")
    assert isinstance(run_id, str) and run_id

    assert checks.get("factor_spec_json", {}).get("exists") is True
    assert checks.get("factor_ast_json", {}).get("exists") is True
    assert checks.get("factor_expression_txt", {}).get("exists") is True
    assert checks.get("factor_scores_json", {}).get("exists") is True
    assert checks.get("factor_pareto_csv", {}).get("exists") is True
    assert checks.get("factor_memory_json", {}).get("exists") is True
    assert checks.get("factor_cards_json", {}).get("exists") is True
    assert checks.get("factor_failure_cards_json", {}).get("exists") is True
    assert checks.get("factor_lineage_json", {}).get("exists") is True

    scores = client.get(f"/flow/factor-scores/{run_id}").json()
    assert scores.get("status") != "error", scores
    assert isinstance(scores.get("items"), list)
    assert scores.get("_run_id") == run_id

    memory = client.get(f"/flow/factor-memory/{run_id}").json()
    assert memory.get("status") != "error", memory
    assert isinstance(memory.get("factor_cards"), list)
    assert memory.get("_run_id") == run_id


def test_api_run_factor_endpoints_start_jobs():
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    resp = client.post(
        "/run/factor_compile",
        json={
            "run_id": "deadbeefdead",
            "out_dir": "artifacts/_tmp_factor_compile_api",
            "spec": {
                "name": "api_demo",
                "version": "1.0",
                "expr": {"op": "zscore", "args": [{"op": "var", "args": ["close"]}, 4]},
                "meta": {"timeframe": "1h", "universe": ["BTC/USDT"], "data_sources": ["ohlcv"]},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload.get("status") == "started"
    assert payload.get("job_id")

    resp2 = client.post(
        "/run/factor_eval",
        json={
            "run_id": "deadbeefdead",
            "out_dir": "artifacts/_tmp_factor_eval_api",
            "expression": "ts_z(close,4)",
            "rows": 32,
            "seed": 7,
        },
    )
    assert resp2.status_code == 200, resp2.text
    payload2 = resp2.json()
    assert payload2.get("status") == "started"
    assert payload2.get("job_id")
