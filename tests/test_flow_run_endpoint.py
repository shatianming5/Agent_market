from __future__ import annotations

from fastapi.testclient import TestClient


def test_flow_run_accepts_missing_steps(monkeypatch):
    """POST /flow/run should not crash when steps is omitted."""
    import server.main as srv  # type: ignore
    import server.api.routes.flow as flow_routes  # type: ignore

    client = TestClient(srv.app)
    monkeypatch.setattr(flow_routes.jobs, "start", lambda *args, **kwargs: "job123")

    resp = client.post(
        "/flow/run",
        json={"config": "configs/agent_flow_kucoin_cpu_nollm_smoke.json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "started"
    assert body.get("job_id") == "job123"
    cmd = body.get("cmd") or []
    assert "--config" in cmd
    assert "--steps" not in cmd

