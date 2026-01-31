from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint():
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"


def test_settings_roundtrip_rejects_invalid_timeframe():
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    res = client.post("/settings", json={"default_timeframe": "4hours"})
    assert res.status_code == 200
    body = res.json()
    assert body.get("status") == "error"
    assert body.get("code") == "INVALID_TIMEFRAME"


def test_settings_roundtrip_accepts_valid_timeframe():
    import server.main as srv  # type: ignore

    client = TestClient(srv.app)
    res = client.post("/settings", json={"default_timeframe": "4h"})
    assert res.status_code == 200
    body = res.json()
    assert body.get("status") == "ok"
    assert "settings" in body

