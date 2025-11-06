from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings
from server.db import DB
from server.secrets import SecretsManager
from server import routes_secrets


def build_app(tmpdir):
    db = DB(tmpdir / "app.db")
    db.init()
    settings = Settings(secret_scopes=["global", "trading"], secret_admin_token="secret-token", secret_read_token="read-token", secrets_master_key=None)
    manager = SecretsManager(db, settings)

    app = FastAPI()
    app.include_router(routes_secrets.router)
    app.dependency_overrides[routes_secrets.get_manager] = lambda: manager
    app.dependency_overrides[routes_secrets.get_settings] = lambda: settings
    return app, manager, db


def test_secret_crud(tmp_path):
    app, manager, db = build_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/secrets",
        headers={"X-Secret-Token": "secret-token"},
        json={"scope": "global", "name": "API_KEY", "value": "123"},
    )
    assert resp.status_code == 201, resp.text

    resp = client.get("/secrets", headers={"X-Secret-Token": "read-token"})
    assert resp.status_code == 200
    assert resp.json()[0]["value"] == "***"

    resp = client.get(
        "/secrets/global/API_KEY",
        params={"reveal": "true"},
        headers={"X-Secret-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["value"] == "123"

    resp = client.delete(
        "/secrets/global/API_KEY",
        headers={"X-Secret-Token": "secret-token"},
    )
    assert resp.status_code == 204


def test_secret_authz(tmp_path):
    app, manager, db = build_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/secrets",
        json={"scope": "global", "name": "API_KEY", "value": "123"},
    )
    assert resp.status_code == 403

    resp = client.get("/secrets")
    assert resp.status_code == 403

    resp = client.get("/secrets", headers={"X-Secret-Token": "read-token"})
    assert resp.status_code == 200

    resp = client.post(
        "/secrets",
        headers={"X-Secret-Token": "secret-token"},
        json={"scope": "forbidden", "name": "KEY", "value": "123"},
    )
    assert resp.status_code == 403
