from __future__ import annotations

from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.config import Settings
from server.db import DB
from server.secrets import SecretsManager
from server.connector_service import ConnectorService
from agent_market import connectors as connector_module
from server import routes_connectors


class DummyJobs:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def start(self, cmd, cwd=None, env=None):  # noqa: ANN001
        job_id = f"job-{len(self.calls) + 1}"
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env, "job_id": job_id})
        return job_id


class DummyTriggers:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def trigger_now(self, trigger_id: str, payload: dict | None = None) -> str:
        job_id = f"trigger-{len(self.calls) + 1}"
        self.calls.append({"trigger_id": trigger_id, "payload": payload or {}, "job_id": job_id})
        return job_id


def build_app(tmpdir):
    connector_module.Web3 = None
    db = DB(tmpdir / "connectors.db")
    db.init()
    settings = Settings(
        secrets_master_key=None,
        secret_admin_token="admintoken",
        secret_read_token="readtoken",
        secret_scopes=["connector:evm", "connector:binance", "connector:mcp_browser"],
    )
    secrets = SecretsManager(db, settings)
    jobs = DummyJobs()
    triggers = DummyTriggers()
    service = ConnectorService(db, secrets, jobs=jobs, triggers=triggers)

    app = FastAPI()
    app.include_router(routes_connectors.router)
    app.dependency_overrides[routes_connectors.get_service] = lambda: service
    return app, service, jobs, triggers


def test_create_and_test_connector(tmp_path, monkeypatch):
    app, service, jobs, triggers = build_app(tmp_path)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, params=None):
            raise AssertionError("unexpected HTTP call")

    monkeypatch.setattr(connector_module, "HTTP_CLIENT_FACTORY", lambda timeout: FakeClient())

    client = TestClient(app)

    resp = client.get("/connectors/types")
    assert resp.status_code == 200
    assert "evm" in resp.json()

    resp = client.post(
        "/connectors",
        json={
            "name": "Local EVM",
            "connector_type": "evm",
            "config": {},
            "credentials": {"rpc_url": "http://localhost:8545"},
            "metadata": {},
        },
    )
    assert resp.status_code == 201, resp.text
    connector_id = resp.json()["id"]

    test_resp = client.post(f"/connectors/{connector_id}/test")
    assert test_resp.status_code == 200
    assert test_resp.json()["success"] is True

    logs = client.get(f"/connectors/{connector_id}/logs").json()
    assert len(logs) >= 1
    assert logs[0]["status"] == "success"


def test_unknown_connector(tmp_path):
    app, service, *_ = build_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/connectors",
        json={
            "name": "Unknown",
            "connector_type": "mystery",
            "config": {},
            "credentials": {},
        },
    )
    assert resp.status_code == 400


def test_binance_connector_with_mock(tmp_path, monkeypatch):
    app, service, *_ = build_app(tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, params=None):
            if "binance" in url:
                return FakeResponse({"serverTime": 123456})
            raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(connector_module, "HTTP_CLIENT_FACTORY", lambda timeout: FakeClient())

    client = TestClient(app)
    resp = client.post(
        "/connectors",
        json={
            "name": "Binance",
            "connector_type": "binance",
            "config": {},
            "credentials": {"api_key": "k", "api_secret": "s"},
        },
    )
    assert resp.status_code == 201, resp.text
    connector_id = resp.json()["id"]

    test_resp = client.post(f"/connectors/{connector_id}/test")
    assert test_resp.status_code == 200
    payload = test_resp.json()
    assert payload["success"] is True
    assert payload["serverTime"] == 123456

    logs = client.get(f"/connectors/{connector_id}/logs").json()
    assert logs and logs[0]["status"] == "success"


def test_connector_run_action(tmp_path, monkeypatch):
    app, service, jobs, triggers = build_app(tmp_path)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, headers=None, params=None):
            return type("Resp", (), {"status_code": 200, "json": lambda self: {"serverTime": 1}})()

    monkeypatch.setattr(connector_module, "HTTP_CLIENT_FACTORY", lambda timeout: FakeClient())

    client = TestClient(app)

    resp = client.post(
        "/connectors",
        json={
            "name": "Runner",
            "connector_type": "binance",
            "config": {},
            "credentials": {"api_key": "k", "api_secret": "s"},
            "metadata": {"job_command": ["echo", "hello"], "job_env": {"X": "1"}},
        },
    )
    assert resp.status_code == 201, resp.text
    connector_id = resp.json()["id"]

    run_resp = client.post(f"/connectors/{connector_id}/run")
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert data["type"] == "job"
    assert jobs.calls and jobs.calls[0]["cmd"][0] == "echo"

    # Trigger metadata
    resp = client.post(
        "/connectors",
        json={
            "name": "Trigger",
            "connector_type": "binance",
            "config": {},
            "credentials": {"api_key": "k", "api_secret": "s"},
            "metadata": {"trigger_id": "trigger-123", "trigger_payload": {"foo": "bar"}},
        },
    )
    trigger_id = resp.json()["id"]
    run_resp = client.post(f"/connectors/{trigger_id}/run")
    assert run_resp.status_code == 200
    assert triggers.calls and triggers.calls[0]["trigger_id"] == "trigger-123"


def test_mcp_browser_connector(tmp_path, monkeypatch):
    app, service, *_ = build_app(tmp_path)

    monkeypatch.setattr(connector_module.shutil, "which", lambda cmd: "C:/bin/npx" if cmd == "npx" else None)

    client = TestClient(app)
    resp = client.post(
        "/connectors",
        json={
            "name": "Playwright MCP",
            "connector_type": "mcp_browser",
            "config": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
            },
            "credentials": {},
        },
    )
    assert resp.status_code == 201, resp.text
    connector_id = resp.json()["id"]

    test_resp = client.post(f"/connectors/{connector_id}/test")
    assert test_resp.status_code == 200
    data = test_resp.json()
    assert data["success"] is True
    assert data["command"] == "npx"
    assert data["args"] == ["@playwright/mcp@latest"]
