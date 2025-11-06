from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import routes_triggers

from server.db import DB
from server.triggers import TriggerManager


class DummyJobs:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def start(self, cmd, cwd=None, env=None) -> str:  # noqa: ANN001
        job_id = f"job-{len(self.calls) + 1}"
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env, "job_id": job_id})
        return job_id


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "app.db"
    database = DB(db_path)
    database.init()
    return database


@pytest.fixture()
def jobs():
    return DummyJobs()


@pytest.fixture()
def manager(db, jobs):
    mgr = TriggerManager(db, jobs, poll_interval=0.1)
    mgr.reload()
    yield mgr
    mgr.stop()


def test_cron_trigger_tick_starts_job(db, jobs, manager):
    record = db.create_trigger(
        "Cron Job",
        "cron",
        {
            "cron": "* * * * *",
            "command": ["echo", "hello"],
        },
    )
    manager.refresh_single(record.id)
    now = datetime.now(timezone.utc)
    manager.tick(now + timedelta(minutes=1, seconds=1))

    assert len(jobs.calls) == 1
    call = jobs.calls[0]
    assert call["cmd"][0] == "echo"
    assert call["env"]["TRIGGER_ID"] == record.id


def test_webhook_trigger_secret(manager, db, jobs):
    record = db.create_trigger(
        "Webhook",
        "webhook",
        {"secret": "top", "command": ["python", "-c", "print('hi')"]},
    )
    manager.refresh_single(record.id)

    with pytest.raises(PermissionError):
        manager.fire_webhook(record.id, {"ok": True}, headers={})

    job_id = manager.fire_webhook(record.id, {"ok": True}, headers={"x-trigger-secret": "top"})
    assert job_id == jobs.calls[0]["job_id"]
    assert json.loads(jobs.calls[0]["env"]["TRIGGER_PAYLOAD"]) == {"ok": True}


def test_trigger_now_uses_job_manager(manager, db, jobs):
    record = db.create_trigger(
        "Manual",
        "webhook",
        {"command": ["python", "-V"]},
    )
    manager.refresh_single(record.id)
    job_id = manager.trigger_now(record.id, {"manual": True})
    assert job_id == jobs.calls[0]["job_id"]
    assert json.loads(jobs.calls[0]["env"]["TRIGGER_PAYLOAD"])["manual"] is True


def test_evm_trigger_updates_from_block(monkeypatch, db, jobs, manager):
    from server import triggers as triggers_module

    class DummyEth:
        def __init__(self):
            self.block_number = 100

        def get_logs(self, params):
            return []

    class DummyProvider:
        def __init__(self, url, request_kwargs=None):  # noqa: ANN001
            self.url = url
            self.request_kwargs = request_kwargs

    class DummyWeb3:
        HTTPProvider = DummyProvider

        def __init__(self, provider, request_kwargs=None):  # noqa: ANN001
            self.provider = provider
            self.eth = DummyEth()

    monkeypatch.setattr(triggers_module, "Web3", DummyWeb3)

    record = db.create_trigger(
        "EVM",
        "evm",
        {
            "provider_url": "http://localhost:8545",
            "address": "0x123",
            "poll_interval": 0.01,
            "command": ["echo", "evm"],
            "from_block": 10,
        },
    )
    manager.refresh_single(record.id)
    managed = manager._get(record.id)
    assert isinstance(managed.handler, triggers_module.EvmTrigger)

    def fake_fetch():
        return [
            {"transactionHash": "0xabc", "blockNumber": 42, "logIndex": 0, "data": "0x0", "address": "0x123", "topics": []}
        ]

    monkeypatch.setattr(managed.handler, "_fetch_events", fake_fetch)
    managed.handler._next_poll = 0  # force immediate poll
    manager.tick(datetime.now(timezone.utc))

    assert jobs.calls[-1]["cmd"][0] == "echo"
    updated = db.get_trigger(record.id)
    assert updated.config["from_block"] == 43


def test_trigger_routes_create_and_fire(tmp_path):
    db = DB(tmp_path / "api.db")
    db.init()
    jobs = DummyJobs()
    manager = TriggerManager(db, jobs, poll_interval=0.1)
    manager.reload()

    app = FastAPI()
    app.include_router(routes_triggers.router)

    def override_db():
        return db

    def override_manager():
        return manager

    app.dependency_overrides[routes_triggers.get_db] = override_db
    app.dependency_overrides[routes_triggers.get_manager] = override_manager

    client = TestClient(app)

    resp = client.post(
        "/triggers",
        json={"name": "Hook", "type": "webhook", "config": {"command": ["echo", "ok"]}},
    )
    assert resp.status_code == 201, resp.text
    trigger_id = resp.json()["id"]

    list_resp = client.get("/triggers")
    assert list_resp.status_code == 200
    assert any(item["id"] == trigger_id for item in list_resp.json())

    fire_resp = client.post(f"/triggers/{trigger_id}/fire", json={"hello": True})
    assert fire_resp.status_code == 200
    assert jobs.calls
    payload_env = json.loads(jobs.calls[0]["env"]["TRIGGER_PAYLOAD"])
    assert payload_env["hello"] is True
