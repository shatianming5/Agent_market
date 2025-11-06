from __future__ import annotations

from pathlib import Path

from server.db import DB


def _init_db(tmp_path: Path) -> DB:
    db = DB(tmp_path / "records.sqlite")
    db.init()
    return db


def test_run_task_artifact_flow(tmp_path):
    db = _init_db(tmp_path)

    run = db.create_run("backtest", payload={"pair": "BTC/USDT"})
    assert run.status == "pending"
    assert run.payload["pair"] == "BTC/USDT"

    run = db.update_run(run.id, status="running", started_at=db.now())
    assert run.status == "running"
    assert run.started_at is not None

    task = db.create_task(run.id, "download", status="running", sequence=1)
    assert task.run_id == run.id
    assert task.sequence == 1

    task = db.update_task(task.id, status="success", finished_at=db.now(), result={"rows": 42})
    assert task.status == "success"
    assert task.result["rows"] == 42

    artifact = db.create_artifact(run.id, task_id=task.id, type="file", uri="s3://bucket/file.json", metadata={"size": 1024})
    assert artifact.run_id == run.id
    assert artifact.metadata["size"] == 1024

    artifacts = db.list_artifacts(run_id=run.id)
    assert len(artifacts) == 1

    tasks = db.list_tasks(run.id)
    assert len(tasks) == 1

    run = db.update_run(run.id, status="success", finished_at=db.now())
    assert run.status == "success"

    db.delete_run(run.id)
    assert db.get_run(run.id) is None
    assert not db.list_tasks(run.id)
    assert not db.list_artifacts(run_id=run.id)


def test_secret_crud(tmp_path):
    db = _init_db(tmp_path)

    created = db.set_secret("global", "API_KEY", "value-1")
    assert created.value == "value-1"

    masked = db.get_secret("global", "API_KEY")
    assert masked.value == "***"

    updated = db.set_secret("global", "API_KEY", "value-2")
    assert updated.value == "value-2"

    secrets = db.list_secrets(include_values=False)
    assert secrets[0].value == "***"

    secrets_full = db.list_secrets(include_values=True)
    assert secrets_full[0].value == "value-2"

    db.delete_secret("global", "API_KEY")
    assert db.get_secret("global", "API_KEY") is None


def test_audit_logging(tmp_path):
    db = _init_db(tmp_path)

    entry = db.log_audit(
        actor="tester",
        action="create",
        resource_type="run",
        resource_id="abc",
        payload={"note": "created run"},
    )
    assert entry.actor == "tester"
    assert entry.payload["note"] == "created run"

    fetched = db.get_audit(entry.id)
    assert fetched is not None

    logs = db.list_audits(resource_type="run")
    assert len(logs) == 1
