from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _json_load(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _json_dump(data: Optional[Dict[str, Any]]) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


@dataclass
class TriggerRecord:
    id: str
    name: str
    type: str
    config: Dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str
    last_fired_at: Optional[str]


@dataclass
class RunRecord:
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    payload: Dict[str, Any]
    context: Dict[str, Any]


@dataclass
class TaskRecord:
    id: str
    run_id: str
    name: str
    status: str
    sequence: Optional[int]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    result: Dict[str, Any]
    error: Optional[str]
    metrics: Dict[str, Any]


@dataclass
class ArtifactRecord:
    id: str
    run_id: str
    task_id: Optional[str]
    type: str
    uri: str
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class SecretRecord:
    id: str
    scope: str
    name: str
    value: str
    created_at: str
    updated_at: str


@dataclass
class AuditRecord:
    id: str
    actor: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    payload: Dict[str, Any]
    created_at: str


@dataclass
class ConnectorRecord:
    id: str
    name: str
    type: str
    config: Dict[str, Any]
    secret_scope: str
    secret_name: str
    status: Optional[str]
    last_tested_at: Optional[str]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


@dataclass
class ConnectorTestLog:
    id: str
    connector_id: str
    status: str
    payload: Dict[str, Any]
    created_at: str


@dataclass
class DB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self.connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(agent_id) REFERENCES agents(id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_steps (
                    job_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    ts TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS triggers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_fired_at TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS connectors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    secret_scope TEXT NOT NULL,
                    secret_name TEXT NOT NULL,
                    status TEXT,
                    last_tested_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            # Ensure metadata column exists for older schemas
            try:
                cur.execute("ALTER TABLE connectors ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_test_logs (
                    id TEXT PRIMARY KEY,
                    connector_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(connector_id) REFERENCES connectors(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    payload TEXT,
                    context TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sequence INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result TEXT,
                    error TEXT,
                    metrics TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    type TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE SET NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS secrets (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scope, name)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    actor TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.commit()

    # Simple helpers
    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_agent(self, name: str) -> Dict[str, Any]:
        agent = {"id": uuid.uuid4().hex[:12], "name": name, "created_at": self.now()}
        with self.connect() as con:
            con.execute(
                "INSERT INTO agents(id,name,created_at) VALUES(?,?,?)",
                (agent["id"], agent["name"], agent["created_at"]),
            )
            con.commit()
        return agent

    def list_agents(self) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.execute("SELECT id,name,created_at FROM agents ORDER BY created_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.execute("SELECT id,name,created_at FROM agents WHERE id=?", (agent_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def create_order(self, agent_id: str, side: str, qty: float) -> Dict[str, Any]:
        order = {
            "id": uuid.uuid4().hex[:12],
            "agent_id": agent_id,
            "side": side,
            "qty": float(qty),
            "status": "created",
            "created_at": self.now(),
        }
        with self.connect() as con:
            con.execute(
                "INSERT INTO orders(id,agent_id,side,qty,status,created_at) VALUES(?,?,?,?,?,?)",
                (order["id"], order["agent_id"], order["side"], order["qty"], order["status"], order["created_at"]),
            )
            con.commit()
        return order

    def list_orders(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connect() as con:
            if agent_id:
                cur = con.execute(
                    "SELECT id,agent_id,side,qty,status,created_at FROM orders WHERE agent_id=? ORDER BY created_at DESC",
                    (agent_id,),
                )
            else:
                cur = con.execute(
                    "SELECT id,agent_id,side,qty,status,created_at FROM orders ORDER BY created_at DESC"
                )
            return [dict(r) for r in cur.fetchall()]

    # Job step logging
    def log_job_step(self, job_id: str, idx: int, total: int, label: str, ts: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO job_steps(job_id, idx, total, label, ts) VALUES(?,?,?,?,?)",
                (job_id, int(idx), int(total), str(label), ts),
            )
            con.commit()

    def get_job_steps(self, job_id: str) -> List[Dict[str, Any]]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT job_id, idx, total, label, ts FROM job_steps WHERE job_id=? ORDER BY idx ASC, ts ASC",
                (job_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_step_stats(self) -> List[Dict[str, Any]]:
        # Compute average duration per label from sequential steps within each job
        with self.connect() as con:
            cur = con.execute("SELECT DISTINCT job_id FROM job_steps")
            jobs = [r[0] for r in cur.fetchall()]
            import datetime as _dt
            durations: Dict[str, List[float]] = {}
            for job_id in jobs:
                cur2 = con.execute(
                    "SELECT idx, total, label, ts FROM job_steps WHERE job_id=? ORDER BY idx ASC, ts ASC",
                    (job_id,),
                )
                rows = [dict(r) for r in cur2.fetchall()]
                for i in range(len(rows)):
                    cur_row = rows[i]
                    next_ts = None
                    if i + 1 < len(rows):
                        next_ts = rows[i + 1]['ts']
                    # No next step: skip duration (unknown)
                    if not next_ts:
                        continue
                    try:
                        t0 = _dt.datetime.fromisoformat(cur_row['ts'].replace('Z','+00:00'))
                        t1 = _dt.datetime.fromisoformat(next_ts.replace('Z','+00:00'))
                        dur = max(0.0, (t1 - t0).total_seconds())
                        durations.setdefault(cur_row['label'], []).append(dur)
                    except Exception:
                        continue
            stats = []
            for label, vals in durations.items():
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                stats.append({'label': label, 'avg_seconds': avg, 'samples': len(vals)})
            return sorted(stats, key=lambda x: x['label'])

    # Trigger management
    def _trigger_from_row(self, row: sqlite3.Row) -> TriggerRecord:
        data = dict(row)
        try:
            config = json.loads(data.get("config") or "{}")
        except json.JSONDecodeError:
            config = {}
        return TriggerRecord(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            config=config,
            enabled=bool(data["enabled"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            last_fired_at=data.get("last_fired_at"),
        )

    def create_trigger(
        self,
        name: str,
        trigger_type: str,
        config: Dict[str, Any],
        enabled: bool = True,
    ) -> TriggerRecord:
        trigger_id = uuid.uuid4().hex[:12]
        now = self.now()
        payload = json.dumps(config, ensure_ascii=False)
        with self.connect() as con:
            con.execute(
                "INSERT INTO triggers(id,name,type,config,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (
                    trigger_id,
                    name,
                    trigger_type,
                    payload,
                    1 if enabled else 0,
                    now,
                    now,
                ),
            )
            con.commit()
        record = self.get_trigger(trigger_id)
        if record is None:
            raise RuntimeError("failed to create trigger")
        return record

    def list_triggers(self) -> List[TriggerRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,name,type,config,enabled,created_at,updated_at,last_fired_at FROM triggers ORDER BY created_at DESC"
            )
            return [self._trigger_from_row(row) for row in cur.fetchall()]

    def get_trigger(self, trigger_id: str) -> Optional[TriggerRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,name,type,config,enabled,created_at,updated_at,last_fired_at FROM triggers WHERE id=?",
                (trigger_id,),
            )
            row = cur.fetchone()
            return self._trigger_from_row(row) if row else None

    def update_trigger(
        self,
        trigger_id: str,
        *,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None,
        last_fired_at: Optional[str] = None,
    ) -> Optional[TriggerRecord]:
        fields: List[str] = []
        values: List[Any] = []
        if name is not None:
            fields.append("name=?")
            values.append(name)
        if config is not None:
            fields.append("config=?")
            values.append(json.dumps(config, ensure_ascii=False))
        if enabled is not None:
            fields.append("enabled=?")
            values.append(1 if enabled else 0)
        if last_fired_at is not None:
            fields.append("last_fired_at=?")
            values.append(last_fired_at)
        if not fields:
            return self.get_trigger(trigger_id)
        fields.append("updated_at=?")
        values.append(self.now())
        values.append(trigger_id)
        with self.connect() as con:
            con.execute(f"UPDATE triggers SET {', '.join(fields)} WHERE id=?", values)
            con.commit()
        return self.get_trigger(trigger_id)

    def delete_trigger(self, trigger_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM triggers WHERE id=?", (trigger_id,))
            con.commit()

    def update_trigger_last_fired(self, trigger_id: str, ts: Optional[str] = None) -> None:
        self.update_trigger(trigger_id, last_fired_at=ts or self.now())

    # Run storage
    def _run_from_row(self, row: sqlite3.Row) -> RunRecord:
        data = dict(row)
        return RunRecord(
            id=data["id"],
            name=data["name"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            payload=_json_load(data.get("payload")),
            context=_json_load(data.get("context")),
        )

    def create_run(self, name: str, *, payload: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, status: str = "pending") -> RunRecord:
        run_id = uuid.uuid4().hex[:12]
        now = self.now()
        with self.connect() as con:
            con.execute(
                "INSERT INTO runs(id,name,status,created_at,updated_at,payload,context) VALUES(?,?,?,?,?,?,?)",
                (run_id, name, status, now, now, _json_dump(payload), _json_dump(context)),
            )
            con.commit()
        record = self.get_run(run_id)
        if record is None:
            raise RuntimeError("failed to create run")
        return record

    def list_runs(self, *, status: Optional[str] = None, limit: Optional[int] = None) -> List[RunRecord]:
        query = "SELECT id,name,status,created_at,updated_at,started_at,finished_at,payload,context FROM runs"
        params: List[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self.connect() as con:
            cur = con.execute(query, params)
            return [self._run_from_row(row) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,name,status,created_at,updated_at,started_at,finished_at,payload,context FROM runs WHERE id=?",
                (run_id,),
            )
            row = cur.fetchone()
            return self._run_from_row(row) if row else None

    def update_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[RunRecord]:
        fields: List[str] = []
        values: List[Any] = []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if started_at is not None:
            fields.append("started_at=?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at=?")
            values.append(finished_at)
        if payload is not None:
            fields.append("payload=?")
            values.append(_json_dump(payload))
        if context is not None:
            fields.append("context=?")
            values.append(_json_dump(context))
        if not fields:
            return self.get_run(run_id)
        fields.append("updated_at=?")
        values.append(self.now())
        values.append(run_id)
        with self.connect() as con:
            con.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id=?", values)
            con.commit()
        return self.get_run(run_id)

    def delete_run(self, run_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM runs WHERE id=?", (run_id,))
            con.commit()

    # Task storage
    def _task_from_row(self, row: sqlite3.Row) -> TaskRecord:
        data = dict(row)
        return TaskRecord(
            id=data["id"],
            run_id=data["run_id"],
            name=data["name"],
            status=data["status"],
            sequence=data.get("sequence"),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=_json_load(data.get("result")),
            error=data.get("error"),
            metrics=_json_load(data.get("metrics")),
        )

    def create_task(
        self,
        run_id: str,
        name: str,
        *,
        status: str = "pending",
        sequence: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex[:12]
        now = self.now()
        with self.connect() as con:
            con.execute(
                "INSERT INTO tasks(id,run_id,name,status,sequence,created_at,result,error,metrics) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    run_id,
                    name,
                    status,
                    sequence,
                    now,
                    _json_dump(result),
                    error,
                    _json_dump(metrics),
                ),
            )
            con.commit()
        record = self.get_task(task_id)
        if record is None:
            raise RuntimeError("failed to create task")
        return record

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,run_id,name,status,sequence,created_at,started_at,finished_at,result,error,metrics FROM tasks WHERE id=?",
                (task_id,),
            )
            row = cur.fetchone()
            return self._task_from_row(row) if row else None

    def list_tasks(self, run_id: str) -> List[TaskRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,run_id,name,status,sequence,created_at,started_at,finished_at,result,error,metrics FROM tasks WHERE run_id=? ORDER BY sequence ASC, created_at ASC",
                (run_id,),
            )
            return [self._task_from_row(row) for row in cur.fetchall()]

    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskRecord]:
        fields: List[str] = []
        values: List[Any] = []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if started_at is not None:
            fields.append("started_at=?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at=?")
            values.append(finished_at)
        if result is not None:
            fields.append("result=?")
            values.append(_json_dump(result))
        if error is not None:
            fields.append("error=?")
            values.append(error)
        if metrics is not None:
            fields.append("metrics=?")
            values.append(_json_dump(metrics))
        if not fields:
            return self.get_task(task_id)
        values.append(task_id)
        with self.connect() as con:
            con.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id=?", values)
            con.commit()
        return self.get_task(task_id)

    def delete_task(self, task_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            con.commit()

    # Artifact storage
    def _artifact_from_row(self, row: sqlite3.Row) -> ArtifactRecord:
        data = dict(row)
        return ArtifactRecord(
            id=data["id"],
            run_id=data["run_id"],
            task_id=data.get("task_id"),
            type=data["type"],
            uri=data["uri"],
            metadata=_json_load(data.get("metadata")),
            created_at=data["created_at"],
        )

    def create_artifact(
        self,
        run_id: str,
        *,
        task_id: Optional[str],
        type: str,
        uri: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        artifact_id = uuid.uuid4().hex[:12]
        now = self.now()
        with self.connect() as con:
            con.execute(
                "INSERT INTO artifacts(id,run_id,task_id,type,uri,metadata,created_at) VALUES(?,?,?,?,?,?,?)",
                (artifact_id, run_id, task_id, type, uri, _json_dump(metadata), now),
            )
            con.commit()
        record = self.get_artifact(artifact_id)
        if record is None:
            raise RuntimeError("failed to create artifact")
        return record

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,run_id,task_id,type,uri,metadata,created_at FROM artifacts WHERE id=?",
                (artifact_id,),
            )
            row = cur.fetchone()
            return self._artifact_from_row(row) if row else None

    def list_artifacts(self, *, run_id: Optional[str] = None, task_id: Optional[str] = None) -> List[ArtifactRecord]:
        query = "SELECT id,run_id,task_id,type,uri,metadata,created_at FROM artifacts"
        params: List[Any] = []
        clauses: List[str] = []
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self.connect() as con:
            cur = con.execute(query, params)
            return [self._artifact_from_row(row) for row in cur.fetchall()]

    def delete_artifact(self, artifact_id: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
            con.commit()

    # Secret storage
    def _secret_from_row(self, row: sqlite3.Row) -> SecretRecord:
        data = dict(row)
        return SecretRecord(
            id=data["id"],
            scope=data["scope"],
            name=data["name"],
            value=data["value"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def set_secret(self, scope: str, name: str, value: str) -> SecretRecord:
        existing = self.get_secret(scope, name, include_value=True)
        now = self.now()
        if existing:
            with self.connect() as con:
                con.execute(
                    "UPDATE secrets SET value=?, updated_at=? WHERE id=?",
                    (value, now, existing.id),
                )
                con.commit()
            return self.get_secret(scope, name, include_value=True)  # type: ignore[return-value]
        secret_id = uuid.uuid4().hex[:12]
        with self.connect() as con:
            con.execute(
                "INSERT INTO secrets(id,scope,name,value,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (secret_id, scope, name, value, now, now),
            )
            con.commit()
        record = self.get_secret(scope, name, include_value=True)
        if record is None:
            raise RuntimeError("failed to persist secret")
        return record

    def get_secret(self, scope: str, name: str, *, include_value: bool = False) -> Optional[SecretRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,scope,name,value,created_at,updated_at FROM secrets WHERE scope=? AND name=?",
                (scope, name),
            )
            row = cur.fetchone()
            if not row:
                return None
            record = self._secret_from_row(row)
            if not include_value:
                record.value = "***"
            return record

    def list_secrets(self, scope: Optional[str] = None, *, include_values: bool = False) -> List[SecretRecord]:
        query = "SELECT id,scope,name,value,created_at,updated_at FROM secrets"
        params: List[Any] = []
        if scope:
            query += " WHERE scope=?"
            params.append(scope)
        query += " ORDER BY scope ASC, name ASC"
        secrets = []
        with self.connect() as con:
            cur = con.execute(query, params)
            for row in cur.fetchall():
                record = self._secret_from_row(row)
                if not include_values:
                    record.value = "***"
                secrets.append(record)
        return secrets

    def delete_secret(self, scope: str, name: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM secrets WHERE scope=? AND name=?", (scope, name))
            con.commit()

    # Connector storage
    def _connector_from_row(self, row: sqlite3.Row) -> ConnectorRecord:
        data = dict(row)
        return ConnectorRecord(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            config=_json_load(data.get('config')),
            secret_scope=data['secret_scope'],
            secret_name=data['secret_name'],
            status=data.get('status'),
            last_tested_at=data.get('last_tested_at'),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            metadata=_json_load(data.get('metadata')),
        )

    def create_connector(
        self,
        name: str,
        connector_type: str,
        config: Dict[str, Any],
        *,
        secret_scope: str,
        secret_name: str,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConnectorRecord:
        connector_id = uuid.uuid4().hex[:12]
        now = self.now()
        with self.connect() as con:
            con.execute(
                "INSERT INTO connectors(id,name,type,config,secret_scope,secret_name,status,last_tested_at,created_at,updated_at,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    connector_id,
                    name,
                    connector_type,
                    _json_dump(config),
                    secret_scope,
                    secret_name,
                    status,
                    None,
                    now,
                    now,
                    _json_dump(metadata),
                ),
            )
            con.commit()
        record = self.get_connector(connector_id)
        if record is None:
            raise RuntimeError('failed to create connector')
        return record

    def list_connectors(self) -> List[ConnectorRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,name,type,config,secret_scope,secret_name,status,last_tested_at,created_at,updated_at,metadata FROM connectors ORDER BY created_at DESC"
            )
            return [self._connector_from_row(row) for row in cur.fetchall()]

    def get_connector(self, connector_id: str) -> Optional[ConnectorRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,name,type,config,secret_scope,secret_name,status,last_tested_at,created_at,updated_at,metadata FROM connectors WHERE id=?",
                (connector_id,),
            )
            row = cur.fetchone()
            return self._connector_from_row(row) if row else None

    def update_connector(
        self,
        connector_id: str,
        *,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        secret_scope: Optional[str] = None,
        secret_name: Optional[str] = None,
        status: Optional[str] = None,
        last_tested_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConnectorRecord]:
        fields: List[str] = []
        values: List[Any] = []
        if name is not None:
            fields.append('name=?')
            values.append(name)
        if config is not None:
            fields.append('config=?')
            values.append(_json_dump(config))
        if secret_scope is not None:
            fields.append('secret_scope=?')
            values.append(secret_scope)
        if secret_name is not None:
            fields.append('secret_name=?')
            values.append(secret_name)
        if status is not None:
            fields.append('status=?')
            values.append(status)
        if last_tested_at is not None:
            fields.append('last_tested_at=?')
            values.append(last_tested_at)
        if metadata is not None:
            fields.append('metadata=?')
            values.append(_json_dump(metadata))
        if not fields:
            return self.get_connector(connector_id)
        fields.append('updated_at=?')
        values.append(self.now())
        values.append(connector_id)
        with self.connect() as con:
            con.execute(f"UPDATE connectors SET {', '.join(fields)} WHERE id=?", values)
            con.commit()
        return self.get_connector(connector_id)

    def delete_connector(self, connector_id: str) -> None:
        with self.connect() as con:
            con.execute('DELETE FROM connectors WHERE id=?', (connector_id,))
            con.commit()

    def update_connector_status(self, connector_id: str, status: str, *, tested_at: Optional[str] = None) -> None:
        self.update_connector(connector_id, status=status, last_tested_at=tested_at or self.now())

    def log_connector_test(self, connector_id: str, status: str, payload: Dict[str, Any]) -> None:
        log_id = uuid.uuid4().hex[:12]
        with self.connect() as con:
            con.execute(
                "INSERT INTO connector_test_logs(id,connector_id,status,payload,created_at) VALUES(?,?,?,?,?)",
                (log_id, connector_id, status, _json_dump(payload), self.now()),
            )
            con.commit()

    def list_connector_tests(self, connector_id: str, limit: int = 20) -> List[ConnectorTestLog]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,connector_id,status,payload,created_at FROM connector_test_logs WHERE connector_id=? ORDER BY created_at DESC LIMIT ?",
                (connector_id, int(limit)),
            )
            rows = cur.fetchall()
            return [
                ConnectorTestLog(
                    id=row['id'],
                    connector_id=row['connector_id'],
                    status=row['status'],
                    payload=_json_load(row['payload']),
                    created_at=row['created_at'],
                )
                for row in rows
            ]

    # Audit logs
    def log_audit(
        self,
        *,
        actor: Optional[str],
        action: str,
        resource_type: Optional[str],
        resource_id: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        audit_id = uuid.uuid4().hex[:12]
        now = self.now()
        with self.connect() as con:
            con.execute(
                "INSERT INTO audit_logs(id,actor,action,resource_type,resource_id,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (audit_id, actor, action, resource_type, resource_id, _json_dump(payload), now),
            )
            con.commit()
        record = self.get_audit(audit_id)
        if record is None:
            raise RuntimeError("failed to write audit log")
        return record

    def _audit_from_row(self, row: sqlite3.Row) -> AuditRecord:
        data = dict(row)
        return AuditRecord(
            id=data["id"],
            actor=data.get("actor"),
            action=data["action"],
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            payload=_json_load(data.get("payload")),
            created_at=data["created_at"],
        )

    def get_audit(self, audit_id: str) -> Optional[AuditRecord]:
        with self.connect() as con:
            cur = con.execute(
                "SELECT id,actor,action,resource_type,resource_id,payload,created_at FROM audit_logs WHERE id=?",
                (audit_id,),
            )
            row = cur.fetchone()
            return self._audit_from_row(row) if row else None

    def list_audits(
        self,
        *,
        limit: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> List[AuditRecord]:
        query = "SELECT id,actor,action,resource_type,resource_id,payload,created_at FROM audit_logs"
        params: List[Any] = []
        clauses: List[str] = []
        if resource_type:
            clauses.append("resource_type=?")
            params.append(resource_type)
        if resource_id:
            clauses.append("resource_id=?")
            params.append(resource_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self.connect() as con:
            cur = con.execute(query, params)
            return [self._audit_from_row(row) for row in cur.fetchall()]
