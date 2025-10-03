from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
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
