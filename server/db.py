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

