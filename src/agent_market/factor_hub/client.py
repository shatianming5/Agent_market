"""High-level Python client for Factor Hub.

Usage:
    from agent_market.factor_hub import Client
    fh = Client()
    fh.init_db()
    fid = fh.propose("z(ema_spread) - z(mfi_28)", origin="my_agent",
                     category="mean_reversion")
    fh.add_evaluation(fid, "ic", "oos_ic", 0.083,
                      period_start="2025-11-01", period_end="2026-04-12",
                      n_samples=12345, sign_agree=8)
    hits = fh.query(ic_gt=0.05, status="active", limit=20)
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from . import db
from .models import Factor, Evaluation, Deployment, Event


def _row_to_factor(row) -> Factor:
    d = dict(row)
    d["features_used"] = json.loads(d.get("features_used") or "[]")
    md = d.get("metadata") or "{}"
    d["metadata"] = json.loads(md) if isinstance(md, str) else (md or {})
    # Drop columns that aren't part of the dataclass
    d.pop("latest_metric", None)
    return Factor(**d)


def _row_to_eval(row) -> Evaluation:
    d = dict(row)
    md = d.get("metadata") or "{}"
    d["metadata"] = json.loads(md) if isinstance(md, str) else (md or {})
    return Evaluation(**d)


def _row_to_deployment(row) -> Deployment:
    d = dict(row)
    d["factor_ids"] = json.loads(d.get("factor_ids") or "[]")
    d["is_active"] = bool(d.get("is_active"))
    return Deployment(**d)


def _row_to_event(row) -> Event:
    d = dict(row)
    pl = d.get("payload") or "{}"
    d["payload"] = json.loads(pl) if isinstance(pl, str) else (pl or {})
    return Event(**d)


class Client:
    """Ergonomic wrapper around the SQLite Factor Hub registry."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path: Optional[Path] = Path(db_path) if db_path else None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def init_db(self) -> Path:
        db.init_schema(self.db_path)
        return db.get_db_path() if self.db_path is None else self.db_path

    @contextmanager
    def connect(self):
        with db.connect(self.db_path) as c:
            yield c

    @property
    def path(self) -> Path:
        return self.db_path or db.get_db_path()

    # ------------------------------------------------------------
    # Factor CRUD
    # ------------------------------------------------------------

    def propose(self, expression: str, *, name: Optional[str] = None,
                category: str = "misc", origin: str = "unknown",
                description: str = "", status: str = "candidate",
                created_by: str = "", source_lib: str = "",
                metadata: Optional[Dict] = None) -> int:
        expr = expression.strip()
        if not name:
            name = f"{origin}_{db._hash_expr(expr)[:10]}"
        with self.connect() as c:
            return db.insert_factor(
                c, name=name, expression=expr, category=category, origin=origin,
                description=description, status=status, created_by=created_by,
                source_lib=source_lib, metadata=metadata,
            )

    add_factor = propose  # alias

    def query(self, *, status: Optional[str] = "active",
              category: Optional[str] = None, origin: Optional[str] = None,
              ic_gt: Optional[float] = None, ic_lt: Optional[float] = None,
              metric_name: str = "oos_ic",
              limit: int = 100, order_by: str = "id DESC",
              as_dict: bool = True) -> List[Union[Dict, Factor]]:
        with self.connect() as c:
            rows = db.query_factors(
                c, status=status, category=category, origin=origin,
                ic_gt=ic_gt, ic_lt=ic_lt, metric_name=metric_name,
                limit=limit, order_by=order_by,
            )
        if as_dict:
            out = []
            for r in rows:
                d = dict(r)
                d["features_used"] = json.loads(d.get("features_used") or "[]")
                md = d.get("metadata") or "{}"
                d["metadata"] = json.loads(md) if isinstance(md, str) else (md or {})
                out.append(d)
            return out
        return [_row_to_factor(r) for r in rows]

    def get(self, ident: Union[int, str]) -> Optional[Factor]:
        with self.connect() as c:
            row = (db.get_factor(c, int(ident)) if isinstance(ident, int)
                   else db.get_factor_by_name(c, ident))
        return _row_to_factor(row) if row else None

    def update_status(self, factor_id: int, status: str, reason: str = "") -> None:
        with self.connect() as c:
            db.update_status(c, factor_id, status, reason=reason)

    def delete(self, factor_id: int) -> None:
        with self.connect() as c:
            db.delete_factor(c, factor_id)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    def add_evaluation(self, factor_id: int, eval_type: str,
                       metric_name: str, metric_value: float, *,
                       period_start: str = "", period_end: str = "",
                       n_samples: int = 0, sign_agree: int = 0,
                       notes: str = "", metadata: Optional[Dict] = None) -> int:
        with self.connect() as c:
            return db.add_evaluation(
                c, factor_id=factor_id, eval_type=eval_type,
                metric_name=metric_name, metric_value=float(metric_value),
                period_start=period_start, period_end=period_end,
                n_samples=int(n_samples), sign_agree=int(sign_agree),
                notes=notes, metadata=metadata,
            )

    def evaluations(self, factor_id: int, metric_name: Optional[str] = None,
                    limit: int = 100) -> List[Evaluation]:
        with self.connect() as c:
            rows = db.get_evaluations(c, factor_id, metric_name=metric_name, limit=limit)
        return [_row_to_eval(r) for r in rows]

    def latest_metric(self, factor_id: int, metric_name: str = "oos_ic") -> Optional[float]:
        with self.connect() as c:
            return db.latest_metric(c, factor_id, metric_name)

    # ------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------

    def deploy(self, name: str, factor_ids: Iterable[int], *,
               activate: bool = False, deployed_by: str = "", notes: str = "") -> int:
        with self.connect() as c:
            return db.create_deployment(
                c, name=name, factor_ids=list(factor_ids),
                activate=activate, deployed_by=deployed_by, notes=notes,
            )

    def active_deployment(self, name: str = "production") -> Optional[Deployment]:
        with self.connect() as c:
            row = db.get_active_deployment(c, name=name)
        return _row_to_deployment(row) if row else None

    def deployments(self, limit: int = 50) -> List[Deployment]:
        with self.connect() as c:
            rows = db.list_deployments(c, limit=limit)
        return [_row_to_deployment(r) for r in rows]

    def activate(self, deployment_id: int) -> None:
        with self.connect() as c:
            db.activate_deployment(c, deployment_id)

    # ------------------------------------------------------------
    # Events
    # ------------------------------------------------------------

    def log(self, event_type: str, factor_id: Optional[int] = None,
            payload: Optional[Dict] = None) -> None:
        with self.connect() as c:
            db.log_event(c, event_type, factor_id=factor_id, payload=payload)

    def events(self, since_id: int = 0, limit: int = 100,
               event_type: Optional[str] = None) -> List[Event]:
        with self.connect() as c:
            rows = db.tail_events(c, since_id=since_id, limit=limit, event_type=event_type)
        return [_row_to_event(r) for r in rows]

    def subscribe(self, event_type: Optional[str] = None,
                  poll_interval: float = 0.5,
                  since_id: Optional[int] = None) -> Iterator[Event]:
        """Poll-based subscription — yields new events as they arrive."""
        if since_id is None:
            with self.connect() as c:
                row = c.execute("SELECT MAX(id) AS mx FROM events").fetchone()
                since_id = int(row["mx"] or 0)
        while True:
            batch = self.events(since_id=since_id, limit=500, event_type=event_type)
            for e in batch:
                since_id = max(since_id, int(e.id or 0))
                yield e
            time.sleep(poll_interval)

    # ------------------------------------------------------------
    # Stats / lineage
    # ------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self.connect() as c:
            return db.get_stats(c)

    def origin_distribution(self) -> List[Dict]:
        with self.connect() as c:
            return db.origin_distribution(c)

    def feature_deps(self, top_n: int = 30) -> List[Dict]:
        with self.connect() as c:
            return db.feature_dependencies(c, top_n=top_n)

    def factors_by_feature(self, feature: str) -> List[Factor]:
        with self.connect() as c:
            rows = db.factors_using_feature(c, feature)
        return [_row_to_factor(r) for r in rows]

    def lineage(self, factor_id: int) -> List[str]:
        """Return the base features referenced by a factor."""
        with self.connect() as c:
            rows = c.execute(
                "SELECT feature FROM lineage WHERE factor_id = ? ORDER BY feature", (factor_id,)
            ).fetchall()
        return [r["feature"] for r in rows]

    # ------------------------------------------------------------
    # Bulk JSON migration
    # ------------------------------------------------------------

    def migrate_json(self, path: Union[str, Path], *, lib_name: str = "",
                     origin: Optional[str] = None, status: str = "active") -> Dict[str, int]:
        from .migrate import migrate_file
        return migrate_file(self, Path(path), lib_name=lib_name,
                            origin=origin, status=status)
