"""SQLite schema + low-level DB operations."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = Path(os.environ.get("FACTOR_HUB_DB") or Path.home() / ".factor_hub" / "registry.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    expression      TEXT    NOT NULL,
    expression_hash TEXT    NOT NULL UNIQUE,
    category        TEXT    DEFAULT 'misc',
    origin          TEXT    DEFAULT 'unknown',
    description     TEXT    DEFAULT '',
    features_used   TEXT    DEFAULT '[]',      -- JSON array
    complexity      INTEGER DEFAULT 0,
    status          TEXT    DEFAULT 'active',
    created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    created_by      TEXT    DEFAULT '',
    source_lib      TEXT    DEFAULT '',
    metadata        TEXT    DEFAULT '{}'       -- JSON object
);

CREATE INDEX IF NOT EXISTS idx_factors_status   ON factors(status);
CREATE INDEX IF NOT EXISTS idx_factors_origin   ON factors(origin);
CREATE INDEX IF NOT EXISTS idx_factors_category ON factors(category);

CREATE TABLE IF NOT EXISTS evaluations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_id    INTEGER NOT NULL,
    eval_type    TEXT    NOT NULL,        -- ic / walk_forward / sub_period / lgb_gain
    period_start TEXT    DEFAULT '',
    period_end   TEXT    DEFAULT '',
    metric_name  TEXT    NOT NULL,        -- 'oos_ic', 'profit_pct', 'sharpe', etc.
    metric_value REAL    NOT NULL,
    n_samples    INTEGER DEFAULT 0,
    sign_agree   INTEGER DEFAULT 0,
    eval_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    notes        TEXT    DEFAULT '',
    metadata     TEXT    DEFAULT '{}',
    FOREIGN KEY (factor_id) REFERENCES factors(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eval_factor      ON evaluations(factor_id);
CREATE INDEX IF NOT EXISTS idx_eval_metric      ON evaluations(factor_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_eval_type        ON evaluations(eval_type);
CREATE INDEX IF NOT EXISTS idx_eval_at          ON evaluations(eval_at);

CREATE TABLE IF NOT EXISTS deployments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,       -- 'production', 'shadow_a', etc.
    factor_ids    TEXT    NOT NULL,       -- JSON array of factor ids
    deployed_at   TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    deployed_by   TEXT    DEFAULT '',
    is_active     INTEGER DEFAULT 0,      -- bool
    notes         TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_deployments_active  ON deployments(is_active);
CREATE INDEX IF NOT EXISTS idx_deployments_name    ON deployments(name);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT    NOT NULL,
    factor_id  INTEGER,
    payload    TEXT    DEFAULT '{}',
    created_at TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_type  ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_at    ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_factor ON events(factor_id);

-- A lineage table mapping factor -> required base features (denormalized for fast lookup)
CREATE TABLE IF NOT EXISTS lineage (
    factor_id  INTEGER NOT NULL,
    feature    TEXT    NOT NULL,
    PRIMARY KEY (factor_id, feature),
    FOREIGN KEY (factor_id) REFERENCES factors(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lineage_feat ON lineage(feature);
"""


# ============================================================
# Connection management
# ============================================================

def get_db_path() -> Path:
    """Resolve DB path, creating parent dir if needed."""
    p = DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect(db_path: Optional[Path] = None):
    """Yield a SQLite connection with WAL mode and foreign-key enforcement."""
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        yield conn
    finally:
        conn.close()


# ============================================================
# Schema init
# ============================================================

def init_schema(db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ============================================================
# Factor operations
# ============================================================

def _hash_expr(expr: str) -> str:
    return hashlib.sha256(expr.strip().encode("utf-8")).hexdigest()


def _extract_features(expr: str) -> List[str]:
    """Parse expression AST, return list of referenced feature names (non-function Names)."""
    import ast
    KNOWN_FUNCS = {
        "z", "ts_z", "abs", "sign", "tanh", "log1p", "log", "sqrt", "exp",
        "decay_linear", "robust_z", "winsorize", "rank_xs", "zscore",
        "ifelse", "ema", "roll_mean", "roll_std", "rolling_max", "rolling_min",
        "rolling_sum", "rolling_mean", "pct_change", "shift", "diff",
        "sum", "mean", "clip", "corr", "skew", "kurtosis", "median", "max", "min",
    }
    PY_BUILTINS = {"True", "False", "None"}
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in KNOWN_FUNCS and node.id not in PY_BUILTINS:
                names.add(node.id)
    return sorted(names)


def _count_complexity(expr: str) -> int:
    import ast
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return 0
    return sum(1 for _ in ast.walk(tree))


def insert_factor(conn: sqlite3.Connection, *, name: str, expression: str,
                  category: str = "misc", origin: str = "unknown",
                  description: str = "", status: str = "active",
                  created_by: str = "", source_lib: str = "",
                  metadata: Optional[Dict] = None) -> int:
    """Insert a factor; returns id. If expression_hash already exists, updates name/meta and returns existing id."""
    expr_norm = expression.strip()
    h = _hash_expr(expr_norm)
    feats = _extract_features(expr_norm)
    complexity = _count_complexity(expr_norm)
    md_json = json.dumps(metadata or {}, ensure_ascii=False)
    feats_json = json.dumps(feats)

    # Dedup by hash — keep existing status unless caller upgrades it.
    # Status precedence (higher index = higher priority, never overwritten by lower):
    STATUS_RANK = {"candidate": 0, "retired": 1, "deprecated": 2, "shadow": 3, "active": 4}
    row = conn.execute(
        "SELECT id, status FROM factors WHERE expression_hash = ?", (h,)
    ).fetchone()
    if row:
        fid = row["id"]
        cur_status = row["status"]
        if STATUS_RANK.get(status, 0) > STATUS_RANK.get(cur_status, 0):
            # Caller upgrades the status (e.g. candidate -> active)
            conn.execute("""
                UPDATE factors SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                                    metadata = ?,
                                    description = COALESCE(NULLIF(?, ''), description),
                                    status = ?
                WHERE id = ?
            """, (md_json, description, status, fid))
            _log_event(conn, "factor.status_changed", fid,
                       {"old": cur_status, "new": status, "reason": "dedup_upgrade"})
        else:
            # Merge metadata/description but preserve existing status silently.
            conn.execute("""
                UPDATE factors SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                                    metadata = ?,
                                    description = COALESCE(NULLIF(?, ''), description)
                WHERE id = ?
            """, (md_json, description, fid))
        return fid

    # Autonumber name conflict handling (append -2, -3, ...)
    base_name = name
    suffix = 0
    while True:
        try_name = base_name if suffix == 0 else f"{base_name}-{suffix}"
        if conn.execute("SELECT id FROM factors WHERE name = ?", (try_name,)).fetchone() is None:
            break
        suffix += 1
    final_name = try_name

    cur = conn.execute("""
        INSERT INTO factors (name, expression, expression_hash, category, origin,
                             description, features_used, complexity, status, created_by, source_lib, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (final_name, expr_norm, h, category, origin, description,
          feats_json, complexity, status, created_by, source_lib, md_json))
    fid = cur.lastrowid

    # Update lineage
    for feat in feats:
        conn.execute("INSERT OR IGNORE INTO lineage (factor_id, feature) VALUES (?, ?)", (fid, feat))

    _log_event(conn, "factor.created", fid, {"name": final_name, "origin": origin})
    return fid


def query_factors(conn: sqlite3.Connection, *, status: Optional[str] = "active",
                  category: Optional[str] = None, origin: Optional[str] = None,
                  ic_gt: Optional[float] = None, ic_lt: Optional[float] = None,
                  metric_name: str = "oos_ic",
                  limit: int = 100, order_by: str = "id DESC") -> List[sqlite3.Row]:
    """Query factors with filters, optionally joined with latest evaluation metric."""
    base = "SELECT f.*"
    joins = ""
    where = []
    params: List[Any] = []

    if ic_gt is not None or ic_lt is not None:
        # Need latest evaluation for this metric
        base += """
        , (SELECT e.metric_value FROM evaluations e
           WHERE e.factor_id = f.id AND e.metric_name = ?
           ORDER BY e.eval_at DESC LIMIT 1) AS latest_metric
        """
        params.append(metric_name)

    base += " FROM factors f"

    if status:
        where.append("f.status = ?"); params.append(status)
    if category:
        where.append("f.category = ?"); params.append(category)
    if origin:
        where.append("f.origin LIKE ?"); params.append(f"%{origin}%")

    sql = base + (" WHERE " + " AND ".join(where) if where else "")

    # For metric filters, use HAVING-like subquery approach
    if ic_gt is not None:
        sql = f"""SELECT * FROM ({sql}) WHERE ABS(COALESCE(latest_metric, 0)) >= ?"""
        params.append(ic_gt)
    if ic_lt is not None:
        sql = f"""SELECT * FROM ({sql}) WHERE ABS(COALESCE(latest_metric, 0)) < ?"""
        params.append(ic_lt)

    sql += f" ORDER BY {order_by} LIMIT ?"
    params.append(int(limit))
    return list(conn.execute(sql, params).fetchall())


def get_factor(conn: sqlite3.Connection, factor_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM factors WHERE id = ?", (factor_id,)).fetchone()


def get_factor_by_name(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM factors WHERE name = ?", (name,)).fetchone()


def update_status(conn: sqlite3.Connection, factor_id: int, status: str, reason: str = "") -> None:
    row = conn.execute("SELECT status FROM factors WHERE id = ?", (factor_id,)).fetchone()
    if not row: return
    old = row["status"]
    conn.execute("""
        UPDATE factors SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?
    """, (status, factor_id))
    _log_event(conn, "factor.status_changed", factor_id,
               {"old": old, "new": status, "reason": reason})


def delete_factor(conn: sqlite3.Connection, factor_id: int) -> None:
    conn.execute("DELETE FROM factors WHERE id = ?", (factor_id,))


# ============================================================
# Evaluation operations
# ============================================================

def add_evaluation(conn: sqlite3.Connection, *, factor_id: int, eval_type: str,
                   metric_name: str, metric_value: float,
                   period_start: str = "", period_end: str = "",
                   n_samples: int = 0, sign_agree: int = 0,
                   notes: str = "", metadata: Optional[Dict] = None) -> int:
    md = json.dumps(metadata or {})
    cur = conn.execute("""
        INSERT INTO evaluations (factor_id, eval_type, period_start, period_end,
                                 metric_name, metric_value, n_samples, sign_agree,
                                 notes, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (factor_id, eval_type, period_start, period_end,
          metric_name, float(metric_value), int(n_samples), int(sign_agree),
          notes, md))
    eid = cur.lastrowid
    _log_event(conn, "factor.evaluated", factor_id,
               {"eval_type": eval_type, "metric": metric_name, "value": metric_value})
    return eid


def get_evaluations(conn: sqlite3.Connection, factor_id: int,
                    metric_name: Optional[str] = None, limit: int = 100) -> List[sqlite3.Row]:
    if metric_name:
        rows = conn.execute("""
            SELECT * FROM evaluations WHERE factor_id = ? AND metric_name = ?
            ORDER BY eval_at DESC LIMIT ?
        """, (factor_id, metric_name, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM evaluations WHERE factor_id = ?
            ORDER BY eval_at DESC LIMIT ?
        """, (factor_id, limit)).fetchall()
    return list(rows)


def latest_metric(conn: sqlite3.Connection, factor_id: int, metric_name: str) -> Optional[float]:
    row = conn.execute("""
        SELECT metric_value FROM evaluations
        WHERE factor_id = ? AND metric_name = ? ORDER BY eval_at DESC LIMIT 1
    """, (factor_id, metric_name)).fetchone()
    return float(row["metric_value"]) if row else None


# ============================================================
# Deployment operations
# ============================================================

def create_deployment(conn: sqlite3.Connection, *, name: str, factor_ids: List[int],
                      activate: bool = False, deployed_by: str = "", notes: str = "") -> int:
    fids_json = json.dumps(list(factor_ids))
    # Deactivate other deployments with same name if activating
    if activate:
        conn.execute("UPDATE deployments SET is_active = 0 WHERE name = ?", (name,))
    cur = conn.execute("""
        INSERT INTO deployments (name, factor_ids, is_active, deployed_by, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (name, fids_json, 1 if activate else 0, deployed_by, notes))
    did = cur.lastrowid
    _log_event(conn, "deployment.switched", None,
               {"deployment_id": did, "name": name, "n_factors": len(factor_ids), "activated": activate})
    return did


def get_active_deployment(conn: sqlite3.Connection, name: str = "production") -> Optional[sqlite3.Row]:
    return conn.execute("""
        SELECT * FROM deployments WHERE name = ? AND is_active = 1
        ORDER BY deployed_at DESC LIMIT 1
    """, (name,)).fetchone()


def list_deployments(conn: sqlite3.Connection, limit: int = 50) -> List[sqlite3.Row]:
    return list(conn.execute(
        "SELECT * FROM deployments ORDER BY deployed_at DESC LIMIT ?", (limit,)
    ).fetchall())


def activate_deployment(conn: sqlite3.Connection, deployment_id: int) -> None:
    row = conn.execute("SELECT name FROM deployments WHERE id = ?", (deployment_id,)).fetchone()
    if not row: raise ValueError(f"No such deployment: {deployment_id}")
    conn.execute("UPDATE deployments SET is_active = 0 WHERE name = ?", (row["name"],))
    conn.execute("UPDATE deployments SET is_active = 1 WHERE id = ?", (deployment_id,))
    _log_event(conn, "deployment.switched", None,
               {"deployment_id": deployment_id, "name": row["name"]})


# ============================================================
# Event operations
# ============================================================

def _log_event(conn: sqlite3.Connection, event_type: str,
                factor_id: Optional[int] = None, payload: Optional[Dict] = None) -> None:
    """Internal helper to log events."""
    pl = json.dumps(payload or {}, ensure_ascii=False, default=str)
    conn.execute("""
        INSERT INTO events (event_type, factor_id, payload) VALUES (?, ?, ?)
    """, (event_type, factor_id, pl))


def log_event(conn: sqlite3.Connection, event_type: str,
              factor_id: Optional[int] = None, payload: Optional[Dict] = None) -> None:
    """Public event logger (for mining / backtest / etc to emit lifecycle events)."""
    _log_event(conn, event_type, factor_id, payload)


def tail_events(conn: sqlite3.Connection, since_id: int = 0, limit: int = 100,
                event_type: Optional[str] = None) -> List[sqlite3.Row]:
    if event_type:
        return list(conn.execute("""
            SELECT * FROM events WHERE id > ? AND event_type = ?
            ORDER BY id ASC LIMIT ?
        """, (since_id, event_type, limit)).fetchall())
    return list(conn.execute("""
        SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT ?
    """, (since_id, limit)).fetchall())


# ============================================================
# Stats helpers
# ============================================================

def get_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    factors = {
        "total":      conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0],
        "active":     conn.execute("SELECT COUNT(*) FROM factors WHERE status='active'").fetchone()[0],
        "deprecated": conn.execute("SELECT COUNT(*) FROM factors WHERE status='deprecated'").fetchone()[0],
        "candidate":  conn.execute("SELECT COUNT(*) FROM factors WHERE status='candidate'").fetchone()[0],
    }
    evals = {
        "total":       conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
        "ic":          conn.execute("SELECT COUNT(*) FROM evaluations WHERE eval_type='ic'").fetchone()[0],
        "walk_forward":conn.execute("SELECT COUNT(*) FROM evaluations WHERE eval_type='walk_forward'").fetchone()[0],
    }
    deployments = {
        "total":  conn.execute("SELECT COUNT(*) FROM deployments").fetchone()[0],
        "active": conn.execute("SELECT COUNT(*) FROM deployments WHERE is_active=1").fetchone()[0],
    }
    return {"factors": factors, "evaluations": evals, "deployments": deployments,
            "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]}


def origin_distribution(conn: sqlite3.Connection) -> List[Dict]:
    rows = conn.execute("""
        SELECT origin, COUNT(*) AS n FROM factors WHERE status='active' GROUP BY origin ORDER BY n DESC
    """).fetchall()
    return [{"origin": r["origin"], "count": r["n"]} for r in rows]


def feature_dependencies(conn: sqlite3.Connection, top_n: int = 30) -> List[Dict]:
    """Return most-referenced base features across active factors."""
    rows = conn.execute("""
        SELECT l.feature, COUNT(*) AS n
        FROM lineage l JOIN factors f ON l.factor_id = f.id
        WHERE f.status = 'active'
        GROUP BY l.feature ORDER BY n DESC LIMIT ?
    """, (top_n,)).fetchall()
    return [{"feature": r["feature"], "dependents": r["n"]} for r in rows]


def factors_using_feature(conn: sqlite3.Connection, feature: str) -> List[sqlite3.Row]:
    return list(conn.execute("""
        SELECT f.* FROM factors f JOIN lineage l ON f.id = l.factor_id
        WHERE l.feature = ? AND f.status = 'active' ORDER BY f.name
    """, (feature,)).fetchall())
