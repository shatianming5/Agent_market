"""Cross-loop tried-expressions log.

Every `simulate` call with --tag appends one JSONL line. Subsequent agent
sessions read recent entries and inject them into the prompt so each loop
learns from prior loops without rerunning the same expressions.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Hashable, Optional

_APPEND_LOCK = threading.Lock()


def append_tried(
    path: Path,
    *,
    expr: str,
    sharpe: Optional[float],
    fitness: Optional[float],
    turnover: Optional[float],
    alpha_id: Optional[str],
    status: str,
    error: Optional[str] = None,
    region: str = "",
    universe: str = "",
    decay: int = 0,
) -> None:
    record = {
        "ts": time.time(),
        "expr": expr,
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "alpha_id": alpha_id,
        "status": status,
        "error": error,
        "region": region,
        "universe": universe,
        "decay": decay,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with _APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def find_recent_revisits(
    records: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Optional[Hashable]],
    *,
    window: int = 5,
    min_revisits: int = 2,
) -> dict[Hashable, list[dict[str, Any]]]:
    """Group last ``window`` records by ``key_fn`` — return groups ≥ min_revisits.

    Anti-flip / cool-down primitive used by :mod:`prompt_builder`'s slot-
    cool-down hint. Records are sorted by ``ts`` ascending and the tail
    sliced to ``window`` before grouping; visits within each group preserve
    chronological order so callers can check fitness trends.

    ``key_fn`` may return ``None`` to mean "skip this record" (e.g. for
    rows with empty expressions). Records without ``ts`` sort to the front.

    Returns ``{key: [record, ...]}`` where each value has length ≥
    ``min_revisits``. Empty dict when no group meets the threshold.
    """
    if min_revisits < 1 or window < 1:
        return {}
    recent = sorted(records, key=lambda r: r.get("ts", 0))[-window:]
    groups: dict[Hashable, list[dict[str, Any]]] = {}
    for r in recent:
        try:
            key = key_fn(r)
        except (KeyError, TypeError, ValueError):
            continue
        if key is None:
            continue
        groups.setdefault(key, []).append(r)
    return {k: v for k, v in groups.items() if len(v) >= min_revisits}


def read_tried(path: Path, *, tail: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: list[str] = []
    try:
        # Cheap tail: read whole file, take last N lines (jsonl is small)
        all_lines = path.read_text(encoding="utf-8").splitlines()
        lines = all_lines[-tail:] if tail else all_lines
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ── Checkpoint sidecar (for tmux resume) ────────────────────────────────


def checkpoint_path(tried_path: Path) -> Path:
    """Return the sidecar checkpoint file path for a tried_exprs.jsonl."""
    return tried_path.with_name(tried_path.name + ".checkpoint.json")


def write_checkpoint(
    tried_path: Path,
    *,
    session_id: str,
    last_iter: int,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically persist {session_id, last_iter, ts} next to tried_exprs.jsonl.

    Used by ``agent_runner`` so that a tmux loop killed mid-iteration can
    resume from the next iteration without re-running prior work. Atomic
    via tmp + replace so a SIGKILL never leaves a half-written file.
    """
    p = checkpoint_path(tried_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "last_iter": int(last_iter),
        "ts": time.time(),
    }
    if extra:
        payload["extra"] = extra
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with _APPEND_LOCK:
        tmp.replace(p)
    return p


def read_checkpoint(tried_path: Path) -> Optional[dict[str, Any]]:
    """Return the last-written checkpoint or None when missing/corrupt."""
    p = checkpoint_path(tried_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def format_for_prompt(records: list[dict[str, Any]], *, max_rows: int = 60) -> str:
    """Render a compact markdown table for prompt injection."""
    if not records:
        return ""
    # De-dupe by expr keeping latest result
    by_expr: dict[str, dict[str, Any]] = {}
    for r in records:
        expr = r.get("expr") or ""
        if not expr:
            continue
        prior = by_expr.get(expr)
        if not prior or float(r.get("ts") or 0) > float(prior.get("ts") or 0):
            by_expr[expr] = r

    rows = list(by_expr.values())
    # Sort: passing first (sh+fi desc), then by recency
    def _key(r: dict[str, Any]) -> tuple:
        sh = r.get("sharpe")
        fi = r.get("fitness")
        score = (float(sh or 0)) + (float(fi or 0))
        return (-score, -float(r.get("ts") or 0))
    rows.sort(key=_key)
    rows = rows[:max_rows]

    lines = ["| expr | sh | fi | to | status |", "|---|---|---|---|---|"]
    for r in rows:
        sh = r.get("sharpe")
        fi = r.get("fitness")
        to = r.get("turnover")
        status = r.get("status") or "?"
        if r.get("error"):
            status = f"{status}:{str(r['error'])[:30]}"
        expr = (r.get("expr") or "")[:80]
        lines.append(
            f"| `{expr}` | "
            f"{sh:.2f} | " if isinstance(sh, (int, float)) else f"| `{expr}` | - | "
        )
    # Rebuild simply (avoid the f-string trick above)
    lines = ["| expr | sh | fi | to | status |", "|---|---|---|---|---|"]
    for r in rows:
        sh = r.get("sharpe")
        fi = r.get("fitness")
        to = r.get("turnover")
        status = r.get("status") or "?"
        err = r.get("error")
        if err:
            status = f"{status}:{str(err)[:30]}"
        expr = (r.get("expr") or "")[:90].replace("|", "/")
        sh_s = f"{sh:.2f}" if isinstance(sh, (int, float)) else "-"
        fi_s = f"{fi:.2f}" if isinstance(fi, (int, float)) else "-"
        to_s = f"{to:.2f}" if isinstance(to, (int, float)) else "-"
        lines.append(f"| `{expr}` | {sh_s} | {fi_s} | {to_s} | {status} |")
    return "\n".join(lines)
