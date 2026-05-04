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
from typing import Any, Optional

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
