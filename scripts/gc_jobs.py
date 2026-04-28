#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_market import paths  # noqa: E402


def _parse_iso8601(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rm(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print("[dry] remove:", path)
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Garbage collect old job logs/registries under user_data/")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument(
        "--keep-days",
        type=int,
        default=int(os.environ.get("AGENT_MARKET_KEEP_JOB_LOG_DAYS", "14") or "14"),
        help="Keep jobs newer than N days (default: env AGENT_MARKET_KEEP_JOB_LOG_DAYS or 14).",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=int(os.environ.get("AGENT_MARKET_KEEP_RECENT_JOBS", "200") or "200"),
        help="Always keep the most recent N jobs regardless of age (default: env AGENT_MARKET_KEEP_RECENT_JOBS or 200).",
    )
    args = parser.parse_args(argv)

    keep_days = max(0, int(args.keep_days or 0))
    keep_recent = max(0, min(int(args.keep or 0), 50_000))
    dry_run = bool(args.dry_run)

    user_root = paths.user_data_root()
    log_dir = (user_root / "job_logs").resolve()
    reg_dir = (user_root / "job_registry").resolve()
    if not reg_dir.exists() and not log_dir.exists():
        print("[gc] job dirs not found:", reg_dir, log_dir)
        return 0

    now = time.time()
    threshold = now - float(keep_days) * 86400.0 if keep_days > 0 else 0.0

    rows: list[tuple[float, str, Path]] = []
    for reg_path in sorted(reg_dir.glob("*.json")) if reg_dir.exists() else []:
        payload = _load_json(reg_path)
        job_id = str(payload.get("id") or reg_path.stem)
        ended = _parse_iso8601(payload.get("ended_at"))
        started = _parse_iso8601(payload.get("started_at"))
        ts = ended or started
        if not ts:
            try:
                ts = reg_path.stat().st_mtime
            except Exception:
                ts = 0.0
        rows.append((float(ts), job_id, reg_path.resolve()))

    rows.sort(key=lambda x: x[0], reverse=True)
    keep_ids = {job_id for _, job_id, _ in rows[:keep_recent]}
    if threshold > 0:
        keep_ids |= {job_id for ts, job_id, _ in rows if ts >= threshold}

    removed = 0
    for ts, job_id, reg_path in rows:
        if job_id in keep_ids:
            continue
        removed += 1
        print("[gc] removing job:", job_id, "ts=", int(ts))
        _rm(reg_path, dry_run=dry_run)
        log_path = (log_dir / f"{job_id}.log").resolve()
        if log_path.exists():
            _rm(log_path, dry_run=dry_run)

    print(
        f"[gc] user_data={user_root} total={len(rows)} keep={len(keep_ids)} removed={removed} keep_days={keep_days} keep_recent={keep_recent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

