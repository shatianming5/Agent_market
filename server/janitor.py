from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent_market import paths  # type: ignore

logger = logging.getLogger(__name__)


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
        logger.info("[janitor] dry-run remove: %s", path)
        return
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        logger.debug("[janitor] failed to remove %s", path, exc_info=True)


def gc_runs(*, keep: int, prune_backtests: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Garbage collect old flow runs under artifacts/runs/ (best-effort)."""
    keep_n = max(1, min(int(keep or 50), 5000))
    runs_root = paths.runs_root()
    if not runs_root.exists():
        return {"runs_root": str(runs_root), "total": 0, "kept": 0, "removed": 0}

    rows: list[tuple[float, Path, dict[str, Any]]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        meta_path = (child / "run_meta.json").resolve()
        payload = _load_json(meta_path)
        ended = _parse_iso8601(payload.get("ended_at"))
        started = _parse_iso8601(payload.get("started_at"))
        ts = ended or started
        if not ts:
            # If run_meta.json isn't written yet (run in progress), fall back to
            # directory mtime so we don't delete active runs.
            try:
                ts = meta_path.stat().st_mtime if meta_path.exists() else child.stat().st_mtime
            except Exception:
                ts = 0.0
        rows.append((float(ts), child.resolve(), payload))

    rows.sort(key=lambda x: x[0], reverse=True)
    kept = rows[:keep_n]
    removed = rows[keep_n:]

    removed_count = 0
    for _ts, run_dir, payload in removed:
        removed_count += 1
        run_id = str(payload.get("run_id") or run_dir.name)
        logger.info("[janitor] removing run: %s %s", run_id, run_dir)
        _rm(run_dir, dry_run=dry_run)

    pruned_backtests = 0
    if prune_backtests:
        keep_names: set[str] = set()
        for _ts, _run_dir, payload in kept:
            artifacts = payload.get("artifacts") or {}
            raw = artifacts.get("backtest_zips") or []
            if isinstance(raw, list):
                for p in raw:
                    name = str(p or "").split("/")[-1].strip()
                    if name:
                        keep_names.add(name)
        bt_dir = (paths.user_data_root() / "backtest_results").resolve()
        if bt_dir.exists():
            for zp in bt_dir.glob("backtest-result-*.zip"):
                if zp.name in keep_names:
                    continue
                pruned_backtests += 1
                logger.info("[janitor] removing backtest zip: %s", zp)
                _rm(zp, dry_run=dry_run)

    return {
        "runs_root": str(runs_root),
        "total": int(len(rows)),
        "kept": int(len(kept)),
        "removed": int(removed_count),
        "pruned_backtests": int(pruned_backtests),
        "dry_run": bool(dry_run),
    }


def gc_jobs(*, keep_recent: int, keep_days: int, dry_run: bool = False) -> dict[str, Any]:
    """Garbage collect old job logs/registries under user_data/ (best-effort)."""
    keep_recent_n = max(0, min(int(keep_recent or 0), 50_000))
    keep_days_n = max(0, int(keep_days or 0))

    user_root = paths.user_data_root()
    log_dir = (user_root / "job_logs").resolve()
    reg_dir = (user_root / "job_registry").resolve()
    if not reg_dir.exists() and not log_dir.exists():
        return {"user_data": str(user_root), "total": 0, "kept": 0, "removed": 0}

    now = time.time()
    threshold = now - float(keep_days_n) * 86400.0 if keep_days_n > 0 else 0.0

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
    keep_ids = {job_id for _, job_id, _ in rows[:keep_recent_n]}
    if threshold > 0:
        keep_ids |= {job_id for ts, job_id, _ in rows if ts >= threshold}

    removed = 0
    for _ts, job_id, reg_path in rows:
        if job_id in keep_ids:
            continue
        removed += 1
        logger.info("[janitor] removing job: %s", job_id)
        _rm(reg_path, dry_run=dry_run)
        log_path = (log_dir / f"{job_id}.log").resolve()
        if log_path.exists():
            _rm(log_path, dry_run=dry_run)

    return {
        "user_data": str(user_root),
        "total": int(len(rows)),
        "kept": int(len(keep_ids)),
        "removed": int(removed),
        "keep_days": int(keep_days_n),
        "keep_recent": int(keep_recent_n),
        "dry_run": bool(dry_run),
    }


def start_janitor_thread() -> Optional[object]:
    """Start background GC janitor if enabled via env.

    Returns the thread object (best-effort) so callers can keep a reference.
    """
    enabled = str(os.environ.get("AGENT_MARKET_JANITOR_ENABLED", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None

    import threading

    interval_sec = float(os.environ.get("AGENT_MARKET_JANITOR_INTERVAL_SEC", "3600") or "3600")
    initial_delay_sec = float(os.environ.get("AGENT_MARKET_JANITOR_INITIAL_DELAY_SEC", "30") or "30")
    keep_runs = int(os.environ.get("AGENT_MARKET_KEEP_LAST_RUNS", "50") or "50")
    prune_backtests = str(os.environ.get("AGENT_MARKET_JANITOR_PRUNE_BACKTESTS", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    keep_jobs = int(os.environ.get("AGENT_MARKET_KEEP_RECENT_JOBS", "200") or "200")
    keep_job_days = int(os.environ.get("AGENT_MARKET_KEEP_JOB_LOG_DAYS", "14") or "14")

    def _loop() -> None:
        try:
            if initial_delay_sec > 0:
                time.sleep(max(0.0, initial_delay_sec))
        except Exception:
            pass
        logger.info(
            "[janitor] enabled interval_sec=%s keep_runs=%s prune_backtests=%s keep_jobs=%s keep_job_days=%s",
            interval_sec,
            keep_runs,
            prune_backtests,
            keep_jobs,
            keep_job_days,
        )
        while True:
            try:
                runs_res = gc_runs(keep=keep_runs, prune_backtests=prune_backtests, dry_run=False)
                jobs_res = gc_jobs(keep_recent=keep_jobs, keep_days=keep_job_days, dry_run=False)
                logger.info("[janitor] gc complete runs=%s jobs=%s", runs_res, jobs_res)
            except Exception:
                logger.exception("[janitor] gc failed")
            try:
                time.sleep(max(30.0, float(interval_sec)))
            except Exception:
                time.sleep(3600.0)

    t = threading.Thread(target=_loop, daemon=True, name="agent_market_janitor")
    t.start()
    return t


__all__ = ["gc_jobs", "gc_runs", "start_janitor_thread"]

