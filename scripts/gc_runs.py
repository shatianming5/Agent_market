#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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


def _rm_tree(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        print("[dry] remove:", path)
        return
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)  # type: ignore[arg-type]
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Garbage collect old flow runs under artifacts/runs/")
    default_keep = int(os.environ.get("AGENT_MARKET_KEEP_LAST_RUNS", "50") or "50")
    parser.add_argument("--keep", type=int, default=default_keep, help="Keep last N runs (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument("--prune-backtests", action="store_true", help="Also delete unreferenced user_data/backtest_results zips")
    args = parser.parse_args(argv)

    keep = int(args.keep or 50)
    keep = max(1, min(keep, 5000))
    dry_run = bool(args.dry_run)

    runs_root = paths.runs_root()
    if not runs_root.exists():
        print("[gc] runs root not found:", runs_root)
        return 0

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
            try:
                ts = meta_path.stat().st_mtime
            except Exception:
                ts = 0.0
        rows.append((float(ts), child.resolve(), payload))

    rows.sort(key=lambda x: x[0], reverse=True)
    kept = rows[:keep]
    removed = rows[keep:]

    print(f"[gc] runs_root={runs_root} total={len(rows)} keep={len(kept)} remove={len(removed)}")
    for _, run_dir, payload in removed:
        run_id = str(payload.get("run_id") or run_dir.name)
        print("[gc] removing run:", run_id, run_dir)
        _rm_tree(run_dir, dry_run=dry_run)

    if args.prune_backtests:
        keep_names: set[str] = set()
        for _, _run_dir, payload in kept:
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
                print("[gc] removing backtest zip:", zp)
                _rm_tree(zp, dry_run=dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
