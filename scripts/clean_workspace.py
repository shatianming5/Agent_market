#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_TARGETS = [
    ".pytest_cache",
    "artifacts",
    "user_data/agent_logs",
    "user_data/backtest_results",
    "user_data/logs",
    "user_data/models",
    "user_data/notebooks",
    "user_data/plot",
    "user_data/freqaimodels",
    "user_data/hyperopts",
    "user_data/hyperopt_results",
]


def rm_rf(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean temporary and generated directories")
    parser.add_argument("targets", nargs="*", help="Optional paths to clean (default uses built-in list)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument("--keep-dirs", action="store_true", help="Recreate empty directories after cleaning")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = args.targets or DEFAULT_TARGETS
    removed = []

    for rel in targets:
        p = (root / rel).resolve()
        if not p.exists():
            continue
        if args.dry_run:
            print(f"[dry] would remove: {p}")
            continue
        print(f"remove: {p}")
        rm_rf(p)
        removed.append(p)
        if args.keep_dirs and rel.endswith(("agent_logs","backtest_results","logs","artifacts")):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    print(f"done. removed {len(removed)} items")


if __name__ == "__main__":
    main()

