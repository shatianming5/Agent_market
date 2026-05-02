#!/usr/bin/env python3
"""Sequential OKX universe download queue.

Runs universes in priority order. Later universes reuse already downloaded files
through the downloader's file-level completeness checks.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", default="core_160,top200_dynamic,all_raw")
    parser.add_argument("--timeframes", default="1h,15m,5m,1m")
    parser.add_argument("--start", default="2025-04-12")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-prefix", default=f"okx_expand_{_utc_stamp()}")
    args = parser.parse_args()

    universes = [item.strip() for item in str(args.universes).replace(";", ",").split(",") if item.strip()]
    for universe in universes:
        run_id = f"{args.run_prefix}_{universe}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "okx_universe_download.py"),
            "--universe",
            universe,
            "--timeframes",
            str(args.timeframes),
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--sleep",
            str(args.sleep),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--run-id",
            run_id,
        ]
        print(f"[queue] start universe={universe} run_id={run_id}", flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        print(f"[queue] done universe={universe} run_id={run_id}", flush=True)


if __name__ == "__main__":
    main()
