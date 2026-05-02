#!/usr/bin/env python3
"""Resumable OKX futures universe downloader.

Example:
  python3 scripts/okx_universe_download.py --universe core_160 --timeframes 15m,5m,1m,1h
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_market.factor_lab import data, okx_universe  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timeframes(raw: str) -> list[str]:
    out = [item.strip() for item in str(raw).replace(";", ",").split(",") if item.strip()]
    allowed = set(data.OKX_BAR_MAP)
    bad = [item for item in out if item not in allowed]
    if bad:
        raise SystemExit(f"unsupported timeframes={bad}; expected {sorted(allowed)}")
    return out


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="core_160", help="universe name or manifest path")
    parser.add_argument("--timeframes", default="15m,5m,1m,1h")
    parser.add_argument("--start", default="2025-04-12")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    payload = okx_universe.load_okx_universe_manifest(args.universe)
    pairs = [str(pair) for pair in payload["pairs"]]
    if args.offset:
        pairs = pairs[int(args.offset) :]
    if args.max_pairs is not None:
        pairs = pairs[: int(args.max_pairs)]
    pair_starts = {str(k): str(v) for k, v in (payload.get("pair_start_dates") or {}).items() if v}
    timeframes = _parse_timeframes(args.timeframes)
    batch_size = max(1, int(args.batch_size))
    workers = max(1, int(args.workers))
    run_id = args.run_id or f"{payload.get('name', 'okx_universe')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "artifacts" / "okx_universe_download" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"

    checkpoint = {
        "run_id": run_id,
        "universe": payload.get("name"),
        "manifest": str(okx_universe.resolve_okx_universe_manifest(args.universe)),
        "start": args.start,
        "end": args.end,
        "timeframes": timeframes,
        "pair_count": len(pairs),
        "batch_size": batch_size,
        "workers": workers,
        "started_at_utc": _utc_now(),
        "status": "RUNNING",
        "completed": [],
    }
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")

    try:
        for timeframe in timeframes:
            batches = _chunk(pairs, batch_size)
            for batch_i, batch_pairs in enumerate(batches, start=1):
                batch_starts = {pair: pair_starts[pair] for pair in batch_pairs if pair in pair_starts}
                print(
                    f"[{_utc_now()}] universe={payload.get('name')} timeframe={timeframe} "
                    f"batch={batch_i}/{len(batches)} workers={workers} pairs={','.join(batch_pairs)}",
                    flush=True,
                )
                if workers <= 1 or len(batch_pairs) <= 1:
                    result = data.download_okx_futures(
                        start=args.start,
                        end=args.end,
                        timeframe=timeframe,
                        pairs=batch_pairs,
                        sleep_s=float(args.sleep),
                        pair_start_dates=batch_starts,
                        prepare_auxiliary_files=False,
                    )
                else:
                    result = {}
                    with ThreadPoolExecutor(max_workers=min(workers, len(batch_pairs))) as pool:
                        futures = {
                            pool.submit(
                                data.download_okx_futures,
                                start=args.start,
                                end=args.end,
                                timeframe=timeframe,
                                pairs=[pair],
                                sleep_s=float(args.sleep),
                                pair_start_dates={pair: batch_starts[pair]} if pair in batch_starts else None,
                                prepare_auxiliary_files=False,
                            ): pair
                            for pair in batch_pairs
                        }
                        for future in as_completed(futures):
                            pair = futures[future]
                            try:
                                result.update(future.result())
                            except Exception as exc:
                                result[pair] = 0
                                print(f"[{_utc_now()}] pair failed timeframe={timeframe} pair={pair} error={exc!r}", flush=True)
                                raise
                data.prepare_okx_futures_auxiliary(timeframe=timeframe)
                checkpoint["completed"].append(
                    {
                        "timeframe": timeframe,
                        "batch": batch_i,
                        "pairs": batch_pairs,
                        "result": result,
                        "finished_at_utc": _utc_now(),
                    }
                )
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
        checkpoint["status"] = "DONE"
        checkpoint["finished_at_utc"] = _utc_now()
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
    except KeyboardInterrupt:
        checkpoint["status"] = "INTERRUPTED"
        checkpoint["finished_at_utc"] = _utc_now()
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
        raise
    except Exception as exc:
        checkpoint["status"] = "FAILED"
        checkpoint["error"] = repr(exc)
        checkpoint["finished_at_utc"] = _utc_now()
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
