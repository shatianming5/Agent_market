#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _lib import parse_csv, resolve_path, utc_now_compact  # noqa: E402
from agent_market import paths  # noqa: E402
from agent_market.runtime_preflight import run_capture_preflight  # noqa: E402


def _ensure_imports() -> None:
    return


def _run_fixture(*, fixture: Path, out_dir: Path, exchange: str, channels: list[str]) -> dict:
    _ensure_imports()
    from agent_market.microstructure.capture.kucoin import (  # noqa: WPS433
        KuCoinLevel2SeqGapTracker,
        infer_channel_from_topic,
    )
    from agent_market.microstructure.capture.writer import CaptureWriter  # noqa: WPS433

    started_at = datetime.now(timezone.utc).isoformat()
    writer = CaptureWriter(out_dir, channels=channels)
    level2_tracker = KuCoinLevel2SeqGapTracker() if "level2" in set(channels) else None
    try:
        for line in fixture.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if not isinstance(msg, dict):
                continue
            topic = str(msg.get("topic") or "")
            ch = infer_channel_from_topic(topic) or "unknown"
            if ch not in channels:
                continue
            if ch == "level2" and level2_tracker is not None:
                level2_tracker.observe(topic=topic, data=msg.get("data"))
            msg["_received_at"] = datetime.now(timezone.utc).isoformat()
            msg["_exchange"] = exchange
            msg["_channel"] = ch
            writer.write(ch, msg)
    finally:
        counts = {k: v.count for k, v in writer.files.items()}
        writer.close()

    ended_at = datetime.now(timezone.utc).isoformat()
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "mode": "fixture",
        "exchange": exchange,
        "channels": channels,
        "counts": counts,
        "level2_seq_gaps": (level2_tracker.meta() if level2_tracker is not None else None),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture KuCoin market data (REST polling or fixture).")
    parser.add_argument("--exchange", default="kucoin")
    parser.add_argument("--symbols", default="BTC-USDT", help="Comma/space separated, e.g. BTC-USDT ETH-USDT")
    parser.add_argument("--channels", default="match,level2", help="Comma/space separated: match level2")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument(
        "--mode",
        choices=["rest"],
        default="rest",
        help="Capture transport (only REST polling is supported).",
    )
    parser.add_argument("--poll-interval-sec", type=float, default=1.0, help="[rest] poll interval seconds")
    parser.add_argument("--rest-timeout-sec", type=float, default=15.0, help="[rest] request timeout seconds")
    parser.add_argument("--out-dir", default=None, help="Session output directory (default: user_data/micro_capture/<ex>/<date>/<session_id>)")
    parser.add_argument("--fixture", default=None, help="Offline replay fixture (jsonl) to avoid network")
    args = parser.parse_args(argv)

    exchange = str(args.exchange or "").strip().lower()
    if exchange != "kucoin":
        raise SystemExit(f"Unsupported exchange: {exchange!r} (only 'kucoin' supported)")

    channels = parse_csv(str(args.channels))
    if not channels:
        raise SystemExit("No channels provided")
    channels = [c.lower() for c in channels]
    allowed = {"match", "level2"}
    unknown = sorted(set(channels) - allowed)
    if unknown:
        raise SystemExit(f"Unsupported channels: {unknown}. Allowed: {sorted(allowed)}")

    session_id = f"{utc_now_compact()}-{uuid.uuid4().hex[:8]}"
    if args.out_dir:
        out_dir = resolve_path(str(args.out_dir))
    else:
        out_dir = (
            paths.user_data_root()
            / "micro_capture"
            / exchange
            / datetime.now(timezone.utc).strftime("%Y%m%d")
            / session_id
        ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = resolve_path(str(args.fixture)) if args.fixture else None
    symbols = parse_csv(str(args.symbols))
    run_capture_preflight(
        exchange=exchange,
        channels=channels,
        symbols=symbols,
        fixture=fixture,
        out_dir=out_dir,
    )

    if fixture is not None:
        meta = _run_fixture(fixture=fixture, out_dir=out_dir, exchange=exchange, channels=channels)
    else:
        if not symbols:
            raise SystemExit("No symbols provided")
        mode = str(args.mode or "ws").strip().lower()
        if mode == "rest":
            from agent_market.microstructure.capture.writer import CaptureWriter  # noqa: WPS433
            from agent_market.microstructure.capture.kucoin_rest import (  # noqa: WPS433
                capture_kucoin_rest,
                fetch_lob_rebuild_snapshot,
            )

            # Ensure a snapshot.json exists for lob_rebuild and to seed stable delta diffs.
            snapshot_path = (out_dir / "snapshot.json").resolve()
            seed_snapshot = None
            if snapshot_path.exists():
                try:
                    seed_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
                except Exception:
                    seed_snapshot = None
            if not isinstance(seed_snapshot, dict):
                seed_snapshot = fetch_lob_rebuild_snapshot(symbol=str(symbols[0]), depth=20, timeout=float(args.rest_timeout_sec))
                snapshot_path.write_text(json.dumps(seed_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

            writer = CaptureWriter(out_dir, channels=channels)
            try:
                meta = capture_kucoin_rest(
                    symbol=str(symbols[0]),
                    channels=channels,
                    duration_sec=float(args.duration_sec),
                    writer=writer,
                    poll_interval_sec=float(args.poll_interval_sec),
                    timeout=float(args.rest_timeout_sec),
                    depth=20,
                    seed_snapshot=seed_snapshot if isinstance(seed_snapshot, dict) else None,
                )
                counts = {k: v.count for k, v in writer.files.items()}
                meta = {"mode": "rest", "exchange": exchange, "counts": counts, **meta}
            finally:
                writer.close()
        else:
            raise SystemExit(f"Unsupported mode: {mode!r} (only 'rest' supported)")

    # Write manifest.
    _ensure_imports()
    counts = meta.get("counts") or {}
    files = []
    for ch in channels:
        try:
            rel = str((out_dir / f"{ch}.ndjson.gz").resolve().relative_to(out_dir))
        except Exception:
            rel = str((out_dir / f"{ch}.ndjson.gz").resolve())
        files.append({"channel": ch, "path": rel, "count": int(counts.get(ch) or 0)})

    manifest = {
        "version": 1,
        "exchange": exchange,
        "session_id": session_id,
        "out_dir": str(out_dir),
        "symbols": parse_csv(str(args.symbols)) if not fixture else meta.get("symbols") or [],
        "channels": channels,
        **{k: v for k, v in meta.items() if k not in {"symbols", "channels", "counts"}},
        "files": files,
    }
    manifest_path = (out_dir / "manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture] out_dir={out_dir}")
    print(f"[capture] wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
