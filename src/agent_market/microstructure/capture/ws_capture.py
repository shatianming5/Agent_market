from __future__ import annotations

"""Capture microstructure data (REST polling or offline fixture).

This module historically supported WebSocket capture. WS support has been removed;
the entrypoint remains for backward compatibility but uses REST polling.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from agent_market.microstructure.capture.kucoin import infer_channel_from_topic
from agent_market.microstructure.capture.kucoin_rest import (
    capture_kucoin_rest,
    fetch_lob_rebuild_snapshot,
)
from agent_market.microstructure.capture.writer import CaptureWriter


@dataclass(frozen=True, slots=True)
class CaptureSession:
    exchange: str
    out_dir: Path
    channels: list[str]
    symbols: list[str]
    mode: str  # rest|fixture
    meta: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    raw = str(value).replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


async def _capture_fixture(
    *,
    fixture: Path,
    out_dir: Path,
    exchange: str,
    channels: Sequence[str],
) -> Dict[str, Any]:
    started_at = _utc_now_iso()
    writer = CaptureWriter(out_dir, channels=list(channels))
    try:
        for line in Path(fixture).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                msg = json.loads(s)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            topic = str(msg.get("topic") or "")
            ch = infer_channel_from_topic(topic) or "unknown"
            if ch not in channels:
                continue
            msg["_received_at"] = _utc_now_iso()
            msg["_exchange"] = exchange
            msg["_channel"] = ch
            writer.write(ch, msg)
    finally:
        counts = {k: v.count for k, v in writer.files.items()}
        writer.close()
    ended_at = _utc_now_iso()
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "mode": "fixture",
        "exchange": exchange,
        "channels": list(channels),
        "counts": counts,
    }


def _capture_rest(
    *,
    out_dir: Path,
    exchange: str,
    symbols: Sequence[str],
    channels: Sequence[str],
    duration_sec: float,
    poll_interval_sec: float = 1.0,
    timeout_sec: float = 15.0,
) -> Dict[str, Any]:
    if exchange != "kucoin":
        raise NotImplementedError(f"Unsupported exchange for rest capture: {exchange!r}")
    syms = [str(s) for s in symbols if str(s).strip()]
    if not syms:
        raise ValueError("No symbols provided")
    if len(syms) != 1:
        raise NotImplementedError("REST capture currently supports exactly 1 symbol per session")

    # Ensure a snapshot.json exists, and seed delta diffs from it.
    snapshot_path = (Path(out_dir) / "snapshot.json").resolve()
    seed_snapshot: Optional[Dict[str, Any]] = None
    if snapshot_path.exists():
        try:
            seed_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
        except Exception:
            seed_snapshot = None
    if not isinstance(seed_snapshot, dict):
        seed_snapshot = fetch_lob_rebuild_snapshot(symbol=str(syms[0]), depth=20, timeout=float(timeout_sec))
        snapshot_path.write_text(json.dumps(seed_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    writer = CaptureWriter(Path(out_dir), channels=list(channels))
    try:
        meta = capture_kucoin_rest(
            symbol=str(syms[0]),
            channels=list(channels),
            duration_sec=float(duration_sec),
            writer=writer,
            poll_interval_sec=float(poll_interval_sec),
            timeout=float(timeout_sec),
            depth=20,
            seed_snapshot=seed_snapshot,
        )
        counts = {k: v.count for k, v in writer.files.items()}
        return {"mode": "rest", "exchange": exchange, "counts": counts, **meta}
    finally:
        writer.close()


def capture_ws(
    *,
    exchange: str = "kucoin",
    out_dir: Path,
    channels: Sequence[str] = ("match", "level2"),
    symbols: Sequence[str] = ("BTC-USDT",),
    duration_sec: float = 60.0,
    fixture: Optional[Path] = None,
) -> CaptureSession:
    """
    Capture microstructure data via REST polling (or offline fixture replay).

    Kept as `capture_ws()` for backward compatibility.
    """

    ex = str(exchange or "").strip().lower() or "kucoin"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = [c.lower() for c in channels if str(c).strip()]
    syms = [s for s in symbols if str(s).strip()]

    if fixture is not None:
        # Fixture capture is synchronous, but historically lived behind an asyncio wrapper.
        import asyncio  # noqa: PLC0415

        meta = asyncio.run(_capture_fixture(fixture=Path(fixture), out_dir=out_dir, exchange=ex, channels=ch))
        mode = "fixture"
    else:
        meta = _capture_rest(out_dir=out_dir, exchange=ex, symbols=syms, channels=ch, duration_sec=float(duration_sec))
        mode = "rest"

    return CaptureSession(exchange=ex, out_dir=out_dir, channels=ch, symbols=syms, mode=mode, meta=dict(meta))


__all__ = ["CaptureSession", "capture_ws"]
