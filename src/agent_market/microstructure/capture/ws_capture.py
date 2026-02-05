from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from agent_market.microstructure.capture.kucoin import capture_kucoin_ws, infer_channel_from_topic
from agent_market.microstructure.capture.writer import CaptureWriter


@dataclass(frozen=True, slots=True)
class CaptureSession:
    exchange: str
    out_dir: Path
    channels: list[str]
    symbols: list[str]
    mode: str  # live|fixture
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


async def _capture_live(
    *,
    out_dir: Path,
    exchange: str,
    symbols: Sequence[str],
    channels: Sequence[str],
    duration_sec: float,
) -> Dict[str, Any]:
    writer = CaptureWriter(out_dir, channels=list(channels))
    try:
        if exchange != "kucoin":
            raise NotImplementedError(f"Unsupported exchange for live ws_capture: {exchange!r}")
        meta = await capture_kucoin_ws(
            symbols=list(symbols),
            channels=list(channels),
            duration_sec=float(duration_sec),
            writer=writer,
        )
        counts = {k: v.count for k, v in writer.files.items()}
        return {"mode": "live", "exchange": exchange, "counts": counts, **meta}
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
    Capture microstructure data via websocket (or offline fixture replay).

    This mirrors the behavior of `scripts/micro_capture.py` but provides a reusable
    module path that matches plan.md (`capture/ws_capture.py`).
    """

    ex = str(exchange or "").strip().lower() or "kucoin"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = [c.lower() for c in channels if str(c).strip()]
    syms = [s for s in symbols if str(s).strip()]

    if fixture is not None:
        meta = asyncio.run(_capture_fixture(fixture=Path(fixture), out_dir=out_dir, exchange=ex, channels=ch))
        mode = "fixture"
    else:
        meta = asyncio.run(
            _capture_live(
                out_dir=out_dir,
                exchange=ex,
                symbols=syms,
                channels=ch,
                duration_sec=float(duration_sec),
            )
        )
        mode = "live"

    return CaptureSession(exchange=ex, out_dir=out_dir, channels=ch, symbols=syms, mode=mode, meta=dict(meta))


__all__ = ["CaptureSession", "capture_ws"]

