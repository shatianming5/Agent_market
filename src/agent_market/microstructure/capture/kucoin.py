from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from .writer import CaptureWriter


@dataclass(frozen=True, slots=True)
class KuCoinBullet:
    endpoint: str
    token: str
    ping_interval_sec: float
    ping_timeout_sec: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _topic_for_channel(channel: str, symbol: str) -> str:
    ch = str(channel or "").strip().lower()
    sym = str(symbol or "").strip()
    if not sym:
        raise ValueError("symbol is required")
    if ch == "match":
        return f"/market/match:{sym}"
    if ch == "level2":
        return f"/market/level2:{sym}"
    raise ValueError(f"Unsupported channel: {channel!r}")


def infer_channel_from_topic(topic: str) -> Optional[str]:
    t = str(topic or "")
    if t.startswith("/market/match:"):
        return "match"
    if t.startswith("/market/level2:"):
        return "level2"
    return None


def _symbol_from_topic(topic: str) -> Optional[str]:
    t = str(topic or "")
    if ":" not in t:
        return None
    _, sym = t.split(":", 1)
    sym = sym.strip()
    return sym or None


def _reconnect_backoff_sec(attempt: int) -> float:
    """
    Exponential backoff with a cap, suitable for long-running capture.

    attempt is 1-based.
    """

    a = max(1, int(attempt))
    base = 0.5
    cap = 30.0
    # 0.5, 1, 2, 4, 8, 16, 30, 30...
    return min(cap, base * (2 ** min(6, a - 1)))


class KuCoinLevel2SeqGapTracker:
    """
    Lightweight sequence-gap tracker for KuCoin level2 updates without a snapshot.

    It tracks per-symbol last sequenceEnd and counts gaps where sequenceStart != last_end + 1.
    """

    def __init__(self, *, example_limit: int = 5) -> None:
        self._example_limit = max(0, int(example_limit))
        self._last_seq_end: dict[str, int] = {}
        self._per_symbol_updates: dict[str, int] = {}
        self._per_symbol_gaps: dict[str, int] = {}
        self._gap_count = 0
        self._examples: list[dict] = []
        self._scanned_updates = 0

    def observe(self, *, topic: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        seq_start = data.get("sequenceStart")
        seq_end = data.get("sequenceEnd")
        if seq_start is None or seq_end is None:
            return
        try:
            seq_start_i = int(seq_start)
            seq_end_i = int(seq_end)
        except Exception:
            return

        sym = str(data.get("symbol") or "").strip() or (_symbol_from_topic(topic) or "")
        if not sym:
            return

        self._scanned_updates += 1
        self._per_symbol_updates[sym] = int(self._per_symbol_updates.get(sym) or 0) + 1

        last_end = self._last_seq_end.get(sym)
        if last_end is not None and seq_start_i != int(last_end) + 1:
            self._gap_count += 1
            self._per_symbol_gaps[sym] = int(self._per_symbol_gaps.get(sym) or 0) + 1
            if self._example_limit and len(self._examples) < self._example_limit:
                self._examples.append(
                    {
                        "symbol": sym,
                        "expected_start": int(last_end) + 1,
                        "got_start": int(seq_start_i),
                        "got_end": int(seq_end_i),
                    }
                )

        self._last_seq_end[sym] = int(seq_end_i)

    def meta(self) -> Dict[str, Any]:
        per_symbol = {}
        for sym in sorted(set(self._per_symbol_updates.keys()) | set(self._per_symbol_gaps.keys())):
            per_symbol[sym] = {
                "updates": int(self._per_symbol_updates.get(sym) or 0),
                "gaps": int(self._per_symbol_gaps.get(sym) or 0),
                "last_seq_end": int(self._last_seq_end.get(sym) or 0),
            }
        return {
            "count": int(self._gap_count),
            "examples": list(self._examples),
            "per_symbol": per_symbol,
            "scanned_updates": int(self._scanned_updates),
        }


async def fetch_bullet_public(*, base_url: str = "https://api.kucoin.com") -> KuCoinBullet:
    import httpx  # noqa: PLC0415

    url = str(base_url).rstrip("/") + "/api/v1/bullet-public"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url)
        resp.raise_for_status()
        payload = resp.json()
    if str(payload.get("code")) != "200000":
        raise RuntimeError(f"KuCoin bullet-public failed: {payload!r}")
    data = payload.get("data") or {}
    token = str(data.get("token") or "").strip()
    servers = data.get("instanceServers") or []
    if not token or not servers:
        raise RuntimeError(f"KuCoin bullet-public missing token/servers: {payload!r}")
    server0 = servers[0] if isinstance(servers, list) else None
    if not isinstance(server0, dict):
        raise RuntimeError(f"KuCoin bullet-public invalid servers: {payload!r}")
    endpoint = str(server0.get("endpoint") or "").strip()
    ping_interval_ms = float(server0.get("pingInterval") or 50000)
    ping_timeout_ms = float(server0.get("pingTimeout") or 10000)
    if not endpoint:
        raise RuntimeError(f"KuCoin bullet-public missing endpoint: {payload!r}")
    return KuCoinBullet(
        endpoint=endpoint,
        token=token,
        ping_interval_sec=max(1.0, ping_interval_ms / 1000.0),
        ping_timeout_sec=max(1.0, ping_timeout_ms / 1000.0),
    )


async def _ping_loop(ws: Any, interval_sec: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(max(1.0, interval_sec))
        try:
            await ws.send_json({"id": uuid.uuid4().hex, "type": "ping"})
        except Exception:
            return


async def capture_kucoin_ws(
    *,
    symbols: Iterable[str],
    channels: Iterable[str],
    duration_sec: float,
    writer: CaptureWriter,
    base_url: str = "https://api.kucoin.com",
    max_reconnects: int = 3,
) -> Dict[str, Any]:
    """Capture KuCoin public websocket data for `duration_sec`.

    Returns meta dict (best-effort) for manifest.
    """
    started_at = _utc_now_iso()

    desired_channels = {str(c).strip().lower() for c in channels if str(c).strip()}
    desired_symbols = [str(s).strip() for s in symbols if str(s).strip()]
    topics = []
    for sym in desired_symbols:
        for ch in sorted(desired_channels):
            topics.append(_topic_for_channel(ch, sym))

    reconnects = 0
    connections = 0
    errors: list[str] = []
    disconnects: list[str] = []
    deadline = asyncio.get_running_loop().time() + float(duration_sec)
    unlimited_reconnects = int(max_reconnects) <= 0

    bullet_last: Optional[KuCoinBullet] = None
    endpoints: list[str] = []
    ws_url_initial: Optional[str] = None
    ws_url_last: Optional[str] = None
    level2_tracker: Optional[KuCoinLevel2SeqGapTracker] = (
        KuCoinLevel2SeqGapTracker() if "level2" in desired_channels else None
    )

    import aiohttp  # noqa: PLC0415

    async with aiohttp.ClientSession() as session:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                # Refresh bullet/token for each (re)connect attempt.
                bullet = await fetch_bullet_public(base_url=base_url)
                bullet_last = bullet
                endpoints.append(str(bullet.endpoint))
                connect_id = uuid.uuid4().hex
                ws_url = f"{bullet.endpoint}?token={bullet.token}&connectId={connect_id}"
                ws_url_last = ws_url
                if ws_url_initial is None:
                    ws_url_initial = ws_url

                async with session.ws_connect(ws_url, heartbeat=None, autoping=False) as ws:
                    connections += 1
                    stop_ping = asyncio.Event()
                    ping_task = asyncio.create_task(
                        _ping_loop(ws, interval_sec=bullet.ping_interval_sec * 0.8, stop=stop_ping)
                    )
                    try:
                        # Subscribe
                        for topic in topics:
                            await ws.send_json(
                                {
                                    "id": uuid.uuid4().hex,
                                    "type": "subscribe",
                                    "topic": topic,
                                    "privateChannel": False,
                                    "response": True,
                                }
                            )

                        # Receive
                        while True:
                            remaining = deadline - asyncio.get_running_loop().time()
                            if remaining <= 0:
                                break
                            try:
                                msg = await ws.receive(timeout=min(5.0, remaining))
                            except asyncio.TimeoutError:
                                continue

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                except Exception:
                                    continue
                                if not isinstance(data, dict):
                                    continue
                                if data.get("type") != "message":
                                    continue
                                ch = infer_channel_from_topic(str(data.get("topic") or ""))
                                if ch is None or ch not in desired_channels:
                                    continue
                                if ch == "level2" and level2_tracker is not None:
                                    level2_tracker.observe(
                                        topic=str(data.get("topic") or ""),
                                        data=data.get("data"),
                                    )
                                data["_received_at"] = _utc_now_iso()
                                data["_exchange"] = "kucoin"
                                data["_channel"] = ch
                                writer.write(ch, data)
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.CLOSE,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break
                    finally:
                        stop_ping.set()
                        try:
                            await ping_task
                        except Exception:
                            pass
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                disconnects.append("ws_closed")
                reconnects += 1
                if (not unlimited_reconnects) and reconnects > int(max_reconnects):
                    break
                await asyncio.sleep(min(_reconnect_backoff_sec(reconnects), max(0.0, remaining)))
            except Exception as exc:
                reconnects += 1
                errors.append(str(exc))
                if (not unlimited_reconnects) and reconnects > int(max_reconnects):
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(_reconnect_backoff_sec(reconnects), max(0.0, remaining)))

    ended_at = _utc_now_iso()
    endpoints_unique = [x for x in endpoints if x]
    if not endpoints_unique and bullet_last is not None:
        endpoints_unique = [str(bullet_last.endpoint)]
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "ws_url": ws_url_last,
        "ws_url_initial": ws_url_initial,
        "endpoints": endpoints_unique,
        "endpoint": (bullet_last.endpoint if bullet_last is not None else None),
        "ping_interval_sec": (bullet_last.ping_interval_sec if bullet_last is not None else None),
        "ping_timeout_sec": (bullet_last.ping_timeout_sec if bullet_last is not None else None),
        "symbols": desired_symbols,
        "channels": sorted(desired_channels),
        "topics": topics,
        "max_reconnects": int(max_reconnects),
        "unlimited_reconnects": bool(unlimited_reconnects),
        "connections": int(connections),
        "reconnects": int(reconnects),
        "errors": errors,
        "disconnects": disconnects[:10],
        "level2_seq_gaps": (level2_tracker.meta() if level2_tracker is not None else None),
    }


__all__ = [
    "KuCoinBullet",
    "KuCoinLevel2SeqGapTracker",
    "capture_kucoin_ws",
    "fetch_bullet_public",
    "infer_channel_from_topic",
]
