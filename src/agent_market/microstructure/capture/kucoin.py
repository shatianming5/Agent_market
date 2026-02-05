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
    bullet = await fetch_bullet_public(base_url=base_url)
    connect_id = uuid.uuid4().hex
    ws_url = f"{bullet.endpoint}?token={bullet.token}&connectId={connect_id}"

    desired_channels = {str(c).strip().lower() for c in channels if str(c).strip()}
    desired_symbols = [str(s).strip() for s in symbols if str(s).strip()]
    topics = []
    for sym in desired_symbols:
        for ch in sorted(desired_channels):
            topics.append(_topic_for_channel(ch, sym))

    reconnects = 0
    errors: list[str] = []
    deadline = asyncio.get_running_loop().time() + float(duration_sec)

    import aiohttp  # noqa: PLC0415

    async with aiohttp.ClientSession() as session:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                async with session.ws_connect(ws_url, heartbeat=None, autoping=False) as ws:
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
                # If we exited normally, we're done.
                break
            except Exception as exc:
                reconnects += 1
                errors.append(str(exc))
                if reconnects > int(max_reconnects):
                    break
                await asyncio.sleep(min(5.0, 0.5 * reconnects))

    ended_at = _utc_now_iso()
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "ws_url": ws_url,
        "endpoint": bullet.endpoint,
        "ping_interval_sec": bullet.ping_interval_sec,
        "ping_timeout_sec": bullet.ping_timeout_sec,
        "symbols": desired_symbols,
        "channels": sorted(desired_channels),
        "topics": topics,
        "reconnects": int(reconnects),
        "errors": errors,
    }


__all__ = [
    "KuCoinBullet",
    "capture_kucoin_ws",
    "fetch_bullet_public",
    "infer_channel_from_topic",
]
