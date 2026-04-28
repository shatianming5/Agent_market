from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

from .writer import CaptureWriter


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _ns_to_ms(ts: Any) -> Optional[int]:
    v = _as_int(ts)
    if v is None:
        return None
    # Heuristic: KuCoin REST trade history uses ns.
    if v > 10_000_000_000_000_000:  # > 1e16 => ns
        return int(v // 1_000_000)
    if v > 10_000_000_000_000:  # > 1e13 => us
        return int(v // 1_000)
    return int(v)  # assume ms


def _snapshot_level2(
    *,
    symbol: str,
    depth: int = 20,
    timeout: float = 15.0,
) -> Tuple[int, int, list, list]:
    d = int(depth)
    if d not in (20, 100):
        d = 20
    url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{d}"
    resp = requests.get(url, params={"symbol": symbol}, timeout=float(timeout))
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or str(payload.get("code") or "") != "200000":
        raise RuntimeError(f"KuCoin snapshot error: {payload!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"KuCoin snapshot missing data: {payload!r}")
    seq = int(data.get("sequence") or 0)
    ts_ms = int(data.get("time") or 0)
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    if not isinstance(bids, list) or not isinstance(asks, list):
        raise RuntimeError("KuCoin snapshot bids/asks not lists")
    return ts_ms, seq, bids, asks


def fetch_lob_rebuild_snapshot(
    *,
    symbol: str,
    depth: int = 20,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Fetch a snapshot JSON compatible with `scripts/lob_rebuild.py --snapshot`."""
    ts_ms, seq, bids, asks = _snapshot_level2(symbol=symbol, depth=int(depth), timeout=float(timeout))
    return {
        "symbol": str(symbol),
        "sequence": int(seq),
        "time": int(ts_ms),
        "bids": bids,
        "asks": asks,
    }


def _histories(
    *,
    symbol: str,
    timeout: float = 15.0,
) -> Iterable[Dict[str, Any]]:
    url = "https://api.kucoin.com/api/v1/market/histories"
    resp = requests.get(url, params={"symbol": symbol}, timeout=float(timeout))
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or str(payload.get("code") or "") != "200000":
        raise RuntimeError(f"KuCoin histories error: {payload!r}")
    data = payload.get("data") or []
    if not isinstance(data, list):
        return []
    return data


def _to_book(levels: list) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in levels:
        try:
            px, sz = item
        except Exception:
            continue
        try:
            p = str(px)
            s = float(sz)
        except Exception:
            continue
        out[p] = s
    return out


def _diff_levels(prev: list, cur: list) -> list:
    """Return KuCoin-style changes array [[px, size], ...] from two snapshots."""
    a = _to_book(prev)
    b = _to_book(cur)
    keys = set(a) | set(b)
    changes = []
    for k in keys:
        old = a.get(k)
        new = b.get(k)
        if old is None and new is None:
            continue
        if old is None:
            changes.append([k, str(new)])
            continue
        if new is None:
            changes.append([k, "0"])
            continue
        try:
            if float(old) == float(new):
                continue
        except Exception:
            pass
        changes.append([k, str(new)])
    return changes


def capture_kucoin_rest(
    *,
    symbol: str,
    channels: list[str],
    duration_sec: float,
    writer: CaptureWriter,
    poll_interval_sec: float = 1.0,
    timeout: float = 15.0,
    depth: int = 20,
    seed_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Best-effort capture via KuCoin REST polling (fallback when WS is unreachable).

    Writes WS-compatible ndjson.gz records for:
      - match: /market/match:{symbol}
      - level2: synthetic /market/level2:{symbol} deltas from snapshot diffs
    """
    started_at = _utc_now_iso()
    desired = {c.lower().strip() for c in (channels or []) if str(c).strip()}
    deadline = time.time() + float(duration_sec)

    last_seq: Optional[int] = None
    last_bids: Optional[list] = None
    last_asks: Optional[list] = None
    last_trade_seq: Optional[int] = None

    snapshot_polls = 0
    trade_polls = 0
    errors: list[str] = []

    seed_sequence = None
    if seed_snapshot:
        try:
            if str(seed_snapshot.get("symbol") or "").strip() == str(symbol):
                last_seq = _as_int(seed_snapshot.get("sequence"))
                last_bids = seed_snapshot.get("bids") if isinstance(seed_snapshot.get("bids"), list) else None
                last_asks = seed_snapshot.get("asks") if isinstance(seed_snapshot.get("asks"), list) else None
                seed_sequence = last_seq
        except Exception:
            pass

    while True:
        now = time.time()
        if now >= deadline:
            break

        loop_started = now
        try:
            if "level2" in desired:
                ts_ms, seq, bids, asks = _snapshot_level2(symbol=symbol, depth=int(depth), timeout=float(timeout))
                snapshot_polls += 1
                if last_seq is not None and last_bids is not None and last_asks is not None:
                    changes = {
                        "bids": _diff_levels(last_bids, bids),
                        "asks": _diff_levels(last_asks, asks),
                    }
                    msg = {
                        "type": "message",
                        "topic": f"/market/level2:{symbol}",
                        "subject": "trade.l2update",
                        "data": {
                            "sequenceStart": int(last_seq) + 1,
                            "sequenceEnd": int(seq),
                            "symbol": symbol,
                            "time": int(ts_ms),
                            "changes": changes,
                        },
                        "_received_at": _utc_now_iso(),
                        "_exchange": "kucoin",
                        "_channel": "level2",
                    }
                    writer.write("level2", msg)
                last_seq = int(seq)
                last_bids = bids
                last_asks = asks
        except Exception as exc:  # noqa: BLE001
            errors.append(f"level2_poll_failed: {exc!s}")

        try:
            if "match" in desired:
                rows = list(_histories(symbol=symbol, timeout=float(timeout)))
                trade_polls += 1
                # KuCoin REST returns newest first; write in time order for nicer downstream.
                rows = list(reversed(rows))
                for row in rows:
                    seq = _as_int(row.get("sequence"))
                    if seq is None:
                        continue
                    if last_trade_seq is not None and seq <= last_trade_seq:
                        continue
                    ts_ms = _ns_to_ms(row.get("time"))
                    if ts_ms is None:
                        continue
                    msg = {
                        "type": "message",
                        "topic": f"/market/match:{symbol}",
                        "subject": "trade.l3match",
                        "data": {
                            "symbol": symbol,
                            "side": row.get("side"),
                            "size": row.get("size"),
                            "price": row.get("price"),
                            "time": int(ts_ms),
                        },
                        "_received_at": _utc_now_iso(),
                        "_exchange": "kucoin",
                        "_channel": "match",
                    }
                    writer.write("match", msg)
                    last_trade_seq = int(seq) if last_trade_seq is None else max(int(last_trade_seq), int(seq))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"match_poll_failed: {exc!s}")

        # sleep to respect poll interval
        sleep_for = float(poll_interval_sec) - (time.time() - loop_started)
        if sleep_for > 0:
            time.sleep(min(sleep_for, max(0.0, deadline - time.time())))

    ended_at = _utc_now_iso()
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "mode": "rest",
        "symbol": symbol,
        "seed_sequence": seed_sequence,
        "poll_interval_sec": float(poll_interval_sec),
        "timeout": float(timeout),
        "depth": int(depth),
        "snapshot_polls": int(snapshot_polls),
        "trade_polls": int(trade_polls),
        "errors": errors[:20],
    }


__all__ = ["capture_kucoin_rest", "fetch_lob_rebuild_snapshot"]
