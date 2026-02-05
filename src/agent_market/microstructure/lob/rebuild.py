from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_ts_from_msg(msg: Dict[str, Any]) -> Optional["datetime"]:
    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
    ts_ms = data.get("time")
    if ts_ms is not None:
        try:
            return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        except Exception:
            pass
    recv = msg.get("_received_at")
    if recv:
        try:
            return datetime.fromisoformat(str(recv).replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def _iter_ndjson_gz(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _topic_symbol(topic: str) -> Optional[str]:
    t = str(topic or "")
    if ":" not in t:
        return None
    return t.split(":", 1)[1].strip() or None


@dataclass
class OrderBook:
    bids: Dict[float, float]
    asks: Dict[float, float]

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> "OrderBook":
        bids_in = snapshot.get("bids") or []
        asks_in = snapshot.get("asks") or []
        bids: Dict[float, float] = {}
        asks: Dict[float, float] = {}
        for px, sz in bids_in:
            p = _as_float(px)
            s = _as_float(sz)
            if p is None or s is None:
                continue
            if s <= 0:
                continue
            bids[float(p)] = float(s)
        for px, sz in asks_in:
            p = _as_float(px)
            s = _as_float(sz)
            if p is None or s is None:
                continue
            if s <= 0:
                continue
            asks[float(p)] = float(s)
        return cls(bids=bids, asks=asks)

    def apply_changes(self, *, bids: Iterable[Iterable[Any]], asks: Iterable[Iterable[Any]]) -> None:
        for px, sz in bids:
            p = _as_float(px)
            s = _as_float(sz)
            if p is None or s is None:
                continue
            if s <= 0:
                self.bids.pop(float(p), None)
            else:
                self.bids[float(p)] = float(s)
        for px, sz in asks:
            p = _as_float(px)
            s = _as_float(sz)
            if p is None or s is None:
                continue
            if s <= 0:
                self.asks.pop(float(p), None)
            else:
                self.asks[float(p)] = float(s)

    def top_n(self, depth: int) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        d = int(depth)
        bid_levels = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:d]
        ask_levels = sorted(self.asks.items(), key=lambda kv: kv[0])[:d]

        bid_px: list[Optional[float]] = [float(px) for px, _ in bid_levels]
        bid_sz: list[Optional[float]] = [float(sz) for _, sz in bid_levels]
        ask_px: list[Optional[float]] = [float(px) for px, _ in ask_levels]
        ask_sz: list[Optional[float]] = [float(sz) for _, sz in ask_levels]

        while len(bid_px) < d:
            bid_px.append(None)
            bid_sz.append(None)
        while len(ask_px) < d:
            ask_px.append(None)
            ask_sz.append(None)
        return bid_px, bid_sz, ask_px, ask_sz

    def summarize(self, depth: int) -> Dict[str, Any]:
        bid_px, bid_sz, ask_px, ask_sz = self.top_n(depth)
        bid1 = bid_px[0]
        ask1 = ask_px[0]
        mid = None
        spread = None
        rel_spread = None
        if bid1 is not None and ask1 is not None:
            mid = (bid1 + ask1) / 2.0
            spread = ask1 - bid1
            rel_spread = spread / (mid + 1e-12)

        depth_bid = sum(float(x or 0.0) for x in bid_sz)
        depth_ask = sum(float(x or 0.0) for x in ask_sz)
        denom = depth_bid + depth_ask
        imbalance = ((depth_bid - depth_ask) / denom) if denom > 0 else None

        return {
            "bid_px": bid_px,
            "bid_sz": bid_sz,
            "ask_px": ask_px,
            "ask_sz": ask_sz,
            "bid1": bid1,
            "ask1": ask1,
            "mid": mid,
            "spread": spread,
            "rel_spread": rel_spread,
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "imbalance": float(imbalance) if imbalance is not None and not math.isnan(float(imbalance)) else None,
        }


def _load_snapshot(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot must be a JSON object")
    return payload


def _parse_level2_update(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if msg.get("type") != "message":
        return None
    topic = str(msg.get("topic") or "")
    sym = _topic_symbol(topic)
    data = msg.get("data")
    if not sym or not isinstance(data, dict):
        return None
    seq_start = data.get("sequenceStart")
    seq_end = data.get("sequenceEnd")
    changes = data.get("changes") or {}
    if not isinstance(changes, dict):
        return None
    asks = changes.get("asks") or []
    bids = changes.get("bids") or []
    if not isinstance(asks, list) or not isinstance(bids, list):
        return None
    return {
        "symbol": sym,
        "seq_start": int(seq_start) if seq_start is not None else None,
        "seq_end": int(seq_end) if seq_end is not None else None,
        "asks": asks,
        "bids": bids,
        "ts": _parse_ts_from_msg(msg),
    }


def rebuild_kucoin_lob(
    *,
    capture_dir: Path,
    snapshot_path: Path,
    symbol: str,
    depth: int = 20,
    out_dir: Path,
) -> Dict[str, Any]:
    """Rebuild KuCoin L2 order book (top-N) from snapshot + level2 deltas."""
    capture_dir = Path(capture_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    level2_path = (capture_dir / "level2.ndjson.gz").resolve()
    if not level2_path.exists():
        raise FileNotFoundError(f"level2.ndjson.gz not found under capture_dir: {capture_dir}")

    snap = _load_snapshot(snapshot_path)
    snap_sym = str(snap.get("symbol") or "").strip()
    if snap_sym and snap_sym != str(symbol):
        raise ValueError(f"snapshot symbol mismatch: {snap_sym!r} != {symbol!r}")
    snap_seq = int(snap.get("sequence") or 0)
    book = OrderBook.from_snapshot(snap)

    rows: list[Dict[str, Any]] = []
    gaps: list[Dict[str, Any]] = []
    last_seq: Optional[int] = snap_seq if snap_seq else None

    # include snapshot as first row
    rows.append(
        {
            "ts": datetime.fromtimestamp(0, tz=timezone.utc),
            "symbol": str(symbol),
            "event": "snapshot",
            "seq_start": None,
            "seq_end": int(snap_seq) if snap_seq else None,
            **book.summarize(depth),
        }
    )

    updates = 0
    for msg in _iter_ndjson_gz(level2_path):
        upd = _parse_level2_update(msg)
        if not upd:
            continue
        if upd["symbol"] != str(symbol):
            continue
        seq_start = upd.get("seq_start")
        seq_end = upd.get("seq_end")
        if seq_start is not None and last_seq is not None and seq_start != last_seq + 1:
            gaps.append(
                {
                    "expected_start": last_seq + 1,
                    "got_start": seq_start,
                    "got_end": seq_end,
                }
            )
        book.apply_changes(bids=upd["bids"], asks=upd["asks"])
        last_seq = seq_end if isinstance(seq_end, int) else last_seq
        updates += 1

        ts = upd.get("ts") or datetime.fromtimestamp(0, tz=timezone.utc)
        rows.append(
            {
                "ts": ts,
                "symbol": str(symbol),
                "event": "delta",
                "seq_start": seq_start,
                "seq_end": seq_end,
                **book.summarize(depth),
            }
        )

    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing pandas dependency: {exc}") from exc

    df = pd.DataFrame(rows)
    df = df.sort_values(["ts", "event"]).reset_index(drop=True)

    lob_path = (out_dir / "lob_state.parquet").resolve()
    df.to_parquet(lob_path, index=False)

    report = {
        "version": 1,
        "exchange": "kucoin",
        "symbol": str(symbol),
        "depth": int(depth),
        "started_at": _utc_now_iso(),
        "snapshot": {
            "path": str(Path(snapshot_path).resolve()),
            "sequence": int(snap_seq) if snap_seq else None,
            "bids": len(book.bids),
            "asks": len(book.asks),
        },
        "capture_dir": str(capture_dir),
        "level2_file": str(level2_path),
        "updates": int(updates),
        "rows": int(df.shape[0]),
        "gaps": {"count": int(len(gaps)), "examples": gaps[:5]},
        "outputs": {
            "lob_state_parquet": str(lob_path),
        },
    }
    report_path = (out_dir / "rebuild_report.json").resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"lob_state": lob_path, "report": report_path}


__all__ = ["rebuild_kucoin_lob"]

