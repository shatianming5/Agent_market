#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _lib import resolve_path  # noqa: E402


def fetch_kucoin_l2_snapshot(*, symbol: str, depth: int = 20, timeout: float = 15.0) -> Dict[str, Any]:
    d = int(depth)
    if d not in (20, 100):
        # KuCoin offers level2_20 and level2_100 for full book snapshots.
        d = 20

    sym = str(symbol).strip()
    if not sym:
        raise ValueError("symbol is required (e.g. BTC-USDT)")

    url = f"https://api.kucoin.com/api/v1/market/orderbook/level2_{d}"
    resp = requests.get(url, params={"symbol": sym}, timeout=float(timeout))
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("KuCoin snapshot response is not an object")
    if str(payload.get("code") or "") not in {"200000", "OK", "0"}:
        raise ValueError(f"KuCoin snapshot error: {payload.get('code')!r} {payload.get('msg')!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("KuCoin snapshot missing data object")

    bids = data.get("bids") or []
    asks = data.get("asks") or []
    seq = data.get("sequence")

    if not isinstance(bids, list) or not isinstance(asks, list):
        raise ValueError("KuCoin snapshot bids/asks must be lists")
    if seq is None:
        raise ValueError("KuCoin snapshot missing sequence")

    return {
        "version": 1,
        "exchange": "kucoin",
        "symbol": sym,
        "depth": int(d),
        "sequence": int(seq),
        "time": data.get("time"),
        "bids": bids,
        "asks": asks,
        "endpoint": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch KuCoin L2 snapshot JSON for lob_rebuild (level2_20/100).")
    p.add_argument("--symbol", default="BTC-USDT", help="KuCoin symbol, e.g. BTC-USDT")
    p.add_argument("--depth", type=int, default=20, choices=[20, 100])
    p.add_argument("--out", required=True, help="Output snapshot JSON path")
    p.add_argument("--timeout", type=float, default=15.0)
    args = p.parse_args(argv)

    out_path = resolve_path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snap = fetch_kucoin_l2_snapshot(symbol=str(args.symbol), depth=int(args.depth), timeout=float(args.timeout))
    out_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[snapshot] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

