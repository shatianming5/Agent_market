#!/usr/bin/env python3
"""Direct ccxt downloader for kucoin 15m — bypasses freqtrade's reload_markets
timeout issue. Fetches 1500-bar chunks and writes feather files compatible
with freqtrade's user_data/data/kucoin/<PAIR>-15m.feather layout.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import ccxt  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "data" / "kucoin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
         "DOGE/USDT", "AVAX/USDT", "ADA/USDT", "LINK/USDT", "DOT/USDT"]
TIMEFRAME = "15m"
SINCE = int(datetime(2023, 5, 15, tzinfo=timezone.utc).timestamp() * 1000)
UNTIL = int(datetime(2026, 4, 12, tzinfo=timezone.utc).timestamp() * 1000)
MS_PER_BAR = 15 * 60 * 1000
LIMIT = 1500  # kucoin max per request


def fetch_pair(ex, pair: str) -> int:
    out_path = OUT_DIR / f"{pair.replace('/', '_')}-{TIMEFRAME}.feather"
    # Resume: if file exists, start after last bar
    start = SINCE
    existing = None
    if out_path.exists():
        existing = pd.read_feather(out_path)
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        start = int(existing["date"].iloc[-1].timestamp() * 1000) + MS_PER_BAR

    rows: list[list] = []
    cursor = start
    fetched = 0
    retries = 0
    while cursor < UNTIL:
        try:
            data = ex.fetch_ohlcv(pair, TIMEFRAME, since=cursor, limit=LIMIT)
        except Exception as exc:
            retries += 1
            if retries > 5:
                print(f"  [{pair}] give up after 5 retries: {exc}")
                break
            time.sleep(2 * retries)
            continue
        retries = 0
        if not data:
            break
        rows.extend(data)
        fetched += len(data)
        # advance cursor
        last_ts = data[-1][0]
        new_cursor = last_ts + MS_PER_BAR
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        # pace: kucoin 5 req/s public
        time.sleep(0.25)
        if fetched % 10000 == 0:
            pct = (cursor - SINCE) / (UNTIL - SINCE) * 100
            print(f"  [{pair}] fetched {fetched:,}  cursor={datetime.fromtimestamp(cursor/1000, tz=timezone.utc).date()}  {pct:.0f}%")

    if not rows and existing is None:
        print(f"  [{pair}] NO DATA"); return 0
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    # dedup on date, sort
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df.to_feather(out_path)
    print(f"  [{pair}] saved {len(df):,} rows → {out_path.name}  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    return len(df)


def main() -> int:
    ex = ccxt.kucoin({"enableRateLimit": True, "timeout": 60000})
    # Skip load_markets — we use symbol strings directly
    total = 0
    for pair in PAIRS:
        total += fetch_pair(ex, pair)
    print(f"[done] total {total:,} rows across {len(PAIRS)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
