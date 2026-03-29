#!/usr/bin/env python3
"""Data downloader — downloads multi-year OHLCV data from exchanges.

Run on a server with unrestricted network access:
    python workspace/download_data.py --days 730 --exchange gate

Supports: gate, binance, okx, bybit, kucoin, mexc
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import ccxt
import pandas as pd


def download_pair(
    ex: ccxt.Exchange,
    pair: str,
    timeframe: str,
    since_ms: int,
    limit: int = 1000,
    max_batches: int = 50,
) -> pd.DataFrame:
    """Download OHLCV data for a single pair."""
    all_data = []
    since = since_ms
    for _ in range(max_batches):
        try:
            batch = ex.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"  error: {e}")
            break
        if not batch:
            break
        all_data.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="gate")
    parser.add_argument("--pairs", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT", "AVAX/USDT"])
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--outdir", default="user_data/data")
    parser.add_argument("--proxy", default=None, help="HTTP proxy, e.g. http://127.0.0.1:10809")
    args = parser.parse_args()

    ex_class = getattr(ccxt, args.exchange)
    config = {}
    if args.proxy:
        config["proxies"] = {"https": args.proxy, "http": args.proxy}
    ex = ex_class(config)
    ex.load_markets()
    print(f"Exchange: {args.exchange}, {len(ex.markets)} markets")

    out_dir = Path(args.outdir) / args.exchange
    out_dir.mkdir(parents=True, exist_ok=True)

    since_ms = int((datetime.now() - timedelta(days=args.days)).timestamp() * 1000)

    for pair in args.pairs:
        if pair not in ex.markets:
            print(f"[SKIP] {pair}")
            continue
        print(f"[DOWN] {pair}...", end=" ", flush=True)
        df = download_pair(ex, pair, args.timeframe, since_ms)
        if df.empty:
            print("no data")
            continue
        slug = pair.replace("/", "_")
        out_file = out_dir / f"{slug}-{args.timeframe}.feather"
        df.to_feather(out_file)
        days_actual = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        print(f"{len(df)} rows, {days_actual} days")

    print("Done!")


if __name__ == "__main__":
    main()
