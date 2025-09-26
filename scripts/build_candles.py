#!/usr/bin/env python
"""Build OHLCV-style candles from swap parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize('UTC') if series.dt.tz is None else series
    return pd.to_datetime(series, unit='s', utc=True, errors='coerce')


def build_candles(input_path: Path, output_dir: Path, rule: str) -> None:
    if not input_path.exists():
        print(f"Input file {input_path} not found; skipping candle build.")
        return
    df = pd.read_parquet(input_path)
    if df.empty:
        print(f"Input file {input_path} is empty; skipping.")
        return
    if 'pool' not in df.columns:
        print("Expected 'pool' column in swaps parquet; aborting.")
        return
    if 'block_time' not in df.columns and 'timestamp' not in df.columns:
        print("Expected 'block_time' or 'timestamp' column for candle aggregation; aborting.")
        return
    time_col = 'block_time' if 'block_time' in df.columns else 'timestamp'
    df['datetime'] = ensure_datetime(df[time_col])
    df = df.dropna(subset=['datetime', 'price'])
    df = df.sort_values('datetime')

    frames = []
    for pool, group in df.groupby('pool'):
        g = group.set_index('datetime')
        o = g['price'].resample(rule).first()
        h = g['price'].resample(rule).max()
        l = g['price'].resample(rule).min()
        c = g['price'].resample(rule).last()
        vol0 = g['amount0_norm'].abs().resample(rule).sum() if 'amount0_norm' in g.columns else None
        vol1 = g['amount1_norm'].abs().resample(rule).sum() if 'amount1_norm' in g.columns else None
        frame = pd.DataFrame({
            'pool': pool,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
        })
        if vol0 is not None:
            frame['vol_token0'] = vol0
        if vol1 is not None:
            frame['vol_token1'] = vol1
        frame = frame.dropna(subset=['open', 'high', 'low', 'close'])
        if not frame.empty:
            frames.append(frame.reset_index())
    if not frames:
        print("No candles generated (possibly insufficient data).")
        return
    out_df = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"candles_{rule}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} candle rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build swap candles")
    parser.add_argument('--input', type=Path, default=Path('data/raw/dex/uniswap-v3_swaps.parquet'))
    parser.add_argument('--output', type=Path, default=Path('data/clean/dex'))
    parser.add_argument('--rule', default='1T', help='Pandas resample rule (e.g. 1T,5T,1H)')
    args = parser.parse_args()
    build_candles(args.input, args.output, args.rule)


if __name__ == '__main__':
    main()