#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _safe_pair_slug_from_feather_path(path: Path) -> str:
    name = path.name
    if name.endswith(".feather"):
        name = name[:-8]
    if "-" in name:
        name = name.split("-", 1)[0]
    return name


def merge_micro_feature_hourly(
    *,
    features_parquet: Path,
    target_feather: Path,
    ts_col: str = "ts",
    target_date_col: str = "date",
    symbol: Optional[str] = None,
    agg: str = "mean",
    prefix: str = "",
    backup_suffix: str = ".feather.pre_microstructure.bak",
) -> dict:
    features_parquet = Path(features_parquet).resolve()
    target_feather = Path(target_feather).resolve()

    if not features_parquet.exists():
        raise FileNotFoundError(f"features parquet not found: {features_parquet}")
    if not target_feather.exists():
        raise FileNotFoundError(f"target feather not found: {target_feather}")

    mf = pd.read_parquet(features_parquet)
    if ts_col not in mf.columns:
        # Best-effort fallback
        if "date" in mf.columns:
            ts_col = "date"
        else:
            raise ValueError(f"micro_feature parquet missing timestamp column: {ts_col!r}")

    mf[ts_col] = pd.to_datetime(mf[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    mf = mf.dropna(subset=[ts_col]).sort_values(ts_col)

    if symbol and "symbol" in mf.columns:
        mf = mf.loc[mf["symbol"].astype(str) == str(symbol)]

    # Choose columns to merge: numeric only, exclude identifiers.
    drop_cols = {ts_col, "symbol", "event", "pair", "date"}
    num_cols = [
        c
        for c in mf.columns
        if c not in drop_cols and np.issubdtype(mf[c].dtype, np.number)
    ]
    if not num_cols:
        raise ValueError("no numeric feature columns found in micro_feature parquet")

    mf_hour = mf.copy()
    mf_hour["__hour__"] = mf_hour[ts_col].dt.floor("1h")
    if agg == "median":
        hourly = mf_hour.groupby("__hour__", sort=True)[num_cols].median().reset_index()
    else:
        hourly = mf_hour.groupby("__hour__", sort=True)[num_cols].mean().reset_index()
    hourly = hourly.rename(columns={"__hour__": target_date_col})

    if prefix:
        rename = {c: f"{prefix}{c}" for c in num_cols}
        hourly = hourly.rename(columns=rename)
        out_cols = [rename[c] for c in num_cols]
    else:
        out_cols = list(num_cols)

    df = pd.read_feather(target_feather)
    if target_date_col not in df.columns:
        raise ValueError(f"target feather missing date col: {target_date_col!r}")
    df[target_date_col] = pd.to_datetime(df[target_date_col], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.sort_values(target_date_col).reset_index(drop=True)

    # Idempotency: drop prior merged cols.
    drop_existing = [c for c in out_cols if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    merged = pd.merge_asof(df, hourly, on=target_date_col, direction="backward")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    bak = target_feather.with_suffix(backup_suffix)
    if not bak.exists():
        df.to_feather(bak)
    merged.to_feather(target_feather)
    return {
        "target": str(target_feather),
        "backup": str(bak),
        "features_parquet": str(features_parquet),
        "symbol": symbol,
        "merged_cols": out_cols,
        "rows": int(merged.shape[0]),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge micro_feature (features.parquet) into a 1h OHLCV feather by hourly aggregation.",
    )
    p.add_argument("--features-parquet", required=True, help="Path to micro_feature features.parquet")
    p.add_argument("--target-feather", required=True, help="Path to target OHLCV feather (e.g. BTC_USDT-1h.feather)")
    p.add_argument("--ts-col", default="ts", help="Timestamp column in micro_feature parquet (default: ts)")
    p.add_argument("--target-date-col", default="date", help="Datetime column in target feather (default: date)")
    p.add_argument("--symbol", default=None, help="Optional symbol filter if parquet contains `symbol`")
    p.add_argument("--agg", choices=["mean", "median"], default="mean")
    p.add_argument("--prefix", default="", help="Optional prefix for merged columns (default: none)")
    args = p.parse_args()

    features_parquet = Path(args.features_parquet)
    target_feather = Path(args.target_feather)
    if not args.symbol and target_feather.name.endswith("-1h.feather"):
        # Provide a gentle hint for common KuCoin symbol format.
        _ = _safe_pair_slug_from_feather_path(target_feather)

    out = merge_micro_feature_hourly(
        features_parquet=features_parquet,
        target_feather=target_feather,
        ts_col=str(args.ts_col),
        target_date_col=str(args.target_date_col),
        symbol=str(args.symbol) if args.symbol else None,
        agg=str(args.agg),
        prefix=str(args.prefix),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

