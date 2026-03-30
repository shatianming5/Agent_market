"""Universe Selector — picks tradeable assets based on liquidity and volatility.

Usage:
    from workspace.universe_selector import select_universe
    pairs = select_universe(exchange="gate", min_daily_volume_usd=1e6)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def scan_available_pairs(
    exchange: str = "kucoin",
    timeframe: str = "1h",
    data_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Scan available data files and compute basic statistics."""
    if data_dir is None:
        data_dir = ROOT / "user_data" / "data"
    data_dir = Path(data_dir) / exchange

    if not data_dir.exists():
        return []

    results = []
    for f in sorted(data_dir.glob(f"*-{timeframe}.feather")):
        try:
            df = pd.read_feather(f)
            df["date"] = pd.to_datetime(df["date"])
            pair = f.stem.replace(f"-{timeframe}", "").replace("_", "/")

            n_bars = len(df)
            n_days = (df["date"].max() - df["date"].min()).days
            avg_volume = float(df["volume"].mean())
            avg_price = float(df["close"].mean())
            avg_daily_volume_usd = avg_volume * avg_price * 24  # hourly vol → daily
            returns = df["close"].pct_change().dropna()
            volatility = float(returns.std())
            annual_vol = volatility * np.sqrt(24 * 365)

            results.append({
                "pair": pair,
                "exchange": exchange,
                "file": str(f),
                "bars": n_bars,
                "days": n_days,
                "avg_price": avg_price,
                "avg_daily_volume_usd": avg_daily_volume_usd,
                "hourly_volatility": volatility,
                "annual_volatility": annual_vol,
            })
        except Exception as _exc:
            import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
            continue

    return results


def select_universe(
    exchanges: Optional[List[str]] = None,
    *,
    timeframe: str = "1h",
    min_bars: int = 500,
    min_daily_volume_usd: float = 1e6,
    max_annual_volatility: float = 2.0,
    min_annual_volatility: float = 0.1,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Select tradeable pairs based on liquidity and volatility filters."""
    if exchanges is None:
        exchanges = ["kucoin", "gate"]

    all_pairs = []
    for exchange in exchanges:
        all_pairs.extend(scan_available_pairs(exchange=exchange, timeframe=timeframe))

    # Filter
    filtered = [
        p for p in all_pairs
        if p["bars"] >= min_bars
        and p["avg_daily_volume_usd"] >= min_daily_volume_usd
        and min_annual_volatility <= p["annual_volatility"] <= max_annual_volatility
    ]

    # Rank by volume (most liquid first)
    filtered.sort(key=lambda x: x["avg_daily_volume_usd"], reverse=True)

    # Deduplicate by pair name (keep highest volume exchange)
    seen = set()
    unique = []
    for p in filtered:
        if p["pair"] not in seen:
            seen.add(p["pair"])
            unique.append(p)

    return unique[:top_n]


__all__ = ["scan_available_pairs", "select_universe"]
