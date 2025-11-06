#!/usr/bin/env python
"""Fetch OHLCV candles via CCXT and store them under data/raw.

Usage:
    python scripts/fetch_ccxt_ohlcv.py --conf conf/symbols.yaml

The script supports incremental updates: if the target file already exists,
fetching will resume from the last completed candle.
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ccxt
import pandas as pd
import yaml
from dateutil.parser import isoparse
from tenacity import retry, stop_after_attempt, wait_exponential

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("ms"):
        return int(tf[:-2])
    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "s":
        return value * 1_000
    if unit == "m":
        return value * 60_000
    if unit == "h":
        return value * 60 * 60_000
    if unit == "d":
        return value * 24 * 60 * 60_000
    if unit == "w":
        return value * 7 * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = {"exchange", "type", "symbols", "timeframes", "start", "store_as"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    return cfg


def ensure_exchange(exchange_id: str, type_: str) -> ccxt.Exchange:
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange id: {exchange_id}")
    klass = getattr(ccxt, exchange_id)
    exchange = klass({
        "enableRateLimit": True,
        "options": {"defaultType": type_},
    })
    exchange.load_markets()
    return exchange


@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(7))
def _fetch_page(exchange: ccxt.Exchange, symbol: str, timeframe: str, since: int, limit: int) -> List[List[float]]:
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)


def _sanitise_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "_")


def _current_millis() -> int:
    return int(time.time() * 1000)


def _load_existing(path: Path, store_as: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if store_as == "parquet":
        df = pd.read_parquet(path)
    elif store_as == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported store_as: {store_as}")
    if "timestamp" not in df.columns:
        raise ValueError(f"Existing file {path} missing 'timestamp' column")
    return df


def _save_dataframe(df: pd.DataFrame, path: Path, store_as: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if store_as == "parquet":
        df.to_parquet(path, index=False)
    elif store_as == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported store_as: {store_as}")


def _merge_data(existing: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        base = new_df
    else:
        base = pd.concat([existing, new_df], ignore_index=True)
    base = base.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return base


def _effective_end(end: Optional[str], timeframe_ms: int) -> int:
    now = _current_millis()
    guard = now - timeframe_ms
    if guard <= 0:
        guard = now
    if end:
        end_ms = int(isoparse(end).astimezone(timezone.utc).timestamp() * 1000)
        return min(end_ms, guard)
    return guard


def fetch_symbol_timeframe(
    exchange: ccxt.Exchange,
    exchange_id: str,
    type_: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str],
    store_as: str,
    base_dir: Path,
    limit: int,
) -> Path:
    tf_ms = timeframe_to_ms(timeframe)
    end_ms = _effective_end(end, tf_ms)
    start_ms = int(isoparse(start).astimezone(timezone.utc).timestamp() * 1000)
    target_dir = base_dir / f"exchange={exchange_id}" / f"type={type_}" / f"symbol={_sanitise_symbol(symbol)}"
    out_path = target_dir / f"tf={timeframe}.{store_as}"

    existing = _load_existing(out_path, store_as)
    if existing is not None and not existing.empty:
        last_ts = int(existing["timestamp"].max())
        start_ms = max(start_ms, last_ts + tf_ms)

    if start_ms >= end_ms:
        print(f"[{symbol}] {timeframe}: up to date (start {start_ms} >= end {end_ms}).")
        return out_path

    all_rows: List[List[float]] = []
    since = start_ms
    start_iso = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
    print(f"[{symbol}] {timeframe}: fetching from {start_iso} UTC")

    while True:
        batch = _fetch_page(exchange, symbol, timeframe, since, limit)
        if not batch:
            break
        filtered = [row for row in batch if row[0] < end_ms]
        if not filtered:
            break
        all_rows.extend(filtered)
        new_last = filtered[-1][0]
        # advance by one full candle to avoid duplicates
        since = new_last + tf_ms
        if new_last + tf_ms >= end_ms:
            break

    if not all_rows:
        print(f"[{symbol}] {timeframe}: no new data fetched.")
        return out_path

    new_df = pd.DataFrame(all_rows, columns=COLUMNS)
    new_df["datetime"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
    merged = _merge_data(existing, new_df)
    _save_dataframe(merged, out_path, store_as)
    print(f"[{symbol}] {timeframe}: wrote {len(new_df)} new rows (total {len(merged)}). -> {out_path}")
    return out_path


def run_from_config(cfg: Dict[str, Any], limit: int) -> List[Path]:
    exchange_id = cfg["exchange"]
    type_ = cfg.get("type", "spot")
    symbols: Iterable[str] = cfg["symbols"]
    timeframes: Iterable[str] = cfg["timeframes"]
    start: str = cfg["start"]
    end: Optional[str] = cfg.get("end")
    store_as: str = cfg.get("store_as", "parquet")

    base_dir = Path("data/raw")
    exchange = ensure_exchange(exchange_id, type_)
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    produced_paths: List[Path] = []

    for symbol in symbols:
        if symbol not in exchange.symbols:
            raise ValueError(f"Exchange {exchange_id} does not list symbol {symbol}")
        for timeframe in timeframes:
            if timeframe not in exchange.timeframes:
                raise ValueError(
                    f"Exchange {exchange_id} does not support timeframe {timeframe}. Available: {sorted(exchange.timeframes.keys())}"
                )
            path = fetch_symbol_timeframe(
                exchange,
                exchange_id,
                type_,
                symbol,
                timeframe,
                start,
                end,
                store_as,
                base_dir,
                limit,
            )
            produced_paths.append(path)
    return produced_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OHLCV data via CCXT")
    parser.add_argument("--conf", type=Path, required=True, help="Path to YAML configuration")
    parser.add_argument("--limit", type=int, default=1000, help="Pagination limit per request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.conf)
    paths = run_from_config(cfg, args.limit)
    print("Completed. Files updated:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()