#!/usr/bin/env python
"""Clean OHLCV data from data/raw and write partitioned parquet files under data/clean."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml

from dateutil.parser import isoparse

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "datetime"]


@dataclass
class SeriesSpec:
    exchange: str
    symbol: str
    timeframe: str

    @property
    def symbol_slug(self) -> str:
        return self.symbol.replace("/", "-")

    @property
    def plain_symbol(self) -> str:
        return self.symbol.replace("/", "")

    @property
    def timeframe_ms(self) -> int:
        unit = self.timeframe[-1].lower()
        value = int(self.timeframe[:-1])
        if unit == "m":
            return value * 60_000
        if unit == "h":
            return value * 3_600_000
        if unit == "d":
            return value * 86_400_000
        if unit == "w":
            return value * 7 * 86_400_000
        raise ValueError(f"Unsupported timeframe {self.timeframe}")


@dataclass
class CleanResult:
    spec: SeriesSpec
    rows: int
    start_ts: Optional[int]
    end_ts: Optional[int]
    coverage: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean OHLCV data")
    parser.add_argument("--conf", type=Path, required=True, help="Path to symbols.yaml")
    parser.add_argument("--min-date", type=str, default=None, help="Optional minimum ISO date for filtering")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_raw_files(spec: SeriesSpec) -> List[Path]:
    if not RAW_DIR.exists():
        return []
    matches: List[Path] = []
    slug = spec.symbol_slug
    plain = spec.plain_symbol
    for parquet in RAW_DIR.rglob("*.parquet"):
        stem = parquet.stem
        name_str = str(parquet)
        if slug not in name_str and plain not in name_str:
            continue
        if stem.startswith("tf="):
            if stem != f"tf={spec.timeframe}":
                continue
        elif f"-{spec.timeframe}" not in stem:
            continue
        matches.append(parquet)
    return matches


def load_frames(paths: Iterable[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        missing = [col for col in COLUMNS if col not in df.columns]
        if missing:
            # ensure datetime exists
            if "timestamp" not in df.columns:
                print(f"Skipping {path}: no timestamp column")
                continue
            df = df.rename(columns={c: c.strip() for c in df.columns})
            if "datetime" not in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df[[col for col in COLUMNS if col in df.columns]]
        frames.append(df[COLUMNS])
    return frames


def concat_and_clean(frames: List[pd.DataFrame], timeframe_ms: int, min_ts: Optional[int]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    if min_ts is not None:
        combined = combined[combined["timestamp"] >= min_ts]
    if combined.empty:
        return combined
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
    # drop any future/incomplete candles
    upper_bound = int(pd.Timestamp.utcnow().timestamp() * 1000) - timeframe_ms
    combined = combined[combined["timestamp"] <= upper_bound]
    return combined.reset_index(drop=True)


def write_partitioned(df: pd.DataFrame, spec: SeriesSpec) -> None:
    if df.empty:
        return
    base = CLEAN_DIR / f"exchange={spec.exchange}" / f"symbol={spec.symbol_slug}" / f"tf={spec.timeframe}"
    base.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["ym"] = df["datetime"].dt.strftime("%Y%m")
    for ym, group in df.groupby("ym"):
        out = base / f"YYYYMM={ym}" / "data.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns=["ym"], inplace=False).to_parquet(out, index=False)


def compute_coverage(df: pd.DataFrame, timeframe_ms: int) -> Optional[float]:
    if df.empty:
        return None
    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    if span <= 0:
        return 1.0
    expected = span // timeframe_ms + 1
    if expected <= 0:
        return None
    return float(len(df) / expected)


def process_series(spec: SeriesSpec, min_ts: Optional[int]) -> CleanResult:
    paths = find_raw_files(spec)
    if not paths:
        print(f"No raw files found for {spec.symbol} {spec.timeframe}")
        return CleanResult(spec, 0, None, None, None)
    frames = load_frames(paths)
    df = concat_and_clean(frames, spec.timeframe_ms, min_ts)
    write_partitioned(df, spec)
    coverage = compute_coverage(df, spec.timeframe_ms)
    start_ts = int(df["timestamp"].iloc[0]) if not df.empty else None
    end_ts = int(df["timestamp"].iloc[-1]) if not df.empty else None
    print(f"Cleaned {spec.symbol} {spec.timeframe}: {len(df)} rows")
    return CleanResult(spec, len(df), start_ts, end_ts, coverage)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.conf)
    exchange = cfg["exchange"]
    symbols = cfg["symbols"]
    timeframes = cfg["timeframes"]
    min_ts = None
    if args.min_date:
        min_ts = int(isoparse(args.min_date).timestamp() * 1000)

    results: List[CleanResult] = []
    for symbol in symbols:
        for timeframe in timeframes:
            spec = SeriesSpec(exchange=exchange, symbol=symbol, timeframe=timeframe)
            result = process_series(spec, min_ts)
            results.append(result)

    if results:
        summary_rows = []
        for res in results:
            summary_rows.append(
                {
                    "exchange": res.spec.exchange,
                    "symbol": res.spec.symbol,
                    "timeframe": res.spec.timeframe,
                    "rows": res.rows,
                    "start_timestamp": res.start_ts,
                    "end_timestamp": res.end_ts,
                    "coverage": res.coverage,
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(CLEAN_DIR / "summary.parquet", index=False)
        print("Wrote data/clean/summary.parquet")


if __name__ == "__main__":
    main()