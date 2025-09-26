#!/usr/bin/env python
"""Generate a simple data-quality report for cleaned OHLCV datasets."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

CLEAN_DIR = Path("data/clean")


@dataclass
class SeriesSpec:
    exchange: str
    symbol: str
    timeframe: str

    @property
    def symbol_slug(self) -> str:
        return self.symbol.replace("/", "-")

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
class DQMetrics:
    spec: SeriesSpec
    rows: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    expected: Optional[int]
    missing: Optional[int]
    max_gap: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create data quality report')
    parser.add_argument('--mode', choices=['ohlcv', 'news'], default='ohlcv')
    parser.add_argument('--conf', type=Path, help='symbols.yaml when mode=ohlcv')
    parser.add_argument('--output', type=Path, default=CLEAN_DIR / 'dq_report.parquet')
    parser.add_argument('--news-path', type=Path, default=None, help='Override news parquet path when mode=news')
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_clean_frame(spec: SeriesSpec) -> pd.DataFrame:
    base = CLEAN_DIR / f"exchange={spec.exchange}" / f"symbol={spec.symbol_slug}" / f"tf={spec.timeframe}"
    if not base.exists():
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for parquet in base.rglob("*.parquet"):
        df = pd.read_parquet(parquet)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    combined["datetime"] = pd.to_datetime(combined["timestamp"], unit="ms", utc=True)
    return combined


def generate_news_report(news_path: Path) -> None:
    if not news_path.exists():
        print(f"News dataset not found: {news_path}")
        return
    df = pd.read_parquet(news_path)
    total = len(df)
    if total == 0:
        print(f"News dataset {news_path} is empty")
        return
    print(f"News records: {total}")
    if 'source' in df.columns:
        print('Top sources:')
        for source, count in df['source'].value_counts().head(10).items():
            print(f"  {source:<20} {count}")
    if 'lang' in df.columns:
        print('Top languages:')
        for lang, count in df['lang'].fillna('unknown').value_counts().head(10).items():
            print(f"  {lang:<10} {count}")
    if 'published_at' in df.columns:
        dates = pd.to_datetime(df['published_at'], errors='coerce')
        dates = dates.dropna()
        if not dates.empty:
            print(f"Published range: {dates.min()} -> {dates.max()}")
    dup = df['url'].duplicated().sum() if 'url' in df.columns else 0
    print(f"Duplicate URLs: {dup}")


def compute_metrics(df: pd.DataFrame, spec: SeriesSpec) -> DQMetrics:
    if df.empty:
        return DQMetrics(spec, 0, None, None, None, None, None)
    diffs = df["timestamp"].diff().dropna()
    expected_ms = spec.timeframe_ms
    missing = int(((diffs // expected_ms) - 1).clip(lower=0).sum()) if not diffs.empty else 0
    max_gap = int(max(((diffs // expected_ms) - 1).max(), 0)) if not diffs.empty else 0
    expected = int(((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) // expected_ms) + 1)
    return DQMetrics(
        spec=spec,
        rows=len(df),
        start=df["datetime"].iloc[0],
        end=df["datetime"].iloc[-1],
        expected=expected,
        missing=missing,
        max_gap=max_gap,
    )


def main() -> None:
    args = parse_args()
    if args.mode == 'ohlcv' and args.conf is None:
        raise SystemExit('--conf is required when mode=ohlcv')
    if args.mode == 'news':
        news_path = args.news_path or Path('data/clean/news/all.parquet')
        generate_news_report(news_path)
        return
    cfg = load_config(args.conf)
    exchange = cfg["exchange"]
    symbols = cfg["symbols"]
    timeframes = cfg["timeframes"]

    metrics: List[DQMetrics] = []
    for symbol in symbols:
        for timeframe in timeframes:
            spec = SeriesSpec(exchange=exchange, symbol=symbol, timeframe=timeframe)
            df = load_clean_frame(spec)
            metrics.append(compute_metrics(df, spec))

    if not metrics:
        print("No datasets to report")
        return

    rows = []
    for m in metrics:
        rows.append(
            {
                "exchange": m.spec.exchange,
                "symbol": m.spec.symbol,
                "timeframe": m.spec.timeframe,
                "rows": m.rows,
                "start": m.start.isoformat() if m.start is not None else None,
                "end": m.end.isoformat() if m.end is not None else None,
                "expected": m.expected,
                "missing": m.missing,
                "max_gap_multiples": m.max_gap,
            }
        )
    report_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_parquet(args.output, index=False)
    print(report_df.to_string(index=False))
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()