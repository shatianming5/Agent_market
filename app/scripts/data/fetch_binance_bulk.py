#!/usr/bin/env python
"""Download historical OHLCV data from Binance data.binance.vision dumps.

Example:
    python scripts/fetch_binance_bulk.py --conf conf/symbols.yaml --monthly --limit-months 1

The script downloads ZIP archives alongside their CHECKSUM files, validates
integrity, extracts the CSV, and stores normalised OHLCV data into data/raw.
"""
import argparse
import hashlib
import io
import sys
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import zipfile

import pandas as pd
import requests
import yaml
from dateutil.parser import isoparse

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
BASE_URL = "https://data.binance.vision"


@dataclass
class Job:
    symbol: str
    interval: str
    period: date
    frequency: str  # "daily" or "monthly"

    @property
    def symbol_plain(self) -> str:
        return self.symbol.replace("/", "")

    def remote_path(self) -> Tuple[str, str]:
        base = f"data/spot/{self.frequency}/klines/{self.symbol_plain}/{self.interval}"
        if self.frequency == "monthly":
            stamp = self.period.strftime("%Y-%m")
        else:
            stamp = self.period.strftime("%Y-%m-%d")
        filename = f"{self.symbol_plain}-{self.interval}-{stamp}.zip"
        checksum = f"{filename}.CHECKSUM"
        return f"{base}/{filename}", f"{base}/{checksum}"

    def local_folder(self, base_dir: Path) -> Path:
        return base_dir / f"exchange=binance" / f"type=spot" / f"symbol={self.symbol_plain}" / f"timeframe={self.interval}" / f"frequency={self.frequency}"


class DownloadError(Exception):
    pass


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance bulk OHLCV data")
    parser.add_argument("--conf", type=Path, required=True, help="Path to symbols.yaml")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly", help="Granularity of bulk files")
    parser.add_argument("--limit-months", type=int, default=None, help="Cap the number of periods (recent first) to download")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even if files exist")
    return parser.parse_args()


def generate_periods(start: date, end: date, frequency: str) -> List[date]:
    periods: List[date] = []
    current = date(start.year, start.month, start.day)
    if frequency == "monthly":
        first = date(start.year, start.month, 1)
        current = first
        while current <= end:
            periods.append(current)
            year = current.year + (current.month // 12)
            month = current.month % 12 + 1
            current = date(year, month, 1)
    else:
        current = start
        while current <= end:
            periods.append(current)
            current += pd.Timedelta(days=1).to_pytimedelta()
    return periods


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def http_get(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404:
        raise DownloadError(f"Remote file not found: {url}")
    resp.raise_for_status()
    return resp.content


def validate_checksum(data: bytes, checksum_text: str, filename: str) -> None:
    expected = None
    for line in checksum_text.splitlines():
        if filename in line:
            expected = line.split()[0]
            break
    if expected is None:
        raise DownloadError(f"Checksum entry for {filename} not found")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise DownloadError(f"Checksum mismatch for {filename}: expected {expected}, got {actual}")


def read_zip_ohlcv(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
        if not csv_files:
            raise DownloadError("ZIP contains no CSV files")
        with zf.open(csv_files[0]) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
    # Binance switched to microsecond timestamps for spot on 2025-01-01
    if df["open_time"].max() > 2_000_000_000_000:
        df["open_time"] = (df["open_time"] // 1_000).astype("int64")
    df = df.rename(columns={"open_time": "timestamp"})
    df = df[COLUMNS]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def merge_store(df: pd.DataFrame, store_as: Path) -> None:
    if store_as.exists():
        existing = pd.read_parquet(store_as)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    store_as.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(store_as, index=False)


def run_jobs(jobs: Iterable[Job], base_dir: Path, limit: Optional[int], overwrite: bool) -> List[Path]:
    updated: List[Path] = []
    ordered_jobs = list(sorted(jobs, key=lambda j: j.period))
    if limit:
        ordered_jobs = ordered_jobs[-limit:]
    for job in ordered_jobs:
        remote_zip, remote_checksum = job.remote_path()
        local_dir = job.local_folder(base_dir)
        ensure_folder(local_dir)
        local_zip = local_dir / Path(remote_zip).name
        parquet_target = local_dir / f"{job.symbol_plain}-{job.interval}.parquet"

        needs_download = overwrite or not local_zip.exists()
        try:
            if needs_download:
                url_zip = f"{BASE_URL}/{remote_zip}"
                url_checksum = f"{BASE_URL}/{remote_checksum}"
                print(f"Downloading {url_zip}")
                zip_bytes = http_get(url_zip)
                checksum_text = http_get(url_checksum).decode("utf-8")
                validate_checksum(zip_bytes, checksum_text, local_zip.name)
                local_zip.write_bytes(zip_bytes)
                print(f"Downloaded and verified {local_zip}")
            else:
                print(f"{local_zip} already exists, skipping download")
                zip_bytes = local_zip.read_bytes()

            df = read_zip_ohlcv(zip_bytes)
            merge_store(df, parquet_target)
            updated.append(parquet_target)
            print(f"Stored {len(df)} rows into {parquet_target}")
        except DownloadError as exc:
            print(f"Skipping {local_zip.name}: {exc}")
            continue
    return updated



def main() -> None:
    args = parse_args()
    cfg = load_config(args.conf)
    start = isoparse(cfg["start"]).date()
    end = date.today()
    if cfg.get("end"):
        candidate_end = isoparse(cfg["end"]).date()
        end = min(end, candidate_end)
    symbols: Iterable[str] = cfg["symbols"]
    intervals: Iterable[str] = cfg["timeframes"]
    jobs: List[Job] = []
    for symbol in symbols:
        for interval in intervals:
            periods = generate_periods(start, end, args.frequency)
            jobs.extend(Job(symbol=symbol, interval=interval, period=p, frequency=args.frequency) for p in periods)
    base_dir = Path("data/raw")
    updated = run_jobs(jobs, base_dir, args.limit_months, args.overwrite)
    print("Completed Binance bulk download. Updated files:")
    for path in updated:
        print("  ", path)


if __name__ == "__main__":
    try:
        main()
    except DownloadError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)