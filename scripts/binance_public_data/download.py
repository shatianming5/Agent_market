from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests

try:
    from .manifest import ManifestDB, ManifestEntry, parse_checksum_file, sha256_file
except Exception:  # allow running as a script
    from manifest import ManifestDB, ManifestEntry, parse_checksum_file, sha256_file

BASE = "https://data.binance.vision/data"


def month_iter(start: str, end: str) -> Iterable[str]:
    """Yield YYYY-MM inclusive."""
    sy, sm = start.split("-")
    ey, em = end.split("-")
    y, m = int(sy), int(sm)
    ey, em = int(ey), int(em)
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def http_get_text(url: str, *, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def http_download(url: str, out_path: Path, *, timeout: int = 120, retries: int = 3) -> None:
    ensure_dir(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    # Resume support
    headers = {}
    mode = "wb"
    pos = 0
    if tmp.exists():
        pos = tmp.stat().st_size
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
            mode = "ab"

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:
                    # Already complete
                    tmp.rename(out_path)
                    return
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.rename(out_path)
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"download failed after {retries} retries: {last_err}")


def build_urls(*, market: str, freq: str, dataset: str, symbol: str, interval: str, yyyymm: str) -> tuple[str, Optional[str], str]:
    """Return (data_url, checksum_url, out_rel_path)."""
    y, m = yyyymm.split("-")
    ym = f"{y}-{m}"

    if dataset == "klines":
        # spot/monthly/klines/SYMBOL/1m/SYMBOL-1m-YYYY-MM.zip
        zip_name = f"{symbol}-{interval}-{ym}.zip"
        rel = f"{market}/{freq}/klines/{symbol}/{interval}/{zip_name}"
    elif dataset == "trades":
        zip_name = f"{symbol}-trades-{ym}.zip"
        rel = f"{market}/{freq}/trades/{symbol}/{zip_name}"
    elif dataset == "aggTrades":
        zip_name = f"{symbol}-aggTrades-{ym}.zip"
        rel = f"{market}/{freq}/aggTrades/{symbol}/{zip_name}"
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    url = f"{BASE}/{rel}"
    checksum_url = url + ".CHECKSUM"
    return url, checksum_url, rel


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/binance_public", help="storage root")
    ap.add_argument("--market", default="spot", choices=["spot", "um", "cm"], help="spot|um|cm")
    ap.add_argument("--freq", default="monthly", choices=["monthly", "daily"], help="monthly|daily")
    ap.add_argument("--symbols", required=True, help="comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--datasets", default="klines,trades,aggTrades", help="klines,trades,aggTrades")
    ap.add_argument("--intervals", default="1m", help="comma-separated, only for klines")
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-files", type=int, default=0, help="0 means no limit (safety)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    mdb = ManifestDB(root / "manifests" / "manifest.sqlite")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    planned: List[ManifestEntry] = []
    for ym in month_iter(args.start, args.end):
        for sym in symbols:
            for ds in datasets:
                if ds == "klines":
                    for itv in intervals:
                        url, checksum_url, rel = build_urls(
                            market=args.market,
                            freq=args.freq,
                            dataset=ds,
                            symbol=sym,
                            interval=itv,
                            yyyymm=ym,
                        )
                        planned.append(
                            ManifestEntry(
                                url=url,
                                checksum_url=checksum_url,
                                out_path=str(root / rel),
                                dataset=ds,
                                market=args.market,
                                freq=args.freq,
                                symbol=sym,
                                interval=itv,
                                yyyymm=ym,
                            )
                        )
                else:
                    url, checksum_url, rel = build_urls(
                        market=args.market,
                        freq=args.freq,
                        dataset=ds,
                        symbol=sym,
                        interval="",
                        yyyymm=ym,
                    )
                    planned.append(
                        ManifestEntry(
                            url=url,
                            checksum_url=checksum_url,
                            out_path=str(root / rel),
                            dataset=ds,
                            market=args.market,
                            freq=args.freq,
                            symbol=sym,
                            interval="",
                            yyyymm=ym,
                        )
                    )

    # Safety: avoid accidental "download the world".
    if args.max_files and len(planned) > args.max_files:
        print(f"Refusing: planned {len(planned)} files > --max-files={args.max_files}")
        return 2

    for e in planned:
        mdb.upsert_planned(e)

    ok = 0
    skipped = 0
    failed = 0

    for idx, e in enumerate(planned, start=1):
        if mdb.is_ok(e.url):
            skipped += 1
            continue

        out_path = Path(e.out_path)
        try:
            # download zip
            http_download(e.url, out_path, timeout=args.timeout, retries=args.retries)

            # checksum verify (best-effort)
            want = None
            if e.checksum_url:
                try:
                    ctext = http_get_text(e.checksum_url, timeout=30)
                    want = parse_checksum_file(ctext)
                except Exception:
                    want = None

            got = sha256_file(out_path)
            if want and want != got:
                raise RuntimeError(f"sha256 mismatch want={want} got={got}")

            mdb.mark_ok(e.url, size_bytes=out_path.stat().st_size, sha256=got)
            ok += 1
        except Exception as ex:
            mdb.mark_failed(e.url, str(ex))
            failed += 1

        if idx % 20 == 0:
            print(f"progress: {idx}/{len(planned)} ok={ok} skipped={skipped} failed={failed}")

    print(f"done: planned={len(planned)} ok={ok} skipped={skipped} failed={failed}")
    mdb.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
