"""Data downloaders (unified: KuCoin spot 1h/4h/1m, OKX futures 1h, Gate funding)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import requests

from .paths import KUCOIN_DIR, OKX_FUTURES_DIR, FUNDING_DIR, DEFAULT_PAIRS

USER_AGENT = "Mozilla/5.0 factor_lab"


# ============================================================
# KuCoin spot OHLCV (1m / 1h / 4h)
# ============================================================

def _ts(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def download_kucoin(
    timeframe: str = "1h",
    start: str = "2023-04-12",
    end: str = "2026-04-18",
    pairs: Sequence[str] = None,
    sleep_s: float = 0.25,
) -> Dict[str, int]:
    """Download KuCoin spot OHLCV via public API. Returns {pair: rows_written}."""
    tf_map = {"1m": ("1min", 60), "1h": ("1hour", 3600), "4h": ("4hour", 14400)}
    if timeframe not in tf_map:
        raise ValueError(f"timeframe must be one of {list(tf_map)}")
    tf_api, tf_sec = tf_map[timeframe]
    pairs = list(pairs or DEFAULT_PAIRS)
    KUCOIN_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    start_s, end_s = _ts(start), _ts(end)
    results = {}

    for pair in pairs:
        sym = pair.replace("/", "-")
        out_path = KUCOIN_DIR / f"{pair.replace('/', '_')}-{timeframe}.feather"
        existing = None
        if out_path.exists():
            try:
                existing = pd.read_feather(out_path)
                existing["date"] = pd.to_datetime(existing["date"], utc=True)
                existing = existing.sort_values("date").reset_index(drop=True)
            except Exception:
                existing = None

        # Incremental update: if an existing file already reaches `end`, skip.
        # Otherwise, only fetch the missing tail with a small overlap.
        effective_start_s = start_s
        if existing is not None and not existing.empty:
            max_dt = existing["date"].max()
            # already up-to-date (or close enough)
            if max_dt >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(seconds=tf_sec):
                results[pair] = int(existing.shape[0])
                print(f"[{pair} {timeframe}] already up-to-date: {len(existing):,} bars")
                continue
            # fetch from a small overlap window to dedupe reliably
            overlap_s = tf_sec * 50  # ~2 days on 1h
            try:
                effective_start_s = max(int(start_s), int(max_dt.timestamp()) - int(overlap_s))
            except Exception:
                effective_start_s = start_s

        print(f"[{pair} {timeframe}] downloading...", flush=True)
        all_rows: Dict[int, list] = {}
        cursor = end_s
        t0 = time.time(); last_log = t0
        while cursor > effective_start_s:
            w_start = max(effective_start_s, cursor - tf_sec * 1490)
            try:
                r = session.get(
                    "https://api.kucoin.com/api/v1/market/candles",
                    params={"symbol": sym, "type": tf_api, "startAt": w_start, "endAt": cursor},
                    timeout=20,
                )
                r.raise_for_status()
                d = r.json()
                if d.get("code") != "200000":
                    time.sleep(2); continue
                batch = d.get("data", [])
                if not batch:
                    cursor = w_start - 1
                    continue
                for row in batch:
                    t = int(row[0])
                    if start_s <= t < end_s:
                        all_rows[t] = row
                oldest = min(int(r[0]) for r in batch)
                cursor = (oldest - 1) if oldest < cursor else (cursor - tf_sec)
                time.sleep(sleep_s)
            except Exception as e:
                print(f"    [err] {e}"); time.sleep(2)
                continue
            if time.time() - last_log > 30:
                last_log = time.time()
                got = len(all_rows)
                pct = (end_s - min(all_rows.keys())) / (end_s - start_s) * 100 if all_rows else 0
                print(f"    [{pair}] {got:,} rows, {pct:.0f}%", flush=True)

        # Serialize
        sorted_keys = sorted(all_rows.keys())
        if not sorted_keys:
            results[pair] = 0; continue
        rows = [all_rows[k] for k in sorted_keys]
        df = pd.DataFrame(rows, columns=["ts", "open", "close", "high", "low", "volume", "turnover"])
        for c in ("open", "close", "high", "low", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
        df = df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df], axis=0, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"], utc=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            combined = combined.loc[
                (combined["date"] >= pd.to_datetime(start, utc=True))
                & (combined["date"] < pd.to_datetime(end, utc=True))
            ]
            combined.to_feather(out_path)
            results[pair] = int(combined.shape[0])
            print(f"[OK {pair}] {len(combined):,} bars  ({combined['date'].min().date()} → {combined['date'].max().date()})")
        else:
            df.to_feather(out_path)
            results[pair] = len(df)
            print(f"[OK {pair}] {len(df):,} bars  ({df['date'].min().date()} → {df['date'].max().date()})")

    return results


# ============================================================
# OKX SWAP futures OHLCV (for backtest)
# ============================================================

def prepare_okx_futures_auxiliary(data_dir: Path = OKX_FUTURES_DIR) -> Dict[str, int]:
    """Create local mark/funding proxy files expected by Freqtrade futures backtests.

    The first rank-portfolio version uses close as a mark-price proxy and zero
    funding when historical funding candles are unavailable.
    """
    root = Path(data_dir)
    results: Dict[str, int] = {}
    for path in sorted(root.glob("*-1h-futures.feather")):
        df = pd.read_feather(path)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        base = path.name[: -len("-1h-futures.feather")]
        mark_path = root / f"{base}-1h-mark.feather"
        funding_path = root / f"{base}-1h-funding_rate.feather"

        mark = df[["date", "open", "high", "low", "close", "volume"]].copy()
        mark.to_feather(mark_path)

        funding = mark.copy()
        for col in ("open", "high", "low", "close", "volume"):
            funding[col] = 0.0
        funding.to_feather(funding_path)
        results[base] = int(len(df))
    return results


def download_okx_futures(
    start: str = "2025-04-12",
    end: str = "2026-04-12",
    pairs: Sequence[str] = None,
    sleep_s: float = 0.15,
) -> Dict[str, int]:
    """Download OKX SWAP 1h OHLCV (for freqtrade futures backtest)."""
    pairs = list(pairs or DEFAULT_PAIRS)
    OKX_FUTURES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    start_ms, end_ms = _ts(start) * 1000, _ts(end) * 1000
    results = {}

    for pair in pairs:
        sym = pair.split("/")[0]
        inst_id = f"{sym}-USDT-SWAP"
        out = OKX_FUTURES_DIR / f"{sym}_USDT_USDT-1h-futures.feather"
        print(f"[{sym}] {inst_id}", flush=True)
        all_rows = {}
        cursor = end_ms
        while cursor > start_ms:
            try:
                r = session.get(
                    "https://www.okx.com/api/v5/market/history-candles",
                    params={"instId": inst_id, "bar": "1H", "limit": 100,
                            "before": str(start_ms), "after": str(cursor)},
                    timeout=20,
                )
                d = r.json()
                if d.get("code") != "0":
                    time.sleep(2); continue
                items = d.get("data", [])
                if not items:
                    break
                for row in items:
                    ts = int(row[0])
                    if start_ms <= ts < end_ms:
                        all_rows[ts] = row
                oldest = min(int(r[0]) for r in items)
                cursor = oldest if oldest < cursor else (cursor - 3600_000)
                time.sleep(sleep_s)
            except Exception as e:
                print(f"    err: {e}"); time.sleep(2)

        if not all_rows:
            results[pair] = 0; continue
        rows = [all_rows[k] for k in sorted(all_rows.keys())]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQ", "confirm"])
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "close"]).reset_index(drop=True)
        df.to_feather(out)
        results[pair] = len(df)
        print(f"    {len(df):,} bars  {df['date'].min().date()} → {df['date'].max().date()}")

    prepare_okx_futures_auxiliary(OKX_FUTURES_DIR)
    return results


# ============================================================
# Gate.io funding rate (historical, 3+ years)
# ============================================================

def download_funding(
    start: str = "2023-04-12",
    end: str = "2026-04-18",
    pairs: Sequence[str] = None,
    sleep_s: float = 0.2,
) -> Dict[str, int]:
    """Download Gate.io funding rate. Returns {pair: rows}."""
    pairs = list(pairs or DEFAULT_PAIRS)
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    start_s, end_s = _ts(start), _ts(end)
    window_sec = 30 * 86400
    results = {}

    for pair in pairs:
        sym = pair.split("/")[0]
        contract = f"{sym}_USDT"
        out = FUNDING_DIR / f"{sym}_USDT-funding.feather"
        print(f"[{sym}] {contract}", flush=True)
        all_rows: Dict[int, float] = {}
        cursor = start_s
        while cursor < end_s:
            win_to = min(cursor + window_sec, end_s)
            try:
                r = session.get(
                    "https://api.gateio.ws/api/v4/futures/usdt/funding_rate",
                    params={"contract": contract, "from": cursor, "to": win_to, "limit": 100},
                    timeout=15,
                )
                if r.status_code == 200:
                    for it in r.json():
                        all_rows[int(it["t"])] = float(it.get("r", 0))
                cursor = win_to + 1
                time.sleep(sleep_s)
            except Exception as e:
                print(f"    err: {e}"); time.sleep(2); cursor = win_to + 1

        if not all_rows:
            results[pair] = 0; continue
        df = pd.DataFrame(
            [{"ts": t, "funding_rate": all_rows[t]} for t in sorted(all_rows.keys())]
        )
        df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df[["date", "funding_rate"]].reset_index(drop=True)
        df.to_feather(out)
        results[pair] = len(df)
        print(f"    {len(df)} rows")

    return results
