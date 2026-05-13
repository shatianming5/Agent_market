"""Data downloaders (unified: KuCoin spot, exchange futures, Gate funding)."""
from __future__ import annotations

import time
import os
import io
import html
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

from .paths import BINANCE_FUTURES_DIR, BYBIT_FUTURES_DIR, KUCOIN_DIR, OKX_FUTURES_DIR, FUNDING_DIR, DEFAULT_PAIRS

USER_AGENT = "Mozilla/5.0 factor_lab"


# ============================================================
# KuCoin spot OHLCV
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
    tf_map = {
        "1m": ("1min", 60),
        "5m": ("5min", 5 * 60),
        "15m": ("15min", 15 * 60),
        "1h": ("1hour", 3600),
        "4h": ("4hour", 14400),
    }
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
# Exchange USDT perpetual futures OHLCV (for backtest/LEAN gate)
# ============================================================

OKX_BAR_MAP = {
    "1m": ("1m", 60_000),
    "5m": ("5m", 5 * 60_000),
    "15m": ("15m", 15 * 60_000),
    "1h": ("1H", 60 * 60_000),
    "4h": ("4H", 4 * 60 * 60_000),
}
BYBIT_BAR_MAP = {
    "1m": ("1", 60_000),
    "5m": ("5", 5 * 60_000),
    "15m": ("15", 15 * 60_000),
    "1h": ("60", 60 * 60_000),
    "4h": ("240", 4 * 60 * 60_000),
}
BINANCE_BAR_MAP = {
    "1m": ("1m", 60_000),
    "5m": ("5m", 5 * 60_000),
    "15m": ("15m", 15 * 60_000),
    "1h": ("1h", 60 * 60_000),
    "4h": ("4h", 4 * 60 * 60_000),
}
BINANCE_VISION_BASE_URL = "https://data.binance.vision/data/futures/um"
BINANCE_VISION_S3_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FUTURES_VENUE_DIRS = {
    "okx": OKX_FUTURES_DIR,
    "bybit": BYBIT_FUTURES_DIR,
    "binance": BINANCE_FUTURES_DIR,
}


def _parse_pairs(raw: Sequence[str] | str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_PAIRS)
    if isinstance(raw, str):
        return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def _is_all_pairs(raw: Sequence[str] | str | None) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() in {"all", "*"}
    return False


def _okx_base_for_pair(pair: str) -> str:
    sym = str(pair).split("/", 1)[0].split("_", 1)[0].strip().upper()
    return f"{sym}_USDT_USDT"


def _linear_symbol_for_pair(pair: str) -> str:
    sym = str(pair).split("/", 1)[0].split("_", 1)[0].strip().upper()
    return f"{sym}USDT"


def _usdt_pair_from_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if not sym.endswith("USDT") or len(sym) <= 4:
        raise ValueError(f"not a USDT linear symbol: {symbol!r}")
    return f"{sym[:-4]}/USDT"


def _futures_path(pair: str, timeframe: str, data_dir: Path) -> Path:
    return Path(data_dir) / f"{_okx_base_for_pair(pair)}-{timeframe}-futures.feather"


def _okx_futures_path(pair: str, timeframe: str, data_dir: Path = OKX_FUTURES_DIR) -> Path:
    return _futures_path(pair, timeframe, Path(data_dir))


def _write_feather_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        df.to_feather(tmp)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _normalize_ohlcv_frame(df: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[["date", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["date", "open", "close"])
    # Pandas 3.14 can hit an Arrow timezone indexing bug when sorting tz-aware
    # datetimes directly. Keep date as UTC for downstream code, but do ordering
    # and filtering on integer nanoseconds.
    out["_date_ns"] = out["date"].astype("int64")
    start_ns = pd.Timestamp(start, tz="UTC").value
    end_ns = pd.Timestamp(end, tz="UTC").value
    out = out.drop_duplicates(subset=["_date_ns"], keep="last")
    out = out.loc[(out["_date_ns"] >= start_ns) & (out["_date_ns"] < end_ns)]
    if out.empty:
        return out.drop(columns=["_date_ns"]).reset_index(drop=True)
    order = np.argsort(out["_date_ns"].to_numpy(dtype=np.int64, copy=False), kind="stable")
    out = out.iloc[order].drop(columns=["_date_ns"]).reset_index(drop=True)
    return out


def _has_regular_bar_gaps(df: pd.DataFrame, *, bar_ms: int) -> bool:
    if df is None or df.empty or len(df) <= 1:
        return False
    dates = pd.to_datetime(df["date"], utc=True).drop_duplicates().sort_values()
    if len(dates) <= 1:
        return False
    step_ns = int(bar_ms) * 1_000_000
    diffs = dates.astype("int64").diff().dropna().to_numpy(dtype=np.int64, copy=False)
    return bool(np.any(diffs != step_ns))


def _fill_internal_ohlcv_gaps(df: pd.DataFrame, *, bar_ms: int) -> pd.DataFrame:
    if df is None or df.empty or len(df) <= 1:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    freq = pd.Timedelta(milliseconds=int(bar_ms))
    full_index = pd.date_range(out["date"].min(), out["date"].max(), freq=freq, tz="UTC")
    if len(full_index) == len(out):
        return out
    indexed = out.set_index("date").reindex(full_index)
    inserted = indexed["close"].isna()
    prev_close = indexed["close"].ffill()
    for col in ("open", "high", "low", "close"):
        indexed[col] = indexed[col].where(~inserted, prev_close)
    indexed["volume"] = indexed["volume"].where(~inserted, 0.0)
    indexed = indexed.dropna(subset=["open", "high", "low", "close"]).reset_index(names="date")
    return indexed[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _binance_vision_zip_frame(session: requests.Session, url: str) -> pd.DataFrame | None:
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, header=None)
    if frame.empty:
        return frame
    return frame


def _binance_archive_symbols_from_listing(xml_text: str) -> list[str]:
    symbols: set[str] = set()
    for raw_prefix in re.findall(
        r"<(?:[A-Za-z_][\w.-]*:)?Prefix\b[^>]*>(.*?)</(?:[A-Za-z_][\w.-]*:)?Prefix>",
        str(xml_text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    ):
        prefix = html.unescape(str(raw_prefix or "")).strip().strip("/")
        if not prefix:
            continue
        symbol = prefix.rsplit("/", 1)[-1].strip().upper()
        if (
            symbol.endswith("USDT")
            and "_" not in symbol
            and not symbol.endswith("SETTLED")
            and len(symbol) > 4
        ):
            symbols.add(symbol)
    return sorted(symbols)


def discover_binance_archive_pairs(session: requests.Session | None = None) -> list[str]:
    """Discover USDT futures symbols that exist in Binance public kline archives."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    r = sess.get(
        BINANCE_VISION_S3_URL,
        params={"delimiter": "/", "prefix": "data/futures/um/monthly/klines/"},
        timeout=30,
    )
    r.raise_for_status()
    symbols = _binance_archive_symbols_from_listing(r.text)
    return [_usdt_pair_from_symbol(symbol) for symbol in symbols]


def discover_binance_futures_pairs(session: requests.Session | None = None) -> list[str]:
    """Discover current Binance USD-M USDT perpetual pairs, with archive fallback."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    try:
        r = sess.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20)
        r.raise_for_status()
        payload = r.json()
        pairs = []
        for item in payload.get("symbols", []) if isinstance(payload, Mapping) else []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("contractType") or "").upper() != "PERPETUAL":
                continue
            if str(item.get("status") or "").upper() != "TRADING":
                continue
            if str(item.get("quoteAsset") or "").upper() != "USDT":
                continue
            pairs.append(_usdt_pair_from_symbol(str(item.get("symbol") or "")))
        if pairs:
            return sorted(dict.fromkeys(pairs))
    except Exception as exc:  # noqa: BLE001
        print(f"[binance discover] exchangeInfo unavailable, using public archive listing: {exc}")
    return discover_binance_archive_pairs(sess)


def discover_bybit_futures_pairs(session: requests.Session | None = None) -> list[str]:
    """Discover current Bybit linear USDT perpetual pairs."""
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    cursor = ""
    pairs: list[str] = []
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = sess.get("https://api.bybit.com/v5/market/instruments-info", params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if int(payload.get("retCode", -1)) != 0:
            raise RuntimeError(f"Bybit instruments retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")
        result = payload.get("result") or {}
        for item in result.get("list") or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("contractType") or "").lower() != "linearperpetual":
                continue
            if str(item.get("status") or "").lower() != "trading":
                continue
            if str(item.get("quoteCoin") or "").upper() != "USDT":
                continue
            pairs.append(_usdt_pair_from_symbol(str(item.get("symbol") or "")))
        cursor = str(result.get("nextPageCursor") or "")
        if not cursor:
            break
    return sorted(dict.fromkeys(pairs))


def _binance_archive_ohlcv(
    session: requests.Session,
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch Binance USD-M kline archives from data.binance.vision.

    REST is blocked in some jurisdictions with HTTP 451. The official public
    archive is separate S3/CloudFront data and remains suitable for historical
    backtest ingestion.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts <= start_ts:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    end_inclusive = end_ts - pd.Timedelta(milliseconds=1)
    frames: list[pd.DataFrame] = []
    months = pd.period_range(
        start_ts.tz_convert(None).to_period("M"),
        end_inclusive.tz_convert(None).to_period("M"),
        freq="M",
    )
    for period in months:
        ym = f"{period.year:04d}-{period.month:02d}"
        monthly_url = (
            f"{BINANCE_VISION_BASE_URL}/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{ym}.zip"
        )
        try:
            frame = _binance_vision_zip_frame(session, monthly_url)
        except Exception as exc:  # noqa: BLE001
            print(f"    [binance archive {symbol}] monthly {ym} err: {exc}")
            frame = None
        if frame is not None:
            frames.append(frame)
            continue

        month_start = pd.Timestamp(period.start_time, tz="UTC")
        month_end = pd.Timestamp(period.end_time, tz="UTC")
        daily_start = max(start_ts.normalize(), month_start)
        daily_end = min(end_inclusive.normalize(), month_end.normalize())
        for day in pd.date_range(daily_start, daily_end, freq="D", tz="UTC"):
            ds = day.strftime("%Y-%m-%d")
            daily_url = (
                f"{BINANCE_VISION_BASE_URL}/daily/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{ds}.zip"
            )
            try:
                daily_frame = _binance_vision_zip_frame(session, daily_url)
            except Exception as exc:  # noqa: BLE001
                print(f"    [binance archive {symbol}] daily {ds} err: {exc}")
                daily_frame = None
            if daily_frame is not None:
                frames.append(daily_frame)
            time.sleep(0.02)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    raw = pd.concat(frames, ignore_index=True)
    if raw.shape[1] < 6:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    raw = raw.iloc[:, :6].copy()
    raw.columns = ["ts", "open", "high", "low", "close", "volume"]
    raw["ts"] = pd.to_numeric(raw["ts"], errors="coerce")
    raw = raw.dropna(subset=["ts"])
    if raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    ts = raw["ts"].astype("int64")
    unit = "us" if float(ts.median()) > 100_000_000_000_000 else "ms"
    raw["date"] = pd.to_datetime(ts, unit=unit, utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return _normalize_ohlcv_frame(raw, start=start, end=end)


def _pair_effective_start_ms(
    pair: str,
    *,
    start_ms: int,
    bar_ms: int,
    pair_start_dates: Mapping[str, object] | None = None,
) -> int:
    if not pair_start_dates:
        return start_ms
    raw = pair_start_dates.get(pair)
    if raw is None:
        raw = pair_start_dates.get(pair.split("/", 1)[0])
    if raw is None:
        return start_ms
    try:
        if isinstance(raw, (int, float)):
            value = int(raw)
            raw_ms = value if value > 10_000_000_000 else value * 1000
        else:
            raw_ms = int(pd.to_datetime(raw, utc=True).timestamp() * 1000)
    except Exception:
        return start_ms
    return max(start_ms, raw_ms - int(bar_ms))


def prepare_okx_futures_auxiliary(data_dir: Path = OKX_FUTURES_DIR, timeframe: str | None = None) -> Dict[str, int]:
    """Create local mark/funding proxy files expected by Freqtrade futures backtests.

    The first rank-portfolio version uses close as a mark-price proxy and zero
    funding when historical funding candles are unavailable.
    """
    root = Path(data_dir)
    results: Dict[str, int] = {}
    pattern = f"*-{timeframe}-futures.feather" if timeframe else "*-futures.feather"
    for path in sorted(root.glob(pattern)):
        try:
            df = pd.read_feather(path)
        except Exception as exc:
            print(f"[aux skip] unreadable futures feather: {path} error={exc!r}")
            continue
        df["date"] = pd.to_datetime(df["date"], utc=True)
        suffix = "-futures.feather"
        stem = path.name[: -len(suffix)]
        mark_path = root / f"{stem}-mark.feather"
        funding_path = root / f"{stem}-funding_rate.feather"

        mark = df[["date", "open", "high", "low", "close", "volume"]].copy()
        _write_feather_atomic(mark, mark_path)

        funding = mark.copy()
        for col in ("open", "high", "low", "close", "volume"):
            funding[col] = 0.0
        _write_feather_atomic(funding, funding_path)
        results[stem] = int(len(df))
    return results


def download_okx_futures(
    start: str = "2025-04-12",
    end: str = "2026-04-12",
    pairs: Sequence[str] | str | None = None,
    timeframe: str = "1h",
    sleep_s: float = 0.15,
    data_dir: Path = OKX_FUTURES_DIR,
    pair_start_dates: Mapping[str, object] | None = None,
    prepare_auxiliary_files: bool = True,
) -> Dict[str, int]:
    """Download OKX USDT-SWAP OHLCV and merge with any existing local feather.

    The endpoint is paged backwards by timestamp, so the function can be safely
    re-run after interruption: completed pair files are merged/deduped and
    already-complete files are skipped.
    """
    if timeframe not in OKX_BAR_MAP:
        raise ValueError(f"timeframe must be one of {sorted(OKX_BAR_MAP)}")
    bar, bar_ms = OKX_BAR_MAP[timeframe]
    pairs = _parse_pairs(pairs)
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    start_ms, end_ms = _ts(start) * 1000, _ts(end) * 1000
    results = {}

    for pair in pairs:
        sym = pair.split("/")[0]
        inst_id = f"{sym}-USDT-SWAP"
        out = _okx_futures_path(pair, timeframe, root)
        pair_start_ms = _pair_effective_start_ms(
            pair,
            start_ms=start_ms,
            bar_ms=bar_ms,
            pair_start_dates=pair_start_dates,
        )
        existing = None
        if out.exists():
            try:
                existing = _normalize_ohlcv_frame(pd.read_feather(out), start=start, end=end)
            except Exception:
                existing = None
        if existing is not None and not existing.empty:
            max_dt = existing["date"].max()
            min_dt = existing["date"].min()
            if (
                min_dt <= pd.Timestamp(pair_start_ms, unit="ms", tz="UTC") + pd.Timedelta(milliseconds=bar_ms)
                and max_dt >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(milliseconds=bar_ms)
            ):
                results[pair] = int(existing.shape[0])
                print(f"[{sym} {timeframe}] already complete: {len(existing):,} bars")
                continue

        print(f"[{sym} {timeframe}] {inst_id}", flush=True)
        all_rows = {}
        cursor = end_ms
        last_log = time.time()
        while cursor > pair_start_ms:
            try:
                r = session.get(
                    "https://www.okx.com/api/v5/market/history-candles",
                    params={"instId": inst_id, "bar": bar, "limit": 300,
                            "before": str(pair_start_ms), "after": str(cursor)},
                    timeout=20,
                )
                r.raise_for_status()
                d = r.json()
                if d.get("code") != "0":
                    print(f"    [{sym} {timeframe}] api code={d.get('code')} msg={d.get('msg')}")
                    time.sleep(2); continue
                items = d.get("data", [])
                if not items:
                    break
                for row in items:
                    ts = int(row[0])
                    if start_ms <= ts < end_ms:
                        all_rows[ts] = row
                oldest = min(int(r[0]) for r in items)
                cursor = oldest if oldest < cursor else (cursor - bar_ms)
                time.sleep(sleep_s)
            except Exception as e:
                print(f"    err: {e}"); time.sleep(2)
            if time.time() - last_log > 30:
                last_log = time.time()
                got = len(all_rows)
                if got:
                    oldest_seen = min(all_rows)
                    pct = (end_ms - oldest_seen) / max(1, end_ms - pair_start_ms) * 100.0
                    print(f"    [{sym} {timeframe}] fetched={got:,} progress={pct:.1f}%", flush=True)

        if not all_rows:
            if existing is not None and not existing.empty:
                _write_feather_atomic(existing, out)
                results[pair] = int(existing.shape[0])
            else:
                results[pair] = 0
            continue
        rows = [all_rows[k] for k in sorted(all_rows.keys())]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQ", "confirm"])
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = _normalize_ohlcv_frame(df, start=start, end=end)
        if existing is not None and not existing.empty:
            df = _normalize_ohlcv_frame(pd.concat([existing, df], ignore_index=True), start=start, end=end)
        _write_feather_atomic(df, out)
        results[pair] = int(len(df))
        if df.empty:
            print(f"    {pair} {timeframe}: 0 bars")
        else:
            print(f"    {len(df):,} bars  {df['date'].min()} -> {df['date'].max()}")

    if prepare_auxiliary_files:
        prepare_okx_futures_auxiliary(root, timeframe=timeframe)
    return results


def download_bybit_futures(
    start: str = "2025-04-12",
    end: str = "2026-04-12",
    pairs: Sequence[str] | str | None = None,
    timeframe: str = "1h",
    sleep_s: float = 0.15,
    data_dir: Path = BYBIT_FUTURES_DIR,
    pair_start_dates: Mapping[str, object] | None = None,
) -> Dict[str, int]:
    """Download Bybit linear USDT perpetual OHLCV and merge local feathers."""
    if timeframe not in BYBIT_BAR_MAP:
        raise ValueError(f"timeframe must be one of {sorted(BYBIT_BAR_MAP)}")
    interval, bar_ms = BYBIT_BAR_MAP[timeframe]
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if _is_all_pairs(pairs):
        pairs = discover_bybit_futures_pairs(session)
        print(f"[bybit discover] {len(pairs)} USDT perpetual pairs")
    else:
        pairs = _parse_pairs(pairs)
    start_ms, end_ms = _ts(start) * 1000, _ts(end) * 1000
    results: Dict[str, int] = {}

    for pair in pairs:
        symbol = _linear_symbol_for_pair(pair)
        out = _futures_path(pair, timeframe, root)
        pair_start_ms = _pair_effective_start_ms(
            pair,
            start_ms=start_ms,
            bar_ms=bar_ms,
            pair_start_dates=pair_start_dates,
        )
        existing = None
        if out.exists():
            try:
                existing = _normalize_ohlcv_frame(pd.read_feather(out), start=start, end=end)
            except Exception:
                existing = None
        if existing is not None and not existing.empty:
            max_dt = existing["date"].max()
            min_dt = existing["date"].min()
            if (
                min_dt <= pd.Timestamp(pair_start_ms, unit="ms", tz="UTC") + pd.Timedelta(milliseconds=bar_ms)
                and max_dt >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(milliseconds=bar_ms)
            ):
                results[pair] = int(existing.shape[0])
                print(f"[bybit {symbol} {timeframe}] already complete: {len(existing):,} bars")
                continue

        print(f"[bybit {symbol} {timeframe}]", flush=True)
        all_rows: Dict[int, list] = {}
        cursor = end_ms
        last_log = time.time()
        consecutive_errors = 0
        while cursor > pair_start_ms:
            window_start = max(pair_start_ms, cursor - bar_ms * 1000)
            try:
                r = session.get(
                    "https://api.bybit.com/v5/market/kline",
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "start": str(window_start),
                        "end": str(cursor),
                        "limit": 1000,
                    },
                    timeout=20,
                )
                r.raise_for_status()
                payload = r.json()
                if int(payload.get("retCode", -1)) != 0:
                    print(f"    [bybit {symbol}] retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    time.sleep(2)
                    continue
                items = ((payload.get("result") or {}).get("list") or [])
                if not items:
                    cursor = window_start - 1
                    continue
                for row in items:
                    ts = int(row[0])
                    if start_ms <= ts < end_ms:
                        all_rows[ts] = row
                oldest = min(int(row[0]) for row in items)
                cursor = oldest - 1 if oldest < cursor else cursor - bar_ms * 1000
                consecutive_errors = 0
                time.sleep(sleep_s)
            except Exception as exc:
                print(f"    [bybit {symbol}] err: {exc}")
                if isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) in {403, 451}:
                    break
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(2)
            if time.time() - last_log > 30:
                last_log = time.time()
                got = len(all_rows)
                if got:
                    oldest_seen = min(all_rows)
                    pct = (end_ms - oldest_seen) / max(1, end_ms - pair_start_ms) * 100.0
                    print(f"    [bybit {symbol} {timeframe}] fetched={got:,} progress={pct:.1f}%", flush=True)

        if not all_rows:
            if existing is not None and not existing.empty:
                _write_feather_atomic(existing, out)
                results[pair] = int(existing.shape[0])
            else:
                results[pair] = 0
            continue
        rows = [all_rows[k] for k in sorted(all_rows)]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _normalize_ohlcv_frame(df, start=start, end=end)
        if existing is not None and not existing.empty:
            df = _normalize_ohlcv_frame(pd.concat([existing, df], ignore_index=True), start=start, end=end)
        _write_feather_atomic(df, out)
        results[pair] = int(len(df))
        if df.empty:
            print(f"    {pair} {timeframe}: 0 bars")
        else:
            print(f"    {len(df):,} bars  {df['date'].min()} -> {df['date'].max()}")

    return results


def download_binance_futures(
    start: str = "2025-04-12",
    end: str = "2026-04-12",
    pairs: Sequence[str] | str | None = None,
    timeframe: str = "1h",
    sleep_s: float = 0.15,
    data_dir: Path = BINANCE_FUTURES_DIR,
    pair_start_dates: Mapping[str, object] | None = None,
) -> Dict[str, int]:
    """Download Binance USD-M perpetual OHLCV and merge local feathers."""
    if timeframe not in BINANCE_BAR_MAP:
        raise ValueError(f"timeframe must be one of {sorted(BINANCE_BAR_MAP)}")
    interval, bar_ms = BINANCE_BAR_MAP[timeframe]
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    archive_only = False
    if _is_all_pairs(pairs):
        pairs = discover_binance_futures_pairs(session)
        archive_only = True
        print(f"[binance discover] {len(pairs)} USDT futures pairs")
    else:
        pairs = _parse_pairs(pairs)
    start_ms, end_ms = _ts(start) * 1000, _ts(end) * 1000
    results: Dict[str, int] = {}

    def _load_complete_existing(out: Path, pair_start_ms: int) -> pd.DataFrame | None:
        if not out.exists():
            return None
        try:
            existing_frame = _normalize_ohlcv_frame(pd.read_feather(out), start=start, end=end)
        except Exception:
            return None
        if existing_frame.empty:
            return existing_frame
        max_dt = existing_frame["date"].max()
        min_dt = existing_frame["date"].min()
        if (
            min_dt <= pd.Timestamp(pair_start_ms, unit="ms", tz="UTC") + pd.Timedelta(milliseconds=bar_ms)
            and max_dt >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(milliseconds=bar_ms)
            and not _has_regular_bar_gaps(existing_frame, bar_ms=bar_ms)
        ):
            return existing_frame
        return None

    def _download_binance_archive_pair(pair: str) -> tuple[str, int, str]:
        symbol = _linear_symbol_for_pair(pair)
        out = _futures_path(pair, timeframe, root)
        pair_start_ms = _pair_effective_start_ms(
            pair,
            start_ms=start_ms,
            bar_ms=bar_ms,
            pair_start_dates=pair_start_dates,
        )
        complete = _load_complete_existing(out, pair_start_ms)
        if complete is not None and not complete.empty:
            return pair, int(complete.shape[0]), f"[binance {symbol} {timeframe}] already complete: {len(complete):,} bars"
        existing = None
        if out.exists():
            try:
                existing = _normalize_ohlcv_frame(pd.read_feather(out), start=start, end=end)
            except Exception:
                existing = None
        local_session = requests.Session()
        local_session.headers.update({"User-Agent": USER_AGENT})
        try:
            archive = _binance_archive_ohlcv(local_session, symbol=symbol, interval=interval, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            archive = pd.DataFrame()
            message = f"[binance archive {symbol}] err: {exc}"
        else:
            message = ""
        if not archive.empty:
            df = archive
            if existing is not None and not existing.empty:
                df = _normalize_ohlcv_frame(pd.concat([existing, df], ignore_index=True), start=start, end=end)
            df = _fill_internal_ohlcv_gaps(df, bar_ms=bar_ms)
            _write_feather_atomic(df, out)
            return pair, int(len(df)), (
                f"[binance {symbol} {timeframe}] [archive] {len(df):,} bars  "
                f"{df['date'].min()} -> {df['date'].max()}"
            )
        if existing is not None and not existing.empty:
            _write_feather_atomic(existing, out)
            return pair, int(existing.shape[0]), f"[binance {symbol} {timeframe}] kept existing partial: {len(existing):,} bars"
        return pair, 0, message or f"[binance {symbol} {timeframe}] no archive rows"

    if archive_only:
        workers = max(1, int(os.environ.get("BINANCE_ARCHIVE_WORKERS", "12") or "12"))
        workers = min(workers, 32)
        print(f"[binance archive] downloading with {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_binance_archive_pair, pair) for pair in pairs]
            completed = 0
            for fut in as_completed(futures):
                pair, rows, message = fut.result()
                results[pair] = int(rows)
                completed += 1
                if message:
                    print(f"{message} ({completed}/{len(futures)})", flush=True)
        return results

    for pair in pairs:
        symbol = _linear_symbol_for_pair(pair)
        out = _futures_path(pair, timeframe, root)
        pair_start_ms = _pair_effective_start_ms(
            pair,
            start_ms=start_ms,
            bar_ms=bar_ms,
            pair_start_dates=pair_start_dates,
        )
        existing = None
        if out.exists():
            try:
                existing = _normalize_ohlcv_frame(pd.read_feather(out), start=start, end=end)
            except Exception:
                existing = None
        if existing is not None and not existing.empty:
            max_dt = existing["date"].max()
            min_dt = existing["date"].min()
            if (
                min_dt <= pd.Timestamp(pair_start_ms, unit="ms", tz="UTC") + pd.Timedelta(milliseconds=bar_ms)
                and max_dt >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(milliseconds=bar_ms)
                and not _has_regular_bar_gaps(existing, bar_ms=bar_ms)
            ):
                results[pair] = int(existing.shape[0])
                print(f"[binance {symbol} {timeframe}] already complete: {len(existing):,} bars")
                continue

        print(f"[binance {symbol} {timeframe}]", flush=True)
        all_rows: Dict[int, list] = {}
        cursor = pair_start_ms
        last_log = time.time()
        consecutive_errors = 0
        while cursor < end_ms and not archive_only:
            try:
                r = session.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": str(cursor),
                        "endTime": str(end_ms - 1),
                        "limit": 1500,
                    },
                    timeout=20,
                )
                r.raise_for_status()
                items = r.json()
                if not isinstance(items, list):
                    print(f"    [binance {symbol}] unexpected response: {items}")
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        break
                    time.sleep(2)
                    continue
                if not items:
                    break
                for row in items:
                    ts = int(row[0])
                    if start_ms <= ts < end_ms:
                        all_rows[ts] = row
                newest = max(int(row[0]) for row in items)
                next_cursor = newest + bar_ms
                cursor = next_cursor if next_cursor > cursor else cursor + bar_ms * 1500
                consecutive_errors = 0
                time.sleep(sleep_s)
            except Exception as exc:
                print(f"    [binance {symbol}] err: {exc}")
                if isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) in {403, 451}:
                    break
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                time.sleep(2)
            if time.time() - last_log > 30:
                last_log = time.time()
                pct = (cursor - pair_start_ms) / max(1, end_ms - pair_start_ms) * 100.0
                print(f"    [binance {symbol} {timeframe}] fetched={len(all_rows):,} progress={pct:.1f}%", flush=True)

        if not all_rows:
            try:
                archive = _binance_archive_ohlcv(session, symbol=symbol, interval=interval, start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                print(f"    [binance archive {symbol}] err: {exc}")
                archive = pd.DataFrame()
            if not archive.empty:
                df = archive
                if existing is not None and not existing.empty:
                    df = _normalize_ohlcv_frame(pd.concat([existing, df], ignore_index=True), start=start, end=end)
                df = _fill_internal_ohlcv_gaps(df, bar_ms=bar_ms)
                _write_feather_atomic(df, out)
                results[pair] = int(len(df))
                print(f"    [archive] {len(df):,} bars  {df['date'].min()} -> {df['date'].max()}")
                continue
            if existing is not None and not existing.empty:
                _write_feather_atomic(existing, out)
                results[pair] = int(existing.shape[0])
            else:
                results[pair] = 0
            continue
        rows = [all_rows[k] for k in sorted(all_rows)]
        df = pd.DataFrame(
            rows,
            columns=[
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_base_volume",
                "taker_quote_volume",
                "ignore",
            ],
        )
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _normalize_ohlcv_frame(df, start=start, end=end)
        if existing is not None and not existing.empty:
            df = _normalize_ohlcv_frame(pd.concat([existing, df], ignore_index=True), start=start, end=end)
        df = _fill_internal_ohlcv_gaps(df, bar_ms=bar_ms)
        _write_feather_atomic(df, out)
        results[pair] = int(len(df))
        if df.empty:
            print(f"    {pair} {timeframe}: 0 bars")
        else:
            print(f"    {len(df):,} bars  {df['date'].min()} -> {df['date'].max()}")

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
