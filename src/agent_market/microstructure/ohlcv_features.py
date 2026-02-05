from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _parse_timerange(timerange: str) -> tuple[datetime, datetime]:
    parts = str(timerange or "").split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid timerange: {timerange!r}")
    a = re.sub(r"\D", "", parts[0])[:8]
    b = re.sub(r"\D", "", parts[1])[:8]
    if len(a) != 8 or len(b) != 8:
        raise ValueError(f"Invalid timerange: {timerange!r}")
    start = datetime.strptime(a, "%Y%m%d")
    end_date = datetime.strptime(b, "%Y%m%d")
    end_exclusive = end_date + timedelta(days=1)
    return start, end_exclusive


def _ensure_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for micro features: {exc}") from exc
    return np, pd


def _load_ohlcv(ex_dir: Path, pair: str, timeframe: str) -> Any:
    _, pd = _ensure_pandas()
    slug = str(pair).replace("/", "_")
    fp = (Path(ex_dir) / f"{slug}-{timeframe}.feather").resolve()
    if not fp.exists():
        raise FileNotFoundError(f"OHLCV feather not found: {fp}")
    df = pd.read_feather(fp)
    if "date" not in df.columns:
        raise ValueError(f"Invalid OHLCV feather (missing date): {fp}")
    dt = pd.to_datetime(df["date"])
    # Normalize to tz-naive for stable slicing with YYYYMMDD timerange.
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    df["date"] = dt
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _compute_features(df: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    np, pd = _ensure_pandas()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    close = out["close"].astype("float64")
    open_ = out["open"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    volume = out["volume"].astype("float64")

    eps = 1e-12

    ret_1 = close.pct_change()
    logret_1 = np.log(close + eps).diff()

    out["ret_1"] = ret_1
    out["logret_1"] = logret_1
    out["range_pct"] = (high - low) / (close + eps)
    out["body_pct"] = (close - open_) / (open_ + eps)
    out["wick_up_pct"] = (high - np.maximum(open_, close)) / (close + eps)
    out["wick_down_pct"] = (np.minimum(open_, close) - low) / (close + eps)

    windows = [12, 24, 72]
    manifest: List[Dict[str, Any]] = [
        {"name": "ret_1", "source": "ohlcv", "kind": "return", "params": {"periods": 1}},
        {"name": "logret_1", "source": "ohlcv", "kind": "logreturn", "params": {"periods": 1}},
        {"name": "range_pct", "source": "ohlcv", "kind": "range", "params": {}},
        {"name": "body_pct", "source": "ohlcv", "kind": "candle_body", "params": {}},
        {"name": "wick_up_pct", "source": "ohlcv", "kind": "candle_wick_up", "params": {}},
        {"name": "wick_down_pct", "source": "ohlcv", "kind": "candle_wick_down", "params": {}},
    ]

    for w in windows:
        w = int(w)
        out[f"rv_{w}"] = logret_1.rolling(w).std(ddof=0)
        out[f"vol_z_{w}"] = (volume - volume.rolling(w).mean()) / (volume.rolling(w).std(ddof=0) + eps)
        out[f"amihud_{w}"] = (ret_1.abs() / (volume + eps)).rolling(w).mean()
        manifest += [
            {"name": f"rv_{w}", "source": "ohlcv", "kind": "realized_vol", "params": {"window": w}},
            {"name": f"vol_z_{w}", "source": "ohlcv", "kind": "volume_z", "params": {"window": w}},
            {"name": f"amihud_{w}", "source": "ohlcv", "kind": "amihud", "params": {"window": w}},
        ]

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, manifest


def build_ohlcv_micro_features(
    *,
    ex_dir: Path,
    exchange: str,
    pairs: Sequence[str],
    timeframe: str,
    timerange: Optional[str],
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Build OHLCV-derived micro feature dataset across pairs.

    Returns: (dataframe, feature_manifest)

    Output dataframe columns:
      date, pair, open/high/low/close/volume, feature columns...
    """
    _, pd = _ensure_pandas()

    start_dt = None
    end_exclusive = None
    if timerange:
        start_dt, end_exclusive = _parse_timerange(timerange)

    frames = []
    feature_manifest: List[Dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, str) or "/" not in pair:
            continue
        df = _load_ohlcv(Path(ex_dir), pair, timeframe)
        if start_dt is not None and end_exclusive is not None:
            df = df.loc[(df["date"] >= start_dt) & (df["date"] < end_exclusive)]
        feat_df, manifest = _compute_features(df)
        feat_df.insert(1, "pair", str(pair))
        frames.append(feat_df)
        if not feature_manifest:
            feature_manifest = manifest

    if not frames:
        raise ValueError("No micro features generated (empty pairs or missing data)")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["date", "pair"]).reset_index(drop=True)
    return out, feature_manifest


__all__ = ["build_ohlcv_micro_features"]

