from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


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


def _infer_periods_per_year(index) -> float:  # noqa: ANN001
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception:
        return 365.0

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 365.0

    try:
        deltas = index.to_series().diff()
        median = deltas.median()
        if pd.isna(median):
            return 365.0
        seconds = float(median.total_seconds())
        if seconds <= 0:
            return 365.0
        return (365.0 * 24.0 * 3600.0) / seconds
    except Exception:
        return 365.0


def load_prices_from_feather(
    root: Path,
    exchange: str,
    pairs: list[str],
    timeframe: str,
    timerange: str | None,
    data_dir: str | Path = "user_data/data",
) -> "pd.DataFrame":
    """Load close prices from local feather OHLCV files.

    Expected layout:
      {root}/{data_dir}/{exchange}/{PAIR_SLUG}-{timeframe}.feather
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing pandas dependency: {exc}") from exc

    ex = str(exchange or "").strip().lower()
    if not ex:
        raise ValueError("exchange is required")
    tf = str(timeframe or "").strip().lower()
    if not tf:
        raise ValueError("timeframe is required")
    if not pairs:
        raise ValueError("pairs must not be empty")

    base = Path(data_dir)
    if not base.is_absolute():
        base = (root / base).resolve()
    ex_dir = (base / ex).resolve()

    start_dt = None
    end_exclusive = None
    if timerange:
        start_dt, end_exclusive = _parse_timerange(str(timerange))

    series_by_pair: dict[str, "pd.Series"] = {}
    for pair in pairs:
        p = str(pair or "").strip()
        if not p or "/" not in p:
            continue
        slug = p.replace("/", "_")
        fp = ex_dir / f"{slug}-{tf}.feather"
        if not fp.exists():
            raise FileNotFoundError(f"Data file not found: {fp}")
        df = pd.read_feather(fp)
        if "date" not in df.columns or "close" not in df.columns:
            raise ValueError(f"Invalid OHLCV feather (missing date/close): {fp}")
        dt = pd.to_datetime(df["date"])
        # Normalize to tz-naive for stable slicing with YYYYMMDD timerange.
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert(None)
        except Exception:
            pass
        close = df["close"].astype("float64")
        s = close.copy()
        s.index = dt
        s.name = p
        s = s.sort_index()
        if start_dt is not None and end_exclusive is not None:
            s = s.loc[(s.index >= start_dt) & (s.index < end_exclusive)]
        series_by_pair[p] = s

    if not series_by_pair:
        raise ValueError("No valid pairs loaded for portfolio optimization")

    prices = pd.concat(series_by_pair.values(), axis=1, join="inner").sort_index()
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("No overlapping price data after alignment")
    if prices.shape[0] < 30:
        raise ValueError("Not enough data for portfolio optimization")
    return prices


def compute_returns(prices: "pd.DataFrame", kind: str) -> "pd.DataFrame":
    try:
        import numpy as np  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing numpy dependency: {exc}") from exc

    k = str(kind or "").strip().lower()
    if k not in ("log", "simple"):
        raise ValueError("returns kind must be 'log' or 'simple'")
    p = prices.sort_index().astype("float64")
    if k == "log":
        returns = np.log(p).diff()
    else:
        returns = p.pct_change()
    returns = returns.dropna(how="any")
    if returns.shape[0] < 30:
        raise ValueError("Not enough data for portfolio optimization")
    return returns


@dataclass
class PortfolioReport:
    weights: Dict[str, float]
    stats: Dict[str, Any]
    inputs: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights, "stats": self.stats, "inputs": self.inputs}


def optimize_hrp(returns: "pd.DataFrame") -> Dict[str, Any]:
    """Optimize weights using Hierarchical Risk Parity (HRP)."""
    try:
        from pypfopt import HRPOpt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Missing PyPortfolioOpt dependency: {exc}. "
            "Install with: pip install PyPortfolioOpt (or pip install -r requirements-full.txt)"
        ) from exc

    r = returns.dropna(how="any")
    if r.empty or r.shape[0] < 30:
        raise ValueError("Not enough data for portfolio optimization")

    opt = HRPOpt(r)
    opt.optimize()
    weights_raw = opt.clean_weights()
    weights: dict[str, float] = {str(k): float(v) for k, v in dict(weights_raw).items()}

    # Stats from returns + weights (annualized best-effort from datetime index)
    periods_per_year = _infer_periods_per_year(r.index)
    w = weights
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    w_series = None
    if pd is not None:
        w_series = pd.Series(w).reindex(r.columns).fillna(0.0).astype("float64")
        port = (r * w_series).sum(axis=1)
        mean = float(port.mean())
        vol = float(port.std(ddof=1))
    else:  # pragma: no cover
        mean = 0.0
        vol = 0.0

    ann_return = mean * float(periods_per_year)
    ann_vol = vol * math.sqrt(float(periods_per_year)) if periods_per_year else 0.0
    sharpe = (ann_return / ann_vol) if ann_vol else None

    stats: dict[str, Any] = {
        "n_assets": int(len(w)),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "max_weight": float(max(w.values())) if w else None,
        "min_weight": float(min(w.values())) if w else None,
        "start": str(r.index.min()) if hasattr(r.index, "min") else None,
        "end": str(r.index.max()) if hasattr(r.index, "max") else None,
        "periods_per_year": float(periods_per_year),
    }
    return {"weights": weights, "stats": stats}


__all__ = [
    "PortfolioReport",
    "compute_returns",
    "load_prices_from_feather",
    "optimize_hrp",
]
