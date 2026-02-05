from __future__ import annotations

from typing import Any


def _ensure_pandas() -> Any:
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency for microstructure features: {exc}") from exc
    return pd


def realized_volatility(series: Any, *, window: int) -> Any:
    """
    Realized volatility proxy on a rolling window.
    """

    pd = _ensure_pandas()
    w = int(window)
    s = pd.Series(series).astype("float64")
    r = s.pct_change()
    return r.rolling(w).std(ddof=0)


__all__ = ["realized_volatility"]

