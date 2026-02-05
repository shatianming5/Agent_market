from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def rolling_ic_std(factor: pd.Series, target: pd.Series, *, window: int = 50) -> Optional[float]:
    """
    Compute rolling IC standard deviation (best-effort).
    """

    if factor is None or target is None:
        return None
    df = pd.concat([factor.rename("x"), target.rename("y")], axis=1).dropna()
    if df.shape[0] < max(3, int(window)):
        return None
    w = int(window)
    ics = df["x"].rolling(w).corr(df["y"]).dropna()
    if ics.empty:
        return None
    value = float(ics.std(ddof=0))
    return value if np.isfinite(value) else None


__all__ = ["rolling_ic_std"]

