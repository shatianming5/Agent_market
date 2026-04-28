from __future__ import annotations

from typing import Any


def future_return(close: Any, horizon: int) -> Any:
    """
    Compute future return label over `horizon` steps.

    This is intentionally explicit (plan.md 3.5.3) so label generation is not a
    scattered inline `shift(-h)` across pipelines.
    """

    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be > 0")
    return (close.shift(-h) / close) - 1.0


def future_classify_3way(close: Any, horizon: int, threshold: float = 0.005) -> Any:
    """
    Three-way classification label: 0=down (<-threshold), 1=flat, 2=up (>+threshold).

    Returns integer-valued series (with NaN preserved for rows without enough lookahead).
    """
    ret = future_return(close, horizon)
    import numpy as np  # local to avoid circular/eager imports
    out = ret.copy() * 0.0 + 1.0  # default flat
    out = out.where(~(ret > float(threshold)), 2.0)
    out = out.where(~(ret < -float(threshold)), 0.0)
    out = out.where(~ret.isna(), np.nan)
    return out


__all__ = ["future_return", "future_classify_3way"]

