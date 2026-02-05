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


__all__ = ["future_return"]

