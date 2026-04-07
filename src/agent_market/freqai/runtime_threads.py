from __future__ import annotations

import os
from typing import Any, Mapping


_THREAD_ENV_KEYS = (
    "AGENT_MARKET_MODEL_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "AGENT_MARKET_SANDBOX_THREADS",
)


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def resolve_num_threads(
    config: Mapping[str, Any] | None,
    *,
    explicit_keys: tuple[str, ...] = ("num_threads", "nthread"),
    default: int = 1,
) -> int:
    """Resolve a stable worker-thread budget from config/env.

    Order:
    1. explicit config keys
    2. shared runtime env overrides
    3. fallback default
    """
    payload = config or {}
    for key in explicit_keys:
        value = _coerce_positive_int(payload.get(key))
        if value is not None:
            return value

    for key in _THREAD_ENV_KEYS:
        value = _coerce_positive_int(os.environ.get(key))
        if value is not None:
            return value

    return max(1, int(default))

