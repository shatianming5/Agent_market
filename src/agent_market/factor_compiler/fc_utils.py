"""Shared utilities for factor_compiler sub-package."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def safe_float(value: Any) -> Optional[float]:
    """Convert *value* to float, returning ``None`` if conversion fails or result is non-finite."""
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None
