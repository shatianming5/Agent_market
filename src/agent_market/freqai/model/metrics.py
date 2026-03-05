"""Shared model evaluation metrics."""
from __future__ import annotations

import numpy as np


def rmse(preds: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((preds - target) ** 2)))
