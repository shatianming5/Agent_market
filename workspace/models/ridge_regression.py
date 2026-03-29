"""Ridge Regression model adapter — example workspace model.

Demonstrates how OpenCode agent can write a custom model that gets
automatically discovered and registered by model_loader.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


class RidgeRegressionAdapter(BaseModelAdapter):
    """Simple Ridge Regression — fast, interpretable, good baseline."""

    registry_name = "ridge_regression"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        alpha = float(self.config.get("alpha", 1.0))

        # Standardize
        self._mean = X_train.mean(axis=0)
        self._std = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - self._mean) / self._std

        # Closed-form ridge: w = (X'X + αI)^{-1} X'y
        n_features = X_norm.shape[1]
        XtX = X_norm.T @ X_norm + alpha * np.eye(n_features)
        Xty = X_norm.T @ y_train
        self._weights = np.linalg.solve(XtX, Xty)
        self._bias = float(y_train.mean())

        # Metrics
        pred_train = self.predict(X_train)
        metrics = {"rmse_train": _rmse(pred_train, y_train)}
        if X_valid is not None and y_valid is not None:
            pred_valid = self.predict(X_valid)
            metrics["rmse_valid"] = _rmse(pred_valid, y_valid)

        # Save
        model_dir = Path(self.config.get("model_dir", "artifacts/models/ridge"))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "ridge_model.pkl"
        self.save(model_path)

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model not trained")
        X_norm = (X - self._mean) / self._std
        return X_norm @ self._weights + self._bias

    def save(self, path: Path) -> None:
        data = {
            "weights": self._weights,
            "bias": self._bias,
            "mean": self._mean,
            "std": self._std,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._weights = data["weights"]
        self._bias = data["bias"]
        self._mean = data["mean"]
        self._std = data["std"]
