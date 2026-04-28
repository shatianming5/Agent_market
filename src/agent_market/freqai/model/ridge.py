"""Ridge-only classification adapter — diagnostic of whether stacked's edge
comes from the Ridge base model alone vs the Ridge+LightGBM+Logistic combo.

Registers as "ridge_classifier" in ModelRegistry so it can be selected by
setting model.name = "ridge_classifier" in the training config.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agent_market.freqai.model.base import BaseModelAdapter, ModelRegistry, TrainResult
from agent_market.freqai.model.metrics import multiclass_metrics

try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return 1.0 / (1.0 + np.exp(-x))
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class RidgeOnlyModel:
    """Pickle-friendly wrapper that matches sklearn predict_proba shape."""

    def __init__(self, ridge: "RidgeClassifier", scaler: "StandardScaler", n_classes: int):
        self.ridge = ridge
        self.scaler = scaler
        self.n_classes = int(n_classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        dec = self.ridge.decision_function(Xs)
        p = _softmax(dec)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        if p.shape[1] < self.n_classes:
            pad = np.zeros((p.shape[0], self.n_classes - p.shape[1]))
            p = np.column_stack([p, pad])
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class RidgeOnlyAdapter(BaseModelAdapter):
    registry_name = "ridge_classifier"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model: Optional[RidgeOnlyModel] = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> TrainResult:
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn required for RidgeOnlyAdapter")
        params = dict(self.config)
        model_dir = Path(params.pop("model_dir", "artifacts/models/ridge"))
        alpha = float(params.get("alpha", params.get("ridge_alpha", 1.0)))
        n_classes = int(params.get("num_class", 3))

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        ridge = RidgeClassifier(alpha=alpha, random_state=42)
        ridge.fit(Xs, y_train)
        model = RidgeOnlyModel(ridge=ridge, scaler=scaler, n_classes=n_classes)
        self.model = model

        p_tr = model.predict_proba(X_train)
        metrics = {f"{k}_train": v for k, v in multiclass_metrics(p_tr, y_train).items()}
        metrics["rmse_train"] = 0.0
        if X_valid is not None and len(X_valid) > 0:
            p_va = model.predict_proba(X_valid)
            for k, v in multiclass_metrics(p_va, y_valid).items():
                metrics[f"{k}_valid"] = v
            metrics["rmse_valid"] = 0.0
        print(f"[ridge] alpha={alpha}  train_size={len(X_train)}  "
              f"acc_train={metrics.get('accuracy_train',0):.3f}  "
              f"acc_valid={metrics.get('accuracy_valid',0):.3f}")

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "ridge_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ridge model not trained")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("ridge model not trained")
        with Path(path).open("wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            self.model = pickle.load(f)


ModelRegistry.register(RidgeOnlyAdapter.registry_name, RidgeOnlyAdapter)
