"""Stacked-ensemble adapter — plugs Ridge + LightGBM + Logistic meta into
the TrainingPipeline via ModelRegistry.

Use by setting model.name = "stacked" in the training config.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market.freqai.model.base import BaseModelAdapter, ModelRegistry, TrainResult
from agent_market.freqai.model.metrics import multiclass_metrics
from agent_market.freqai.stacking import StackedClassifier, StackConfig


class StackedAdapter(BaseModelAdapter):
    registry_name = "stacked"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model: Optional[StackedClassifier] = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> TrainResult:
        params = dict(self.config)
        model_dir = Path(params.pop("model_dir", "artifacts/models/stacked"))
        cfg = StackConfig(
            lgb_params={k: v for k, v in params.items()
                        if k in {"num_leaves", "learning_rate", "min_child_samples",
                                  "subsample", "colsample_bytree", "reg_alpha",
                                  "reg_lambda", "max_depth"}},
            ridge_alpha=float(params.get("ridge_alpha", 1.0)),
            meta_C=float(params.get("meta_C", 1.0)),
            n_classes=int(params.get("num_class", 3)),
            cv_folds=int(params.get("cv_folds", 3)),
        )
        clf = StackedClassifier(cfg)
        print(f"[stacked] fit size={len(X_train)} classes={cfg.n_classes} cv_folds={cfg.cv_folds}")
        clf.fit(X_train, y_train)
        self.model = clf

        # Metrics: use LightGBM-like probability output for compatibility
        tr_p = clf.predict_proba(X_train)
        metrics = {f"{k}_train": v for k, v in multiclass_metrics(tr_p, y_train).items()}
        metrics["rmse_train"] = 0.0
        if X_valid is not None and len(X_valid) > 0:
            va_p = clf.predict_proba(X_valid)
            for k, v in multiclass_metrics(va_p, y_valid).items():
                metrics[f"{k}_valid"] = v
            metrics["rmse_valid"] = 0.0
        # Log base weights
        w = clf.base_weights()
        for k, v in w.items():
            metrics[f"weight_{k}"] = float(v)
        print(f"[stacked] meta base weights: {w}")

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "stacked_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(clf, f)
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("stacked model not trained")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("stacked model not trained")
        with Path(path).open("wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            self.model = pickle.load(f)


ModelRegistry.register(StackedAdapter.registry_name, StackedAdapter)
