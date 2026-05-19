"""Stacked ensemble: Ridge + LightGBM base learners + Logistic meta.

Exports a single `StackedClassifier` that implements the sklearn-compatible
`fit(X, y)` / `predict_proba(X)` interface, so downstream callers (freqtrade
FreqAI, train_pipeline, etc.) can use it as a drop-in replacement for the
current LightGBM model.

Design choices:
    - Ridge on raw features: captures linear relationships, fast, low-variance.
    - LightGBM on raw features: captures tree non-linearities & regime shifts.
    - Meta (logistic regression) learns how to combine the two base predictions
      with a single set of coefs — much less overfitting than a second LightGBM.
    - All 3 models retrained on every window (inside walk-forward).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


@dataclass
class StackConfig:
    lgb_params: Optional[Dict[str, Any]] = None
    ridge_alpha: float = 10.0      # ↑ regularize Ridge harder — previously dominated meta
    meta_C: float = 1.0
    n_classes: int = 3
    cv_folds: int = 3
    random_state: int = 42
    use_ridge: bool = True         # escape hatch — set False to run pure LGB
    lgb_weight: float = 1.0        # neutral by default; set >1 to upweight LGB in meta


class StackedClassifier:
    """Ridge + LightGBM → Logistic meta, sklearn-compatible."""

    def __init__(self, config: Optional[StackConfig] = None):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn required for StackedClassifier")
        if not _HAS_LGB:
            raise ImportError("lightgbm required for StackedClassifier")
        self.cfg = config or StackConfig()
        self._ridge: Optional[RidgeClassifier] = None
        self._lgb: Optional[Any] = None  # lgb.Booster
        self._meta: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._n_classes: int = self.cfg.n_classes
        self._fit_size: int = 0

    # ------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------

    def _oof_ridge_proba(self, Xs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OOF Ridge probabilities — same time-series split as LGB OOF."""
        folds = int(max(2, self.cfg.cv_folds))
        n = Xs.shape[0]
        out = np.zeros((n, self._n_classes), dtype=np.float64)
        fold_size = n // folds
        for k in range(folds):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < folds - 1 else n
            val_idx = np.arange(val_start, val_end)
            tr_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
            if len(tr_idx) < 100 or len(val_idx) < 10:
                continue
            r = RidgeClassifier(alpha=self.cfg.ridge_alpha)
            r.fit(Xs[tr_idx], y[tr_idx])
            p = _softmax(r.decision_function(Xs[val_idx]))
            if p.ndim == 1:
                p = np.column_stack([1.0 - p, p])
            if p.shape[1] < self._n_classes:
                pad = np.zeros((p.shape[0], self._n_classes - p.shape[1]))
                p = np.column_stack([p, pad])
            out[val_idx] = p[:, :self._n_classes]
        return out

    def _oof_lgb_proba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Out-of-fold LightGBM probabilities — used as meta features."""
        folds = int(max(2, self.cfg.cv_folds))
        n = X.shape[0]
        out = np.zeros((n, self._n_classes), dtype=np.float64)
        # Time-series split: no shuffling, sequential folds
        fold_size = n // folds
        for k in range(folds):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < folds - 1 else n
            val_idx = np.arange(val_start, val_end)
            tr_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
            if len(tr_idx) < 100 or len(val_idx) < 10:
                continue
            params = {
                "objective": "multiclass", "num_class": self._n_classes,
                "metric": "multi_logloss", "num_leaves": 15,
                "learning_rate": 0.03, "min_child_samples": 100,
                "subsample": 0.7, "colsample_bytree": 0.5,
                "reg_alpha": 2.0, "reg_lambda": 2.0,
                "verbosity": -1, "random_state": self.cfg.random_state,
            }
            if self.cfg.lgb_params:
                params.update(self.cfg.lgb_params)
            ds = lgb.Dataset(X[tr_idx], label=y[tr_idx])
            booster = lgb.train(params, ds, num_boost_round=150)
            out[val_idx] = booster.predict(X[val_idx])
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._fit_size = int(X.shape[0])
        n_classes = int(np.max(y)) + 1
        self._n_classes = max(n_classes, self.cfg.n_classes)

        # 1. Ridge on standardized features — IN-SAMPLE predictions are a leak source.
        #    Fix: use OOF Ridge prob too (same CV scheme as LGB) so meta sees honest signals.
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        if self.cfg.use_ridge:
            self._ridge = RidgeClassifier(alpha=self.cfg.ridge_alpha)
            self._ridge.fit(Xs, y)
            ridge_oof = self._oof_ridge_proba(Xs, y)
            # Pad to n_classes if needed
            if ridge_oof.shape[1] < self._n_classes:
                pad = np.zeros((ridge_oof.shape[0], self._n_classes - ridge_oof.shape[1]))
                ridge_oof = np.column_stack([ridge_oof, pad])
        else:
            self._ridge = None
            ridge_oof = None

        # 2. LightGBM OOF predictions (for meta training)
        lgb_oof = self._oof_lgb_proba(X, y) * float(self.cfg.lgb_weight)

        # 3. Full-sample LightGBM (for inference)
        params = {
            "objective": "multiclass", "num_class": self._n_classes,
            "metric": "multi_logloss", "num_leaves": 15,
            "learning_rate": 0.03, "min_child_samples": 100,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_alpha": 2.0, "reg_lambda": 2.0,
            "verbosity": -1, "random_state": self.cfg.random_state,
        }
        if self.cfg.lgb_params:
            params.update(self.cfg.lgb_params)
        self._lgb = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=200)

        # 4. Meta: logistic on concatenated OOF preds (LGB ± Ridge)
        if ridge_oof is not None:
            meta_X = np.hstack([ridge_oof, lgb_oof])
        else:
            meta_X = lgb_oof
        self._meta = LogisticRegression(C=self.cfg.meta_C, max_iter=1000)
        self._meta.fit(meta_X, y)
        return self

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._lgb is None or self._meta is None:
            raise RuntimeError("StackedClassifier not fitted")
        X = np.asarray(X, dtype=np.float64)
        lgb_p = self._lgb.predict(X) * float(self.cfg.lgb_weight)
        if self._ridge is not None:
            Xs = self._scaler.transform(X)
            ridge_p = _softmax(self._ridge.decision_function(Xs))
            if ridge_p.ndim == 1:
                ridge_p = np.column_stack([1.0 - ridge_p, ridge_p])
            if ridge_p.shape[1] < self._n_classes:
                pad = np.zeros((ridge_p.shape[0], self._n_classes - ridge_p.shape[1]))
                ridge_p = np.column_stack([ridge_p, pad])
            meta_X = np.hstack([ridge_p, lgb_p])
        else:
            meta_X = lgb_p
        return self._meta.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------
    # Introspection — gain per base model (avg abs coef of meta)
    # ------------------------------------------------------------

    def base_weights(self) -> Dict[str, float]:
        """Rough proxy for how much each base contributes to final prediction."""
        if self._meta is None: return {}
        coefs = np.abs(self._meta.coef_)  # [n_classes, 2*n_classes]
        n = self._n_classes
        ridge_coefs = coefs[:, :n]
        lgb_coefs = coefs[:, n:]
        total = ridge_coefs.sum() + lgb_coefs.sum() + 1e-9
        return {
            "ridge": float(ridge_coefs.sum() / total),
            "lightgbm": float(lgb_coefs.sum() / total),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        # Binary case — convert to sigmoid proba of positive class
        return 1.0 / (1.0 + np.exp(-x))
    x_shift = x - x.max(axis=1, keepdims=True)
    e = np.exp(x_shift)
    return e / e.sum(axis=1, keepdims=True)
