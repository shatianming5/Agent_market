from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


class AutoML_v1(BaseModelAdapter):
    registry_name = "auto_ml_v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            super().__init__(config)
        except Exception:
            self.config = config or {}

        cfg = getattr(self, "config", None) or {}
        self._cfg: Dict[str, Any] = dict(cfg)

        self.n_models: int = int(self._cfg.get("n_models", 5))
        self.bootstrap: bool = bool(self._cfg.get("bootstrap", True))
        self.random_state: int = int(self._cfg.get("random_state", 42))

        alpha_grid = self._cfg.get("alpha_grid", None)
        if alpha_grid is None:
            alpha_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        self.alpha_grid = (
            [float(a) for a in alpha_grid]
            if isinstance(alpha_grid, (list, tuple))
            else [float(alpha_grid)]
        )

        self.clip_pred: Optional[float] = self._cfg.get("clip_pred", None)
        self._is_fitted: bool = False

        self._n_features: Optional[int] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_scale: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None  # shape: (n_models, n_features + 1)

    @staticmethod
    def _nan_to_num_inplace(x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        x = x.astype(np.float32, copy=False)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    @staticmethod
    def _design_matrix(x_std: np.ndarray) -> np.ndarray:
        n = x_std.shape[0]
        bias = np.ones((n, 1), dtype=np.float32)
        return np.concatenate([bias, x_std], axis=1)

    @staticmethod
    def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        xtx = x.T @ x
        xty = x.T @ y
        reg = np.eye(xtx.shape[0], dtype=np.float32) * np.float32(alpha)
        reg[0, 0] = np.float32(0.0)  # do not regularize bias
        a = xtx + reg
        try:
            w = np.linalg.solve(a.astype(np.float64), xty.astype(np.float64)).astype(
                np.float32
            )
        except Exception:
            w = np.linalg.lstsq(
                a.astype(np.float64), xty.astype(np.float64), rcond=None
            )[0].astype(np.float32)
        return w

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.float64, copy=False)
        y_pred = y_pred.astype(np.float64, copy=False)
        e = y_pred - y_true
        return float(np.sqrt(np.mean(e * e)))

    @staticmethod
    def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.float64, copy=False)
        y_pred = y_pred.astype(np.float64, copy=False)
        yt = y_true - np.mean(y_true)
        yp = y_pred - np.mean(y_pred)
        denom = np.sqrt(np.sum(yt * yt) * np.sum(yp * yp))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(yt * yp) / denom)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        x = self._nan_to_num_inplace(np.asarray(X_train))
        y = self._nan_to_num_inplace(np.asarray(y_train)).reshape(-1)

        if x.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape={x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y_train must be 1D, got shape={y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_train and y_train length mismatch: {x.shape[0]} vs {y.shape[0]}"
            )

        n, d = x.shape
        self._n_features = int(d)

        x_mean = np.mean(x, axis=0).astype(np.float32)
        x_std = np.std(x, axis=0).astype(np.float32)
        x_std = np.where(x_std > np.float32(1e-8), x_std, np.float32(1.0)).astype(
            np.float32
        )

        x_stdzd = ((x - x_mean) / x_std).astype(np.float32)
        x_design = self._design_matrix(x_stdzd)

        rng = np.random.default_rng(self.random_state)
        weights = np.zeros((self.n_models, d + 1), dtype=np.float32)

        for m in range(self.n_models):
            alpha = self.alpha_grid[m % len(self.alpha_grid)]
            alpha_eff = float(alpha) * float(d)

            if self.bootstrap:
                idx = rng.integers(0, n, size=n, endpoint=False)
                xb = x_design[idx]
                yb = y[idx]
            else:
                xb = x_design
                yb = y

            weights[m] = self._ridge_solve(xb, yb, alpha_eff)

        self._x_mean = x_mean
        self._x_scale = x_std
        self._weights = weights
        self._is_fitted = True

        yhat_train = self.predict(x)
        metrics: Dict[str, float] = {
            "train_rmse": self._rmse(y, yhat_train),
            "train_corr": self._corr(y, yhat_train),
            "n_models": float(self.n_models),
            "n_features": float(d),
        }

        if X_valid is not None and y_valid is not None:
            xv = self._nan_to_num_inplace(np.asarray(X_valid))
            yv = self._nan_to_num_inplace(np.asarray(y_valid)).reshape(-1)
            if xv.ndim != 2 or yv.ndim != 1 or xv.shape[0] != yv.shape[0]:
                raise ValueError(
                    f"Invalid validation shapes: X_valid={xv.shape}, y_valid={yv.shape}"
                )
            yhat_valid = self.predict(xv)
            metrics["valid_rmse"] = self._rmse(yv, yhat_valid)
            metrics["valid_corr"] = self._corr(yv, yhat_valid)

        model_dir = Path((getattr(self, "config", None) or {}).get("model_dir", "."))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "auto_ml_v1.pkl"
        self.save(model_path)

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if (
            not self._is_fitted
            or self._weights is None
            or self._x_mean is None
            or self._x_scale is None
        ):
            raise RuntimeError("Model is not fitted. Call fit() first.")

        x = self._nan_to_num_inplace(np.asarray(X))
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        if self._n_features is not None and x.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: got {x.shape[1]}, expected {self._n_features}"
            )

        x_stdzd = ((x - self._x_mean) / self._x_scale).astype(np.float32)
        x_design = self._design_matrix(x_stdzd)

        preds = x_design @ self._weights.T  # (n, n_models)
        yhat = np.mean(preds, axis=1).astype(np.float32)

        if self.clip_pred is not None:
            c = float(self.clip_pred)
            yhat = np.clip(yhat, -c, c).astype(np.float32)

        return np.asarray(yhat).reshape(-1)

    def save(self, path: Any) -> None:
        p = Path(path)
        if p.exists() and p.is_dir():
            p = p / "auto_ml_v1.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "registry_name": self.registry_name,
            "config": getattr(self, "config", None),
            "n_models": self.n_models,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "alpha_grid": self.alpha_grid,
            "clip_pred": self.clip_pred,
            "_is_fitted": self._is_fitted,
            "_n_features": self._n_features,
            "_x_mean": self._x_mean,
            "_x_scale": self._x_scale,
            "_weights": self._weights,
        }

        with p.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Any) -> None:
        """Load model state from disk into this instance.

        Note: BaseModelAdapter.load is an instance method. Older versions of this
        adapter implemented load() as a @classmethod returning a new instance,
        which breaks callers that expect in-place mutation.
        """
        p = Path(path)
        if p.exists() and p.is_dir():
            p = p / "auto_ml_v1.pkl"

        with p.open("rb") as f:
            state = pickle.load(f)

        self.n_models = int(state.get("n_models", self.n_models))
        self.bootstrap = bool(state.get("bootstrap", self.bootstrap))
        self.random_state = int(state.get("random_state", self.random_state))
        self.alpha_grid = list(state.get("alpha_grid", self.alpha_grid))
        self.clip_pred = state.get("clip_pred", self.clip_pred)

        self._is_fitted = bool(state.get("_is_fitted", False))
        self._n_features = state.get("_n_features", None)
        self._x_mean = state.get("_x_mean", None)
        self._x_scale = state.get("_x_scale", None)
        self._weights = state.get("_weights", None)

    @classmethod
    def load_from_file(cls, path: Any) -> "AutoML_v1":
        """Backward-compatible loader returning a new instance."""
        model = cls(config={})
        model.load(path)
        return model
