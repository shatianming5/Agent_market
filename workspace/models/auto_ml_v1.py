from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


class AutoML_v1(BaseModelAdapter):
    registry_name = "auto_ml_v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None

        self._models: Optional[list[Any]] = None
        self._model_names: Optional[list[str]] = None
        self._model_weights: Optional[np.ndarray] = None

        self._clip_pred: Optional[float] = self.config.get("clip_pred", None)

    @staticmethod
    def _as_float32_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={x.shape}")
        x = x.astype(np.float32, copy=False)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    @staticmethod
    def _as_float32_1d(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        y = y.astype(np.float32, copy=False)
        np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return y

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = y_true.astype(np.float64, copy=False)
        yp = y_pred.astype(np.float64, copy=False)
        e = yp - yt
        return float(np.sqrt(np.mean(e * e)))

    @staticmethod
    def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = y_true.astype(np.float64, copy=False)
        yp = y_pred.astype(np.float64, copy=False)
        yt = yt - float(np.mean(yt))
        yp = yp - float(np.mean(yp))
        denom = float(np.sqrt(np.sum(yt * yt) * np.sum(yp * yp)))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(yt * yp) / denom)

    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = np.where(np.isfinite(w), w, 0.0)
        s = float(np.sum(w))
        if s <= 0.0:
            return (np.ones_like(w) / float(len(w))).astype(np.float32)
        return (w / s).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        xtr = self._as_float32_2d(X_train)
        ytr = self._as_float32_1d(y_train)
        if xtr.shape[0] != ytr.shape[0]:
            raise ValueError(f"X_train and y_train length mismatch: {xtr.shape[0]} vs {ytr.shape[0]}")

        n_samples, n_features = xtr.shape
        self._n_features = int(n_features)

        xva: Optional[np.ndarray] = None
        yva: Optional[np.ndarray] = None
        if X_valid is not None and y_valid is not None:
            xva = self._as_float32_2d(X_valid)
            yva = self._as_float32_1d(y_valid)
            if xva.shape[0] != yva.shape[0] or xva.shape[1] != n_features:
                raise ValueError(f"Invalid validation shapes: X_valid={xva.shape}, y_valid={yva.shape}")

        rs = int(self.config.get("random_state", 42))

        enet = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", ElasticNet(alpha=float(self.config.get("enet_alpha", 1e-3)),
                                    l1_ratio=float(self.config.get("enet_l1_ratio", 0.15)),
                                    fit_intercept=True,
                                    max_iter=int(self.config.get("enet_max_iter", 5000)),
                                    random_state=rs)),
            ]
        )

        gbr = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(
                    random_state=rs,
                    n_estimators=int(self.config.get("gbr_n_estimators", 300)),
                    learning_rate=float(self.config.get("gbr_learning_rate", 0.03)),
                    max_depth=int(self.config.get("gbr_max_depth", 3)),
                    subsample=float(self.config.get("gbr_subsample", 0.8)),
                    min_samples_leaf=int(self.config.get("gbr_min_samples_leaf", 5)),
                )),
            ]
        )

        rfr = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    random_state=rs,
                    n_estimators=int(self.config.get("rf_n_estimators", 400)),
                    max_depth=self.config.get("rf_max_depth", None),
                    min_samples_leaf=int(self.config.get("rf_min_samples_leaf", 2)),
                    min_samples_split=int(self.config.get("rf_min_samples_split", 4)),
                    max_features=self.config.get("rf_max_features", "sqrt"),
                    n_jobs=int(self.config.get("n_jobs", -1)),
                    bootstrap=True,
                )),
            ]
        )

        models: list[Any] = [enet, gbr, rfr]
        names: list[str] = ["elasticnet", "gbr", "rf"]

        for m in models:
            m.fit(xtr, ytr)

        self._models = models
        self._model_names = names

        def ensemble_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
            preds = []
            for m in models:
                p = np.asarray(m.predict(x), dtype=np.float32).reshape(-1)
                preds.append(p)
            P = np.stack(preds, axis=1)  # (n, k)
            return (P @ weights.astype(np.float32)).astype(np.float32)

        k = len(models)
        if xva is not None and yva is not None:
            rmse_scores = []
            corr_scores = []
            for m in models:
                p = np.asarray(m.predict(xva), dtype=np.float32).reshape(-1)
                rmse_scores.append(self._rmse(yva, p))
                corr_scores.append(max(0.0, self._corr(yva, p)))
            inv_rmse = np.array([1.0 / (r + 1e-8) for r in rmse_scores], dtype=np.float64)
            corr_arr = np.array(corr_scores, dtype=np.float64)
            w = (0.65 * inv_rmse) + (0.35 * corr_arr)
            weights = self._normalize_weights(w)
        else:
            weights = (np.ones((k,), dtype=np.float32) / float(k)).astype(np.float32)

        self._model_weights = weights
        self._is_fitted = True

        yhat_tr = ensemble_predict(xtr, weights)
        metrics: Dict[str, float] = {
            "train_rmse": self._rmse(ytr, yhat_tr),
            "train_corr": self._corr(ytr, yhat_tr),
            "n_models": float(k),
            "n_features": float(n_features),
        }

        if xva is not None and yva is not None:
            yhat_va = ensemble_predict(xva, weights)
            metrics["valid_rmse"] = self._rmse(yva, yhat_va)
            metrics["valid_corr"] = self._corr(yva, yhat_va)

        model_dir = Path(self.config.get("model_dir", "."))
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "auto_ml_v1.pkl"
        self.save(model_path)

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._models is None or self._model_weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        x = self._as_float32_2d(X)
        if self._n_features is not None and x.shape[1] != self._n_features:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[1]}, expected {self._n_features}")

        preds = []
        for m in self._models:
            p = np.asarray(m.predict(x), dtype=np.float32).reshape(-1)
            preds.append(p)
        P = np.stack(preds, axis=1)  # (n, k)
        yhat = (P @ self._model_weights.reshape(-1, 1)).reshape(-1).astype(np.float32)

        if self._clip_pred is not None:
            c = float(self._clip_pred)
            yhat = np.clip(yhat, -c, c).astype(np.float32)

        return np.asarray(yhat).reshape(-1)

    def save(self, path: Path) -> None:
        p = Path(path)
        if p.exists() and p.is_dir():
            p = p / "auto_ml_v1.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "registry_name": self.registry_name,
            "config": self.config,
            "_is_fitted": self._is_fitted,
            "_n_features": self._n_features,
            "_model_names": self._model_names,
            "_model_weights": self._model_weights,
            "_models": self._models,
            "_clip_pred": self._clip_pred,
        }

        with p.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        p = Path(path)
        if p.exists() and p.is_dir():
            p = p / "auto_ml_v1.pkl"

        with p.open("rb") as f:
            state = pickle.load(f)

        self.config = state.get("config", self.config)
        self._is_fitted = bool(state.get("_is_fitted", False))
        self._n_features = state.get("_n_features", None)
        self._model_names = state.get("_model_names", None)
        self._model_weights = state.get("_model_weights", None)
        self._models = state.get("_models", None)
        self._clip_pred = state.get("_clip_pred", self.config.get("clip_pred", None))