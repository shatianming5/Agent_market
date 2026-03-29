from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


class AutoML_v2(BaseModelAdapter):
    registry_name = "auto_ml_v2"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None

        self._models: Optional[list[Any]] = None
        self._model_names: Optional[list[str]] = None
        self._model_weights: Optional[np.ndarray] = None

        self._clip_pred: Optional[float] = self.config.get("clip_pred", None)
        self._model_path: Optional[Path] = None

    @staticmethod
    def _as_float32_2d(x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    @staticmethod
    def _as_float32_1d(y: Any) -> np.ndarray:
        arr = np.asarray(y).reshape(-1)
        arr = arr.astype(np.float32, copy=False)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return 0.0
        e = yp - yt
        return float(np.sqrt(np.mean(e * e)))

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return 0.0
        return float(np.mean(np.abs(yp - yt)))

    @staticmethod
    def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return 0.0

        m = np.isfinite(yt) & np.isfinite(yp)
        if int(m.sum()) < 3:
            return 0.0

        a = yt[m]
        b = yp[m]
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 0.0 or (not np.isfinite(denom)):
            return 0.0
        c = float(np.sum(a * b) / denom)
        return 0.0 if not np.isfinite(c) else c

    @staticmethod
    def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
        yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return 0.0
        m = np.isfinite(yt) & np.isfinite(yp)
        if int(m.sum()) == 0:
            return 0.0
        return float(np.mean((np.sign(yt[m]) == np.sign(yp[m])).astype(np.float64)))

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
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        xtr = self._as_float32_2d(X_train)
        ytr = self._as_float32_1d(y_train)
        if xtr.shape[0] != ytr.shape[0]:
            raise ValueError(f"X_train and y_train length mismatch: {xtr.shape[0]} vs {ytr.shape[0]}")
        self._n_features = int(xtr.shape[1])

        xva: Optional[np.ndarray] = None
        yva: Optional[np.ndarray] = None
        if X_valid is not None and y_valid is not None:
            xva = self._as_float32_2d(X_valid)
            yva = self._as_float32_1d(y_valid)
            if xva.shape[1] != self._n_features:
                raise ValueError(f"X_valid has {xva.shape[1]} features, expected {self._n_features}")
            if xva.shape[0] != yva.shape[0]:
                raise ValueError(f"X_valid and y_valid length mismatch: {xva.shape[0]} vs {yva.shape[0]}")

        model_dir_val = self.config.get("model_dir")
        if model_dir_val is None:
            raise ValueError('config["model_dir"] is required')
        model_dir = Path(model_dir_val)
        model_dir.mkdir(parents=True, exist_ok=True)

        rs = int(self.config.get("random_state", 42))
        n_jobs = int(self.config.get("n_jobs", -1))

        enet_alpha = float(self.config.get("enet_alpha", 5e-4))
        enet_l1_ratio = float(self.config.get("enet_l1_ratio", 0.15))
        enet_max_iter = int(self.config.get("enet_max_iter", 40000))

        et_n_estimators = int(self.config.get("et_n_estimators", 800))
        et_min_samples_leaf = int(self.config.get("et_min_samples_leaf", 10))
        et_max_features = self.config.get("et_max_features", "sqrt")

        hgb_learning_rate = float(self.config.get("hgb_learning_rate", 0.03))
        hgb_max_depth = int(self.config.get("hgb_max_depth", 3))
        hgb_max_leaf_nodes = int(self.config.get("hgb_max_leaf_nodes", 31))
        hgb_min_samples_leaf = int(self.config.get("hgb_min_samples_leaf", 30))
        hgb_l2_regularization = float(self.config.get("hgb_l2_regularization", 0.0))
        hgb_n_iter_no_change = int(self.config.get("hgb_n_iter_no_change", 25))
        hgb_validation_fraction = float(self.config.get("hgb_validation_fraction", 0.1))
        hgb_early_stopping = bool(self.config.get("hgb_early_stopping", True))
        hgb_loss = str(self.config.get("hgb_loss", "absolute_error")).lower()
        if hgb_loss not in ("squared_error", "absolute_error", "poisson", "gamma"):
            hgb_loss = "absolute_error"

        lin = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))),
                (
                    "model",
                    ElasticNet(
                        alpha=enet_alpha,
                        l1_ratio=enet_l1_ratio,
                        fit_intercept=True,
                        max_iter=enet_max_iter,
                        random_state=rs,
                    ),
                ),
            ]
        )

        et = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=et_n_estimators,
                        max_depth=None,
                        min_samples_leaf=et_min_samples_leaf,
                        min_samples_split=max(2, 2 * et_min_samples_leaf),
                        max_features=et_max_features,
                        n_jobs=n_jobs,
                        random_state=rs,
                        bootstrap=True,
                    ),
                ),
            ]
        )

        hgb = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss=hgb_loss,
                        learning_rate=hgb_learning_rate,
                        max_depth=hgb_max_depth,
                        max_leaf_nodes=hgb_max_leaf_nodes,
                        min_samples_leaf=hgb_min_samples_leaf,
                        l2_regularization=hgb_l2_regularization,
                        early_stopping=hgb_early_stopping,
                        validation_fraction=hgb_validation_fraction,
                        n_iter_no_change=hgb_n_iter_no_change,
                        random_state=rs,
                    ),
                ),
            ]
        )

        models: list[Any] = [lin, et, hgb]
        names: list[str] = ["elasticnet", "extratrees", "histgbr"]

        for m in models:
            m.fit(xtr, ytr)

        def _pred_ensemble(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
            preds = []
            for mm in models:
                p = np.asarray(mm.predict(x), dtype=np.float32).reshape(-1)
                np.nan_to_num(p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                preds.append(p)
            P = np.stack(preds, axis=1)  # (n, k)
            yhat = (P @ weights.astype(np.float32)).astype(np.float32)
            return yhat.reshape(-1)

        k = len(models)
        if xva is not None and yva is not None:
            scores = []
            for m in models:
                pv = np.asarray(m.predict(xva), dtype=np.float32).reshape(-1)
                np.nan_to_num(pv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                rmse = self._rmse(yva, pv)
                corr = self._corr(yva, pv)
                score = (max(0.0, corr) + 1e-6) / (rmse + 1e-6)
                scores.append(score)
            weights = self._normalize_weights(np.asarray(scores, dtype=np.float64))
        else:
            weights = (np.ones((k,), dtype=np.float32) / float(k)).astype(np.float32)

        self._models = models
        self._model_names = names
        self._model_weights = weights
        self._is_fitted = True

        yhat_tr = _pred_ensemble(xtr, weights)
        metrics: Dict[str, float] = {
            "train_rmse": self._rmse(ytr, yhat_tr),
            "train_mae": self._mae(ytr, yhat_tr),
            "train_corr": self._corr(ytr, yhat_tr),
            "train_dir_acc": self._dir_acc(ytr, yhat_tr),
            "n_models": float(k),
            "n_features": float(self._n_features),
        }

        if xva is not None and yva is not None:
            yhat_va = _pred_ensemble(xva, weights)
            metrics["valid_rmse"] = self._rmse(yva, yhat_va)
            metrics["valid_mae"] = self._mae(yva, yhat_va)
            metrics["valid_corr"] = self._corr(yva, yhat_va)
            metrics["valid_dir_acc"] = self._dir_acc(yva, yhat_va)

        model_path = model_dir / "auto_ml_v2.pkl"
        self.save(model_path)
        self._model_path = model_path

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if (not self._is_fitted) or (self._models is None) or (self._model_weights is None):
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        x = self._as_float32_2d(X)
        if self._n_features is not None and x.shape[1] != self._n_features:
            raise ValueError(f"Feature dimension mismatch: got {x.shape[1]}, expected {self._n_features}")
        if x.size == 0:
            return np.asarray([], dtype=np.float32)

        preds = []
        for m in self._models:
            p = np.asarray(m.predict(x), dtype=np.float32).reshape(-1)
            np.nan_to_num(p, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            preds.append(p)

        P = np.stack(preds, axis=1)  # (n, k)
        w = self._model_weights.astype(np.float32).reshape(-1)
        yhat = (P @ w).astype(np.float32).reshape(-1)

        if self._clip_pred is not None:
            c = float(self._clip_pred)
            yhat = np.clip(yhat, -c, c).astype(np.float32)

        if yhat.shape[0] != x.shape[0]:
            raise RuntimeError("predict() returned wrong length")
        return yhat

    def save(self, path: Path) -> None:
        if self._models is None or self._model_weights is None:
            raise RuntimeError("Nothing to save: model is not fitted/loaded")
        p = Path(path)
        if p.exists() and p.is_dir():
            p = p / "auto_ml_v2.pkl"
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
            p = p / "auto_ml_v2.pkl"

        with p.open("rb") as f:
            state = pickle.load(f)

        self.config = state.get("config", self.config)
        self._is_fitted = bool(state.get("_is_fitted", False))
        self._n_features = state.get("_n_features", None)
        self._model_names = state.get("_model_names", None)
        self._model_weights = state.get("_model_weights", None)
        self._models = state.get("_models", None)
        self._clip_pred = state.get("_clip_pred", self.config.get("clip_pred", None))
        self._model_path = p