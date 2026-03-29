from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


def _as_2d_float_array(X: Any) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D array-like, got shape={arr.shape}")
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _as_1d_float_array(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        raise ValueError("corrcoef inputs must have same length")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if not np.isfinite(a_std) or not np.isfinite(b_std) or a_std == 0.0 or b_std == 0.0:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


class AutoML_v1(BaseModelAdapter):
    registry_name = "auto_ml_v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_: Any = None
        self.model_path_: Optional[Path] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        X_tr = _as_2d_float_array(X_train)
        y_tr = _as_1d_float_array(y_train)

        if X_tr.shape[0] != y_tr.shape[0]:
            raise ValueError(f"X_train and y_train length mismatch: {X_tr.shape[0]} vs {y_tr.shape[0]}")
        self.n_features_in_ = int(X_tr.shape[1])

        model_dir = self.config.get("model_dir")
        if model_dir is None:
            raise ValueError('config["model_dir"] is required')
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        random_state = int(self.config.get("random_state", 42))
        enet_alpha = float(self.config.get("enet_alpha", 1e-3))
        enet_l1_ratio = float(self.config.get("enet_l1_ratio", 0.1))
        rf_n_estimators = int(self.config.get("rf_n_estimators", 400))
        rf_max_depth = self.config.get("rf_max_depth", None)
        rf_min_samples_leaf = int(self.config.get("rf_min_samples_leaf", 5))
        hgb_max_depth = int(self.config.get("hgb_max_depth", 3))
        hgb_learning_rate = float(self.config.get("hgb_learning_rate", 0.05))
        hgb_max_leaf_nodes = int(self.config.get("hgb_max_leaf_nodes", 31))
        weights = self.config.get("ensemble_weights", None)

        # Lazy sklearn imports (import block is fixed by platform requirement)
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        lin = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "enet",
                    ElasticNet(
                        alpha=enet_alpha,
                        l1_ratio=enet_l1_ratio,
                        fit_intercept=True,
                        max_iter=20000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        rf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        min_samples_leaf=rf_min_samples_leaf,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        hgb = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=hgb_learning_rate,
                        max_depth=hgb_max_depth,
                        max_leaf_nodes=hgb_max_leaf_nodes,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        estimators = [("lin", lin), ("rf", rf), ("hgb", hgb)]
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        else:
            if not isinstance(weights, (list, tuple)) or len(weights) != 3:
                raise ValueError('config["ensemble_weights"] must be a list/tuple of length 3: [lin, rf, hgb]')
            weights = [float(w) for w in weights]

        model = VotingRegressor(estimators=estimators, weights=weights)
        model.fit(X_tr, y_tr)
        self.model_ = model

        metrics: Dict[str, float] = {}

        yhat_tr = np.asarray(self.model_.predict(X_tr), dtype=np.float64).reshape(-1)
        resid_tr = yhat_tr - y_tr.astype(np.float64, copy=False)
        metrics["train_mse"] = float(np.mean(resid_tr**2)) if y_tr.size else 0.0
        metrics["train_mae"] = float(np.mean(np.abs(resid_tr))) if y_tr.size else 0.0
        metrics["train_corr"] = _safe_corrcoef(yhat_tr, y_tr)
        metrics["train_dir_acc"] = (
            float(np.mean((np.sign(yhat_tr) == np.sign(y_tr.astype(np.float64, copy=False))).astype(np.float64)))
            if y_tr.size
            else 0.0
        )

        if X_valid is not None and y_valid is not None:
            X_va = _as_2d_float_array(X_valid)
            y_va = _as_1d_float_array(y_valid)
            if X_va.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X_valid has different feature count: {X_va.shape[1]} vs expected {self.n_features_in_}"
                )
            if X_va.shape[0] != y_va.shape[0]:
                raise ValueError(f"X_valid and y_valid length mismatch: {X_va.shape[0]} vs {y_va.shape[0]}")

            yhat_va = np.asarray(self.model_.predict(X_va), dtype=np.float64).reshape(-1)
            resid_va = yhat_va - y_va.astype(np.float64, copy=False)
            metrics["valid_mse"] = float(np.mean(resid_va**2)) if y_va.size else 0.0
            metrics["valid_mae"] = float(np.mean(np.abs(resid_va))) if y_va.size else 0.0
            metrics["valid_corr"] = _safe_corrcoef(yhat_va, y_va)
            metrics["valid_dir_acc"] = (
                float(np.mean((np.sign(yhat_va) == np.sign(y_va.astype(np.float64, copy=False))).astype(np.float64)))
                if y_va.size
                else 0.0
            )

        model_path = model_dir / "auto_ml_v1.pkl"
        self.save(model_path)
        self.model_path_ = model_path

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not loaded/fitted yet")
        X_in = _as_2d_float_array(X)
        if X_in.size == 0:
            return np.asarray([], dtype=np.float32)

        if self.n_features_in_ is not None and X_in.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_in.shape[1]} features, expected {self.n_features_in_}")

        yhat = self.model_.predict(X_in)
        yhat = np.asarray(yhat).reshape(-1)
        if yhat.shape[0] != X_in.shape[0]:
            raise RuntimeError("predict() returned wrong length")
        return yhat.astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Nothing to save: model is not fitted/loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "registry_name": self.registry_name,
            "model": self.model_,
            "n_features_in": self.n_features_in_,
        }

        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        self.model_ = payload.get("model")
        self.n_features_in_ = payload.get("n_features_in", None)
        self.model_path_ = path