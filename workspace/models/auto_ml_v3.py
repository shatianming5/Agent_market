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
    return arr.astype(np.float32, copy=False)


def _as_1d_float_array(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def _finite_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return np.zeros((0,), dtype=bool)
    if a.size != b.size:
        raise ValueError("metric inputs must have same length")
    return np.isfinite(a) & np.isfinite(b)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64, copy=False)
    b = np.asarray(b).reshape(-1).astype(np.float64, copy=False)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    m = _finite_pair_mask(a, b)
    if m.sum() < 3:
        return 0.0
    aa = a[m]
    bb = b[m]
    a_std = float(np.std(aa))
    b_std = float(np.std(bb))
    if a_std == 0.0 or b_std == 0.0 or (not np.isfinite(a_std)) or (not np.isfinite(b_std)):
        return 0.0
    c = float(np.corrcoef(aa, bb)[0, 1])
    return 0.0 if not np.isfinite(c) else c


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64, copy=False)
    b = np.asarray(b).reshape(-1).astype(np.float64, copy=False)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    m = _finite_pair_mask(a, b)
    if m.sum() == 0:
        return 0.0
    d = a[m] - b[m]
    return float(np.mean(d * d))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64, copy=False)
    b = np.asarray(b).reshape(-1).astype(np.float64, copy=False)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    m = _finite_pair_mask(a, b)
    if m.sum() == 0:
        return 0.0
    d = a[m] - b[m]
    return float(np.mean(np.abs(d)))


def _dir_acc(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1).astype(np.float64, copy=False)
    y = np.asarray(y).reshape(-1).astype(np.float64, copy=False)
    if pred.size == 0 or y.size == 0 or pred.size != y.size:
        return 0.0
    m = _finite_pair_mask(pred, y)
    if m.sum() == 0:
        return 0.0
    pp = pred[m]
    yy = y[m]
    return float(np.mean((np.sign(pp) == np.sign(yy)).astype(np.float64)))


class AutoML_v3(BaseModelAdapter):
    registry_name = "auto_ml_v3"

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

        enet_alpha = float(self.config.get("enet_alpha", 5e-4))
        enet_l1_ratio = float(self.config.get("enet_l1_ratio", 0.15))

        rf_n_estimators = int(self.config.get("rf_n_estimators", 500))
        rf_max_depth = self.config.get("rf_max_depth", None)
        rf_min_samples_leaf = int(self.config.get("rf_min_samples_leaf", 8))
        rf_max_features = self.config.get("rf_max_features", "sqrt")

        et_n_estimators = int(self.config.get("et_n_estimators", 700))
        et_max_depth = self.config.get("et_max_depth", None)
        et_min_samples_leaf = int(self.config.get("et_min_samples_leaf", 6))
        et_max_features = self.config.get("et_max_features", "sqrt")

        hgb_learning_rate = float(self.config.get("hgb_learning_rate", 0.03))
        hgb_max_depth = int(self.config.get("hgb_max_depth", 3))
        hgb_max_leaf_nodes = int(self.config.get("hgb_max_leaf_nodes", 31))
        hgb_l2_regularization = float(self.config.get("hgb_l2_regularization", 0.0))
        hgb_min_samples_leaf = int(self.config.get("hgb_min_samples_leaf", 30))
        hgb_early_stopping = bool(self.config.get("hgb_early_stopping", True))
        hgb_validation_fraction = float(self.config.get("hgb_validation_fraction", 0.1))
        hgb_n_iter_no_change = int(self.config.get("hgb_n_iter_no_change", 25))
        hgb_loss = str(self.config.get("hgb_loss", "absolute_error")).lower()

        weights = self.config.get("ensemble_weights", None)

        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        lin = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))),
                (
                    "enet",
                    ElasticNet(
                        alpha=enet_alpha,
                        l1_ratio=enet_l1_ratio,
                        fit_intercept=True,
                        max_iter=40000,
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
                        max_features=rf_max_features,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        et = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=et_n_estimators,
                        max_depth=et_max_depth,
                        min_samples_leaf=et_min_samples_leaf,
                        max_features=et_max_features,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        if hgb_loss not in ("squared_error", "absolute_error", "poisson", "gamma"):
            hgb_loss = "absolute_error"

        hgb = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "hgb",
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
                        random_state=random_state,
                    ),
                ),
            ]
        )

        estimators = [("lin", lin), ("rf", rf), ("et", et), ("hgb", hgb)]
        if weights is None:
            weights = [0.15, 0.25, 0.25, 0.35]
        else:
            if not isinstance(weights, (list, tuple)) or len(weights) != 4:
                raise ValueError('config["ensemble_weights"] must be a list/tuple of length 4: [lin, rf, et, hgb]')
            weights = [float(w) for w in weights]

        model = VotingRegressor(estimators=estimators, weights=weights)
        model.fit(X_tr, y_tr)
        self.model_ = model

        metrics: Dict[str, float] = {}

        pred_tr = np.asarray(self.model_.predict(X_tr), dtype=np.float64).reshape(-1)
        y_tr64 = y_tr.astype(np.float64, copy=False)
        metrics["train_mse"] = _mse(pred_tr, y_tr64)
        metrics["train_mae"] = _mae(pred_tr, y_tr64)
        metrics["train_rmse"] = float(np.sqrt(metrics["train_mse"])) if metrics["train_mse"] >= 0.0 else 0.0
        metrics["train_corr"] = _safe_corrcoef(pred_tr, y_tr64)
        metrics["train_dir_acc"] = _dir_acc(pred_tr, y_tr64)

        if X_valid is not None and y_valid is not None:
            X_va = _as_2d_float_array(X_valid)
            y_va = _as_1d_float_array(y_valid)

            if X_va.shape[1] != self.n_features_in_:
                raise ValueError(f"X_valid has different feature count: {X_va.shape[1]} vs expected {self.n_features_in_}")
            if X_va.shape[0] != y_va.shape[0]:
                raise ValueError(f"X_valid and y_valid length mismatch: {X_va.shape[0]} vs {y_va.shape[0]}")

            pred_va = np.asarray(self.model_.predict(X_va), dtype=np.float64).reshape(-1)
            y_va64 = y_va.astype(np.float64, copy=False)
            metrics["valid_mse"] = _mse(pred_va, y_va64)
            metrics["valid_mae"] = _mae(pred_va, y_va64)
            metrics["valid_rmse"] = float(np.sqrt(metrics["valid_mse"])) if metrics["valid_mse"] >= 0.0 else 0.0
            metrics["valid_corr"] = _safe_corrcoef(pred_va, y_va64)
            metrics["valid_dir_acc"] = _dir_acc(pred_va, y_va64)

        model_path = model_dir / "auto_ml_v3.pkl"
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