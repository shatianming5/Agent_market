from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from agent_market.freqai.model.base import BaseModelAdapter, ModelRegistry, TrainResult


def _rmse(preds: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - target) ** 2)))


class LightGBMAdapter(BaseModelAdapter):
    registry_name = 'lightgbm'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        import lightgbm as lgb  # type: ignore

        params = dict(self.config)
        model_dir = Path(params.pop('model_dir', 'artifacts/models/lightgbm'))
        num_boost_round = int(params.pop('num_boost_round', 200))
        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        if X_valid is not None and X_valid.size:
            dvalid = lgb.Dataset(X_valid, label=y_valid)
            valid_sets.append(dvalid)
        # progress callback for Flow percent estimation
        def _progress_cb(env):
            try:
                cur = int(getattr(env, 'iteration', 0)) + 1
                tot = max(1, int(num_boost_round))
                if cur == 1 or cur == tot or (tot >= 10 and cur % max(1, tot // 10) == 0):
                    pct = int(cur * 100 / tot)
                    print(f"[FLOW] EPOCH {cur}/{tot} PROGRESS {pct}%")
            except Exception:
                pass
        booster = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=[_progress_cb])
        self.model = booster
        metrics = {'rmse_train': _rmse(booster.predict(X_train), y_train)}
        if X_valid is not None and X_valid.size:
            metrics['rmse_valid'] = _rmse(booster.predict(X_valid), y_valid)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'lightgbm_model.txt'
        booster.save_model(str(model_path))
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('LightGBM model not trained')
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError('LightGBM model not trained')
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        import lightgbm as lgb  # type: ignore

        booster = lgb.Booster(model_file=str(path))
        self.model = booster


class XGBoostAdapter(BaseModelAdapter):
    registry_name = 'xgboost'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        import xgboost as xgb  # type: ignore

        params = dict(self.config)
        model_dir = Path(params.pop('model_dir', 'artifacts/models/xgboost'))
        num_boost_round = int(params.pop('num_boost_round', 300))
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, 'train')]
        dvalid = None
        if X_valid is not None and X_valid.size:
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            evals.append((dvalid, 'valid'))
        # progress callback for Flow percent estimation
        try:
            from xgboost.callback import TrainingCallback  # type: ignore
            class _ProgressCB(TrainingCallback):
                def __init__(self, tot: int):
                    self.tot = max(1, int(tot))
                def after_iteration(self, model, epoch: int, evals_log=None):  # noqa: D401
                    cur = epoch + 1
                    if cur == 1 or cur == self.tot or (self.tot >= 10 and cur % max(1, self.tot // 10) == 0):
                        pct = int(cur * 100 / self.tot)
                        print(f"[FLOW] EPOCH {cur}/{self.tot} PROGRESS {pct}%")
                    return False
            callbacks = [_ProgressCB(num_boost_round)]
        except Exception:
            callbacks = []
        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=False, callbacks=callbacks)
        self.model = booster
        metrics = {'rmse_train': _rmse(booster.predict(dtrain), y_train)}
        if dvalid is not None:
            metrics['rmse_valid'] = _rmse(booster.predict(dvalid), y_valid)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'xgboost_model.json'
        booster.save_model(str(model_path))
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('XGBoost model not trained')
        import xgboost as xgb  # type: ignore

        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError('XGBoost model not trained')
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        import xgboost as xgb  # type: ignore

        booster = xgb.Booster()
        booster.load_model(str(path))
        self.model = booster


class CatBoostAdapter(BaseModelAdapter):
    registry_name = 'catboost'

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        from catboost import CatBoostRegressor  # type: ignore

        params = dict(self.config)
        model_dir = Path(params.pop('model_dir', 'artifacts/models/catboost'))
        iterations = int(params.pop('iterations', 500))
        params.setdefault('depth', 6)
        params.setdefault('learning_rate', 0.03)
        model = CatBoostRegressor(iterations=iterations, **params)
        eval_set = None
        if X_valid is not None and X_valid.size:
            eval_set = (X_valid, y_valid)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.model = model
        metrics = {'rmse_train': _rmse(model.predict(X_train), y_train)}
        if X_valid is not None and X_valid.size:
            metrics['rmse_valid'] = _rmse(model.predict(X_valid), y_valid)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'catboost_model.cbm'
        model.save_model(model_path)
        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('CatBoost model not trained')
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError('CatBoost model not trained')
        self.model.save_model(path)

    def load(self, path: Path) -> None:
        from catboost import CatBoostRegressor  # type: ignore

        model = CatBoostRegressor()
        model.load_model(path)
        self.model = model


ModelRegistry.register(LightGBMAdapter.registry_name, LightGBMAdapter)
ModelRegistry.register(XGBoostAdapter.registry_name, XGBoostAdapter)
ModelRegistry.register(CatBoostAdapter.registry_name, CatBoostAdapter)

__all__ = ['LightGBMAdapter', 'XGBoostAdapter', 'CatBoostAdapter']
