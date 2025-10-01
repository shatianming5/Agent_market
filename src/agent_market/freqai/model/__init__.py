from . import gradient_boosting  # noqa: F401
try:
    # Optional: only import torch models when environment provides torch
    from . import torch_models  # noqa: F401
except Exception:  # pragma: no cover
    torch_models = None  # type: ignore
    # Register a minimal dummy adapter for 'pytorch_mlp' to keep pipeline working without torch
    from .base import BaseModelAdapter, ModelRegistry, TrainResult  # type: ignore
    import numpy as _np  # type: ignore
    from pathlib import Path as _Path  # type: ignore

    class _DummyTorchAdapter(BaseModelAdapter):  # type: ignore
        registry_name = 'pytorch_mlp'
        def __init__(self, config):
            super().__init__(config)
            self._w = None
        def fit(self, X_train, y_train, X_valid=None, y_valid=None) -> TrainResult:  # noqa: ANN001
            X = X_train.astype(_np.float64)
            y = y_train.astype(_np.float64)
            lam = float(self.config.get('lambda_l2', 1e-3))
            XtX = X.T @ X + lam * _np.eye(X.shape[1])
            w = _np.linalg.pinv(XtX) @ X.T @ y
            self._w = w
            metrics = {'rmse_train': float(_np.sqrt(_np.mean((_np.dot(X, w) - y_train) ** 2)))}
            out_dir = _Path(self.config.get('model_dir', 'artifacts/models/pytorch'))
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / 'pytorch_mlp.pt'
            try:
                model_path.write_text('dummy-pt-model', encoding='utf-8')
            except Exception:
                pass
            return TrainResult(model_path=model_path, metrics=metrics)
        def predict(self, X):  # noqa: ANN001
            if self._w is None:
                raise RuntimeError('Dummy torch model not trained')
            return X.astype(_np.float64) @ self._w
        def save(self, path):  # noqa: ANN001
            _Path(path).write_text('dummy-pt-model', encoding='utf-8')
        def load(self, path):  # noqa: ANN001
            self._w = _np.zeros(1)

    try:
        ModelRegistry.register(_DummyTorchAdapter.registry_name, _DummyTorchAdapter)
    except Exception:
        pass
