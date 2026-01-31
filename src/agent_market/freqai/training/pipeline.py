from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import agent_market.freqai.model  # noqa: F401
import numpy as np
import pandas as pd

from agent_market.freqai.expression_engine import apply_expressions, load_expression_file
from agent_market.freqai.features import apply_configured_features
from agent_market.freqai.model.base import ModelRegistry, TrainResult

try:  # optional deps
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_file(src: Path, dst: Path) -> Dict[str, Any]:
    payload = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    return {
        "path": str(dst),
        "sha256": _sha256_bytes(payload),
        "size_bytes": int(len(payload)),
    }


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    columns: List[str]


class FeatureDatasetBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_file = Path(config['feature_file'])
        self.expressions_file = Path(config['expressions_file']) if config.get('expressions_file') else None
        self.data_dir = Path(config.get('data_dir', 'user_data/data'))
        self.exchange = config.get('exchange') or 'binanceus'
        self.timeframe = config.get('timeframe') or '1h'
        self.pairs = config.get('pairs') or ['BTC/USDT']
        self.label_period = int(config.get('label_period') or 12)
        self._expression_specs = (
            load_expression_file(self.expressions_file)
            if self.expressions_file is not None
            else []
        )

    def build(self) -> Dataset:
        feature_cfg = json.loads(self.feature_file.read_text(encoding="utf-8-sig"))
        datasets: List[Tuple[np.ndarray, np.ndarray]] = []
        feature_cols: List[str] = []
        for pair in self.pairs:
            df = self._load_pair_dataframe(pair)
            df = apply_configured_features(df, feature_cfg)
            if self._expression_specs:
                df, _added = apply_expressions(df, self._expression_specs, on_error='raise')
            columns = self._select_feature_columns(df)
            if not columns:
                continue
            labels = (df['close'].shift(-self.label_period) / df['close']) - 1
            features = df[columns]
            features, labels = self._align(features, labels)
            if features.empty:
                continue
            datasets.append((features.to_numpy(dtype=np.float32), labels.to_numpy(dtype=np.float32)))
            feature_cols = columns
        if not datasets:
            raise ValueError('No training data available; check feature configuration and data paths.')
        X = np.vstack([item[0] for item in datasets])
        y = np.concatenate([item[1] for item in datasets])
        return Dataset(X, y, feature_cols)

    def _load_pair_dataframe(self, pair: str) -> pd.DataFrame:
        sanitized = pair.replace('/', '_')
        file_path = self.data_dir / self.exchange / f"{sanitized}-{self.timeframe}.feather"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_feather(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)

    @staticmethod
    def _select_feature_columns(df: pd.DataFrame) -> List[str]:
        exclude = {'date', 'open', 'high', 'low', 'close', 'volume'}
        return [col for col in df.columns if col not in exclude]

    def _align(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        drop_count = self.label_period
        if drop_count > 0 and drop_count < len(features):
            features = features.iloc[:-drop_count]
            labels = labels.iloc[:-drop_count]
        combined = pd.concat([features, labels.rename('__target__')], axis=1)
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
        if combined.empty:
            return pd.DataFrame(columns=features.columns), pd.Series(dtype=float)
        target = combined.pop('__target__')
        return combined, target


class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get('model', {})
        self.data_cfg = config.get('data', {})
        self.training_cfg = config.get('training', {})
        self.output_cfg = config.get('output', {})

    def run(self) -> TrainResult:
        builder = FeatureDatasetBuilder(self.data_cfg)
        dataset = builder.build()

        # Standardization (optional)
        scaler_name = str(self.training_cfg.get('scaler', 'none')).lower()
        X = dataset.features
        scaler_obj = None
        if _HAS_SKLEARN and scaler_name in ('standard', 'robust'):
            scaler_obj = StandardScaler() if scaler_name == 'standard' else RobustScaler()
            X = scaler_obj.fit_transform(X)

        # Primary time split
        X_train, y_train, X_valid, y_valid = self._split(
            X,
            dataset.labels,
            float(self.training_cfg.get('validation_ratio', 0.2)),
        )

        model_dir = Path(self.output_cfg.get('model_dir', 'artifacts/models'))
        params = dict(self.model_cfg.get('params', {}))
        params.setdefault('model_dir', str(model_dir))
        model_name = self.model_cfg.get('name', 'lightgbm')
        adapter = ModelRegistry.create(model_name, params)

        def _fit_with_hint(ad, Xt, yt, Xv, yv):  # noqa: ANN001
            try:
                return ad.fit(Xt, yt, X_valid=Xv, y_valid=yv)
            except ImportError as exc:
                pkg = {
                    'lightgbm': 'lightgbm',
                    'xgboost': 'xgboost',
                    'catboost': 'catboost',
                    'torch': 'torch',
                }.get(str(model_name))
                if pkg:
                    raise RuntimeError(
                        f"Missing optional dependency for model={model_name!r}: {exc}. "
                        f"Install with: pip install {pkg} (or pip install -r requirements-full.txt)"
                    ) from exc
                raise

        result = _fit_with_hint(adapter, X_train, y_train, X_valid, y_valid)

        # Rolling validation (optional)
        rolling_splits = int(self.training_cfg.get('rolling_splits', 0) or 0)
        rolling_metrics = None
        if _HAS_SKLEARN and rolling_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=rolling_splits)
            rmses = []
            for tr_idx, va_idx in tscv.split(X):
                Xt, Xv = X[tr_idx], X[va_idx]
                yt, yv = dataset.labels[tr_idx], dataset.labels[va_idx]
                ad = ModelRegistry.create(model_name, params)
                r = _fit_with_hint(ad, Xt, yt, Xv, yv)
                rmse_v = float(r.metrics.get('rmse_valid') or r.metrics.get('rmse_train') or 0.0)
                rmses.append(rmse_v)
            if rmses:
                rolling_metrics = {
                    'rmse_valid_mean': float(np.mean(rmses)),
                    'rmse_valid_std': float(np.std(rmses)),
                    'splits': int(rolling_splits),
                }

        # Persist scaler
        if scaler_obj is not None:
            try:
                import pickle  # noqa: PLC0415
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(model_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump({'type': scaler_name, 'features': dataset.columns, 'scaler': scaler_obj}, f)
            except Exception:
                pass

        # Summary
        summary_path = model_dir / 'training_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        feature_snapshot_info = _snapshot_file(builder.feature_file, model_dir / 'feature_snapshot.json')
        summary = {
            'model': self.model_cfg.get('name', 'lightgbm'),
            'features': dataset.columns,
            'data': {
                'data_dir': str(builder.data_dir),
                'exchange': str(builder.exchange),
                'pairs': list(builder.pairs),
                'timeframe': str(builder.timeframe),
                'label_period': int(builder.label_period),
            },
            'train_size': int(X_train.shape[0]),
            'valid_size': int(X_valid.shape[0]),
            'metrics': result.metrics,
            'model_path': str(result.model_path),
            'feature_file': str(builder.feature_file),
            'feature_snapshot': feature_snapshot_info.get('path'),
            'feature_sha256': feature_snapshot_info.get('sha256'),
        }
        if builder.expressions_file is not None:
            expressions_snapshot_info = _snapshot_file(builder.expressions_file, model_dir / 'expressions_snapshot.json')
            summary['expressions_file'] = str(builder.expressions_file)
            summary['expressions_snapshot'] = expressions_snapshot_info.get('path')
            summary['expressions_sha256'] = expressions_snapshot_info.get('sha256')
        if rolling_metrics:
            summary['rolling'] = rolling_metrics
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _split(
        features: np.ndarray,
        labels: np.ndarray,
        validation_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if features.shape[0] != labels.shape[0]:
            raise ValueError('Feature and label size mismatch')
        if features.shape[0] == 0:
            raise ValueError('Empty dataset')
        validation_ratio = max(0.0, min(0.9, validation_ratio))
        split_index = int(features.shape[0] * (1.0 - validation_ratio))
        split_index = max(1, min(split_index, features.shape[0] - 1))
        X_train = features[:split_index]
        y_train = labels[:split_index]
        X_valid = features[split_index:]
        y_valid = labels[split_index:]
        return X_train, y_train, X_valid, y_valid


__all__ = ['TrainingPipeline', 'FeatureDatasetBuilder', 'Dataset']
