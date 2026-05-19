from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import agent_market.freqai.model  # noqa: F401
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from agent_market.utils import sha256_bytes
from agent_market.freqai.expression_engine import apply_expressions, load_expression_file
from agent_market.freqai.features import apply_configured_features
from agent_market.freqai.external_data import apply_external_features
from agent_market.freqai.model.base import ModelRegistry, TrainResult
from agent_market.freqai.training.labels import future_return, future_classify_3way
from agent_market import paths

try:  # optional deps
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA as _PCA
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def _snapshot_file(src: Path, dst: Path) -> Dict[str, Any]:
    payload = src.read_bytes()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    return {
        "path": str(dst),
        "sha256": sha256_bytes(payload),
        "size_bytes": int(len(payload)),
    }


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    columns: List[str]
    prices: np.ndarray
    pair_ids: np.ndarray
    dates: np.ndarray
    ohlcv: Optional[np.ndarray] = None  # shape (N, 5): open, high, low, close, volume


def _parse_timerange_bounds(timerange: str) -> tuple:
    raw = str(timerange or "").strip()
    if not raw:
        return None, None
    parts = raw.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid timerange: {timerange!r}")
    start_dt = datetime.strptime(parts[0], "%Y%m%d").replace(tzinfo=timezone.utc)
    end_dt = (datetime.strptime(parts[1], "%Y%m%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return pd.Timestamp(start_dt), pd.Timestamp(end_dt)


class FeatureDatasetBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_file = paths.resolve_repo_path(config['feature_file'])
        self.expressions_file = (
            paths.resolve_repo_path(config['expressions_file'])
            if config.get('expressions_file')
            else None
        )
        self.data_dir = paths.resolve_repo_path(config.get('data_dir', 'user_data/data'))
        self.exchange = config.get('exchange') or 'binanceus'
        self.timeframe = config.get('timeframe') or '1h'
        self.pairs = config.get('pairs') or ['BTC/USDT']
        self.label_period = int(config.get('label_period') or 12)
        self._expression_specs = (
            load_expression_file(self.expressions_file)
            if self.expressions_file is not None
            else []
        )
        self.train_timerange = str(config.get("train_timerange") or "").strip()
        self.test_timerange = str(config.get("test_timerange") or "").strip()
        self.task = str(config.get("task") or "regression").strip().lower()
        self.class_threshold = float(config.get("class_threshold", 0.005))

    def build(self) -> Dataset:
        feature_cfg = json.loads(self.feature_file.read_text(encoding="utf-8-sig"))
        datasets: List[Tuple] = []
        feature_cols: List[str] = []
        for pair in self.pairs:
            df = self._load_pair_dataframe(pair)
            df = apply_configured_features(df, feature_cfg)
            if self._expression_specs:
                df, _added = apply_expressions(df, self._expression_specs, on_error='raise')
            df = apply_external_features(df, self.config)
            # Filter to train_timerange if specified (strict training/backtest separation)
            if self.train_timerange:
                start_ts, end_ts = _parse_timerange_bounds(self.train_timerange)
                if start_ts is not None and end_ts is not None:
                    date_col = pd.to_datetime(df["date"], utc=True)
                    mask = (date_col >= start_ts) & (date_col < end_ts)
                    df = df.loc[mask].reset_index(drop=True)
                    if df.empty:
                        continue
            columns = self._select_feature_columns(df)
            if not columns:
                continue
            if self.task in {"classify_3way", "classify", "classification"}:
                labels = future_classify_3way(df["close"], self.label_period, self.class_threshold)
            else:
                labels = future_return(df["close"], self.label_period)
            features = df[columns]
            prices = df["close"]
            dates = df["date"]
            features, labels, extras = self._align(
                features,
                labels,
                extras={
                    "__price__": prices,
                    "__date__": dates,
                    "__open__": df["open"],
                    "__high__": df["high"],
                    "__low__": df["low"],
                    "__close__": df["close"],
                    "__volume__": df["volume"],
                },
            )
            if features.empty:
                continue
            pair_ids = np.asarray([pair] * len(features), dtype=object)
            ohlcv_arr = np.stack([
                extras["__open__"].to_numpy(dtype=np.float32),
                extras["__high__"].to_numpy(dtype=np.float32),
                extras["__low__"].to_numpy(dtype=np.float32),
                extras["__close__"].to_numpy(dtype=np.float32),
                extras["__volume__"].to_numpy(dtype=np.float32),
            ], axis=1)  # (N, 5)
            datasets.append(
                (
                    features.to_numpy(dtype=np.float32),
                    labels.to_numpy(dtype=np.float32),
                    extras["__price__"].to_numpy(dtype=np.float32),
                    pair_ids,
                    extras["__date__"].to_numpy(dtype="datetime64[ns]"),
                    ohlcv_arr,
                )
            )
            feature_cols = columns
        if not datasets:
            raise ValueError('No training data available; check feature configuration and data paths.')
        X = np.vstack([item[0] for item in datasets])
        y = np.concatenate([item[1] for item in datasets])
        prices = np.concatenate([item[2] for item in datasets])
        pair_ids = np.concatenate([item[3] for item in datasets])
        dates = np.concatenate([item[4] for item in datasets])
        ohlcv = np.vstack([item[5] for item in datasets])
        return Dataset(X, y, feature_cols, prices, pair_ids, dates, ohlcv=ohlcv)

    def _load_pair_dataframe(self, pair: str) -> pd.DataFrame:
        sanitized = pair.replace('/', '_')
        file_path = self.data_dir / self.exchange / f"{sanitized}-{self.timeframe}.feather"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_feather(file_path)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        return df.sort_values('date').reset_index(drop=True)

    @staticmethod
    def _select_feature_columns(df: pd.DataFrame) -> List[str]:
        exclude = {'date', 'open', 'high', 'low', 'close', 'volume'}
        return [col for col in df.columns if col not in exclude]

    def _align(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        extras: dict[str, pd.Series] | None = None,
    ) -> Tuple[pd.DataFrame, pd.Series, dict[str, pd.Series]]:
        drop_count = self.label_period
        if drop_count > 0 and drop_count < len(features):
            features = features.iloc[:-drop_count]
            labels = labels.iloc[:-drop_count]
            if extras:
                extras = {key: series.iloc[:-drop_count] for key, series in extras.items()}
        extra_names = list((extras or {}).keys())
        combined = pd.concat(
            [features, labels.rename('__target__'), *[(extras or {})[name].rename(name) for name in extra_names]],
            axis=1,
        )
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
        if combined.empty:
            return pd.DataFrame(columns=features.columns), pd.Series(dtype=float), {
                key: pd.Series(dtype=float) for key in extra_names
            }
        target = combined.pop('__target__')
        extra_out = {key: combined.pop(key) for key in extra_names}
        return combined, target, extra_out


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

        # Sort globally by date before splitting to ensure valid temporal ordering
        # across multi-pair datasets (fixes invalid purge/embargo for concatenated pairs).
        sort_idx = np.argsort(dataset.dates)
        X = dataset.features[sort_idx]
        y = dataset.labels[sort_idx]
        dataset.prices = dataset.prices[sort_idx]
        dataset.pair_ids = dataset.pair_ids[sort_idx]
        dataset.dates = dataset.dates[sort_idx]

        # Primary time split with purge/embargo (plan.md §3.5.3 mandatory gate).
        # purge defaults to label_period to prevent label leakage at boundary.
        label_period = int(builder.label_period)
        purge = int(self.training_cfg.get('purge', label_period))
        embargo = int(self.training_cfg.get('embargo', 0))
        X_train, y_train, X_valid, y_valid = self._split(
            X,
            y,
            float(self.training_cfg.get('validation_ratio', 0.2)),
            purge=purge,
            embargo=embargo,
            dates=dataset.dates,
        )

        # Standardization: fit on TRAIN ONLY to prevent data leakage.
        scaler_name = str(self.training_cfg.get('scaler', 'none')).lower()
        scaler_obj = None
        if _HAS_SKLEARN and scaler_name in ('standard', 'robust'):
            scaler_obj = StandardScaler() if scaler_name == 'standard' else RobustScaler()
            X_train = scaler_obj.fit_transform(X_train)
            X_valid = scaler_obj.transform(X_valid)

        # Optional PCA on expression-derived features (fit on TRAIN ONLY).
        # Activated when training config contains: "pca": {"n_components": 0.95}
        pca_obj = None
        pca_col_names: List[str] = []
        pca_expr_cols: List[str] = []
        pca_n_components = (self.training_cfg.get('pca') or {}).get('n_components')
        if _HAS_SKLEARN and pca_n_components and builder.expressions_file:
            try:
                _ed = json.loads(builder.expressions_file.read_text(encoding='utf-8-sig'))
                _expr_set = {e['name'] for e in _ed.get('expressions', [])}
            except Exception:
                _expr_set = set()
            _expr_idx = [i for i, c in enumerate(dataset.columns) if c in _expr_set]
            if _expr_idx:
                _non_idx = [i for i in range(X_train.shape[1]) if i not in set(_expr_idx)]
                pca_obj = _PCA(n_components=pca_n_components, random_state=42)
                _Xe_tr = pca_obj.fit_transform(X_train[:, _expr_idx])
                _Xe_vl = pca_obj.transform(X_valid[:, _expr_idx])
                k_pca = _Xe_tr.shape[1]
                X_train = np.hstack([X_train[:, _non_idx], _Xe_tr.astype(np.float32)])
                X_valid = np.hstack([X_valid[:, _non_idx], _Xe_vl.astype(np.float32)])
                pca_expr_cols = [dataset.columns[i] for i in _expr_idx]
                pca_col_names = [f'pca_{i+1:03d}' for i in range(k_pca)]
                _non_cols = [dataset.columns[i] for i in _non_idx]
                dataset.columns = _non_cols + pca_col_names
                _var_pct = round(float(np.sum(pca_obj.explained_variance_ratio_)) * 100, 1)
                logger.info("[PCA] %d expr cols → %d components (%.1f%% variance)",
                            len(_expr_idx), k_pca, _var_pct)

        raw_model_dir = (
            self.output_cfg.get("model_dir")
            or (self.model_cfg.get("params") or {}).get("model_dir")
            or str(paths.models_root())
        )
        model_dir = paths.resolve_repo_path(raw_model_dir)
        params = dict(self.model_cfg.get('params', {}))
        # Normalize model_dir even when explicitly provided in params/output config,
        # so test/e2e runs can safely redirect output roots without overwriting
        # tracked evidence artifacts.
        params["model_dir"] = str(model_dir)
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

        # Rolling validation (optional) — date-group aware, scaler fit per fold
        rolling_splits = int(self.training_cfg.get('rolling_splits', 0) or 0)
        rolling_metrics = None
        if _HAS_SKLEARN and rolling_splits >= 2:
            unique_dates = np.unique(dataset.dates)
            n_dates = len(unique_dates)
            rmses = []
            if n_dates >= rolling_splits + 1:
                # Date-group aware CV with purge/embargo at fold boundaries
                fold_size = max(1, n_dates // (rolling_splits + 1))
                for fold_i in range(rolling_splits):
                    train_end_date_idx = fold_size * (fold_i + 1)
                    # Apply purge + embargo gap between train and valid folds
                    valid_start_date_idx = min(train_end_date_idx + purge + embargo, n_dates)
                    valid_end_date_idx = min(train_end_date_idx + fold_size + purge + embargo, n_dates)
                    if valid_start_date_idx >= n_dates or valid_end_date_idx <= valid_start_date_idx:
                        continue
                    train_cutoff = unique_dates[train_end_date_idx - 1]
                    valid_start = unique_dates[valid_start_date_idx]
                    valid_end = unique_dates[valid_end_date_idx - 1]
                    tr_mask = dataset.dates <= train_cutoff
                    va_mask = (dataset.dates >= valid_start) & (dataset.dates <= valid_end)
                    Xt, Xv = X[tr_mask], X[va_mask]
                    yt, yv = y[tr_mask], y[va_mask]
                    if len(Xt) == 0 or len(Xv) == 0:
                        continue
                    if scaler_obj is not None:
                        fold_scaler = StandardScaler() if scaler_name == 'standard' else RobustScaler()
                        Xt = fold_scaler.fit_transform(Xt)
                        Xv = fold_scaler.transform(Xv)
                    ad = ModelRegistry.create(model_name, params)
                    r = _fit_with_hint(ad, Xt, yt, Xv, yv)
                    rmse_v = float(r.metrics.get('rmse_valid') or r.metrics.get('rmse_train') or 0.0)
                    rmses.append(rmse_v)
            if rmses:
                rolling_metrics = {
                    'rmse_valid_mean': float(np.mean(rmses)),
                    'rmse_valid_std': float(np.std(rmses)),
                    'splits': int(len(rmses)),
                }

        # Persist scaler
        if scaler_obj is not None:
            try:
                import pickle  # noqa: PLC0415
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(model_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump({'type': scaler_name, 'features': dataset.columns, 'scaler': scaler_obj}, f)
            except Exception:
                logger.warning("Failed to persist scaler to %s", model_dir / 'scaler.pkl', exc_info=True)

        # Persist PCA
        if pca_obj is not None:
            try:
                import pickle  # noqa: PLC0415
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(model_dir / 'pca.pkl', 'wb') as f:
                    pickle.dump({
                        'pca': pca_obj,
                        'expr_cols': pca_expr_cols,
                        'pca_col_names': pca_col_names,
                    }, f)
            except Exception:
                logger.warning("Failed to persist PCA to %s", model_dir / 'pca.pkl', exc_info=True)

        # Summary
        summary_path = model_dir / 'training_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        feature_snapshot_info = _snapshot_file(builder.feature_file, model_dir / 'feature_snapshot.json')
        summary = {
            'model': self.model_cfg.get('name', 'lightgbm'),
            'task': str(builder.task),
            'class_threshold': float(builder.class_threshold),
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
        if pca_obj is not None:
            summary['pca'] = {
                'n_components': len(pca_col_names),
                'n_original': len(pca_expr_cols),
                'explained_variance_pct': round(
                    float(np.sum(pca_obj.explained_variance_ratio_)) * 100, 1),
            }
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
        *,
        purge: int = 0,
        embargo: int = 0,
        dates: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series split with mandatory purge/embargo gap.

        When ``dates`` is provided, the split is done on unique timestamp
        boundaries so that no timestamp straddles train/valid sets (important
        for multi-pair datasets where the same candle time appears for each pair).

        The purge gap (rows dropped between train and valid) prevents
        label leakage when the label horizon overlaps the boundary.
        The embargo gap drops rows at the start of the validation set
        to guard against autocorrelation bleed-through.
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError('Feature and label size mismatch')
        if features.shape[0] == 0:
            raise ValueError('Empty dataset')
        validation_ratio = max(0.0, min(0.9, validation_ratio))
        purge = max(0, int(purge))
        embargo = max(0, int(embargo))

        if dates is not None:
            if len(dates) != features.shape[0]:
                raise ValueError(
                    f"dates length ({len(dates)}) must match features rows ({features.shape[0]})"
                )
            # Split on unique timestamp groups to prevent same-timestamp leakage
            unique_dates = np.unique(dates)
            n_dates = len(unique_dates)
            split_date_idx = int(n_dates * (1.0 - validation_ratio))
            split_date_idx = max(1, min(split_date_idx, n_dates - 1))

            train_cutoff = unique_dates[split_date_idx - 1]
            # purge/embargo: skip timestamps after train cutoff
            valid_date_idx = min(split_date_idx + purge + embargo, n_dates)
            if valid_date_idx >= n_dates:
                valid_date_idx = split_date_idx
            valid_start_date = unique_dates[valid_date_idx]

            train_mask = dates <= train_cutoff
            valid_mask = dates >= valid_start_date
        else:
            # Fallback: row-based split
            split_index = int(features.shape[0] * (1.0 - validation_ratio))
            split_index = max(1, min(split_index, features.shape[0] - 1))
            train_end = split_index
            valid_start = min(split_index + purge + embargo, features.shape[0])
            if valid_start >= features.shape[0]:
                valid_start = split_index
            train_mask = np.zeros(features.shape[0], dtype=bool)
            train_mask[:train_end] = True
            valid_mask = np.zeros(features.shape[0], dtype=bool)
            valid_mask[valid_start:] = True

        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_valid = features[valid_mask]
        y_valid = labels[valid_mask]
        return X_train, y_train, X_valid, y_valid


__all__ = ['TrainingPipeline', 'FeatureDatasetBuilder', 'Dataset']
