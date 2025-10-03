from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SummaryInfo:
    name: str
    summary_path: Path


class AuxModelPredictor:
    """Load auxiliary ML/RL models (PyTorch, PPO) and generate signals for strategies."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._summary_cache: Dict[str, Dict[str, Any]] = {}
        self._model_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._candidates = {
            'pytorch_mlp': SummaryInfo('pytorch_mlp', project_root / 'artifacts/models/pytorch_real/training_summary.json'),
            'rl_ppo': SummaryInfo('rl_ppo', project_root / 'artifacts/models/rl_real/training_summary.json'),
            'lightgbm_multi': SummaryInfo('lightgbm_multi', project_root / 'artifacts/models/lightgbm_multi/training_summary.json'),
        }

    # ------------------------------------------------------------------
    def augment(self, dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        """Return computed signals and attach them to dataframe in-place."""
        results: Dict[str, pd.Series] = {}
        for key, info in self._candidates.items():
            summary = self._load_summary(key, info.summary_path)
            if not summary:
                continue
            features = summary.get('features') or []
            if not features:
                continue
            matrix = self._extract_feature_matrix(dataframe, features)
            if matrix is None:
                continue
            model_path = self._resolve_model_path(summary.get('model_path'))
            if model_path is None:
                continue
            if key == 'pytorch_mlp':
                preds = self._predict_pytorch(key, model_path, matrix)
                if preds is None:
                    continue
                series = pd.Series(preds, index=dataframe.index, name='mlp_prediction').astype(float)
                z_series = self._zscore(series, window=200)
                results['mlp_prediction'] = series
                results['mlp_zscore'] = z_series
                dataframe['mlp_prediction'] = series
                dataframe['mlp_zscore'] = z_series
            elif key == 'rl_ppo':
                probs = self._predict_rl(key, model_path, matrix)
                if probs is None:
                    continue
                long_prob = pd.Series(probs[:, 1] if probs.shape[1] > 1 else 0.0, index=dataframe.index, name='rl_long_prob').astype(float)
                short_prob = pd.Series(probs[:, 2] if probs.shape[1] > 2 else 0.0, index=dataframe.index, name='rl_short_prob').astype(float)
                action = pd.Series(np.argmax(probs, axis=1), index=dataframe.index, name='rl_action')
                results['rl_long_prob'] = long_prob
                results['rl_short_prob'] = short_prob
                results['rl_action'] = action
                dataframe['rl_long_prob'] = long_prob
                dataframe['rl_short_prob'] = short_prob
                dataframe['rl_action'] = action
            elif key == 'lightgbm_multi':
                preds = self._predict_lightgbm(key, model_path, matrix)
                if preds is None:
                    continue
                series = pd.Series(preds, index=dataframe.index, name='lgbm_prediction').astype(float)
                z_series = self._zscore(series, window=200)
                results['lgbm_prediction'] = series
                results['lgbm_zscore'] = z_series
                dataframe['lgbm_prediction'] = series
                dataframe['lgbm_zscore'] = z_series
        return results

    # ------------------------------------------------------------------
    def _load_summary(self, key: str, path: Path) -> Optional[Dict[str, Any]]:
        try:
            cache = self._summary_cache.get(key)
            if cache and cache.get('__mtime__') == path.stat().st_mtime:
                return cache
            if not path.exists():
                logger.debug("AuxModelPredictor: summary missing for %s (%s)", key, path)
                return None
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['__mtime__'] = path.stat().st_mtime
            self._summary_cache[key] = payload
            return payload
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("AuxModelPredictor: failed to load summary %s (%s): %s", key, path, exc)
            return None

    def _resolve_model_path(self, raw: Any) -> Optional[Path]:
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        if not path.exists():
            logger.debug("AuxModelPredictor: model file missing %s", path)
            return None
        return path

    def _extract_feature_matrix(self, dataframe: pd.DataFrame, columns: Iterable[str]) -> Optional[np.ndarray]:
        missing = [col for col in columns if col not in dataframe.columns]
        if missing:
            logger.debug("AuxModelPredictor: dataframe missing features %s", ', '.join(missing))
            return None
        matrix = dataframe[list(columns)].astype(float).replace([np.inf, -np.inf], np.nan)
        matrix = matrix.ffill().bfill().fillna(0.0)
        return matrix.to_numpy(dtype=np.float32)

    @staticmethod
    def _zscore(series: pd.Series, window: int = 200) -> pd.Series:
        rolling_mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
        rolling_std = series.rolling(window, min_periods=max(10, window // 5)).std(ddof=0)
        z = (series - rolling_mean) / (rolling_std + 1e-9)
        return z.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-6, 6)

    def _predict_pytorch(self, key: str, model_path: Path, matrix: np.ndarray) -> Optional[np.ndarray]:
        try:
            from agent_market.freqai.model.torch_models import PyTorchMLPAdapter
        except ImportError as exc:
            logger.warning("AuxModelPredictor: PyTorch not available (%s)", exc)
            return None
        cache_key = (key, str(model_path))
        cached = self._model_cache.get(cache_key)
        mtime = model_path.stat().st_mtime
        if not cached or cached.get('mtime') != mtime:
            adapter = PyTorchMLPAdapter({'use_cuda': False})
            adapter.load(model_path)
            cached = {'model': adapter, 'mtime': mtime}
            self._model_cache[cache_key] = cached
        adapter = cached['model']
        try:
            preds = adapter.predict(matrix)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AuxModelPredictor: PyTorch predict failed: %s", exc)
            return None
        return preds.astype(np.float32)

    def _predict_rl(self, key: str, model_path: Path, matrix: np.ndarray) -> Optional[np.ndarray]:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            logger.warning("AuxModelPredictor: Stable-Baselines3 not available (%s)", exc)
            return None
        cache_key = (key, str(model_path))
        cached = self._model_cache.get(cache_key)
        mtime = model_path.stat().st_mtime
        if not cached or cached.get('mtime') != mtime:
            model = PPO.load(model_path, device='cpu')
            cached = {'model': model, 'mtime': mtime}
            self._model_cache[cache_key] = cached
        model = cached['model']
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - defensive
            logger.warning("AuxModelPredictor: torch not available for RL inference (%s)", exc)
            return None
        try:
            obs = torch.as_tensor(matrix, dtype=torch.float32, device=model.policy.device)
            distribution = model.policy.get_distribution(obs)
            if hasattr(distribution.distribution, 'probs'):
                probs = distribution.distribution.probs.detach().cpu().numpy()
            else:  # pragma: no cover - fallback
                actions, _ = model.predict(matrix, deterministic=False)
                probs = np.zeros((matrix.shape[0], model.action_space.n), dtype=np.float32)
                probs[np.arange(matrix.shape[0]), actions] = 1.0
            return probs.astype(np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AuxModelPredictor: RL prediction failed: %s", exc)
            return None

    def _predict_lightgbm(self, key: str, model_path: Path, matrix: np.ndarray) -> Optional[np.ndarray]:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:
            logger.warning("AuxModelPredictor: LightGBM not available (%s)", exc)
            return None
        cache_key = (key, str(model_path))
        cached = self._model_cache.get(cache_key)
        mtime = model_path.stat().st_mtime
        if not cached or cached.get('mtime') != mtime:
            booster = lgb.Booster(model_file=str(model_path))
            cached = {'model': booster, 'mtime': mtime}
            self._model_cache[cache_key] = cached
        booster = cached['model']
        try:
            preds = booster.predict(matrix)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AuxModelPredictor: LightGBM predict failed: %s", exc)
            return None
        return preds.astype(np.float32)


__all__ = ['AuxModelPredictor']
