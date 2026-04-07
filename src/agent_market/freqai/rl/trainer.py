from __future__ import annotations

import inspect
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore

from agent_market.freqai.rl.env import TradingEnv, TradingEnvConfig
from agent_market.freqai.runtime_threads import resolve_num_threads
from agent_market.freqai.training.pipeline import FeatureDatasetBuilder


@dataclass
class RLTrainingResult:
    model_path: Path
    timesteps: int


class RLTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = Path(config.get('output', {}).get('model_dir', 'artifacts/models/rl'))
        self.total_timesteps = int(config.get('training', {}).get('total_timesteps', 10_000))
        self.algo_params = dict(config.get('algo', {}))
        self.policy = config.get('policy', 'MlpPolicy')

    @staticmethod
    def _normalize_policy_kwargs(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload = dict(value)
        activation = payload.get('activation_fn')
        if isinstance(activation, str):
            mapping = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'elu': nn.ELU,
                'gelu': nn.GELU,
                'silu': nn.SiLU,
                'swish': nn.SiLU,
                'leaky_relu': nn.LeakyReLU,
            }
            payload['activation_fn'] = mapping.get(activation.strip().lower(), nn.Tanh)

        net_arch = payload.get('net_arch')
        if isinstance(net_arch, dict):
            cleaned: Dict[str, list[int]] = {}
            for key in ('pi', 'vf'):
                layers = net_arch.get(key)
                if isinstance(layers, list):
                    cleaned[key] = [int(x) for x in layers if isinstance(x, (int, float)) or str(x).isdigit()]
            if cleaned:
                payload['net_arch'] = cleaned
        return payload

    @classmethod
    def _sanitize_algo_params(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload = dict(value)
        payload.pop('policy', None)
        payload.pop('env', None)
        payload.pop('policy_type', None)
        payload.pop('total_timesteps', None)
        payload.pop('model_dir', None)
        policy_kwargs = dict(payload.get('policy_kwargs') or {}) if isinstance(payload.get('policy_kwargs'), dict) else {}
        for key in (
            'net_arch',
            'activation_fn',
            'ortho_init',
            'features_extractor_class',
            'features_extractor_kwargs',
        ):
            if key in payload and key not in policy_kwargs:
                policy_kwargs[key] = payload.pop(key)
        if policy_kwargs:
            payload['policy_kwargs'] = cls._normalize_policy_kwargs(policy_kwargs)
        if 'learning_rate' in payload:
            payload['learning_rate'] = float(payload['learning_rate'])
        for key in ('n_steps', 'batch_size', 'n_epochs'):
            if key in payload:
                payload[key] = int(payload[key])
        for key in ('gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'clip_range', 'max_grad_norm'):
            if key in payload:
                payload[key] = float(payload[key])
        return payload

    @staticmethod
    def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a kwargs dict so it only contains parameters accepted by a callable.

        This makes RL training resilient to LLM-generated configs that include keys from
        other algorithms (e.g., `train_freq`) or unrelated metadata (e.g., `timeframe`).
        """
        if not isinstance(kwargs, dict):
            return {}
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return dict(kwargs)

        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return dict(kwargs)

        allowed = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        allowed.discard('self')
        return {k: v for k, v in kwargs.items() if k in allowed}

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): RLTrainer._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RLTrainer._json_safe(v) for v in value]
        if isinstance(value, type):
            return value.__name__
        return value

    def train(self) -> RLTrainingResult:
        if PPO is None:
            raise ImportError('stable-baselines3 is not installed; RL training unavailable')
        num_threads = resolve_num_threads(self.config, explicit_keys=('num_threads',))
        if num_threads > 0:
            torch.set_num_threads(num_threads)
            try:
                torch.set_num_interop_threads(num_threads)
            except RuntimeError:
                pass
        data_cfg = self.config.get('data')
        if not isinstance(data_cfg, dict):
            raise ValueError('rl_training.config.data must be provided')
        builder = FeatureDatasetBuilder(data_cfg)
        dataset = builder.build()
        reward_cfg = self.config.get('reward') if isinstance(self.config.get('reward'), dict) else {}
        env_cfg = TradingEnvConfig(
            data=data_cfg,
            fee_bps=float(reward_cfg.get('fee_bps', 10.0)),
            holding_penalty_bps=float(reward_cfg.get('holding_penalty_bps', 0.2)),
            drawdown_penalty=float(reward_cfg.get('drawdown_penalty', 0.05)),
            invalid_action_penalty=float(reward_cfg.get('invalid_action_penalty', 0.0005)),
        )
        env = TradingEnv(dataset, env_cfg)
        algo_params = self._sanitize_algo_params(self.algo_params)
        algo_params.setdefault('verbose', 0)
        algo_params = self._filter_kwargs_for_callable(PPO.__init__, algo_params)
        policy = self.policy
        model = PPO(policy, env, **algo_params)
        model.learn(total_timesteps=self.total_timesteps)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / 'ppo_trading_env.zip'
        model.save(model_path)
        summary = {
            'model': 'ppo',
            'features': dataset.columns,
            'train_size': int(dataset.features.shape[0]),
            'timesteps': self.total_timesteps,
            'model_path': str(model_path),
            'data': data_cfg,
            'reward': reward_cfg,
            'feature_file': data_cfg.get('feature_file'),
            'expressions_file': data_cfg.get('expressions_file'),
            'policy': policy,
            'algo': self._json_safe(algo_params),
        }
        summary_path = self.model_dir / 'training_summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return RLTrainingResult(model_path=model_path, timesteps=self.total_timesteps)
