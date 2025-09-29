from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore

from agent_market.freqai.rl.env import TradingEnv, TradingEnvConfig
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

    def train(self) -> RLTrainingResult:
        if PPO is None:
            raise ImportError('stable-baselines3 is not installed; RL training unavailable')
        data_cfg = self.config.get('data')
        if not isinstance(data_cfg, dict):
            raise ValueError('rl_training.config.data must be provided')
        builder = FeatureDatasetBuilder(data_cfg)
        dataset = builder.build()
        env = TradingEnv(dataset, TradingEnvConfig(data=data_cfg))
        algo_params = dict(self.algo_params)
        algo_params.setdefault('verbose', 0)
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
        }
        summary_path = self.model_dir / 'training_summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return RLTrainingResult(model_path=model_path, timesteps=self.total_timesteps)






