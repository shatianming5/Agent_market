from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Env, spaces

from agent_market.freqai.training.pipeline import Dataset, FeatureDatasetBuilder


@dataclass
class TradingEnvConfig:
    data: Dict[str, Any]
    reward_positive: float = 1.0
    reward_negative: float = -0.5


class TradingEnv(Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, dataset: Dataset, config: Optional[TradingEnvConfig] = None):
        super().__init__()
        self.dataset = dataset
        self.config = config or TradingEnvConfig(data={})
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: long, 2: short
        obs_dim = dataset.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._index = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._index = 0
        return self.dataset.features[self._index], {}

    def step(self, action: int):
        self._index += 1
        terminated = self._index >= self.dataset.features.shape[0] - 1
        observation = self.dataset.features[min(self._index, self.dataset.features.shape[0] - 1)]
        reward = self._compute_reward(action, self.dataset.labels[self._index - 1])
        return observation, reward, terminated, False, {}

    def _compute_reward(self, action: int, label: float) -> float:
        if action == 1:  # long
            return float(label * self.config.reward_positive)
        if action == 2:  # short
            return float(-label * abs(self.config.reward_negative))
        return 0.0


class TradingEnvFactory:
    @staticmethod
    def from_config(config: Dict[str, Any]) -> TradingEnv:
        builder = FeatureDatasetBuilder(config['data'])
        dataset = builder.build()
        return TradingEnv(dataset, TradingEnvConfig(data=config['data']))
