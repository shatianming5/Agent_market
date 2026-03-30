from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from gymnasium import Env, spaces  # type: ignore
    _HAS_GYMNASIUM = True
except Exception:  # pragma: no cover
    logger.debug("gymnasium not available, TradingEnv will be disabled", exc_info=True)
    Env = object  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYMNASIUM = False

from agent_market.freqai.training.pipeline import Dataset, FeatureDatasetBuilder


@dataclass
class TradingEnvConfig:
    data: Dict[str, Any]
    reward_positive: float = 1.0
    reward_negative: float = -0.5


if _HAS_GYMNASIUM:
    class TradingEnv(Env[np.ndarray, int]):
        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[TradingEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or TradingEnvConfig(data={})
            self.action_space = spaces.Discrete(3)  # type: ignore[attr-defined]
            obs_dim = dataset.features.shape[1]
            self.observation_space = spaces.Box(  # type: ignore[attr-defined]
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self._index = 0

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ):
            super().reset(seed=seed)
            self._index = 0
            return self.dataset.features[self._index], {}

        def step(self, action: int):
            self._index += 1
            terminated = self._index >= self.dataset.features.shape[0] - 1
            observation = self.dataset.features[
                min(self._index, self.dataset.features.shape[0] - 1)
            ]
            reward = self._compute_reward(action, self.dataset.labels[self._index - 1])
            return observation, reward, terminated, False, {}

        def _compute_reward(self, action: int, label: float) -> float:
            if action == 1:  # long
                return float(label * self.config.reward_positive)
            if action == 2:  # short
                return float(-label * abs(self.config.reward_negative))
            return 0.0
else:
    class TradingEnv:  # pragma: no cover
        def __init__(self, dataset: Dataset, config: Optional[TradingEnvConfig] = None):
            raise ImportError("gymnasium is not installed; TradingEnv unavailable")


class TradingEnvFactory:
    @staticmethod
    def from_config(config: Dict[str, Any]) -> TradingEnv:
        builder = FeatureDatasetBuilder(config['data'])
        dataset = builder.build()
        return TradingEnv(dataset, TradingEnvConfig(data=config['data']))
