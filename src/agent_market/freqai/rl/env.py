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
    fee_bps: float = 10.0
    holding_penalty_bps: float = 0.2
    drawdown_penalty: float = 0.05
    invalid_action_penalty: float = 0.0005


if _HAS_GYMNASIUM:
    class TradingEnv(Env[np.ndarray, int]):
        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[TradingEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or TradingEnvConfig(data={})
            self.action_space = spaces.Discrete(3)  # type: ignore[attr-defined]
            obs_dim = dataset.features.shape[1] + 2
            self.observation_space = spaces.Box(  # type: ignore[attr-defined]
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ):
            super().reset(seed=seed)
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0
            return self._observation(self._index), {}

        def step(self, action: int):
            fee = float(self.config.fee_bps) / 10_000.0
            hold_penalty = float(self.config.holding_penalty_bps) / 10_000.0
            invalid_action_penalty = float(self.config.invalid_action_penalty)
            drawdown_penalty = float(self.config.drawdown_penalty)

            cur_idx = int(self._index)
            next_idx = min(cur_idx + 1, self.dataset.features.shape[0] - 1)
            current_price = float(self.dataset.prices[cur_idx])
            next_price = float(self.dataset.prices[next_idx])
            pair_changed = bool(self.dataset.pair_ids[next_idx] != self.dataset.pair_ids[cur_idx])
            reward = 0.0

            if action == 1:
                if self._position == 0:
                    self._position = 1
                    self._entry_price = current_price
                    reward -= fee
                    self._equity *= max(1e-9, 1.0 - fee)
                else:
                    reward -= invalid_action_penalty
            elif action == 2:
                if self._position == 1:
                    reward -= fee
                    self._equity *= max(1e-9, 1.0 - fee)
                    self._position = 0
                    self._entry_price = 0.0
                else:
                    reward -= invalid_action_penalty

            if self._position == 1:
                if current_price > 0 and not pair_changed:
                    step_return = (next_price / current_price) - 1.0
                else:
                    step_return = 0.0
                reward += step_return
                reward -= hold_penalty
                self._equity *= max(1e-9, 1.0 + step_return - hold_penalty)
                if pair_changed:
                    reward -= fee
                    self._equity *= max(1e-9, 1.0 - fee)
                    self._position = 0
                    self._entry_price = 0.0

            self._peak_equity = max(self._peak_equity, self._equity)
            if self._peak_equity > 0:
                drawdown = max(0.0, 1.0 - (self._equity / self._peak_equity))
                reward -= drawdown * drawdown_penalty

            self._index = next_idx
            terminated = self._index >= self.dataset.features.shape[0] - 1
            observation = self._observation(self._index)
            info = {
                "equity": float(self._equity),
                "position": int(self._position),
            }
            return observation, float(reward), terminated, False, info

        def _observation(self, index: int) -> np.ndarray:
            base = self.dataset.features[index]
            price = float(self.dataset.prices[index]) if index < len(self.dataset.prices) else 0.0
            unrealized = 0.0
            if self._position == 1 and self._entry_price > 0 and price > 0:
                unrealized = (price / self._entry_price) - 1.0
            return np.concatenate(
                [
                    base.astype(np.float32, copy=False),
                    np.asarray([float(self._position), float(unrealized)], dtype=np.float32),
                ]
            )
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
