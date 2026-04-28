"""TargetPositionEnv — action = target leverage ∈ {-1, 0, +1}.

Core difference vs TradingEnv:
    - TradingEnv: action is {hold, buy, sell} — an *event*. PPO easily learns
      to fire buy every bar and burn on fees.
    - TargetPositionEnv: action is the *desired position*. If last step
      already in long and action=long, no trade happens — fee only charged
      when position actually changes. This removes the systematic incentive
      to churn.

Reward stays nearly identical (step return × position - fee × |Δposition|)
but the fee term only fires on transitions, not on every step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from gymnasium import Env, spaces  # type: ignore
    _HAS_GYMNASIUM = True
except Exception:
    _HAS_GYMNASIUM = False
    Env = object  # type: ignore
    spaces = None  # type: ignore

from agent_market.freqai.training.pipeline import Dataset, FeatureDatasetBuilder


@dataclass
class TargetPosEnvConfig:
    data: Dict[str, Any]
    fee_bps: float = 25.0              # 0.25% per transition (round-trip ≈ 50 bps)
    drawdown_penalty: float = 0.15
    holding_penalty_bps: float = 0.1
    allow_short: bool = True


if _HAS_GYMNASIUM:
    class TargetPositionEnv(Env[np.ndarray, int]):
        """Discrete-3 target-position env. action maps:
            0 → target = 0  (flat)
            1 → target = +1 (long)
            2 → target = -1 (short, only if allow_short)
        Reward each step = position × step_return - hold_penalty × |position|
                         - fee × |new_pos - old_pos|
                         - drawdown_penalty × max(0, 1 - equity/peak)
        """
        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[TargetPosEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or TargetPosEnvConfig(data={})
            self.action_space = spaces.Discrete(3)  # type: ignore[attr-defined]
            obs_dim = dataset.features.shape[1] + 2  # features + pos + unrealized
            self.observation_space = spaces.Box(  # type: ignore[attr-defined]
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self._index = 0
            self._position = 0       # {-1, 0, +1}
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0

        def _action_to_target(self, action: int) -> int:
            if action == 0: return 0
            if action == 1: return 1
            if action == 2:
                return -1 if self.config.allow_short else 0
            return 0

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0
            return self._observation(self._index), {}

        def step(self, action: int):
            fee = float(self.config.fee_bps) / 10_000.0
            hold_pen = float(self.config.holding_penalty_bps) / 10_000.0
            dd_pen = float(self.config.drawdown_penalty)

            cur_idx = int(self._index)
            next_idx = min(cur_idx + 1, self.dataset.features.shape[0] - 1)
            cur_price = float(self.dataset.prices[cur_idx])
            nxt_price = float(self.dataset.prices[next_idx])
            pair_changed = bool(
                self.dataset.pair_ids[next_idx] != self.dataset.pair_ids[cur_idx]
            )

            target = self._action_to_target(int(action))
            # At pair boundaries, force flat to prevent position carry-over
            if pair_changed:
                target = 0

            delta = target - self._position
            trade_cost = fee * abs(delta)
            self._equity *= max(1e-9, 1.0 - trade_cost)

            # Update entry price on new position open
            if delta != 0:
                self._position = target
                self._entry_price = cur_price if target != 0 else 0.0

            reward = -trade_cost
            if self._position != 0:
                if cur_price > 0:
                    step_ret = (nxt_price / cur_price) - 1.0
                else:
                    step_ret = 0.0
                pnl = self._position * step_ret
                reward += pnl - hold_pen * abs(self._position)
                self._equity *= max(1e-9, 1.0 + pnl - hold_pen * abs(self._position))

            self._peak_equity = max(self._peak_equity, self._equity)
            if self._peak_equity > 0:
                drawdown = max(0.0, 1.0 - (self._equity / self._peak_equity))
                reward -= drawdown * dd_pen

            self._index = next_idx
            terminated = self._index >= self.dataset.features.shape[0] - 1
            obs = self._observation(self._index)
            info = {"equity": float(self._equity),
                    "position": int(self._position),
                    "trade_cost": float(trade_cost)}
            return obs, float(reward), terminated, False, info

        def _observation(self, index: int) -> np.ndarray:
            base = self.dataset.features[index]
            price = float(self.dataset.prices[index]) if index < len(self.dataset.prices) else 0.0
            unrealized = 0.0
            if self._position != 0 and self._entry_price > 0 and price > 0:
                unrealized = self._position * ((price / self._entry_price) - 1.0)
            return np.concatenate([
                base.astype(np.float32, copy=False),
                np.asarray([float(self._position), float(unrealized)], dtype=np.float32),
            ])
else:
    class TargetPositionEnv:  # pragma: no cover
        def __init__(self, *a, **k):
            raise ImportError("gymnasium required")
