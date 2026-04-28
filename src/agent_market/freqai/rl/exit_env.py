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
    logger.debug("gymnasium not available", exc_info=True)
    Env = object  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYMNASIUM = False

from agent_market.freqai.training.pipeline import Dataset


@dataclass
class ExitEnvConfig:
    data: Dict[str, Any]
    fee_bps: float = 10.0
    window_size: int = 24      # OHLCV look-back candles
    max_hold_steps: int = 48   # max candles before forced exit


if _HAS_GYMNASIUM:
    class ExitOnlyEnv(Env[np.ndarray, int]):
        """2-action RL environment for exit timing only.

        The agent is ALWAYS in position (position=1 from the first step).
        Actions: 0 = hold, 1 = exit (and immediately re-enter).
        Reward: accumulated step returns, paid out at exit minus fee.
        No holding penalty — the RL learns purely from P&L quality.

        Observation: [indicators(N), ohlcv_window(window_size*5)]
        This is context-free (no entry_price, no unrealized_pnl) so that
        signal generation at inference time is identical to training.
        """

        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[ExitEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or ExitEnvConfig(data={})
            self.action_space = spaces.Discrete(2)  # 0=hold, 1=exit
            self._window_size = max(1, int(self.config.window_size))
            indicator_dim = dataset.features.shape[1]
            ohlcv_dim = self._window_size * 5
            obs_dim = indicator_dim + ohlcv_dim
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self._index = 0
            self._entry_price = 0.0
            self._steps_in_trade = 0
            self._accumulated_return = 0.0

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
            super().reset(seed=seed)
            self._index = 0
            self._entry_price = float(self.dataset.prices[0])
            self._steps_in_trade = 0
            self._accumulated_return = 0.0
            return self._observation(0), {}

        def step(self, action: int):
            fee = float(self.config.fee_bps) / 10_000.0
            max_hold = int(self.config.max_hold_steps)
            cur_idx = int(self._index)
            next_idx = min(cur_idx + 1, self.dataset.features.shape[0] - 1)
            current_price = float(self.dataset.prices[cur_idx])
            next_price = float(self.dataset.prices[next_idx])

            # Accumulate step return while holding
            if current_price > 0:
                step_return = (next_price / current_price) - 1.0
            else:
                step_return = 0.0
            self._accumulated_return += step_return
            self._steps_in_trade += 1

            reward = 0.0
            force_exit = self._steps_in_trade >= max_hold
            if action == 1 or force_exit:
                # Exit: reward = accumulated return - fee
                reward = self._accumulated_return - fee
                # Re-enter immediately at current price
                self._entry_price = current_price
                self._accumulated_return = 0.0
                self._steps_in_trade = 0

            self._index = next_idx
            terminated = self._index >= self.dataset.features.shape[0] - 1
            return self._observation(self._index), float(reward), terminated, False, {}

        def _observation(self, index: int) -> np.ndarray:
            base = self.dataset.features[index].astype(np.float32, copy=False)
            if self.dataset.ohlcv is not None:
                start = max(0, index - self._window_size + 1)
                window = self.dataset.ohlcv[start: index + 1].astype(np.float32)
                if len(window) < self._window_size:
                    pad = np.zeros((self._window_size - len(window), 5), dtype=np.float32)
                    window = np.vstack([pad, window])
                cur_close = float(self.dataset.prices[index])
                if cur_close > 0:
                    window = window.copy()
                    window[:, :4] /= cur_close
                    vol_max = float(np.max(np.abs(window[:, 4]))) + 1e-9
                    window[:, 4] /= vol_max
                ohlcv_flat = window.flatten()
            else:
                ohlcv_flat = np.zeros(self._window_size * 5, dtype=np.float32)
            return np.concatenate([base, ohlcv_flat])

else:
    class ExitOnlyEnv:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("gymnasium is not installed; ExitOnlyEnv unavailable")
