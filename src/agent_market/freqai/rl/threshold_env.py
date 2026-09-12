"""ThresholdPolicyEnv — action selects a signal-strength threshold; env
decides position automatically from the g-factors consensus.

Rationale: in TradingEnv / TargetPositionEnv the policy must learn
"given these 13 g-factors, should I open a long?". That is a hard
mapping to learn from sparse rewards. Here we collapse the decision to
a single scalar:
    action ∈ {0..6}  →  k ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}
The env precomputes a per-bar `consensus` score as the mean of the
rank-transformed g-factors; it then sets target_position = sign(consensus)
whenever |consensus| > k. Low k → trade often; high k → rarely trade.

The policy's job becomes "pick the right threshold for current regime"
— a much simpler 1-D search than full open/close decisions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from gymnasium import Env, spaces  # type: ignore
    _HAS_GYMNASIUM = True
except Exception:
    _HAS_GYMNASIUM = False
    Env = object  # type: ignore
    spaces = None  # type: ignore

from agent_market.freqai.training.pipeline import Dataset


THRESHOLD_LEVELS: List[float] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


@dataclass
class ThresholdEnvConfig:
    data: Dict[str, Any]
    fee_bps: float = 25.0
    holding_penalty_bps: float = 0.1
    drawdown_penalty: float = 0.15
    allow_short: bool = True
    thresholds: List[float] = field(default_factory=lambda: list(THRESHOLD_LEVELS))


def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Fast per-column percentile rank in [-1, 1] on finite values."""
    out = np.full(arr.shape, 0.0, dtype=np.float64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 50:
            continue
        vals = col[mask]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals), dtype=np.float64)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)
        # Map to [-1, 1]
        out[mask, j] = (ranks / len(vals)) * 2.0 - 1.0
    return out


if _HAS_GYMNASIUM:
    class ThresholdPolicyEnv(Env[np.ndarray, int]):
        """Discrete-7 env. action picks threshold level; env fires trades
        only when |consensus_signal| > threshold."""
        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[ThresholdEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or ThresholdEnvConfig(data={})
            # Precompute consensus = mean of rank-transformed factors
            ranks = _percentile_rank(dataset.features.astype(np.float64))
            self._consensus = np.mean(ranks, axis=1)
            self.action_space = spaces.Discrete(len(self.config.thresholds))  # type: ignore[attr-defined]
            # obs: factor features + position + unrealized + consensus + prev_threshold
            obs_dim = dataset.features.shape[1] + 4
            self.observation_space = spaces.Box(  # type: ignore[attr-defined]
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0
            self._last_threshold = 0.0

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._equity = 1.0
            self._peak_equity = 1.0
            self._last_threshold = 0.0
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

            # Action → threshold → target position
            idx_action = int(max(0, min(len(self.config.thresholds) - 1, action)))
            k = float(self.config.thresholds[idx_action])
            consensus = float(self._consensus[cur_idx])
            if consensus > k:
                target = 1
            elif consensus < -k and self.config.allow_short:
                target = -1
            else:
                target = 0
            if pair_changed:
                target = 0

            self._last_threshold = k
            delta = target - self._position
            trade_cost = fee * abs(delta)
            self._equity *= max(1e-9, 1.0 - trade_cost)
            if delta != 0:
                self._position = target
                self._entry_price = cur_price if target != 0 else 0.0

            reward = -trade_cost
            if self._position != 0:
                step_ret = (nxt_price / cur_price) - 1.0 if cur_price > 0 else 0.0
                pnl = self._position * step_ret
                reward += pnl - hold_pen * abs(self._position)
                self._equity *= max(1e-9, 1.0 + pnl - hold_pen * abs(self._position))

            self._peak_equity = max(self._peak_equity, self._equity)
            if self._peak_equity > 0:
                drawdown = max(0.0, 1.0 - (self._equity / self._peak_equity))
                reward -= drawdown * dd_pen

            self._index = next_idx
            terminated = self._index >= self.dataset.features.shape[0] - 1
            return (self._observation(self._index), float(reward),
                    terminated, False,
                    {"equity": float(self._equity),
                     "position": int(self._position),
                     "threshold": float(k),
                     "consensus": float(consensus)})

        def _observation(self, index: int) -> np.ndarray:
            base = self.dataset.features[index].astype(np.float32, copy=False)
            price = float(self.dataset.prices[index]) if index < len(self.dataset.prices) else 0.0
            unrealized = 0.0
            if self._position != 0 and self._entry_price > 0 and price > 0:
                unrealized = self._position * ((price / self._entry_price) - 1.0)
            consensus = float(self._consensus[index]) if index < len(self._consensus) else 0.0
            return np.concatenate([
                base,
                np.asarray([float(self._position), float(unrealized),
                             float(consensus), float(self._last_threshold)],
                            dtype=np.float32),
            ])
else:
    class ThresholdPolicyEnv:  # pragma: no cover
        def __init__(self, *a, **k):
            raise ImportError("gymnasium required")
