"""TradeLevelEnv (E5) — trade-as-unit action space.

Root cause of E1-E4 failures: TradingEnv forces the policy to decide at
every bar; with 25bps fees and ±0.1% per-bar returns, any "open often"
policy burns equity. Real strategies hold a position for many bars then
exit on a stop/target — a trade-level view.

This env emulates that:
    action (Discrete 3):
        0 = noop (do nothing / stay flat)
        1 = open_long  (only effective when currently flat)
        2 = open_short (only effective when currently flat)

    While in a position, the env ignores further actions and checks on
    each bar whether one of these auto-exits fires:
        - Stop loss:     price drops stop_loss_pct below entry (for long)
        - Take profit:   price rises take_profit_pct above entry (for long)
        - Max hold bars: N bars since entry
        - Pair boundary: dataset rolls to next pair

    Reward is the net PnL of each *completed* trade (realized at exit).
    Intermediate bars give zero reward. The policy learns "when to enter".
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

from agent_market.freqai.training.pipeline import Dataset


@dataclass
class TradeEnvConfig:
    data: Dict[str, Any]
    fee_bps: float = 8.0
    stop_loss_pct: float = 0.015       # -1.5% from entry
    take_profit_pct: float = 0.03      # +3% from entry (2:1 RR)
    max_hold_bars: int = 24            # force exit after 1 day on 1h
    allow_short: bool = True
    no_trade_penalty_bps: float = 0.0  # small pressure to actually open — 0 disables


if _HAS_GYMNASIUM:
    class TradeLevelEnv(Env[np.ndarray, int]):
        metadata = {"render_modes": []}

        def __init__(self, dataset: Dataset, config: Optional[TradeEnvConfig] = None):
            super().__init__()
            self.dataset = dataset
            self.config = config or TradeEnvConfig(data={})
            self.action_space = spaces.Discrete(3)  # type: ignore[attr-defined]
            # obs: factor features + {position, bars_in_pos, unrealized_pnl, last_trade_pnl}
            obs_dim = dataset.features.shape[1] + 4
            self.observation_space = spaces.Box(  # type: ignore[attr-defined]
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._bars_in_pos = 0
            self._equity = 1.0
            self._peak_equity = 1.0
            self._last_trade_pnl = 0.0
            self._trades: list = []

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            self._index = 0
            self._position = 0
            self._entry_price = 0.0
            self._bars_in_pos = 0
            self._equity = 1.0
            self._peak_equity = 1.0
            self._last_trade_pnl = 0.0
            self._trades = []
            return self._observation(self._index), {}

        def _exit_position(self, cur_price: float, reason: str) -> float:
            """Close any open position at cur_price. Returns PnL (fraction)."""
            if self._position == 0 or self._entry_price <= 0:
                return 0.0
            fee = float(self.config.fee_bps) / 10_000.0
            direction = self._position
            raw_ret = direction * ((cur_price / self._entry_price) - 1.0)
            # Round-trip fees: one on entry, one on exit
            net_pnl = raw_ret - 2.0 * fee
            # Apply to equity
            self._equity *= max(1e-9, 1.0 + net_pnl)
            self._trades.append({
                "direction": direction, "bars": self._bars_in_pos,
                "raw_ret": raw_ret, "net_pnl": net_pnl, "reason": reason,
            })
            self._last_trade_pnl = net_pnl
            self._position = 0
            self._entry_price = 0.0
            self._bars_in_pos = 0
            return net_pnl

        def step(self, action: int):
            cfg = self.config
            cur_idx = int(self._index)
            next_idx = min(cur_idx + 1, self.dataset.features.shape[0] - 1)
            cur_price = float(self.dataset.prices[cur_idx])
            nxt_price = float(self.dataset.prices[next_idx])
            pair_changed = bool(
                self.dataset.pair_ids[next_idx] != self.dataset.pair_ids[cur_idx]
            )

            reward = 0.0
            exit_reason = None

            if self._position == 0:
                # Flat — action decides whether to open
                if action == 1:
                    self._position = 1
                    self._entry_price = cur_price
                    self._bars_in_pos = 0
                elif action == 2 and cfg.allow_short:
                    self._position = -1
                    self._entry_price = cur_price
                    self._bars_in_pos = 0
                elif cfg.no_trade_penalty_bps > 0:
                    reward -= cfg.no_trade_penalty_bps / 10_000.0
            else:
                # In position — ignore action, check exit triggers
                self._bars_in_pos += 1
                direction = self._position
                ret_from_entry = direction * ((cur_price / self._entry_price) - 1.0)
                if ret_from_entry <= -cfg.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif ret_from_entry >= cfg.take_profit_pct:
                    exit_reason = "take_profit"
                elif self._bars_in_pos >= cfg.max_hold_bars:
                    exit_reason = "max_hold"
                elif pair_changed:
                    exit_reason = "pair_change"

                if exit_reason is not None:
                    pnl = self._exit_position(cur_price, exit_reason)
                    reward = pnl  # only non-zero reward in the env

            self._peak_equity = max(self._peak_equity, self._equity)
            self._index = next_idx
            terminated = self._index >= self.dataset.features.shape[0] - 1
            obs = self._observation(self._index)
            info = {
                "equity": float(self._equity),
                "position": int(self._position),
                "bars_in_pos": int(self._bars_in_pos),
                "last_trade_pnl": float(self._last_trade_pnl),
                "trades_closed": len(self._trades),
                "exit_reason": exit_reason or "",
            }
            return obs, float(reward), terminated, False, info

        def _observation(self, index: int) -> np.ndarray:
            base = self.dataset.features[index].astype(np.float32, copy=False)
            price = float(self.dataset.prices[index]) if index < len(self.dataset.prices) else 0.0
            unrealized = 0.0
            if self._position != 0 and self._entry_price > 0 and price > 0:
                unrealized = self._position * ((price / self._entry_price) - 1.0)
            extra = np.asarray([
                float(self._position),
                float(self._bars_in_pos) / 100.0,
                float(unrealized),
                float(self._last_trade_pnl),
            ], dtype=np.float32)
            return np.concatenate([base, extra])
else:
    class TradeLevelEnv:  # pragma: no cover
        def __init__(self, *a, **k):
            raise ImportError("gymnasium required")
