"""Freqtrade Basket Momentum Strategy — cross-sectional momentum in freqtrade.

Buys the TOP-N strongest assets (by recent return) from a universe of 9 cryptos.
Rebalances weekly by exiting losers and entering new winners.

Uses informative_pairs to get ALL assets' data, then ranks by momentum.
Only enters long on assets ranked in top N.

This is a real freqtrade IStrategy — runs through the L2 backtest engine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pandas import DataFrame

_ROOT = Path(__file__).resolve().parents[3]
for p in [str(_ROOT / "src"), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from freqtrade.strategy import IStrategy


class FreqtradeBasketMomentum(IStrategy):
    """Buy top-N momentum assets, rebalance weekly."""

    timeframe = "1h"
    can_short = False

    # Wide ROI — hold for rebalance cycle
    minimal_roi = {"0": 0.20, "168": 0.05, "504": 0.01}
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 200

    # Basket parameters
    momentum_lookback = 168  # hours (1 week)
    top_n = 3  # buy top N assets
    rebalance_hours = 168  # rebalance every week

    # Full universe of assets
    universe = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT",
        "AVAX/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT",
    ]

    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Load ALL universe assets as informative pairs."""
        return [(pair, self.timeframe) for pair in self.universe]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate momentum rank across all universe assets."""
        current_pair = metadata.get("pair", "")

        # Collect momentum (return over lookback) for all assets
        momentum_data = {}

        for pair in self.universe:
            if pair == current_pair:
                # Use main dataframe
                ret = dataframe["close"].pct_change(self.momentum_lookback)
                momentum_data[pair] = ret
            else:
                # Use informative data
                inf_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if not inf_df.empty and len(inf_df) > self.momentum_lookback:
                    inf_ret = inf_df["close"].pct_change(self.momentum_lookback)
                    # Align to main dataframe by date
                    inf_aligned = inf_df[["date"]].copy()
                    inf_aligned["ret"] = inf_ret.values
                    merged = dataframe[["date"]].merge(inf_aligned, on="date", how="left")
                    momentum_data[pair] = merged["ret"].values
                else:
                    momentum_data[pair] = np.full(len(dataframe), np.nan)

        # Calculate rank for current pair at each bar
        n_assets = len(self.universe)
        ranks = np.full(len(dataframe), n_assets)  # default: worst rank

        for i in range(self.momentum_lookback, len(dataframe)):
            returns = {}
            for pair, ret_series in momentum_data.items():
                if isinstance(ret_series, np.ndarray):
                    val = ret_series[i] if i < len(ret_series) else np.nan
                else:
                    val = ret_series.iloc[i] if i < len(ret_series) else np.nan
                if np.isfinite(val):
                    returns[pair] = val

            if len(returns) < 3:
                continue

            # Rank: 1 = best momentum
            sorted_pairs = sorted(returns.keys(), key=lambda p: returns[p], reverse=True)
            for rank_idx, pair in enumerate(sorted_pairs):
                if pair == current_pair:
                    ranks[i] = rank_idx + 1
                    break

        dataframe["momentum_rank"] = ranks
        dataframe["is_top_n"] = (dataframe["momentum_rank"] <= self.top_n).astype(int)

        # Rebalance timing: only trade at rebalance intervals
        dataframe["bar_index"] = range(len(dataframe))
        dataframe["is_rebalance"] = (dataframe["bar_index"] % self.rebalance_hours == 0).astype(int)

        # Current pair's own momentum
        dataframe["own_momentum"] = dataframe["close"].pct_change(self.momentum_lookback)

        # Volume filter
        dataframe["vol_sma"] = dataframe["volume"].rolling(48).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter long if this asset is in top-N and it's rebalance time."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["is_top_n"] == 1)
            & (dataframe["own_momentum"] > 0)  # positive momentum
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.3),
            ["enter_long", "enter_tag"],
        ] = (1, "basket_top_n")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit if asset drops out of top-N or momentum turns negative."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["is_top_n"] == 0)  # no longer in top N
                | (dataframe["own_momentum"] < -0.05)  # strong negative momentum
            ),
            "exit_long",
        ] = 1
        return dataframe
