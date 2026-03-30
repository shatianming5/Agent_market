"""Freqtrade IStrategy for ADA/AVAX Relative-Value Pairs Trading.

This is the REAL freqtrade strategy (not L1 simulator).
It uses informative_pairs to get AVAX data while trading ADA.

Signal: When ADA is cheap relative to AVAX (zscore < -entry_z), buy ADA.
        When spread reverts (|zscore| < exit_z), sell ADA.

This is a single-leg execution (buy/sell ADA only),
using AVAX as a reference signal. Not true market-neutral.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

# Inject project paths so we can import from user_data/strategies/
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]  # workspace/strategies/type_C_pairs/ → project root
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from freqtrade.strategy import IStrategy


class FreqtradePairsADAVAX(IStrategy):
    """ADA/AVAX pairs trading via freqtrade (L2 backtest)."""

    # --- Core settings ---
    timeframe = "1h"
    can_short = False

    # Wider ROI for pairs (holds longer for spread reversion)
    minimal_roi = {"0": 0.15, "240": 0.05, "720": 0.02}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 120  # need lookback for spread calc

    # --- Pairs parameters ---
    pair_b = "AVAX/USDT"  # reference asset (signal source)
    lookback = 80
    entry_z = 2.5
    exit_z = 0.7

    def informative_pairs(self):
        """Tell freqtrade to also load AVAX/USDT data."""
        return [(self.pair_b, self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate spread z-score between ADA and AVAX."""
        pair = metadata.get("pair", "")

        # Get informative data for pair_b (AVAX)
        inf_df = self.dp.get_pair_dataframe(pair=self.pair_b, timeframe=self.timeframe)

        if inf_df.empty or len(inf_df) < self.lookback + 10:
            dataframe["zscore"] = 0.0
            return dataframe

        # Align by date — merge informative into main dataframe
        inf_df = inf_df[["date", "close"]].rename(columns={"close": "close_b"})
        dataframe = dataframe.merge(inf_df, on="date", how="left")
        dataframe["close_b"] = dataframe["close_b"].ffill().fillna(method="bfill")

        if dataframe["close_b"].isna().all():
            dataframe["zscore"] = 0.0
            return dataframe

        # Log prices
        log_a = np.log(dataframe["close"].astype(float).values + 1e-10)
        log_b = np.log(dataframe["close_b"].astype(float).values + 1e-10)

        # Rolling OLS hedge ratio: log_a = beta * log_b + alpha
        n = len(dataframe)
        hedge_ratios = np.full(n, np.nan)
        for i in range(self.lookback, n):
            la = log_a[i - self.lookback:i + 1]
            lb = log_b[i - self.lookback:i + 1]
            try:
                X = np.column_stack([lb, np.ones(len(lb))])
                beta = np.linalg.lstsq(X, la, rcond=None)[0][0]
                hedge_ratios[i] = beta
            except Exception:
                pass

        # Spread and z-score
        spread = log_a - hedge_ratios * log_b
        spread_series = DataFrame({"s": spread})["s"]
        spread_mean = spread_series.rolling(self.lookback).mean().values
        spread_std = spread_series.rolling(self.lookback).std().values + 1e-10
        zscore = (spread - spread_mean) / spread_std

        dataframe["spread"] = spread
        dataframe["zscore"] = zscore
        dataframe["hedge_ratio"] = hedge_ratios

        # Volume filter
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Buy ADA when it's cheap relative to AVAX (zscore < -entry_z)."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["zscore"] < -self.entry_z)
            & (dataframe["zscore"].notna())
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.3),
            ["enter_long", "enter_tag"],
        ] = (1, "pairs_cheap")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Sell ADA when spread reverts to mean (|zscore| < exit_z)."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["zscore"].abs() < self.exit_z)  # reversion
                | (dataframe["zscore"] > self.entry_z)  # spread reversed
            ),
            "exit_long",
        ] = 1
        return dataframe
