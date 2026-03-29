"""RSI + Bollinger Bands Mean Reversion Strategy.

Hypothesis: When price touches the lower Bollinger Band AND RSI is oversold,
price tends to revert to the mean. Exit when price reaches the middle band
or RSI becomes overbought.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pandas import DataFrame

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from freqtrade.strategy import IStrategy


class RSIBBMeanReversion(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03, "360": 0.01}
    stoploss = -0.04
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 30
    can_short = False

    # Parameters
    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    rsi_buy_threshold = 35
    rsi_sell_threshold = 65

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma = dataframe["close"].rolling(self.bb_period).mean()
        std = dataframe["close"].rolling(self.bb_period).std()
        dataframe["bb_upper"] = sma + self.bb_std * std
        dataframe["bb_middle"] = sma
        dataframe["bb_lower"] = sma - self.bb_std * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (sma + 1e-10)

        # Volume filter
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < self.rsi_buy_threshold)
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.5),
            ["enter_long", "enter_tag"],
        ] = (1, "rsi_bb_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] > dataframe["bb_middle"])
                | (dataframe["rsi"] > self.rsi_sell_threshold)
            ),
            "exit_long",
        ] = 1
        return dataframe
