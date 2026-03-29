"""Momentum Breakout Strategy.

Hypothesis: When price breaks above the Donchian channel high with increasing
volume and positive momentum (ADX > 20), it signals a trend continuation.
Exit when momentum fades or price drops below EMA.
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


class MomentumBreakout(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.15, "180": 0.05, "480": 0.02}
    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 50
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian Channel
        dataframe["dc_high_20"] = dataframe["high"].rolling(20).max()
        dataframe["dc_low_20"] = dataframe["low"].rolling(20).min()
        dataframe["dc_mid"] = (dataframe["dc_high_20"] + dataframe["dc_low_20"]) / 2

        # EMA
        dataframe["ema_12"] = dataframe["close"].ewm(span=12).mean()
        dataframe["ema_26"] = dataframe["close"].ewm(span=26).mean()

        # MACD-like momentum
        dataframe["momentum"] = dataframe["ema_12"] - dataframe["ema_26"]
        dataframe["momentum_signal"] = dataframe["momentum"].ewm(span=9).mean()

        # Volume
        dataframe["vol_sma_20"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / (dataframe["vol_sma_20"] + 1e-10)

        # ADX approximation (using directional movement)
        high_diff = dataframe["high"].diff()
        low_diff = -dataframe["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
        tr = np.maximum(
            dataframe["high"] - dataframe["low"],
            np.maximum(
                abs(dataframe["high"] - dataframe["close"].shift(1)),
                abs(dataframe["low"] - dataframe["close"].shift(1)),
            ),
        )
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_14 + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_14 + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        dataframe["adx"] = dx.rolling(14).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] > dataframe["dc_high_20"].shift(1))  # breakout
            & (dataframe["adx"] > 20)  # trending
            & (dataframe["momentum"] > dataframe["momentum_signal"])  # momentum positive
            & (dataframe["vol_ratio"] > 1.0),  # above-average volume
            ["enter_long", "enter_tag"],
        ] = (1, "momentum_breakout")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] < dataframe["ema_26"])  # below slow EMA
                | (dataframe["momentum"] < dataframe["momentum_signal"])  # momentum fading
            ),
            "exit_long",
        ] = 1
        return dataframe
