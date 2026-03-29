"""RSI+BB Mean Reversion adapted for 5-minute timeframe.

Same logic as the 1H version that passed walk-forward,
but with parameters scaled for 5m bars:
- Faster RSI (7 instead of 14)
- Tighter BB (1.5 std instead of 2.0)
- Much tighter TP/SL (0.5% / 0.8%)
- Quick trades: in/out within 1-3 hours
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


class RSIBB5m(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.008, "30": 0.004, "60": 0.002}
    stoploss = -0.008
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.005
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 50
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fast RSI (7-period for 5m)
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(7).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi_7"] = 100 - (100 / (1 + rs))

        # Tight Bollinger Bands
        sma = dataframe["close"].rolling(20).mean()
        std = dataframe["close"].rolling(20).std()
        dataframe["bb_upper"] = sma + 1.5 * std
        dataframe["bb_middle"] = sma
        dataframe["bb_lower"] = sma - 1.5 * std

        # Volume filter
        dataframe["vol_sma"] = dataframe["volume"].rolling(30).mean()

        # Momentum
        dataframe["roc_12"] = dataframe["close"].pct_change(12) * 100

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi_7"] < 25)
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.8),
            ["enter_long", "enter_tag"],
        ] = (1, "5m_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] > dataframe["bb_middle"]) | (dataframe["rsi_7"] > 55)),
            "exit_long",
        ] = 1
        return dataframe
