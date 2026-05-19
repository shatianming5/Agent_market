"""Long-Short RSI + Bollinger Bands Strategy.

Core idea: RSI+BB mean reversion works for longs (oversold bounce).
Mirror it for shorts: overbought + above upper BB = short entry.
In a -31% bear market, short side should capture the downtrend.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pandas import DataFrame

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from freqtrade.strategy import IStrategy


class LongShortRSIBB(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03, "360": 0.01}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 30
    can_short = True  # ENABLE SHORT SELLING

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0

    # Long: oversold entry
    rsi_long_entry = 35
    rsi_long_exit = 60

    # Short: overbought entry
    rsi_short_entry = 65
    rsi_short_exit = 40

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

        # Trend filter: EMA slope
        dataframe["ema_50"] = dataframe["close"].ewm(span=50).mean()
        dataframe["ema_slope"] = dataframe["ema_50"].pct_change(12)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG: oversold + below lower BB
        long_cond = (
            (dataframe["volume"] > 0)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < self.rsi_long_entry)
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.5)
        )
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "bb_oversold_long")

        # SHORT: overbought + above upper BB + downtrend
        short_cond = (
            (dataframe["volume"] > 0)
            & (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["rsi"] > self.rsi_short_entry)
            & (dataframe["ema_slope"] < 0)  # only short in downtrend
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.5)
        )
        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "bb_overbought_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long: back to middle band or RSI recovering
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] > dataframe["bb_middle"]) | (dataframe["rsi"] > self.rsi_long_exit)),
            "exit_long",
        ] = 1

        # Exit short: back to middle band or RSI dropping
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["close"] < dataframe["bb_middle"]) | (dataframe["rsi"] < self.rsi_short_exit)),
            "exit_short",
        ] = 1

        return dataframe
