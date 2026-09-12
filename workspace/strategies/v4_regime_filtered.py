"""V4 + Regime Filter — only trades in bearish/falling regimes.

Based on analysis: v4 earns +1.21% in mild decline, +0.32% in crash,
but LOSES in rebounds. Solution: disable entries when market is rising.
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


class V4RegimeFiltered(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03, "360": 0.01}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 55
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # BB
        sma = dataframe["close"].rolling(20).mean()
        std = dataframe["close"].rolling(20).std()
        dataframe["bb_lower"] = sma - 2.0 * std
        dataframe["bb_middle"] = sma

        # v4's RSI divergence
        dataframe["rsi_up"] = dataframe["rsi"] > dataframe["rsi"].shift(1)
        dataframe["rsi_prev_low"] = dataframe["rsi"].shift(1) < dataframe["rsi"].shift(2)
        dataframe["price_down"] = dataframe["close"] < dataframe["close"].shift(1)
        dataframe["divergence"] = dataframe["rsi_up"] & dataframe["rsi_prev_low"] & dataframe["price_down"]

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # === REGIME FILTER ===
        # 20-bar return: negative = we're in a decline (where v4 works)
        dataframe["ret_20"] = dataframe["close"].pct_change(20)
        # 50-bar EMA slope: negative = downtrend
        dataframe["ema_50"] = dataframe["close"].ewm(span=50).mean()
        dataframe["ema_slope"] = dataframe["ema_50"].pct_change(10)

        # Regime: bearish when both ret_20 < 0 AND ema slope < 0
        # Or: extreme selling (many RSI < 35 recently)
        dataframe["rsi_below_35_count"] = (dataframe["rsi"] < 35).rolling(48).sum()

        dataframe["is_bearish"] = (
            (dataframe["ret_20"] < -0.01)  # market fell > 1% in last 20 bars
            | (dataframe["ema_slope"] < -0.001)  # EMA trending down
            | (dataframe["rsi_below_35_count"] > 5)  # lots of oversold readings recently
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # v4 core signal: BB lower + RSI divergence
        v4_signal = (
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < 40)
            & (dataframe["divergence"] | (dataframe["rsi"] < 28))
        )

        dataframe.loc[
            (dataframe["volume"] > 0)
            & v4_signal
            & dataframe["is_bearish"]  # ONLY in bearish regime
            & (dataframe["volume"] > dataframe["vol_sma"] * 0.5),
            ["enter_long", "enter_tag"],
        ] = (1, "v4_bear_regime")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] > dataframe["bb_middle"])
                | (dataframe["rsi"] > 62)
            ),
            "exit_long",
        ] = 1
        return dataframe
