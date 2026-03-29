"""Multi-Timeframe RSI+BB — 1H entries confirmed by higher TF trend.

Uses 1H for entry signals (RSI+BB mean reversion).
Uses informative 4H-equivalent (4-bar rolling) for trend filter.
Only enters long when higher TF is not in strong downtrend.
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


class MultiTFRSIBB(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.06, "120": 0.02, "360": 0.005}
    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.015
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 1H signals (entry timeframe) ===
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        sma20 = dataframe["close"].rolling(20).mean()
        std20 = dataframe["close"].rolling(20).std()
        dataframe["bb_upper"] = sma20 + 2.0 * std20
        dataframe["bb_middle"] = sma20
        dataframe["bb_lower"] = sma20 - 2.0 * std20

        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        # RSI divergence (from v4)
        dataframe["rsi_up"] = dataframe["rsi"] > dataframe["rsi"].shift(1)
        dataframe["price_down"] = dataframe["close"] < dataframe["close"].shift(1)
        dataframe["bullish_divergence"] = dataframe["rsi_up"] & dataframe["price_down"]

        # === 4H-equivalent signals (trend filter) ===
        # Use 4-bar rolling to approximate 4H from 1H data
        dataframe["close_4h"] = dataframe["close"].rolling(4).mean()
        dataframe["ema_4h_20"] = dataframe["close_4h"].ewm(span=20).mean()
        dataframe["ema_4h_50"] = dataframe["close_4h"].ewm(span=50).mean()

        # 4H trend direction
        dataframe["trend_4h"] = np.where(
            dataframe["ema_4h_20"] > dataframe["ema_4h_50"], 1,
            np.where(dataframe["ema_4h_20"] < dataframe["ema_4h_50"], -1, 0)
        )

        # 4H momentum
        dataframe["roc_4h"] = dataframe["close_4h"].pct_change(20)

        # ATR for volatility regime
        tr = np.maximum(
            dataframe["high"] - dataframe["low"],
            np.maximum(
                abs(dataframe["high"] - dataframe["close"].shift(1)),
                abs(dataframe["low"] - dataframe["close"].shift(1)),
            ),
        )
        dataframe["atr"] = tr.rolling(14).mean()
        atr_fast = tr.rolling(7).mean()
        atr_slow = tr.rolling(28).mean()
        dataframe["squeeze"] = atr_fast < atr_slow  # low vol = potential breakout

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Primary: RSI+BB oversold + bullish divergence (proven in WF)
        primary = (
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < 38)
            & (dataframe["bullish_divergence"] | (dataframe["rsi"] < 28))
        )

        # Higher TF filter: don't buy into strong downtrend
        tf_filter = (
            (dataframe["trend_4h"] >= 0)  # uptrend or neutral
            | (dataframe["roc_4h"] > -0.05)  # or downtrend but slowing
        )

        # Volume confirmation
        vol_ok = dataframe["volume"] > dataframe["vol_sma"] * 0.6

        dataframe.loc[
            (dataframe["volume"] > 0) & primary & tf_filter & vol_ok,
            ["enter_long", "enter_tag"],
        ] = (1, "mtf_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["close"] > dataframe["bb_middle"])
                | (dataframe["rsi"] > 62)
                | ((dataframe["trend_4h"] == -1) & (dataframe["rsi"] > 50))  # quick exit in downtrend
            ),
            "exit_long",
        ] = 1
        return dataframe
