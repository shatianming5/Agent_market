"""Conservative Trend Filter Strategy.

Hypothesis: In a choppy/bearish market, fewer high-quality trades beat
many low-quality ones. Enter ONLY when multiple confirmations align:
1. EMA alignment (fast > medium > slow = uptrend)
2. RSI in bullish zone (50-70, not overbought)
3. Volume above average (smart money participating)
4. Price above VWAP-like indicator (institutional support)

Use ATR-based trailing stop to let winners run.
Target: <30 trades, high PF, controlled DD.
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


class ConservativeTrend(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.20, "360": 0.05, "720": 0.02}
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.025
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Triple EMA alignment
        dataframe["ema_8"] = dataframe["close"].ewm(span=8).mean()
        dataframe["ema_21"] = dataframe["close"].ewm(span=21).mean()
        dataframe["ema_55"] = dataframe["close"].ewm(span=55).mean()

        # RSI
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # ATR for volatility
        tr = np.maximum(
            dataframe["high"] - dataframe["low"],
            np.maximum(
                abs(dataframe["high"] - dataframe["close"].shift(1)),
                abs(dataframe["low"] - dataframe["close"].shift(1)),
            ),
        )
        dataframe["atr_14"] = tr.rolling(14).mean()

        # Volume filter
        dataframe["vol_sma_30"] = dataframe["volume"].rolling(30).mean()

        # VWAP approximation (cumulative volume-weighted price)
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["vwap_20"] = (
            (typical * dataframe["volume"]).rolling(20).sum()
            / (dataframe["volume"].rolling(20).sum() + 1e-10)
        )

        # Trend strength: distance between fast and slow EMA normalized by ATR
        dataframe["trend_strength"] = (
            (dataframe["ema_8"] - dataframe["ema_55"]) / (dataframe["atr_14"] + 1e-10)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            # EMA alignment: fast > medium > slow
            & (dataframe["ema_8"] > dataframe["ema_21"])
            & (dataframe["ema_21"] > dataframe["ema_55"])
            # RSI in bullish but not overbought zone
            & (dataframe["rsi"] > 50)
            & (dataframe["rsi"] < 70)
            # Volume confirmation
            & (dataframe["volume"] > dataframe["vol_sma_30"] * 1.1)
            # Price above VWAP
            & (dataframe["close"] > dataframe["vwap_20"])
            # Meaningful trend (not just noise)
            & (dataframe["trend_strength"] > 0.5),
            ["enter_long", "enter_tag"],
        ] = (1, "conservative_trend")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                # EMA crossover down
                (dataframe["ema_8"] < dataframe["ema_21"])
                # RSI overbought
                | (dataframe["rsi"] > 78)
                # Trend weakening significantly
                | (dataframe["trend_strength"] < -0.3)
            ),
            "exit_long",
        ] = 1
        return dataframe
