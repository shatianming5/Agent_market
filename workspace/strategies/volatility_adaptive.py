"""Volatility-Adaptive Strategy.

Hypothesis: Adjust entry/exit thresholds based on current volatility regime.
In low-vol: tighter entries (mean-reversion). In high-vol: wider stops (trend-following).
Uses ATR for position sizing signals and Keltner channels for entries.
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


class VolatilityAdaptive(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.12, "240": 0.04, "720": 0.01}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 50
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR
        tr = np.maximum(
            dataframe["high"] - dataframe["low"],
            np.maximum(
                abs(dataframe["high"] - dataframe["close"].shift(1)),
                abs(dataframe["low"] - dataframe["close"].shift(1)),
            ),
        )
        dataframe["atr_14"] = tr.rolling(14).mean()
        dataframe["atr_pct"] = dataframe["atr_14"] / (dataframe["close"] + 1e-10) * 100

        # Volatility regime: compare short vs long ATR
        dataframe["atr_7"] = tr.rolling(7).mean()
        dataframe["atr_28"] = tr.rolling(28).mean()
        dataframe["vol_regime"] = dataframe["atr_7"] / (dataframe["atr_28"] + 1e-10)

        # Keltner Channel (EMA + ATR)
        dataframe["ema_20"] = dataframe["close"].ewm(span=20).mean()
        dataframe["kc_upper"] = dataframe["ema_20"] + 1.5 * dataframe["atr_14"]
        dataframe["kc_lower"] = dataframe["ema_20"] - 1.5 * dataframe["atr_14"]

        # RSI for confirmation
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # Price relative to EMA
        dataframe["price_vs_ema"] = (dataframe["close"] - dataframe["ema_20"]) / (dataframe["atr_14"] + 1e-10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Low volatility: mean reversion at Keltner lower band
        low_vol_entry = (
            (dataframe["vol_regime"] < 1.0)
            & (dataframe["close"] < dataframe["kc_lower"])
            & (dataframe["rsi"] < 40)
        )
        # High volatility: breakout above Keltner upper band
        high_vol_entry = (
            (dataframe["vol_regime"] >= 1.0)
            & (dataframe["close"] > dataframe["kc_upper"])
            & (dataframe["rsi"] > 50)
            & (dataframe["rsi"] < 80)
        )

        dataframe.loc[
            (dataframe["volume"] > 0) & low_vol_entry,
            ["enter_long", "enter_tag"],
        ] = (1, "low_vol_reversion")

        dataframe.loc[
            (dataframe["volume"] > 0) & high_vol_entry,
            ["enter_long", "enter_tag"],
        ] = (1, "high_vol_breakout")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                # Mean-reversion exit: back to EMA
                ((dataframe["price_vs_ema"] > 0.5) & (dataframe["vol_regime"] < 1.0))
                # Trend exit: momentum fading
                | ((dataframe["rsi"] > 75) & (dataframe["vol_regime"] >= 1.0))
                | (dataframe["close"] < dataframe["ema_20"] - 2 * dataframe["atr_14"])
            ),
            "exit_long",
        ] = 1
        return dataframe
