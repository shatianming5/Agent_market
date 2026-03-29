"""Inverse Short Strategy — validates short-side alpha in spot mode.

Since freqtrade spot mode doesn't support can_short=True,
this strategy INVERTS the short signal: it buys when the original
strategy would short (overbought → expecting drop, but we BUY the dip
that follows).

Alternatively: this acts as a "sell the rip" strategy — enters long
at overbought levels expecting mean reversion back DOWN, with very
tight stop-loss. In a bear market, the short-term bounces after
overbought can still be caught.

More practically: this strategy enters when price DROPS from
overbought levels (RSI was >65, now declining) — catching the
beginning of the mean reversion move downward by going long on
the initial dead-cat bounce.
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


class InverseShort(IStrategy):
    """Contrarian: buy extreme weakness expecting dead-cat bounce."""
    timeframe = "1h"
    minimal_roi = {"0": 0.015, "60": 0.005}  # very tight TP — quick scalp
    stoploss = -0.02  # tight stop
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 50
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
        dataframe["bb_lower"] = sma - 2 * std
        dataframe["bb_upper"] = sma + 2 * std
        dataframe["bb_middle"] = sma

        # Rate of change (momentum)
        dataframe["roc_6"] = dataframe["close"].pct_change(6) * 100

        # ATR for volatility filter
        tr = np.maximum(
            dataframe["high"] - dataframe["low"],
            np.maximum(
                abs(dataframe["high"] - dataframe["close"].shift(1)),
                abs(dataframe["low"] - dataframe["close"].shift(1)),
            ),
        )
        dataframe["atr"] = tr.rolling(14).mean()
        dataframe["atr_pct"] = dataframe["atr"] / (dataframe["close"] + 1e-10)

        # Volume
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy after sharp drop (dead-cat bounce entry)
        # Condition: RSI was very low (extreme selling), now ticking up
        rsi_recovering = (dataframe["rsi"] > dataframe["rsi"].shift(1)) & (dataframe["rsi"].shift(1) < 25)

        # Or: price crashed below BB lower by a wide margin
        extreme_dip = (dataframe["close"] < dataframe["bb_lower"]) & (dataframe["roc_6"] < -3)

        # Volume spike (panic selling = potential bounce)
        vol_spike = dataframe["volume"] > dataframe["vol_sma"] * 1.5

        dataframe.loc[
            (dataframe["volume"] > 0)
            & (rsi_recovering | extreme_dip)
            & (dataframe["atr_pct"] > 0.002),  # need some volatility
            ["enter_long", "enter_tag"],
        ] = (1, "dead_cat_bounce")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Quick exit at any sign of resistance
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (
                (dataframe["rsi"] > 55)  # quick RSI recovery
                | (dataframe["close"] > dataframe["bb_middle"])  # back to mean
            ),
            "exit_long",
        ] = 1
        return dataframe
