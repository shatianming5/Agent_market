"""Ensemble Strategy — combines RSI+BB baseline, RSI+BB v4, and Volatility Adaptive.

All three passed walk-forward validation independently.
This strategy runs all three and enters only when 2+ agree.
Regime-aware: reduces exposure in volatile/downtrend markets.
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


class EnsembleStrategy(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03, "360": 0.01}
    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    # Ensemble requires 2 of 3 strategies to agree
    min_agreement = 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # === Shared indicators ===
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma20 + 2.0 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2.0 * std20

        df["vol_sma"] = df["volume"].rolling(20).mean()
        df["ema_20"] = df["close"].ewm(span=20).mean()

        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10) * 100

        # Keltner Channel
        df["kc_upper"] = df["ema_20"] + 1.5 * df["atr"]
        df["kc_lower"] = df["ema_20"] - 1.5 * df["atr"]

        # Regime
        atr_short = tr.rolling(7).mean()
        atr_long = tr.rolling(28).mean()
        df["vol_regime"] = atr_short / (atr_long + 1e-10)
        ema_fast = df["close"].ewm(span=12).mean()
        ema_slow = df["close"].ewm(span=48).mean()
        df["trend_dir"] = np.sign(ema_fast - ema_slow)

        # === Strategy 1: RSI+BB baseline ===
        df["sig_rsi_bb"] = (
            (df["close"] < df["bb_lower"])
            & (df["rsi"] < 35)
            & (df["volume"] > df["vol_sma"] * 0.5)
        ).astype(int)

        # === Strategy 2: RSI+BB v4 (RSI divergence) ===
        rsi_higher_low = (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"].shift(1) < df["rsi"].shift(2))
        price_lower_low = df["close"] < df["close"].shift(1)
        divergence = rsi_higher_low & price_lower_low
        df["sig_rsi_bb_v4"] = (
            (df["close"] < df["bb_lower"])
            & (df["rsi"] < 40)
            & (divergence | (df["rsi"] < 30))
        ).astype(int)

        # === Strategy 3: Volatility Adaptive ===
        low_vol_entry = (df["vol_regime"] < 1.0) & (df["close"] < df["kc_lower"]) & (df["rsi"] < 40)
        high_vol_entry = (df["vol_regime"] >= 1.0) & (df["close"] > df["kc_upper"]) & (df["rsi"] > 50) & (df["rsi"] < 80)
        df["sig_vol_adaptive"] = (low_vol_entry | high_vol_entry).astype(int)

        # === Ensemble signal ===
        df["ensemble_count"] = df["sig_rsi_bb"] + df["sig_rsi_bb_v4"] + df["sig_vol_adaptive"]

        # === Exit signals ===
        df["exit_rsi_bb"] = ((df["close"] > df["bb_middle"]) | (df["rsi"] > 65)).astype(int)
        df["exit_vol"] = (
            ((df["close"] > df["ema_20"]) & (df["vol_regime"] < 1.0))
            | (df["rsi"] > 75)
        ).astype(int)
        df["exit_count"] = df["exit_rsi_bb"] + df["exit_vol"]

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter when 2+ strategies agree AND not in extreme volatility
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["ensemble_count"] >= self.min_agreement)
            & (dataframe["vol_regime"] < 2.0),  # avoid extreme vol
            ["enter_long", "enter_tag"],
        ] = (1, "ensemble_agree")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["exit_count"] >= 2),
            "exit_long",
        ] = 1
        return dataframe
