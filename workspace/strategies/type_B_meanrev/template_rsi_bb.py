"""Template: RSI + Bollinger Bands Mean Reversion. Timeframe: 1h/4h."""
from __future__ import annotations
import sys; from pathlib import Path; from pandas import DataFrame
_R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_R/"src")); sys.path.insert(0, str(_R))
from freqtrade.strategy import IStrategy

class RSIBBMeanRev(IStrategy):
    timeframe = "1h"; minimal_roi = {"0": 0.08, "120": 0.03}; stoploss = -0.04
    trailing_stop = True; trailing_stop_positive = 0.01; trailing_stop_positive_offset = 0.02
    startup_candle_count = 30; can_short = False; use_exit_signal = True

    # Adaptive: override these via adaptive_params.optimize_directional()
    rsi_period = 14; bb_period = 20; bb_std = 2.0; rsi_entry = 35; rsi_exit = 65

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        sma = df["close"].rolling(self.bb_period).mean()
        std = df["close"].rolling(self.bb_period).std()
        df["bb_lower"] = sma - self.bb_std * std
        df["bb_middle"] = sma
        df["vol_sma"] = df["volume"].rolling(20).mean()
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] < df["bb_lower"]) & (df["rsi"] < self.rsi_entry) & (df["volume"] > df["vol_sma"] * 0.5), ["enter_long", "enter_tag"]] = (1, "oversold")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] > df["bb_middle"]) | (df["rsi"] > self.rsi_exit), "exit_long"] = 1
        return df
