"""Example: RSI+BB Mean Reversion — copy and modify."""
from __future__ import annotations
import sys
from pathlib import Path
from pandas import DataFrame
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(_ROOT).parent / "src"))
sys.path.insert(0, str(Path(_ROOT).parent))
from freqtrade.strategy import IStrategy

class ExampleRSIBB(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.08, "120": 0.03}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    use_exit_signal = True
    startup_candle_count = 30
    can_short = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_lower"] = sma - 2.0 * std
        df["bb_middle"] = sma
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] < df["bb_lower"]) & (df["rsi"] < 35), ["enter_long", "enter_tag"]] = (1, "oversold")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["close"] > df["bb_middle"]) | (df["rsi"] > 65), "exit_long"] = 1
        return df
