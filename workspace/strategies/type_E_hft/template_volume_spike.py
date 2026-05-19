"""Template: Volume Spike Scalper. Timeframe: 5m.
Enters on volume spike + price dip, exits quickly.
NOTE: High frequency — verify with L3 event-driven backtest."""
from __future__ import annotations
import sys; from pathlib import Path; from pandas import DataFrame
_R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_R/"src")); sys.path.insert(0, str(_R))
from freqtrade.strategy import IStrategy

class VolumeSpikeScalper(IStrategy):
    timeframe = "5m"; minimal_roi = {"0": 0.005, "15": 0.003}; stoploss = -0.005
    trailing_stop = True; trailing_stop_positive = 0.002; trailing_stop_positive_offset = 0.004
    startup_candle_count = 50; can_short = False; use_exit_signal = True

    vol_spike_mult = 3.0; rsi_entry = 30

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["vol_sma"] = df["volume"].rolling(30).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_sma"] + 1e-10)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        df["rsi_7"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df["price_drop"] = df["close"].pct_change(3) * 100
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["vol_ratio"] > self.vol_spike_mult) & (df["rsi_7"] < self.rsi_entry) & (df["price_drop"] < -0.3), ["enter_long", "enter_tag"]] = (1, "vol_spike_dip")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["rsi_7"] > 55) | (df["vol_ratio"] < 1.0), "exit_long"] = 1
        return df
