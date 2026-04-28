"""Template: EMA Crossover Trend Following."""
from __future__ import annotations
import sys; from pathlib import Path; import numpy as np; from pandas import DataFrame
_R = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_R/"src")); sys.path.insert(0, str(_R))
from freqtrade.strategy import IStrategy
class EMACrossTrend(IStrategy):
    timeframe = "1h"; minimal_roi = {"0": 0.15, "480": 0.05}; stoploss = -0.05
    trailing_stop = True; trailing_stop_positive = 0.02; startup_candle_count = 60; can_short = False
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["ema_fast"] = df["close"].ewm(span=12).mean()
        df["ema_slow"] = df["close"].ewm(span=26).mean()
        return df
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["ema_fast"]>df["ema_slow"])&(df["volume"]>0),["enter_long","enter_tag"]]=(1,"ema_up")
        return df
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[(df["ema_fast"]<df["ema_slow"])&(df["volume"]>0),"exit_long"]=1
        return df
