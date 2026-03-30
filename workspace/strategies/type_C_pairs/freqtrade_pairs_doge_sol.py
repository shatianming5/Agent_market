"""Freqtrade L2 strategy: DOGE relative to SOL."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from pandas import DataFrame
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from freqtrade.strategy import IStrategy

class FreqtradePairsDOGESOL(IStrategy):
    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.03, "120": 0.015, "360": 0.005}
    stoploss = -0.035
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.015
    use_exit_signal = True
    startup_candle_count = 100
    process_only_new_candles = True

    pair_b = "SOL/USDT"
    lookback = 80
    entry_z = 3.0
    exit_z = 0.5

    def informative_pairs(self):
        return [(self.pair_b, self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        inf = self.dp.get_pair_dataframe(pair=self.pair_b, timeframe=self.timeframe)
        if inf.empty or len(inf) < self.lookback + 10:
            dataframe["zscore"] = 0.0
            return dataframe
        inf = inf[["date", "close"]].rename(columns={"close": "close_b"})
        dataframe = dataframe.merge(inf, on="date", how="left")
        dataframe["close_b"] = dataframe["close_b"].ffill().fillna(0)
        if (dataframe["close_b"] == 0).all():
            dataframe["zscore"] = 0.0
            return dataframe

        log_a = np.log(dataframe["close"].astype(float).values + 1e-10)
        log_b = np.log(dataframe["close_b"].astype(float).values + 1e-10)
        n = len(dataframe)
        hr = np.full(n, np.nan)
        for i in range(self.lookback, n):
            la = log_a[i-self.lookback:i+1]
            lb = log_b[i-self.lookback:i+1]
            try:
                X = np.column_stack([lb, np.ones(len(lb))])
                hr[i] = np.linalg.lstsq(X, la, rcond=None)[0][0]
            except: pass
        spread = log_a - hr * log_b
        s = DataFrame({"s": spread})["s"]
        sm = s.rolling(self.lookback).mean().values
        ss = s.rolling(self.lookback).std().values + 1e-10
        dataframe["zscore"] = (spread - sm) / ss
        dataframe["vol_sma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["zscore"] < -self.entry_z)
            & (dataframe["zscore"].notna()),
            ["enter_long", "enter_tag"],
        ] = (1, "pairs_cheap")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["volume"] > 0)
            & ((dataframe["zscore"].abs() < self.exit_z) | (dataframe["zscore"] > self.entry_z)),
            "exit_long",
        ] = 1
        return dataframe
