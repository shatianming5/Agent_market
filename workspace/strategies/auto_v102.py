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


class AutoStrategy_v102(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.018, "36": 0.012, "72": 0.0}  # wide TP, time-decay
    stoploss = -0.027  # tight 2-3% stop
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01  # do not exit on tiny bounces (<1%)
    ignore_roi_if_entry_signal = False

    @staticmethod
    def _ema(s, n: int):
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rsi(close, n: int = 14):
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
        roll_down = down.ewm(alpha=1.0 / n, adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _atr(df: DataFrame, n: int = 14):
        prev_close = df["close"].shift(1)
        tr1 = (df["high"] - df["low"]).abs()
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = np.maximum(tr1.to_numpy(), np.maximum(tr2.to_numpy(), tr3.to_numpy()))
        tr = DataFrame({"tr": tr}, index=df.index)["tr"]
        return tr.ewm(alpha=1.0 / n, adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"]
        dataframe["ema20"] = self._ema(close, 20)
        dataframe["ema50"] = self._ema(close, 50)
        dataframe["rsi14"] = self._rsi(close, 14)

        mid = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        dataframe["bb_mid"] = mid
        dataframe["bb_upper"] = mid + 2.0 * std
        dataframe["bb_lower"] = mid - 2.0 * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]

        dataframe["atr14"] = self._atr(dataframe, 14)

        vol_sma20 = dataframe["volume"].rolling(20).mean()
        dataframe["vol_sma20"] = vol_sma20
        dataframe["volume_ratio"] = dataframe["volume"] / vol_sma20

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        rsi = dataframe["rsi14"]
        bb_lower = dataframe["bb_lower"]
        bb_width = dataframe["bb_width"]
        ema50 = dataframe["ema50"]
        vol_ratio = dataframe["volume_ratio"]

        rebound = (dataframe["close"] > dataframe["open"]) | (dataframe["close"] > dataframe["close"].shift(1))
        oversold_extreme = (rsi < 31.5) & (rsi > rsi.shift(1))
        below_band = dataframe["close"] < (bb_lower * 1.006)
        regime_ok = dataframe["close"] > (ema50 * 0.985)  # relaxed filter to avoid worst downswings
        vol_ok = (dataframe["volume"] > 0) & (vol_ratio > 0.6)
        volat_ok = bb_width > 0.006  # avoid ultra-low vol scalps

        cond = vol_ok & volat_ok & regime_ok & below_band & oversold_extreme & rebound
        dataframe.loc[cond, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        rsi = dataframe["rsi14"]
        exit_mean = (dataframe["close"] >= dataframe["bb_mid"]) & (rsi > 50.0)
        exit_over = rsi > 60.0
        exit_spike = dataframe["close"] >= dataframe["bb_upper"]

        cond = (dataframe["volume"] > 0) & (exit_spike | exit_over | exit_mean)
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe