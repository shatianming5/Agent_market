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


class AutoStrategy_v3(IStrategy):
    timeframe = "1h"
    can_short = False

    process_only_new_candles = True
    startup_candle_count = 240

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 0.012, "48": 0.006, "120": 0}
    stoploss = -0.08

    @staticmethod
    def _ema(s: DataFrame, span: int):
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: DataFrame, period: int = 14):
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(period)
        roll_up = up.ewm(alpha=alpha, adjust=False).mean()
        roll_down = down.ewm(alpha=alpha, adjust=False).mean()
        rs = roll_up / (roll_down.replace(0.0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _atr(df: DataFrame, period: int = 14):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        return tr.ewm(alpha=1.0 / float(period), adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        vol = dataframe["volume"]

        dataframe["ret_1h"] = close.pct_change().fillna(0.0)
        dataframe["ema20"] = self._ema(close, 20)
        dataframe["ema50"] = self._ema(close, 50)
        dataframe["ema100"] = self._ema(close, 100)
        dataframe["ema200"] = self._ema(close, 200)

        ema200 = dataframe["ema200"]
        dataframe["ema200_slope"] = (ema200 - ema200.shift(24)) / ema200.replace(0.0, np.nan)
        dataframe["rsi14"] = self._rsi(close, 14)
        dataframe["atr14"] = self._atr(dataframe, 14)

        mid = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        dataframe["bb_mid"] = mid
        dataframe["bb_upper"] = mid + 2.0 * std
        dataframe["bb_lower"] = mid - 2.0 * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / mid.replace(0.0, np.nan)

        vma = vol.rolling(20).mean()
        dataframe["volume_ratio"] = (vol / vma.replace(0.0, np.nan)).fillna(1.0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""

        close = dataframe["close"]
        rsi = dataframe["rsi14"]
        atr = dataframe["atr14"]
        ema20 = dataframe["ema20"]
        ema200 = dataframe["ema200"]
        slope = dataframe["ema200_slope"]
        vr = dataframe["volume_ratio"]

        volume_ok = vr > 0.55
        regime_ok = (slope > -0.0045) & (close > ema200 * 0.93)
        regime_soft = close > ema200 * 0.90

        cond_bb = (rsi < 43.0) & (close <= dataframe["bb_lower"] * 1.002)
        cond_kelt = (rsi < 48.0) & (close < (ema20 - 0.85 * atr))
        cond_dump = (dataframe["ret_1h"] < -0.012) & (rsi < 50.0) & (vr > 1.2) & regime_soft

        enter = volume_ok & ((regime_ok & (cond_bb | cond_kelt)) | cond_dump)
        dataframe.loc[enter, "enter_long"] = 1
        dataframe.loc[enter & cond_dump, "enter_tag"] = "dump_rebound"
        dataframe.loc[enter & ~cond_dump & cond_bb, "enter_tag"] = "bb_oversold"
        dataframe.loc[enter & ~cond_dump & ~cond_bb, "enter_tag"] = "atr_pullback"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""

        close = dataframe["close"]
        rsi = dataframe["rsi14"]
        atr = dataframe["atr14"]
        ema20 = dataframe["ema20"]
        ema50 = dataframe["ema50"]
        bb_mid = dataframe["bb_mid"]
        bb_upper = dataframe["bb_upper"]

        cross_mid_up = (close > bb_mid) & (close.shift(1) <= bb_mid.shift(1))

        take_profit = ((rsi > 55.0) & (close > ema20)) | (cross_mid_up & (rsi > 50.0)) | ((close > bb_upper * 0.998) & (rsi > 60.0))
        stop_guard = ((close < ema50) & (rsi < 40.0)) | (close < (ema20 - 1.5 * atr))

        exit_sig = take_profit | stop_guard
        dataframe.loc[exit_sig, "exit_long"] = 1
        dataframe.loc[stop_guard, "exit_tag"] = "risk_off"
        dataframe.loc[exit_sig & ~stop_guard, "exit_tag"] = "reversion_exit"

        return dataframe