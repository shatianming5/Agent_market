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


class AutoStrategy_v103(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    minimal_roi = {
        "0": 0.015,
        "24": 0.010,
        "48": 0.0,
    }
    stoploss = -0.025

    @staticmethod
    def _ema(s, period: int):
        return s.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def _rsi(close, period: int):
        d = close.diff()
        up = d.clip(lower=0.0)
        dn = (-d).clip(lower=0.0)
        au = up.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        ad = dn.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
        rs = au / (ad + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(df: DataFrame, period: int):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        return tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"]
        vol = dataframe["volume"]

        mid = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std(ddof=0)
        lower = mid - 2.0 * std
        upper = mid + 2.0 * std
        dataframe["bb_mid"] = mid
        dataframe["bb_lower"] = lower
        dataframe["bb_upper"] = upper
        dataframe["bb_width"] = (upper - lower) / (mid + 1e-12)
        dataframe["z"] = (close - mid) / (std + 1e-12)

        dataframe["rsi14"] = self._rsi(close, 14)
        dataframe["rsi2"] = self._rsi(close, 2)

        dataframe["ema20"] = self._ema(close, 20)
        dataframe["ema50"] = self._ema(close, 50)
        dataframe["ema200"] = self._ema(close, 200)
        ema200 = dataframe["ema200"]
        dataframe["ema200_slope_24"] = (ema200 / (ema200.shift(24) + 1e-12)) - 1.0

        atr14 = self._atr(dataframe, 14)
        dataframe["atr14"] = atr14
        dataframe["atr_pct"] = atr14 / (close + 1e-12)

        vma20 = vol.rolling(20, min_periods=20).mean()
        dataframe["vol_ratio"] = vol / (vma20 + 1e-12)

        dataframe["ret1"] = close.pct_change(1)
        dataframe["ret3"] = close.pct_change(3)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol = dataframe["volume"]
        close = dataframe["close"]

        base = vol > 0

        oversold = (
            (dataframe["ret1"] <= -0.010)
            | (close <= dataframe["bb_lower"] * 0.998)
            | (dataframe["z"] <= -1.8)
        )

        rsi_ok = (dataframe["rsi14"] < 42.0) | (dataframe["rsi2"] < 18.0)

        vol_ok = dataframe["vol_ratio"] >= 0.80

        atr_ok = dataframe["atr_pct"] >= 0.0018

        rebound = (close > dataframe["open"]) | (dataframe["rsi2"] > dataframe["rsi2"].shift(1))

        trend_ok = (dataframe["ema200_slope_24"] > -0.020) | (close > dataframe["ema200"] * 0.88)

        enter = (base & oversold & rsi_ok & vol_ok & atr_ok & rebound & trend_ok).fillna(False)

        dataframe.loc[enter, "enter_long"] = 1
        dataframe.loc[enter, "enter_tag"] = "capitulation_rebound"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol = dataframe["volume"]
        close = dataframe["close"]

        base = vol > 0

        mean_revert = (close >= dataframe["bb_mid"]) | (dataframe["ret3"] >= 0.010)
        momentum_cool = (dataframe["rsi14"] >= 55.0) | (close >= dataframe["ema50"])

        exit_cond = (base & mean_revert & momentum_cool).fillna(False)

        dataframe.loc[exit_cond, "exit_long"] = 1
        dataframe.loc[exit_cond, "exit_tag"] = "mean_revert_takeprofit"
        return dataframe