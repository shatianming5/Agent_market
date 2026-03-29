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

import pandas as pd


class AutoStrategy_v2(IStrategy):
    timeframe = "1h"
    can_short = False
    startup_candle_count = 240

    minimal_roi = {"0": 0.012, "24": 0.006, "72": 0.0}
    stoploss = -0.035
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(period)
        roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _atr(df: DataFrame, period: int) -> pd.Series:
        h = df["high"]
        l = df["low"]
        c = df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        alpha = 1.0 / float(period)
        return tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    @staticmethod
    def _bbands(close: pd.Series, period: int = 20, stds: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = close.rolling(period, min_periods=period).mean()
        sd = close.rolling(period, min_periods=period).std(ddof=0)
        upper = mid + stds * sd
        lower = mid - stds * sd
        return lower, mid, upper

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema20"] = self._ema(df["close"], 20)
        df["ema50"] = self._ema(df["close"], 50)
        df["ema200"] = self._ema(df["close"], 200)

        df["rsi14"] = self._rsi(df["close"], 14)
        df["rsi2"] = self._rsi(df["close"], 2)

        df["atr14"] = self._atr(df, 14)

        bb_low, bb_mid, bb_high = self._bbands(df["close"], 20, 2.0)
        df["bb_low"] = bb_low
        df["bb_mid"] = bb_mid
        df["bb_high"] = bb_high
        df["bb_width"] = ((bb_high - bb_low) / bb_mid.replace(0.0, np.nan)).fillna(0.0)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma.replace(0.0, np.nan)).fillna(1.0)

        df["ema200_slope"] = (df["ema200"] / df["ema200"].shift(24) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        trend_ok = (
            (df["close"] > df["ema200"] * 0.992) &
            (df["ema50"] > df["ema200"] * 0.992) &
            (df["ema200_slope"] > -0.0015)
        )

        pullback_fast = (
            (df["rsi2"] < 30.0) &
            ((df["close"] < df["ema20"]) | (df["close"] < df["ema20"] - 0.25 * df["atr14"]))
        )

        bb_reclaim = (
            (df["close"] < df["bb_low"] * 1.01) &
            (df["rsi14"] < 46.0)
        )

        range_meanrev = (
            (df["close"] > df["ema200"] * 0.982) &
            (df["bb_width"].between(0.004, 0.030)) &
            (df["close"] < df["bb_low"] * 1.008) &
            (df["rsi14"] < 49.0)
        )

        liq_ok = (df["vol_ratio"] > 0.45)

        enter = liq_ok & (range_meanrev | (trend_ok & (pullback_fast | bb_reclaim)))

        df.loc[enter, "enter_long"] = 1
        df.loc[enter, "buy"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        take_profit = (
            ((df["close"] > df["bb_mid"]) & (df["rsi14"] > 52.0)) |
            (df["close"] > df["bb_high"] * 0.995) |
            (df["rsi2"] > 85.0)
        )

        risk_off = (
            ((df["close"] < df["ema200"] * 0.985) & (df["rsi14"] < 36.0)) |
            (df["close"] < df["bb_low"] * 0.992)
        )

        exit_ = take_profit | risk_off

        df.loc[exit_, "exit_long"] = 1
        df.loc[exit_, "sell"] = 1
        return df