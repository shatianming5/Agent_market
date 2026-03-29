from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))


class AutoStrategy_v8(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.004, "24": 0.0}
    stoploss = -0.045
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 240

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "gtc", "exit": "gtc"}

    def _ema(self, s, period: int):
        return s.ewm(span=period, adjust=False, min_periods=period).mean()

    def _rsi(self, close, period: int = 14):
        d = close.diff()
        up = d.clip(lower=0.0)
        dn = (-d).clip(lower=0.0)
        au = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        ad = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = au / ad.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50.0)

    def _atr(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        tr = pd.Series(tr, index=df.index)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        tr = pd.Series(tr, index=df.index)
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_dm_s = pd.Series(plus_dm, index=df.index)
        minus_dm_s = pd.Series(minus_dm, index=df.index)
        plus_di = 100 * (plus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema_200"] = self._ema(df["close"], 200)
        df["ema_50"] = self._ema(df["close"], 50)

        bb_len, bb_std = 20, 2.0
        mid = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        df["bb_mid"] = mid
        df["bb_low"] = mid - bb_std * sd
        df["bb_up"] = mid + bb_std * sd

        df["rsi"] = self._rsi(df["close"], 14)
        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, 14)
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        v = df["volume"].astype(float)
        pv = (tp * v).rolling(24, min_periods=24).sum()
        vv = v.rolling(24, min_periods=24).sum()
        df["vwap_24"] = (pv / vv.replace(0, np.nan)).fillna(df["close"])

        rng = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_pos"] = ((df["close"] - df["low"]) / rng).fillna(0.5)

        df["under_bb_low"] = df["close"] < df["bb_low"]
        df["under_bb_low_2"] = df["under_bb_low"] & df["under_bb_low"].shift(1).fillna(False)

        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            idx_series = idx.to_series()
        else:
            idx_series = pd.to_datetime(idx, errors="coerce").to_series()

        if pd.api.types.is_datetime64_any_dtype(idx_series):
            df["age_36"] = idx_series.diff().dt.total_seconds().div(3600.0).fillna(1.0).cumsum()
        else:
            df["age_36"] = pd.Series(np.arange(len(df), dtype=float), index=df.index)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        regime_ok = (df["close"] > df["ema_200"] * 0.94) & (df["adx"] < 26) & (df["atr_pct"] > 0.0055)

        wick_flush = df["low"] < (df["bb_low"] - 0.55 * df["atr"])
        rsi_ext = df["rsi"] < 33
        vol_spike = df["vol_ratio"] > 1.25

        reclaim = (df["close"] > df["open"]) & (df["close"] > df["bb_low"])
        close_strong = df["close_pos"] > 0.55

        entry = regime_ok & wick_flush & rsi_ext & vol_spike & reclaim & close_strong

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp_mid = (df["close"] >= df["bb_mid"]) | (df["close"] >= df["vwap_24"])
        tp_rsi = (df["rsi"] > 56) & (df["close"] > df["ema_50"])
        tp_fast = tp_mid & tp_rsi

        breakdown = df["under_bb_low_2"] | (
            (df["adx"] > 28) & (df["close"] < df["ema_50"]) & (df["rsi"] < 45)
        )

        bar_36_ago = df["close"].shift(36)
        time_stop = (
            (bar_36_ago.notna())
            & (df["close"] < df["bb_mid"])
            & (df["close"] < df["ema_50"])
            & (df["rsi"] < 52)
        )

        exit_sig = tp_fast | breakdown | time_stop

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df