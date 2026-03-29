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


class AutoStrategy_v9(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.015}
    stoploss = -0.045
    trailing_stop = True
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.006
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    startup_candle_count = 240

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
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
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
        tr = DataFrame(tr, index=df.index)[0]
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: DataFrame, period: int = 14):
        h, l, c = df["high"], df["low"], df["close"]
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        pc = c.shift(1)
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
        tr = DataFrame(tr, index=df.index)[0]
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        p = DataFrame(plus_dm, index=df.index)[0].ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        m = DataFrame(minus_dm, index=df.index)[0].ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_di = 100 * (p / atr)
        minus_di = 100 * (m / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema_20"] = self._ema(df["close"], 20)
        df["ema_50"] = self._ema(df["close"], 50)
        df["ema_200"] = self._ema(df["close"], 200)
        df["ema200_slope"] = (df["ema_200"] - df["ema_200"].shift(24)).fillna(0.0)

        bb_len, bb_std = 20, 2.0
        bb_mid = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        bb_sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        df["bb_mid"] = bb_mid
        df["bb_low"] = bb_mid - bb_std * bb_sd
        df["bb_up"] = bb_mid + bb_std * bb_sd
        df["bb_width"] = ((df["bb_up"] - df["bb_low"]) / df["bb_mid"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["atr"] = self._atr(df, 14)
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["rsi"] = self._rsi(df["close"], 14)
        df["adx"] = self._adx(df, 14)

        df["kc_mid"] = df["ema_20"]
        df["kc_low"] = df["kc_mid"] - 1.5 * df["atr"]
        df["kc_up"] = df["kc_mid"] + 1.5 * df["atr"]

        vma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["green"] = df["close"] > df["open"]
        df["flush"] = df["low"] < (df["bb_low"] - 0.25 * df["atr"])
        df["reclaim"] = df["close"] > df["bb_low"]
        df["rsi_up"] = df["rsi"] > df["rsi"].shift(1)
        df["ret_24"] = df["close"].pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        excursion = df["flush"] & df["reclaim"]
        decel = ((df["rsi"] < 48) & df["rsi_up"]) | (df["green"] & (df["rsi"] < 52))
        chop_ok = df["adx"] < 30
        vol_ok = df["vol_ratio"] > 0.75

        kill_switch = (df["close"] < df["ema_200"] * 0.94) & (df["ema200_slope"] < 0) & (df["atr_pct"] > 0.018)
        avoid_freefall = (df["close"] > df["close"].shift(1) * 0.975) | (df["adx"] < 22)

        entry = excursion & decel & chop_ok & vol_ok & (~kill_switch) & avoid_freefall

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp_mean = (df["close"] >= df["bb_mid"]) | (df["close"] >= df["kc_mid"])
        tp_stretch = (df["close"] >= df["bb_up"]) | (df["rsi"] >= 62)

        back_under_low_2 = (df["close"] < df["bb_low"]) & (df["close"].shift(1) < df["bb_low"].shift(1))
        trend_accel_down = (df["adx"] > 32) & (df["close"] < df["ema_20"])
        volatility_spike_fail = (df["atr_pct"] > 0.024) & (df["close"] < df["kc_low"])

        time_decay_proxy = (df["ret_24"] < -0.003) & (df["rsi"] < 48) & (df["close"] < df["kc_mid"])

        exit_sig = tp_stretch | (tp_mean & (df["rsi"] > 50)) | back_under_low_2 | trend_accel_down | volatility_spike_fail | time_decay_proxy

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df