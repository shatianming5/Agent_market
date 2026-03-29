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


class AutoStrategy_v1(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.02}
    stoploss = -0.20
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
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_market_ratio": 0.99,
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
        rs = au / (ad.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _atr(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        tr = DataFrame(tr, index=df.index)[0]
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        return atr

    def _adx(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        tr = DataFrame(tr, index=df.index)[0]
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        plus_dm_s = DataFrame(plus_dm, index=df.index)[0]
        minus_dm_s = DataFrame(minus_dm, index=df.index)[0]

        plus_di = 100 * (plus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        return adx

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema_200"] = self._ema(df["close"], 200)
        df["rsi"] = self._rsi(df["close"], 14)

        bb_len = 20
        bb_std = 2.0
        m = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        df["bb_mid"] = m
        df["bb_low"] = m - bb_std * sd
        df["bb_up"] = m + bb_std * sd
        df["bb_width"] = ((df["bb_up"] - df["bb_low"]) / df["bb_mid"]).replace([np.inf, -np.inf], np.nan)

        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, 14)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan)

        df["stretch"] = (df["close"] < df["bb_low"]) & (df["rsi"] < 35)
        df["recent_stretch"] = df["stretch"].rolling(6, min_periods=1).max().astype(bool)

        df["rsi_cross_up"] = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
        df["higher_low"] = df["low"] > df["low"].shift(1)
        df["bull_candle"] = df["close"] > df["open"]

        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        chop_ok = df["adx"] < 27
        confirm = df["rsi_cross_up"] & (df["higher_low"] | df["bull_candle"])
        vol_ok = (df["bb_width"].fillna(0) > 0.004) & (df["bb_width"].fillna(0) < 0.05)
        volratio_ok = df["vol_ratio"].fillna(1.0) > 0.6
        ema_ok = (df["close"] > df["ema_200"] * 0.985) | (df["adx"] < 20)

        entry = df["recent_stretch"] & confirm & chop_ok & vol_ok & volratio_ok & ema_ok

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp_signal = (df["close"] >= df["bb_mid"]) | (df["rsi"] > 55)
        risk_off = (df["close"] < df["bb_low"]) & (df["rsi"] < 28) & (df["adx"] > 25)

        ema50 = self._ema(df["close"], 50)
        stale = (~df["recent_stretch"]) & (df["rsi"] < 45) & (df["close"] < ema50)

        exit_sig = tp_signal | risk_off | stale

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df