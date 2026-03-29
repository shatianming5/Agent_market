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


class AutoStrategy_v4(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.03}
    stoploss = -0.12
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

    def _ema(self, s, period: int) -> DataFrame:
        return s.ewm(span=period, adjust=False, min_periods=period).mean()

    def _rsi(self, close, period: int = 14):
        d = close.diff()
        up = d.clip(lower=0.0)
        dn = (-d).clip(lower=0.0)
        au = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        ad = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = au / ad.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _atr(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        tr = DataFrame(tr, index=df.index)[0]
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _adx(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        tr = DataFrame(tr, index=df.index)[0]
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        plus_dm_s = DataFrame(plus_dm, index=df.index)[0]
        minus_dm_s = DataFrame(minus_dm, index=df.index)[0]
        plus_di = 100 * (plus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm_s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema_200"] = self._ema(df["close"], 200)
        df["ema_50"] = self._ema(df["close"], 50)

        bb_len = 20
        bb_std = 2.0
        mid = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        df["bb_mid"] = mid
        df["bb_low"] = mid - bb_std * sd
        df["bb_up"] = mid + bb_std * sd
        df["bb_width"] = ((df["bb_up"] - df["bb_low"]) / df["bb_mid"]).replace([np.inf, -np.inf], np.nan)

        df["rsi"] = self._rsi(df["close"], 14)
        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, 14)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["rsi_up"] = df["rsi"] > df["rsi"].shift(1)
        df["close_reclaim"] = df["close"] > df["bb_low"]
        df["flush"] = df["low"] < (df["bb_low"] - 0.25 * df["atr"])

        df["bb_squeeze"] = df["bb_width"] < 0.014
        df["bb_squeeze_recent"] = df["bb_squeeze"].rolling(8, min_periods=1).max().astype(bool)

        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        chop_ok = df["adx"] < 26
        squeeze_ok = df["bb_squeeze_recent"]
        vol_ok = df["vol_ratio"] > 0.65

        rsi_ok = (df["rsi"] < 48) & df["rsi_up"]
        trigger = df["flush"] & df["close_reclaim"]

        soft_trend_ok = (df["close"] > df["ema_200"] * 0.965) | (df["adx"] < 18)
        avoid_freefall = (df["close"] > df["close"].shift(1) * 0.985) | (df["adx"] < 20)

        entry = chop_ok & squeeze_ok & vol_ok & rsi_ok & trigger & soft_trend_ok & avoid_freefall

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp1 = df["close"] >= df["bb_mid"]
        tp2 = (df["close"] >= df["bb_up"]) | (df["rsi"] > 62)

        breakdown = (df["adx"] > 28) & (df["close"] < df["bb_low"]) & (df["rsi"] < 42)
        fail_reclaim = (df["close"] < df["ema_50"]) & (df["rsi"] < 45) & (df["bb_width"] > 0.02)

        exit_sig = tp2 | (tp1 & (df["rsi"] > 52)) | breakdown | fail_reclaim

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df