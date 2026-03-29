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


class AutoStrategy_v7(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.02}
    stoploss = -0.06
    trailing_stop = False

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

        bb_len, bb_std = 20, 2.0
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

        df["squeeze"] = (df["bb_width"] < 0.015).fillna(False)
        df["squeeze_recent"] = df["squeeze"].rolling(12, min_periods=1).max().astype(bool)

        df["flush"] = df["low"] < (df["bb_low"] - 0.35 * df["atr"])
        df["reclaim"] = (df["close"] > df["bb_low"]) & (df["close"] > df["open"])
        df["rsi_turn"] = (df["rsi"] < 50) & (df["rsi"] > df["rsi"].shift(1))

        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["fwd_ret_24"] = df["close"].pct_change(24)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        chop_ok = df["adx"] < 24
        squeeze_ok = df["squeeze_recent"]
        flush_ok = df["flush"] & df["reclaim"]
        rsi_ok = df["rsi_turn"]
        vol_ok = df["vol_ratio"] > 0.8

        soft_trend_ok = df["close"] > df["ema_200"] * 0.96
        not_capitulating = (df["close"] > df["close"].shift(1) * 0.982) | (df["adx"] < 18)

        entry = chop_ok & squeeze_ok & flush_ok & rsi_ok & vol_ok & soft_trend_ok & not_capitulating

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # Approx ATR-normalized profit exits using current close distance to mid/upper
        tp_atr = df["close"] >= (df["bb_low"] + 0.95 * df["atr"])
        tp_mid = (df["close"] >= df["bb_mid"]) & (df["rsi"] > 52)
        tp_fast = (df["close"] >= df["bb_up"]) | (df["rsi"] > 64)

        # Failure mode exits: back under range + rising trendiness / weakness
        back_under_low = (df["close"] < df["bb_low"]) & (df["close"].shift(1) < df["bb_low"].shift(1))
        trend_down = (df["adx"] > 28) & (df["close"] < df["ema_50"])
        breakdown = back_under_low | trend_down

        # Time-decay proxy: after a day without momentum (use 24h change and weak RSI)
        time_decay = (df["fwd_ret_24"] < 0.0) & (df["rsi"] < 50) & (df["adx"] > 18)

        exit_sig = tp_fast | tp_mid | tp_atr | breakdown | time_decay

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df