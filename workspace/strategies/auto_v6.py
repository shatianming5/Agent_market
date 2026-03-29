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


class AutoStrategy_v6(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.02, "8": 0.008, "16": 0.0}
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
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["adx"] = self._adx(df, 14)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["sweep"] = df["low"] < (df["bb_low"] - 0.30 * df["atr"])
        df["reclaim"] = df["close"] > df["bb_low"]
        df["sweep_recent3"] = (df["sweep"].shift(1).fillna(False) | df["sweep"].shift(2).fillna(False) | df["sweep"].shift(3).fillna(False))
        df["rsi_cross45"] = (df["rsi"] > 45) & (df["rsi"].shift(1) <= 45)
        df["rsi_rise2"] = (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"].shift(1) > df["rsi"].shift(2))
        df["rsi_turn"] = df["rsi_cross45"] | ((df["rsi"] < 50) & df["rsi_rise2"])

        df["enter_tag"] = ""
        df["exit_tag"] = ""
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        bb_ok = (df["bb_width"] > 0.010) & (df["bb_width"] < 0.032)
        regime_ok = (df["adx"] < 30) & ((df["close"] > df["ema_200"] * 0.955) | (df["adx"] < 18))
        vol_ok = df["vol_ratio"] > 0.60
        not_freefall = (df["close"] > df["close"].shift(1) * 0.985) | (df["adx"] < 20)

        setup = df["sweep_recent3"] & df["reclaim"] & df["rsi_turn"]
        entry = setup & bb_ok & regime_ok & vol_ok & not_freefall

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        df.loc[entry, "enter_tag"] = "sweep_reclaim_rsi"
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp_mid = df["close"] >= df["bb_mid"]
        tp_strong = (df["close"] >= df["bb_up"]) | (df["rsi"] > 62)

        fail = (df["close"] < df["bb_low"]) & (df["rsi"] < 43)
        drift_bear = (df["close"] < df["ema_50"]) & (df["rsi"] < 44) & (df["adx"] > 26) & (df["bb_width"] > 0.018)

        time_stop = (df["close"] < df["bb_mid"]) & (df["rsi"] < 56) & (df["adx"] > 16)

        exit_sig = tp_strong | (tp_mid & (df["rsi"] > 50)) | fail | drift_bear | time_stop

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        df.loc[tp_strong, "exit_tag"] = "tp_strong"
        df.loc[tp_mid & (df["rsi"] > 50), "exit_tag"] = "tp_mid"
        df.loc[fail, "exit_tag"] = "fail"
        df.loc[drift_bear, "exit_tag"] = "drift_bear"
        df.loc[time_stop & ~tp_mid, "exit_tag"] = "time_stop"
        return df