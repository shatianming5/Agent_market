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


class AutoStrategy_v5(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.02}
    stoploss = -0.05
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
        df["rsi_up"] = df["rsi"] > df["rsi"].shift(1)

        df["atr"] = self._atr(df, 14)
        df["atr20"] = self._atr(df, 20)
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ema_kc = self._ema(df["close"], 20)
        df["kc_mid"] = ema_kc
        df["kc_up"] = ema_kc + 1.5 * df["atr20"]
        df["kc_low"] = ema_kc - 1.5 * df["atr20"]

        df["squeeze"] = (df["bb_up"] < df["kc_up"]) & (df["bb_low"] > df["kc_low"])
        df["squeeze_recent"] = df["squeeze"].rolling(12, min_periods=1).max().astype(bool)

        ll = df["low"].rolling(24, min_periods=24).min()
        df["ll24"] = ll
        df["sweep"] = df["low"] < (df["ll24"].shift(1) - 0.15 * df["atr"])

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["reclaim"] = (df["close"] > df["bb_low"]) & (df["close"] > df["open"])
        df["ema50_slope6"] = (df["ema_50"] / df["ema_50"].shift(6) - 1.0).replace([np.inf, -np.inf], np.nan)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        squeeze_ok = df["squeeze_recent"]
        vol_ok = df["vol_ratio"] > 0.8

        rsi_ok = (df["rsi"] < 52) & df["rsi_up"]

        regime_ok = (df["close"] > df["ema_200"] * 0.955) & (df["ema50_slope6"] > -0.012)
        avoid_freefall = df["close"] > df["close"].shift(1) * 0.985

        entry = squeeze_ok & df["sweep"] & df["reclaim"] & rsi_ok & vol_ok & regime_ok & avoid_freefall

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        tp_mid = (df["close"] >= df["bb_mid"]) & (df["rsi"] > 50)
        tp_strong = (df["close"] >= df["bb_up"]) | (df["rsi"] > 60)
        breakdown = (df["close"] < df["bb_low"]) & (df["rsi"] < 44)
        vol_spike_fail = (df["bb_width"] > 0.022) & (df["close"] < df["ema_50"]) & (df["rsi"] < 48)

        exit_sig = tp_strong | tp_mid | breakdown | vol_spike_fail

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df