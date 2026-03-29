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

    minimal_roi = {"0": 0.03}
    stoploss = -0.12
    trailing_stop = False

    process_only_new_candles = True
    startup_candle_count = 80

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema50"] = self._ema(df["close"], 50)
        df["rsi"] = self._rsi(df["close"], 14)
        df["atr"] = self._atr(df, 14)
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        bb_len = 20
        bb_std = 2.0
        mid = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        up = mid + bb_std * sd
        lo = mid - bb_std * sd
        df["bb_mid"], df["bb_up"], df["bb_low"] = mid, up, lo
        bw = ((up - lo) / mid).replace([np.inf, -np.inf], np.nan)
        df["bb_width"] = bw.fillna(0.0)
        denom = (up - lo).replace(0, np.nan)
        df["pctb"] = ((df["close"] - lo) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.5)

        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        df["ret1"] = (df["close"] / df["close"].shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        atrp_prev = df["atr_pct"].shift(1).fillna(df["atr_pct"])
        df["shock_down"] = df["ret1"] <= (-0.8 * atrp_prev)

        df["bull"] = df["close"] > df["open"]
        df["higher_low"] = df["low"] >= df["low"].shift(1)
        df["mom_turn"] = (df["rsi"] > df["rsi"].shift(1)) & (df["close"] > df["close"].shift(1))
        df["stabilize"] = (df["bull"] & df["higher_low"]) | df["mom_turn"]

        df["cooldown"] = (df["shock_down"]).rolling(3, min_periods=1).max().shift(1).fillna(0).astype(bool)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        oversold = (df["rsi"] < 34) | (df["pctb"] < 0.12) | (df["close"] < df["bb_low"] * 1.01)
        liquidity = df["vol_ratio"] > 0.75
        soft_trend = df["close"] > (df["ema50"] * 0.965)
        vol_ok = df["bb_width"] > 0.002

        recent_shock = df["shock_down"].rolling(2, min_periods=1).max().astype(bool)
        entry = recent_shock & oversold & df["stabilize"] & liquidity & soft_trend & vol_ok & (~df["cooldown"])

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        snapback = (df["pctb"] > 0.55) | (df["close"] > df["bb_mid"]) | (df["rsi"] > 52)

        atrp = df["atr_pct"].replace(0, np.nan).ffill().fillna(0.01)
        renewed_flush = (df["ret1"] < (-0.9 * atrp.shift(1).fillna(atrp))) & (df["rsi"] < 30) & (
            df["close"] < df["bb_low"] * 0.995
        )

        weak = (df["rsi"] < 45) & (df["close"] < df["ema50"])
        stale = weak & (df["rsi"].rolling(18, min_periods=18).max() < 52)

        exit_sig = (snapback & (df["bull"] | (df["rsi"] > df["rsi"].shift(1)))) | renewed_flush | stale

        df["exit_long"] = 0
        df.loc[exit_sig, "exit_long"] = 1
        return df