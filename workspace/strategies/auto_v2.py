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


class AutoStrategy_v2(IStrategy):
    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.015}
    stoploss = -0.08
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

    def _adx_di(self, df: DataFrame, period: int = 14):
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        tr = DataFrame(tr, index=df.index)[0]
        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        plus_dm_s = DataFrame(plus_dm, index=df.index)[0].ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        minus_dm_s = DataFrame(minus_dm, index=df.index)[0].ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        plus_di = (100 * plus_dm_s / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        minus_di = (100 * minus_dm_s / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)
        return adx, plus_di, minus_di

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        df["ema_20"] = self._ema(df["close"], 20)
        df["ema_200"] = self._ema(df["close"], 200)
        df["rsi"] = self._rsi(df["close"], 14)

        bb_len = 20
        bb_std = 2.2
        mid = df["close"].rolling(bb_len, min_periods=bb_len).mean()
        sd = df["close"].rolling(bb_len, min_periods=bb_len).std(ddof=0)
        df["bb_mid"] = mid
        df["bb_low"] = mid - bb_std * sd
        df["bb_up"] = mid + bb_std * sd
        df["bb_width"] = ((df["bb_up"] - df["bb_low"]) / df["bb_mid"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["atr"] = self._atr(df, 14)
        df["atr_pct"] = (df["atr"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        adx, pdi, mdi = self._adx_di(df, 14)
        df["adx"] = adx
        df["di_plus"] = pdi
        df["di_minus"] = mdi

        vol_ma = df["volume"].rolling(30, min_periods=30).mean()
        df["vol_ratio"] = (df["volume"] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        rng = (df["high"] - df["low"]).replace(0, np.nan)
        lower_wick = (np.minimum(df["open"], df["close"]) - df["low"]).clip(lower=0.0)
        body = (df["close"] - df["open"]).abs()
        df["range"] = rng.fillna(0.0)
        df["lw_frac"] = (lower_wick / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["body_frac"] = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        imp = (df["close"] < df["bb_low"]) | (df["rsi"] < 32)
        df["recent_impulse"] = imp.rolling(8, min_periods=1).max().astype(bool)

        # Reversal / exhaustion candle (to be used on next candle via shift)
        bull = df["close"] > df["open"]
        range_expand = df["range"] > (1.2 * df["atr"])
        vol_expand = df["vol_ratio"] > 1.15
        wick_ok = df["lw_frac"] > 0.42
        not_doji = df["body_frac"] > 0.12
        df["exhaust"] = (bull & range_expand & vol_expand & wick_ok & not_doji).fillna(False)

        df["break_prev_high"] = df["close"] > df["high"].shift(1)
        df["adx_rising"] = (df["adx"] > df["adx"].shift(1)).fillna(False)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        trend_guard = (df["close"] > df["ema_200"] * 0.972) | (df["adx"] < 23)
        down_dom = df["di_minus"] > df["di_plus"] * 0.95

        # Enter on break of prior exhaustion candle high
        entry = (
            df["break_prev_high"]
            & df["exhaust"].shift(1).fillna(False)
            & df["recent_impulse"].shift(1).fillna(False)
            & down_dom.shift(1).fillna(False)
            & trend_guard
            & (df["bb_width"] > 0.002)
            & (df["bb_width"] < 0.06)
        )

        df["enter_long"] = 0
        df.loc[entry, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # Take profit into normalization
        tp = (df["close"] >= df["bb_mid"]) | (df["rsi"] > 56)

        # Fast risk-off if continuation risk returns after entry
        risk_off = (
            (df["rsi"] < 29)
            & (df["close"] < df["bb_low"])
            & (df["di_minus"] > df["di_plus"])
            & (df["adx"] > 24)
            & df["adx_rising"]
        )

        # Stale bounce: no follow-through
        stale = (df["close"] < df["ema_20"]) & (df["rsi"] < 48) & (df["adx"] > 18)

        df["exit_long"] = 0
        df.loc[tp | risk_off | stale, "exit_long"] = 1
        return df