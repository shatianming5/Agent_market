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


class AutoStrategy_v105(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    stoploss = -0.025
    minimal_roi = {"0": 0.015, "24": 0.008, "48": 0.0}
    use_exit_signal = True
    ignore_buying_expired_candle_after = 0

    @staticmethod
    def _rsi(close: DataFrame, period: int = 14):
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(period)
        avg_up = up.ewm(alpha=alpha, adjust=False).mean()
        avg_down = down.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_up / (avg_down.replace(0.0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14):
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        alpha = 1.0 / float(period)
        return tr.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _adx(dataframe: DataFrame, period: int = 14):
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        alpha = 1.0 / float(period)
        atr = DataFrame(tr).ewm(alpha=alpha, adjust=False).mean().iloc[:, 0]
        plus_di = 100.0 * (DataFrame(plus_dm).ewm(alpha=alpha, adjust=False).mean().iloc[:, 0] / atr.replace(0.0, np.nan))
        minus_di = 100.0 * (DataFrame(minus_dm).ewm(alpha=alpha, adjust=False).mean().iloc[:, 0] / atr.replace(0.0, np.nan))
        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx, plus_di, minus_di

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None:
            return dataframe
        if dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        volume = dataframe["volume"].astype(float)

        dataframe["rsi"] = self._rsi(close, 14)
        dataframe["ema9"] = close.ewm(span=9, adjust=False).mean()
        dataframe["ema21"] = close.ewm(span=21, adjust=False).mean()
        dataframe["ema55"] = close.ewm(span=55, adjust=False).mean()

        bb_len = 20
        bb_mid = close.rolling(bb_len).mean()
        bb_std = close.rolling(bb_len).std(ddof=0)
        dataframe["bb_mid"] = bb_mid
        dataframe["bb_lower"] = bb_mid - 2.0 * bb_std
        dataframe["bb_upper"] = bb_mid + 2.0 * bb_std
        dataframe["bb_width"] = ((dataframe["bb_upper"] - dataframe["bb_lower"]) / bb_mid.replace(0.0, np.nan))

        atr = self._atr(dataframe, 14)
        dataframe["atr"] = atr
        dataframe["atrp"] = (atr / close.replace(0.0, np.nan))

        adx, pdi, mdi = self._adx(dataframe, 14)
        dataframe["adx"] = adx
        dataframe["pdi"] = pdi
        dataframe["mdi"] = mdi

        vma = volume.rolling(20).mean()
        dataframe["vol_ratio"] = (volume / vma.replace(0.0, np.nan))

        dataframe["roc24"] = (close / close.shift(24) - 1.0)
        dataframe["roc48"] = (close / close.shift(48) - 1.0)

        zden = bb_std.replace(0.0, np.nan)
        dataframe["z"] = ((close - bb_mid) / zden)

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = dataframe["volume"] > 0
        not_panic = dataframe["roc48"] > -0.07
        not_trending_hard = dataframe["adx"] < 30
        vol_sane = dataframe["vol_ratio"] > 0.35
        move_ok = dataframe["atrp"] > 0.003

        oversold_core = (dataframe["rsi"] < 30) & (dataframe["z"] < -1.6)
        band_flush = (dataframe["close"] < dataframe["bb_lower"] * 1.006) & (dataframe["rsi"] < 33)
        reversal_hint = dataframe["rsi"] > dataframe["rsi"].shift(1)

        entry = vol_ok & vol_sane & move_ok & not_panic & not_trending_hard & (oversold_core | band_flush) & (reversal_hint | (dataframe["rsi"] < 24))
        entry = entry & (~entry.shift(1).fillna(False))

        dataframe.loc[entry, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = dataframe["volume"] > 0

        tp_band = (dataframe["close"] > dataframe["bb_upper"] * 0.995) | (dataframe["rsi"] > 58)
        recover = (dataframe["close"] > dataframe["ema21"]) & (dataframe["z"] > -0.2) & (dataframe["roc24"] > -0.005)
        risk_off = (dataframe["roc48"] < -0.06) & (dataframe["adx"] > 28) & (dataframe["mdi"] > dataframe["pdi"]) & (dataframe["close"] < dataframe["ema21"])

        exit_sig = vol_ok & (tp_band | recover | risk_off)

        dataframe.loc[exit_sig, "exit_long"] = 1
        return dataframe