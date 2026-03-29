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

from datetime import datetime, timedelta
from typing import Optional


class AutoStrategy_v1(IStrategy):
    timeframe = "1h"
    can_short = False

    process_only_new_candles = True
    startup_candle_count = 240

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 0.012, "8": 0.006, "18": 0.0}
    stoploss = -0.04

    def _ema(self, s, span: int):
        return s.ewm(span=span, adjust=False).mean()

    def _rsi(self, close, period: int = 14):
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _atr(self, dataframe: DataFrame, period: int = 14):
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"]
        volume = dataframe["volume"]

        dataframe["ema50"] = self._ema(close, 50)
        dataframe["ema200"] = self._ema(close, 200)
        dataframe["ema200_slope_24"] = dataframe["ema200"] / dataframe["ema200"].shift(24) - 1.0

        mid = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        dataframe["bb_mid"] = mid
        dataframe["bb_upper"] = mid + 2.0 * std
        dataframe["bb_lower"] = mid - 2.0 * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / mid.replace(0.0, np.nan)

        dataframe["rsi14"] = self._rsi(close, 14)
        dataframe["atr14"] = self._atr(dataframe, 14)
        dataframe["atr_pct"] = dataframe["atr14"] / close.replace(0.0, np.nan)

        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std(ddof=0)
        dataframe["vol_ratio"] = (volume / vol_mean.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        dataframe["vol_z"] = ((volume - vol_mean) / vol_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        close = df["close"]

        trend_ok = (df["ema200_slope_24"] > -0.015) & (close > df["ema200"] * 0.93)
        vol_ok = (df["atr_pct"] > 0.002) | (df["bb_width"] > 0.006)
        liquidity_ok = (df["vol_ratio"].fillna(1.0) > 0.75) | (df["vol_z"].fillna(0.0) > -0.3)

        strong_oversold = (close < df["bb_lower"] * 1.01) & (df["rsi14"] < 33)
        mild_pullback = (close < df["bb_mid"] * 0.988) & (df["rsi14"] < 41)

        df["enter_long"] = 0
        df.loc[
            trend_ok
            & vol_ok
            & liquidity_ok
            & (strong_oversold | mild_pullback)
            & (df["volume"] > 0),
            "enter_long",
        ] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        close = df["close"]

        take_profit = (close >= df["bb_mid"]) | (df["rsi14"] > 56) | (close > df["ema50"] * 1.003)
        risk_off = (close < df["bb_lower"] * 0.985) & (df["rsi14"] < 34)
        trend_break = close < df["ema200"] * 0.915

        df["exit_long"] = 0
        df.loc[(take_profit | risk_off | trend_break) & (df["volume"] > 0), "exit_long"] = 1
        return df

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        if trade.open_date_utc and current_time - trade.open_date_utc >= timedelta(hours=18):
            return "time_exit"
        if current_profit > 0.0:
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                last = df.iloc[-1]
                if float(last.get("close", current_rate)) >= float(last.get("bb_mid", current_rate)):
                    return "bb_mid_tp"
            except Exception:
                return None
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = df.iloc[-1]
            atr = float(last.get("atr14", np.nan))
            close = float(last.get("close", current_rate))
            if np.isfinite(atr) and close > 0:
                sl = -1.2 * (atr / close)
                return float(np.clip(sl, -0.06, -0.015))
        except Exception:
            pass
        return self.stoploss