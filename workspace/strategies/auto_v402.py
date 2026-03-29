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


class AutoStrategy_v402(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 55

    minimal_roi = {"0": 0.0}
    stoploss = -0.05
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        volume = dataframe["volume"].astype(float)

        ret = close.pct_change()
        dataframe["ret"] = ret

        # RSI(14) - Wilder's smoothing via EWM
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0.0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        dataframe["rsi"] = rsi.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        # Bollinger Bands(20, 2)
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std(ddof=0)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        dataframe["bb_mid"] = bb_mid.ffill().fillna(0.0)
        dataframe["bb_upper"] = bb_upper.ffill().fillna(0.0)
        dataframe["bb_lower"] = bb_lower.ffill().fillna(0.0)
        dataframe["bb_width"] = ((bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).ffill().fillna(0.0)

        # Volume ratio vs 20-bar average
        vol_ma = volume.rolling(20, min_periods=1).mean()
        dataframe["volume_ratio"] = (volume / vol_ma.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        ).ffill().fillna(0.0)

        # Annualized volatility from hourly returns (20 bars)
        ann_factor = float(np.sqrt(24.0 * 365.0))
        ann_vol_20 = ret.rolling(20, min_periods=20).std(ddof=0) * ann_factor
        dataframe["ann_vol_20"] = ann_vol_20.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        # Divergence proxy: price makes a new low, RSI does NOT make a new low
        prev_close_min = close.shift(1).rolling(18, min_periods=18).min()
        prev_rsi_min = dataframe["rsi"].shift(1).rolling(18, min_periods=18).min()
        dataframe["div_bull"] = (
            (close <= prev_close_min * 0.997) & (dataframe["rsi"] >= (prev_rsi_min + 2.5))
        ).astype(int)

        # Oversold "frequency" - relaxed
        dataframe["rsi_os_count_20"] = (dataframe["rsi"] < 35.0).rolling(20, min_periods=1).sum().ffill().fillna(0.0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        dataframe["enter_long"] = 0

        rsi = dataframe["rsi"]
        close = dataframe["close"].astype(float)
        bb_lower = dataframe["bb_lower"]
        ann_vol = dataframe["ann_vol_20"]
        vol_ratio = dataframe["volume_ratio"]

        cond = (
            (dataframe["volume"] > 0)
            & (rsi < 37.0)
            & (close < bb_lower * 1.010)
            & (dataframe["div_bull"] > 0)
            & (dataframe["rsi_os_count_20"] >= 2.0)
            & (ann_vol > 0.0)
            & (ann_vol < 0.65)
            & (vol_ratio > 0.20)
        )

        dataframe.loc[cond, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        dataframe["exit_long"] = 0

        close = dataframe["close"].astype(float)
        rsi = dataframe["rsi"]
        bb_mid = dataframe["bb_mid"]
        bb_upper = dataframe["bb_upper"]
        ann_vol = dataframe["ann_vol_20"]

        vol_high = ann_vol > 0.50

        exit_low_vol = (close >= bb_upper * 0.995) | ((rsi > 54.0) & (close >= bb_mid * 1.002))
        exit_high_vol = (close >= bb_mid * 0.997) | (rsi > 58.0)

        cond = (dataframe["volume"] > 0) & (
            ((~vol_high) & exit_low_vol) | (vol_high & exit_high_vol)
        )

        dataframe.loc[cond, "exit_long"] = 1
        return dataframe