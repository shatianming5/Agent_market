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


class AutoStrategy_v203(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.02, "240": 0.006, "720": 0.0}
    stoploss = -0.05
    use_exit_signal = True

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    ema_fast = 20
    ema_slow = 50

    rsi_buy_1 = 38.0
    rsi_buy_2 = 45.0
    rsi_sell = 68.0

    min_bb_width = 0.006
    min_vol_ratio = 0.7
    max_downtrend_dist = 0.97

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None:
            return dataframe
        if dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        vol = dataframe["volume"].astype(float)

        ret1 = close.pct_change()
        dataframe["ret1"] = ret1.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(self.rsi_period)
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        dataframe["rsi"] = (100.0 - (100.0 / (1.0 + rs))).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        dataframe["rsi_chg"] = dataframe["rsi"].diff().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=0)
        dataframe["bb_middle"] = sma
        dataframe["bb_upper"] = sma + self.bb_std * std
        dataframe["bb_lower"] = sma - self.bb_std * std
        dataframe["bb_width"] = ((dataframe["bb_upper"] - dataframe["bb_lower"]) / (sma + 1e-10)).replace(
            [np.inf, -np.inf], np.nan
        ).ffill().fillna(0.0)

        dataframe["vol_sma"] = vol.rolling(20).mean()
        dataframe["vol_ratio"] = (vol / (dataframe["vol_sma"] + 1e-10)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        dataframe["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        dataframe["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()

        dataframe["bb_pos"] = ((close - dataframe["bb_lower"]) / (dataframe["bb_upper"] - dataframe["bb_lower"] + 1e-10)).replace(
            [np.inf, -np.inf], np.nan
        ).ffill().fillna(0.0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        base = (
            (dataframe["volume"] > 0)
            & (dataframe["bb_width"] > self.min_bb_width)
            & (dataframe["vol_ratio"] > self.min_vol_ratio)
            & (dataframe["close"] > dataframe["ema_slow"] * self.max_downtrend_dist)
        )

        overshoot = dataframe["close"] < (dataframe["bb_lower"] * 1.0015)

        entry_a = base & overshoot & (dataframe["rsi"] < self.rsi_buy_1)
        entry_b = base & overshoot & (dataframe["rsi"] < self.rsi_buy_2) & (dataframe["ret1"] < -0.006)
        entry_c = base & overshoot & (dataframe["rsi"] < (self.rsi_buy_2 + 2.0)) & (dataframe["rsi_chg"] > 0.0) & (
            dataframe["bb_pos"] < 0.08
        )

        cond = entry_a | entry_b | entry_c
        dataframe.loc[cond, ["enter_long", "enter_tag"]] = (1, "rsi_bb_mr_trail_v203")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        base = dataframe["volume"] > 0

        tp_soft = (dataframe["close"] > dataframe["bb_middle"]) & (dataframe["rsi"] > 55.0) & (dataframe["bb_pos"] > 0.45)
        tp_strong = (dataframe["close"] > dataframe["bb_upper"] * 0.995) | (dataframe["rsi"] > self.rsi_sell)
        fail_bounce = (dataframe["close"] < dataframe["bb_lower"] * 0.992) & (dataframe["rsi"] < 28.0)

        dataframe.loc[base & (tp_soft | tp_strong | fail_bounce), "exit_long"] = 1
        return dataframe