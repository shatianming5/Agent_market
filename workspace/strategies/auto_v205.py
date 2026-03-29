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


class AutoStrategy_v205(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    use_exit_signal = True
    stoploss = -0.06
    minimal_roi = {"0": 0.05, "180": 0.02, "540": 0.01}

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0

    rsi_buy = 38
    rsi_buy_deep = 28
    rsi_sell_soft = 60
    rsi_sell_hard = 70

    squeeze_lookback = 200
    squeeze_q = 0.45
    expand_ratio = 1.06
    near_lower_eps = 0.003

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)

        delta = close.diff()
        gain = delta.where(delta > 0.0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0.0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        dataframe["rsi"] = (100.0 - (100.0 / (1.0 + rs))).astype(float)

        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std

        dataframe["bb_middle"] = mid.astype(float)
        dataframe["bb_upper"] = upper.astype(float)
        dataframe["bb_lower"] = lower.astype(float)

        width = (upper - lower) / (mid.abs() + 1e-10)
        dataframe["bb_width"] = width.astype(float)

        vol_sma = dataframe["volume"].rolling(20).mean()
        dataframe["vol_sma"] = vol_sma.astype(float)
        dataframe["vol_ratio"] = (dataframe["volume"] / (vol_sma + 1e-10)).astype(float)

        dataframe["bb_squeeze_thr"] = (
            dataframe["bb_width"]
            .rolling(self.squeeze_lookback, min_periods=max(30, self.bb_period))
            .quantile(self.squeeze_q)
            .astype(float)
        )

        dataframe["bb_touch_lower"] = (close <= (lower * (1.0 + self.near_lower_eps))).astype(int)
        dataframe["bb_n_std"] = ((close - mid) / (std + 1e-10)).astype(float)

        dataframe["rsi_slope"] = (dataframe["rsi"] - dataframe["rsi"].shift(1)).astype(float)

        for c in (
            "rsi",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "vol_sma",
            "vol_ratio",
            "bb_squeeze_thr",
            "bb_touch_lower",
            "bb_n_std",
            "rsi_slope",
        ):
            dataframe[c] = (
                dataframe[c]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0.0)
            )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        base = (dataframe["volume"] > 0) & (dataframe["vol_ratio"] > 0.55)

        squeeze = dataframe["bb_width"] < (dataframe["bb_squeeze_thr"] + 1e-12)
        expanding = (dataframe["bb_width"] / (dataframe["bb_width"].shift(3) + 1e-12)) > self.expand_ratio

        oversold = dataframe["rsi"] < float(self.rsi_buy)
        deep_oversold = dataframe["rsi"] < float(self.rsi_buy_deep)
        touch_lower = (dataframe["bb_touch_lower"] > 0) | (dataframe["bb_n_std"] < -1.8)

        setup_a = base & touch_lower & oversold & (squeeze.shift(1).fillna(0.0) > 0.0) & expanding
        setup_b = base & touch_lower & deep_oversold

        dataframe.loc[setup_a, ["enter_long", "enter_tag"]] = (1, "squeeze_expand_revert")
        dataframe.loc[setup_b & ~setup_a, ["enter_long", "enter_tag"]] = (1, "deep_oversold_revert")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        base = dataframe["volume"] > 0

        to_mid = dataframe["close"] > dataframe["bb_middle"]
        to_upper = dataframe["close"] > (dataframe["bb_upper"] * 0.995)

        rsi_rebound = dataframe["rsi"] > float(self.rsi_sell_soft)
        rsi_overbought = dataframe["rsi"] > float(self.rsi_sell_hard)

        weakness = (dataframe["rsi_slope"] < -0.8) & (dataframe["rsi"] > 55.0) & (dataframe["close"] > dataframe["bb_middle"] * 0.99)

        exit_cond = base & (to_upper | rsi_overbought | (to_mid & rsi_rebound) | weakness)

        dataframe.loc[exit_cond, "exit_long"] = 1
        return dataframe