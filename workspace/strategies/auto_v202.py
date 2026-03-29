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


class AutoStrategy_v202(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.06, "120": 0.025, "360": 0.01}
    stoploss = -0.055
    use_exit_signal = True

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    trend_period = 50

    rsi_buy = 40
    rsi_buy_deep = 32
    rsi_sell = 62
    rsi_sell_hard = 70

    vol_ratio_confirm = 1.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)

        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        gain = up.ewm(alpha=1.0 / float(self.rsi_period), adjust=False).mean()
        loss = down.ewm(alpha=1.0 / float(self.rsi_period), adjust=False).mean()
        rs = gain / (loss + 1e-12)
        dataframe["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=0)
        dataframe["bb_middle"] = mid
        dataframe["bb_upper"] = mid + self.bb_std * std
        dataframe["bb_lower"] = mid - self.bb_std * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (mid.abs() + 1e-12)

        z = (close - mid) / (std + 1e-12)
        dataframe["bb_z"] = z
        dataframe["bb_percent"] = (close - dataframe["bb_lower"]) / (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) + 1e-12
        )

        vol_sma = dataframe["volume"].rolling(self.vol_period).mean()
        dataframe["vol_sma"] = vol_sma
        dataframe["vol_ratio"] = dataframe["volume"] / (vol_sma + 1e-12)

        sma_trend = close.rolling(self.trend_period).mean()
        dataframe["sma_trend"] = sma_trend
        dataframe["trend_dist"] = (close - sma_trend) / (sma_trend.abs() + 1e-12)

        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.ffill(inplace=True)
        dataframe.fillna(0.0, inplace=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        touch_lower = dataframe["close"] < dataframe["bb_lower"]
        oversold = dataframe["rsi"] < float(self.rsi_buy)
        deep_oversold = dataframe["rsi"] < float(self.rsi_buy_deep)
        vol_confirm = dataframe["vol_ratio"] > float(self.vol_ratio_confirm)
        far_below = dataframe["bb_z"] < -2.4

        not_extreme_bear = dataframe["trend_dist"] > -0.04
        allow_in_bear = deep_oversold | far_below

        entry = (
            (dataframe["volume"] > 0)
            & touch_lower
            & oversold
            & (
                (vol_confirm & not_extreme_bear)
                | allow_in_bear
                | ((dataframe["bb_z"] < -2.0) & (dataframe["vol_ratio"] > 1.15))
            )
        )

        dataframe.loc[entry, ["enter_long", "enter_tag"]] = (1, "rsi_bb_mr_v2")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        width = dataframe["bb_width"]
        near_upper = dataframe["close"] > (dataframe["bb_upper"] * 0.995)
        revert_mid = (dataframe["close"] > dataframe["bb_middle"]) & (dataframe["rsi"] > float(self.rsi_sell))
        low_vol_exit = (width < 0.010) & (dataframe["bb_percent"] > 0.55) & (dataframe["rsi"] > 55.0)
        hard_rsi = dataframe["rsi"] > float(self.rsi_sell_hard)

        exit_cond = (dataframe["volume"] > 0) & (near_upper | hard_rsi | revert_mid | low_vol_exit)
        dataframe.loc[exit_cond, "exit_long"] = 1
        return dataframe