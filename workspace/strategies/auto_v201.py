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


class AutoStrategy_v201(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 50

    use_exit_signal = True
    minimal_roi = {"0": 0.0}
    stoploss = -0.07

    # Prefer letting exits "run" to upper band; avoid tiny-profit exits
    sell_profit_only = True
    sell_profit_offset = 0.003

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0

    rsi_buy = 40.0
    rsi_sell = 72.0

    vol_lookback = 20
    vol_mult = 0.6
    min_bb_width = 0.004

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)

        # RSI (Wilder-style smoothing)
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(self.rsi_period)
        avg_gain = up.ewm(alpha=alpha, adjust=False, min_periods=self.rsi_period).mean()
        avg_loss = down.ewm(alpha=alpha, adjust=False, min_periods=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        dataframe["rsi"] = (100.0 - (100.0 / (1.0 + rs))).astype(float)

        # Bollinger Bands
        mid = close.rolling(self.bb_period, min_periods=self.bb_period).mean()
        std = close.rolling(self.bb_period, min_periods=self.bb_period).std(ddof=0)
        upper = mid + float(self.bb_std) * std
        lower = mid - float(self.bb_std) * std
        dataframe["bb_middle"] = mid
        dataframe["bb_upper"] = upper
        dataframe["bb_lower"] = lower
        denom = (upper - lower).replace(0.0, np.nan)
        dataframe["bb_width"] = ((upper - lower) / (mid + 1e-10)).astype(float)
        dataframe["percent_b"] = ((close - lower) / denom).astype(float)

        # Volume baseline + mild trend context
        dataframe["vol_sma"] = dataframe["volume"].rolling(self.vol_lookback, min_periods=self.vol_lookback).mean()
        dataframe["sma50"] = close.rolling(50, min_periods=50).mean()

        cols = ["rsi", "bb_middle", "bb_upper", "bb_lower", "bb_width", "percent_b", "vol_sma", "sma50"]
        for c in cols:
            if c in dataframe.columns:
                dataframe[c] = dataframe[c].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = (dataframe["volume"] > 0) & (dataframe["volume"] > (dataframe["vol_sma"] * float(self.vol_mult)))
        vol_ok = vol_ok | ((dataframe["volume"] > 0) & (dataframe["vol_sma"] == 0.0))

        # Main mean-reversion trigger: near/under lower band + oversold RSI
        near_lower = dataframe["close"] <= (dataframe["bb_lower"] * 1.003)
        oversold = dataframe["rsi"] < float(self.rsi_buy)

        # Ensure some movement potential; keep loose to avoid 0 trades
        vol_regime = dataframe["bb_width"] > float(self.min_bb_width)

        # Very mild downtrend guard (avoid extreme freefall), still permissive in bearish chop
        trend_guard = (dataframe["sma50"] == 0.0) | (dataframe["close"] > (dataframe["sma50"] * 0.90))

        cond = vol_ok & near_lower & oversold & vol_regime & trend_guard
        dataframe.loc[cond, ["enter_long", "enter_tag"]] = (1, "rsi_bb_revert_v201")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = dataframe["volume"] > 0

        # Let winners run: take profit at (near) upper band; fallback exit if mean reversion stalls
        hit_upper = dataframe["close"] >= (dataframe["bb_upper"] * 0.997)
        rsi_hot = dataframe["rsi"] > float(self.rsi_sell)
        stalled_revert = (dataframe["close"] > dataframe["bb_middle"]) & (dataframe["rsi"] > 58.0)

        cond = vol_ok & (hit_upper | rsi_hot | stalled_revert)
        dataframe.loc[cond, "exit_long"] = 1
        return dataframe