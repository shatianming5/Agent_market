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


class AutoStrategy_v403(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 55

    minimal_roi = {"0": 0.0}
    stoploss = -0.03
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    @staticmethod
    def _rsi(close, period: int = 14):
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        volume = dataframe["volume"].astype(float)

        dataframe["ret1h"] = close.pct_change()
        dataframe["ret20"] = close / close.shift(20) - 1.0

        dataframe["rsi"] = self._rsi(close, 14)

        mid = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        dataframe["bb_mid"] = mid
        dataframe["bb_upper"] = mid + 2.0 * std
        dataframe["bb_lower"] = mid - 2.0 * std
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / (mid.replace(0.0, np.nan))

        vol_ma = volume.rolling(20).mean()
        dataframe["vol_ratio"] = volume / (vol_ma.replace(0.0, np.nan))

        ann_factor = float(np.sqrt(24.0 * 365.0))
        dataframe["ann_vol24"] = dataframe["ret1h"].rolling(24).std(ddof=0) * ann_factor

        rsi_min12 = dataframe["rsi"].rolling(12).min()
        dataframe["rsi_os_recent"] = (rsi_min12 < 35.0).astype(float)

        lb = 6
        price_ll = close < close.shift(lb)
        rsi_hl = dataframe["rsi"] > dataframe["rsi"].shift(lb)
        dataframe["rsi_div"] = (price_ll & rsi_hl).astype(float)

        for c in ("ret1h", "ret20", "rsi", "bb_mid", "bb_upper", "bb_lower", "bb_width", "vol_ratio", "ann_vol24", "rsi_os_recent", "rsi_div"):
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

        rebound_skip = dataframe["ret20"] > 0.03
        oversold = (dataframe["close"] < dataframe["bb_lower"] * 1.01) & ((dataframe["rsi"] < 42.0) | (dataframe["rsi_os_recent"] > 0.0))
        div_ok = dataframe["rsi_div"] > 0.0
        vol_ok = dataframe["ann_vol24"] < 0.80
        liq_ok = dataframe["volume"] > 0
        volr_ok = dataframe["vol_ratio"] > 0.25

        enter = liq_ok & volr_ok & vol_ok & (~rebound_skip) & oversold & div_ok
        dataframe.loc[enter, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        liq_ok = dataframe["volume"] > 0
        low_vol = dataframe["ann_vol24"] < 0.45

        exit_low = low_vol & (dataframe["close"] > dataframe["bb_upper"])
        exit_high = (~low_vol) & (dataframe["close"] > dataframe["bb_mid"])

        exit_cond = liq_ok & (exit_low | exit_high)
        dataframe.loc[exit_cond, "exit_long"] = 1
        return dataframe