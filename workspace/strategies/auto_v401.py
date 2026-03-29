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


class AutoStrategy_v401(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 55

    minimal_roi = {"0": 0.03, "12": 0.015, "36": 0.0}
    stoploss = -0.05
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    @staticmethod
    def _rsi(close, period: int = 14):
        d = close.diff()
        gain = d.clip(lower=0.0)
        loss = (-d).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        low = dataframe["low"].astype(float)
        volume = dataframe["volume"].astype(float)

        rsi = self._rsi(close, 14)
        n = 20
        mid = close.rolling(n).mean()
        sd = close.rolling(n).std(ddof=0)
        upper = mid + 2.0 * sd
        lower = mid - 2.0 * sd
        bb_width = (upper - lower) / (mid.replace(0.0, np.nan))

        vol_mean = volume.rolling(n).mean()
        vol_ratio = volume / (vol_mean.replace(0.0, np.nan))

        ret1 = close.pct_change()
        ret20 = close / close.shift(20) - 1.0
        vol_ann = ret1.rolling(20).std(ddof=0) * np.sqrt(24.0 * 365.0)

        w = 10
        low_w = low.rolling(w).min()
        low_prev = low.shift(w).rolling(w).min()
        rsi_low_w = rsi.rolling(w).min()
        rsi_low_prev = rsi.shift(w).rolling(w).min()

        div1 = (low_w < (low_prev * 0.999)) & (rsi_low_w > (rsi_low_prev + 0.5)) & (close <= (low_w * 1.01))
        div2 = (low < (low.shift(5) * 0.999)) & (rsi > (rsi.shift(5) + 0.5)) & (rsi < 45.0)

        dataframe["rsi"] = rsi
        dataframe["bb_middle"] = mid
        dataframe["bb_upper"] = upper
        dataframe["bb_lower"] = lower
        dataframe["bb_width"] = bb_width
        dataframe["volume_ratio"] = vol_ratio
        dataframe["ret1"] = ret1
        dataframe["ret20"] = ret20
        dataframe["vol_ann"] = vol_ann
        dataframe["div_bull"] = (div1 | div2).astype(int)

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        rsi = dataframe["rsi"]
        close = dataframe["close"]
        lower = dataframe["bb_lower"]
        mid = dataframe["bb_middle"]

        regime_ok = dataframe["ret20"] <= 0.03  # soft filter: only skip strong rebound
        vol_ok = (dataframe["volume"] > 0) & (dataframe["volume_ratio"] > 0.5)
        vol_guard = dataframe["vol_ann"] < 0.60

        oversold = (rsi < 38.0) | (close < (lower * 1.02))
        bb_pressure = close < (mid * 0.995)
        divergence = dataframe["div_bull"] > 0

        enter = regime_ok & vol_ok & vol_guard & divergence & oversold & bb_pressure
        dataframe.loc[enter, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"]
        rsi = dataframe["rsi"]
        mid = dataframe["bb_middle"]
        upper = dataframe["bb_upper"]

        take_mid = (close > mid) & (rsi > 48.0)
        take_upper = close > (upper * 0.995)
        rebound_spike = (dataframe["ret1"] > 0.012) & (close > mid) & (rsi > 52.0)
        exit_cond = (dataframe["volume"] > 0) & (take_mid | take_upper | rebound_spike)

        dataframe.loc[exit_cond, "exit_long"] = 1
        return dataframe