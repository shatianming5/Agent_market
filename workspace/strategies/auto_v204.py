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


class AutoStrategy_v204(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 50

    minimal_roi = {"0": 0.06, "240": 0.03, "720": 0.012}
    stoploss = -0.06
    use_exit_signal = True

    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    atr_period = 14

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        high = dataframe["high"].astype(float)
        low = dataframe["low"].astype(float)
        vol = dataframe["volume"].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(self.rsi_period)
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        dataframe["rsi"] = (100.0 - (100.0 / (1.0 + rs))).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=0)
        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std
        dataframe["bb_middle"] = mid
        dataframe["bb_upper"] = upper
        dataframe["bb_lower"] = lower
        bb_range = (upper - lower).replace(0.0, np.nan)
        dataframe["bb_width"] = (bb_range / (mid + 1e-12)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        dataframe["bb_percent"] = ((close - lower) / (bb_range + 1e-12)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        vol_sma = vol.rolling(20).mean()
        dataframe["vol_sma"] = vol_sma
        dataframe["vol_ratio"] = (vol / (vol_sma + 1e-12)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        dataframe["atr"] = tr.rolling(self.atr_period).mean().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        lb = 12
        dataframe["bull_div"] = (
            (low < low.shift(lb) * 0.997)
            & (dataframe["rsi"] > dataframe["rsi"].shift(lb) + 2.0)
        ).astype(int)

        dataframe["hh_6"] = high.rolling(6).max().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        rsi = dataframe["rsi"]
        close = dataframe["close"]
        lower = dataframe["bb_lower"]
        width = dataframe["bb_width"]
        volr = dataframe["vol_ratio"]
        bbp = dataframe["bb_percent"]

        base_ok = (dataframe["volume"] > 0) & (volr > 0.3) & (width > 0.004)

        touch = (close < lower * 1.003) & (rsi < 41.0) & (bbp < 0.20)
        deep = ((close < lower * 0.995) | (bbp < 0.08)) & (rsi < 34.0)
        div = (dataframe["bull_div"] > 0) & (close < dataframe["bb_middle"] * 0.995) & (rsi < 46.0)

        enter = base_ok & (touch | deep | div)
        dataframe.loc[enter, ["enter_long", "enter_tag"]] = (1, "rsi_bb_mr_v204")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        rsi = dataframe["rsi"]
        close = dataframe["close"]
        mid = dataframe["bb_middle"]
        upper = dataframe["bb_upper"]
        atr = dataframe["atr"]
        hh6 = dataframe["hh_6"]
        bbp = dataframe["bb_percent"]

        base_ok = dataframe["volume"] > 0

        take_profit = ((close > upper * 0.995) & (rsi > 55.0)) | ((bbp > 0.85) & (rsi > 58.0))
        delayed_mean = (close > mid * 1.002) & (rsi > 60.0)
        overbought = rsi > 72.0
        trail_after_push = (close > mid) & (close < (hh6 - atr * 1.2)) & (rsi > 50.0)

        exit_cond = base_ok & (take_profit | delayed_mean | overbought | trail_after_push)
        dataframe.loc[exit_cond, "exit_long"] = 1
        return dataframe