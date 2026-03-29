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


class AutoStrategy_v104(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    minimal_roi = {"0": 0.012, "12": 0.006, "36": 0.0}
    stoploss = -0.028
    use_exit_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        close = dataframe["close"].astype(float)
        high = dataframe["high"].astype(float)
        low = dataframe["low"].astype(float)
        vol = dataframe["volume"].astype(float)

        ret = close.pct_change()

        def ema(s: DataFrame | np.ndarray, n: int):
            return s.ewm(span=n, adjust=False).mean()

        def rsi(s, n: int = 14):
            d = s.diff()
            up = d.clip(lower=0.0)
            dn = (-d).clip(lower=0.0)
            au = up.ewm(alpha=1.0 / n, adjust=False).mean()
            ad = dn.ewm(alpha=1.0 / n, adjust=False).mean()
            rs = au / (ad + 1e-12)
            return 100.0 - (100.0 / (1.0 + rs))

        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
        plus_dm = DataFrame({"v": plus_dm}, index=dataframe.index)["v"]
        minus_dm = DataFrame({"v": minus_dm}, index=dataframe.index)["v"]

        tr_s = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
        p_s = plus_dm.ewm(alpha=1.0 / 14, adjust=False).mean()
        m_s = minus_dm.ewm(alpha=1.0 / 14, adjust=False).mean()

        plus_di = 100.0 * (p_s / (tr_s + 1e-12))
        minus_di = 100.0 * (m_s / (tr_s + 1e-12))
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
        adx = dx.ewm(alpha=1.0 / 14, adjust=False).mean()

        bb_n = 20
        bb_mid = close.rolling(bb_n).mean()
        bb_std = close.rolling(bb_n).std(ddof=0)
        bb_up = bb_mid + 2.0 * bb_std
        bb_low = bb_mid - 2.0 * bb_std
        bb_width = (bb_up - bb_low) / (bb_mid.abs() + 1e-12)

        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        ema100 = ema(close, 100)
        ema100_slope = (ema100 / (ema100.shift(24) + 1e-12)) - 1.0

        vol_ma = vol.rolling(20).mean()
        vol_ratio = vol / (vol_ma + 1e-12)

        dataframe["ret"] = ret
        dataframe["rsi14"] = rsi(close, 14)
        dataframe["atr14"] = atr
        dataframe["adx14"] = adx
        dataframe["bb_mid"] = bb_mid
        dataframe["bb_low"] = bb_low
        dataframe["bb_up"] = bb_up
        dataframe["bb_width"] = bb_width
        dataframe["ema20"] = ema20
        dataframe["ema50"] = ema50
        dataframe["ema100"] = ema100
        dataframe["ema100_slope"] = ema100_slope
        dataframe["vol_ratio"] = vol_ratio

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = dataframe["volume"] > 0
        regime_ok = (dataframe["adx14"] < 32.0) & (dataframe["ema100_slope"] > -0.006)
        not_too_deep = dataframe["close"] > dataframe["ema100"] * 0.975

        bb_touch = dataframe["close"] < dataframe["bb_low"] * 1.005
        vol_spike = dataframe["vol_ratio"] > 1.10
        volat_ok = dataframe["bb_width"] > 0.006
        gentle_drop = dataframe["ret"] > -0.02
        reversal = dataframe["close"] > dataframe["close"].shift(1)

        setup_revert = regime_ok & not_too_deep & bb_touch & vol_spike & volat_ok & gentle_drop & (
            (dataframe["rsi14"] < 28.0) | ((dataframe["rsi14"] < 32.0) & reversal)
        )

        setup_panic = (dataframe["rsi14"] < 24.0) & (dataframe["vol_ratio"] > 1.30) & (dataframe["adx14"] < 38.0) & (
            dataframe["close"] < (dataframe["ema50"] - 1.30 * dataframe["atr14"])
        )

        dataframe.loc[vol_ok & (setup_revert | setup_panic), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        vol_ok = dataframe["volume"] > 0

        take_mean = (dataframe["close"] > dataframe["bb_mid"]) & (dataframe["rsi14"] > 52.0)
        take_strength = (dataframe["close"] > dataframe["ema20"] * 1.003) & (dataframe["rsi14"] > 55.0)
        take_upper = dataframe["close"] > dataframe["bb_up"] * 0.995

        risk_off = (dataframe["adx14"] > 40.0) & (dataframe["close"] < dataframe["ema100"] * 0.985) & (dataframe["rsi14"] < 42.0)

        dataframe.loc[vol_ok & (take_mean | take_strength | take_upper | risk_off), "exit_long"] = 1
        return dataframe