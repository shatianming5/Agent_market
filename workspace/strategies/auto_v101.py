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

import pandas as pd


class AutoStrategy_v101(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 60

    stoploss = -0.028
    minimal_roi = {"0": 0.02, "24": 0.012, "72": 0.006}
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    rsi_len = 14
    bb_len = 20
    bb_k = 2.0
    atr_len = 14
    ema_len = 50
    vol_len = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        df = dataframe
        for c in ("open", "high", "low", "close", "volume"):
            if c not in df.columns:
                df[c] = 0.0

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        vol = df["volume"].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / float(self.rsi_len)
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(0.0)

        mid = close.rolling(self.bb_len, min_periods=2).mean()
        std = close.rolling(self.bb_len, min_periods=2).std(ddof=0)
        upper = mid + (self.bb_k * std)
        lower = mid - (self.bb_k * std)
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_width"] = ((upper - lower) / mid.replace(0.0, np.nan)).fillna(0.0)

        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=(1.0 / float(self.atr_len)), adjust=False).mean().fillna(0.0)

        df["ema"] = close.ewm(span=self.ema_len, adjust=False).mean()
        df["ema_slope24"] = (df["ema"] - df["ema"].shift(24)).fillna(0.0)

        vol_ma = vol.rolling(self.vol_len, min_periods=2).mean()
        df["vol_ratio"] = (vol / vol_ma.replace(0.0, np.nan)).fillna(0.0)

        df["ret_24"] = (close / prev_close.shift(23).replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        df = dataframe
        for c in ("rsi", "bb_lower", "bb_mid", "bb_width", "atr", "ema", "ema_slope24", "vol_ratio"):
            if c not in df.columns:
                df[c] = 0.0

        close = df["close"].astype(float)
        rsi = df["rsi"].astype(float)
        bbl = df["bb_lower"].astype(float)
        bbm = df["bb_mid"].astype(float)
        atr = df["atr"].astype(float)
        ema = df["ema"].astype(float)
        slope24 = df["ema_slope24"].astype(float)
        vratio = df["vol_ratio"].astype(float)

        oversold = (close < (bbl * 1.002)) & (rsi < 30.5)
        rebound = (close > close.shift(1)) | (rsi > rsi.shift(1))
        regime_ok = (slope24 > (-1.15 * atr)) | (close > (ema * 0.985))
        liquidity_ok = (df["volume"] > 0) & (vratio > 0.6)
        not_too_far = close > (bbm * 0.94)

        enter = oversold & rebound & regime_ok & liquidity_ok & not_too_far

        df.loc[enter, "enter_long"] = 1
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe is None or dataframe.empty:
            return dataframe

        df = dataframe
        for c in ("rsi", "bb_mid", "bb_upper", "atr", "ema", "ema_slope24"):
            if c not in df.columns:
                df[c] = 0.0

        close = df["close"].astype(float)
        rsi = df["rsi"].astype(float)
        bbm = df["bb_mid"].astype(float)
        bbu = df["bb_upper"].astype(float)
        atr = df["atr"].astype(float)
        slope24 = df["ema_slope24"].astype(float)

        mean_revert_done = (close > bbm) & (rsi > 55.0)
        big_pop = (close > (bbu * 0.998)) | (rsi > 62.0)
        risk_off = (slope24 < (-1.6 * atr)) & (rsi < 45.0)

        exit_sig = (df["volume"] > 0) & (mean_revert_done | big_pop | risk_off)
        df.loc[exit_sig, "exit_long"] = 1
        return df

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        try:
            age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        except Exception:
            return None

        if age_h >= 18.0 and current_profit > 0.008:
            return "time_tp"
        if age_h >= 30.0 and current_profit > -0.004:
            return "time_flat"
        if age_h >= 48.0:
            return "time_max"
        return None