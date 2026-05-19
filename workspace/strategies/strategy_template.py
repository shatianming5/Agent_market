"""Base strategy template for OpenCode workspace.

Copy this file and modify to create a new strategy.
The backtest API will auto-discover strategies by class name.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pandas import DataFrame

# Inject project paths so agent_market imports work
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT))

from freqtrade.strategy import IStrategy  # noqa: E402


class StrategyTemplate(IStrategy):
    """
    Template strategy — agent should copy and modify this.

    Key parameters to customize:
    - timeframe: candle period ("1h", "5m", "1d")
    - minimal_roi: take-profit schedule {minutes: target_ratio}
    - stoploss: max loss per trade (negative decimal, e.g. -0.05 = -5%)
    - populate_indicators: add features/indicators to dataframe
    - populate_entry_trend: define entry conditions
    - populate_exit_trend: define exit conditions
    """

    timeframe = "1h"
    minimal_roi = {"0": 0.10, "240": -1}
    stoploss = -0.05
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 60
    can_short = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators to the dataframe.

        Available columns: date, open, high, low, close, volume
        You can compute anything here using pandas/numpy/ta-lib.
        """
        # Example: simple moving averages
        dataframe["sma_20"] = dataframe["close"].rolling(20).mean()
        dataframe["sma_50"] = dataframe["close"].rolling(50).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions. Set enter_long=1 where you want to buy."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["sma_20"] > dataframe["sma_50"]),
            ["enter_long", "enter_tag"],
        ] = (1, "sma_cross_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions. Set exit_long=1 where you want to sell."""
        dataframe.loc[
            (dataframe["volume"] > 0)
            & (dataframe["sma_20"] < dataframe["sma_50"]),
            "exit_long",
        ] = 1
        return dataframe
