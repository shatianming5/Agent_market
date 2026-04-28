"""Strategy Ensemble + Market Regime Detection.

Combines multiple strategy signals with regime-aware weighting.

Usage:
    from workspace.ensemble import RegimeDetector, StrategyEnsemble
    regime = RegimeDetector()
    ensemble = StrategyEnsemble(strategies=[...], regime_detector=regime)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeState:
    """Current market regime classification."""
    regime: str          # "trending_up", "trending_down", "ranging", "volatile"
    confidence: float    # 0-1
    volatility: float    # current vol level
    trend_strength: float  # ADX-like measure


class RegimeDetector:
    """Detects market regime from OHLCV data."""

    def __init__(
        self,
        *,
        vol_lookback: int = 24,
        trend_lookback: int = 48,
        vol_threshold_high: float = 1.5,  # multiplier of median vol
        adx_threshold: float = 25.0,
    ):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.vol_threshold_high = vol_threshold_high
        self.adx_threshold = adx_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime columns to dataframe.

        Adds: regime, regime_confidence, volatility_regime, trend_strength
        """
        df = df.copy()

        # Volatility: ATR-based
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        atr = tr.rolling(self.vol_lookback).mean()
        atr_median = atr.rolling(self.vol_lookback * 4).median()
        df["vol_ratio"] = atr / (atr_median + 1e-10)

        # Trend: EMA slope + directional movement
        ema_fast = df["close"].ewm(span=12).mean()
        ema_slow = df["close"].ewm(span=self.trend_lookback).mean()
        df["trend_strength"] = ((ema_fast - ema_slow) / (atr + 1e-10)).abs()

        trend_direction = np.sign(ema_fast - ema_slow)

        # Classify regime
        is_volatile = df["vol_ratio"] > self.vol_threshold_high
        is_trending = df["trend_strength"] > 1.0
        is_up = trend_direction > 0

        conditions = [
            is_trending & is_up & ~is_volatile,      # trending up, normal vol
            is_trending & ~is_up & ~is_volatile,     # trending down, normal vol
            is_volatile,                              # high volatility
        ]
        choices = ["trending_up", "trending_down", "volatile"]
        df["regime"] = np.select(conditions, choices, default="ranging")

        # Confidence: how clear is the regime signal
        df["regime_confidence"] = np.clip(df["trend_strength"] / 2.0, 0, 1)
        df.loc[is_volatile, "regime_confidence"] = np.clip(df.loc[is_volatile, "vol_ratio"] / 3.0, 0, 1)

        return df

    def current_regime(self, df: pd.DataFrame) -> RegimeState:
        """Get the current (last bar) regime."""
        enriched = self.detect(df)
        last = enriched.iloc[-1]
        return RegimeState(
            regime=str(last.get("regime", "ranging")),
            confidence=float(last.get("regime_confidence", 0)),
            volatility=float(last.get("vol_ratio", 1)),
            trend_strength=float(last.get("trend_strength", 0)),
        )


class StrategyEnsemble:
    """Combines multiple strategy signals with regime-aware weighting.

    Each strategy contributes a signal (-1 to +1).
    Ensemble aggregates with weights that depend on market regime.
    """

    def __init__(
        self,
        strategy_names: List[str],
        *,
        regime_weights: Optional[Dict[str, Dict[str, float]]] = None,
        default_weights: Optional[Dict[str, float]] = None,
    ):
        self.strategy_names = strategy_names

        # Default: equal weights
        n = len(strategy_names)
        self.default_weights = default_weights or {s: 1.0 / n for s in strategy_names}

        # Regime-specific weights
        self.regime_weights = regime_weights or {
            "trending_up": self.default_weights,
            "trending_down": {s: 0.5 / n for s in strategy_names},  # reduce exposure
            "ranging": self.default_weights,
            "volatile": {s: 0.3 / n for s in strategy_names},  # minimal exposure
        }

    def combine_signals(
        self,
        signals: Dict[str, float],
        regime: str = "ranging",
    ) -> Dict[str, Any]:
        """Combine strategy signals based on regime.

        Args:
            signals: {strategy_name: signal_value} where -1=short, 0=flat, +1=long
            regime: current market regime

        Returns:
            {
                "combined_signal": float (-1 to +1),
                "position_size_multiplier": float (0 to 1),
                "active_strategies": int,
                "regime": str,
                "weights_used": dict,
            }
        """
        weights = self.regime_weights.get(regime, self.default_weights)
        total_signal = 0.0
        total_weight = 0.0
        active = 0

        for name in self.strategy_names:
            sig = signals.get(name, 0.0)
            w = weights.get(name, 0.0)
            if abs(sig) > 0.01:
                active += 1
            total_signal += sig * w
            total_weight += abs(w)

        combined = total_signal / (total_weight + 1e-10)
        combined = np.clip(combined, -1.0, 1.0)

        # Position size: reduce in volatile/downtrend regimes
        size_mult = {
            "trending_up": 1.0,
            "trending_down": 0.5,
            "ranging": 0.7,
            "volatile": 0.3,
        }.get(regime, 0.5)

        return {
            "combined_signal": float(combined),
            "position_size_multiplier": float(size_mult),
            "active_strategies": active,
            "regime": regime,
            "weights_used": weights,
        }


__all__ = ["RegimeDetector", "RegimeState", "StrategyEnsemble"]
