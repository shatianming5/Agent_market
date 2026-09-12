"""Signal Validator — Gate 1 automated signal quality check.

Before wasting time on full backtests, check if the raw signal has any
predictive power (IC > 0.02, hit_rate > 52%).

Usage:
    from workspace.signal_validator import validate_signal, validate_pairs_signal
    result = validate_signal(signal_series, target_series)
    result = validate_pairs_signal("ADA/USDT", "AVAX/USDT", lookback=80, entry_z=2.5)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_signal(
    signal: pd.Series,
    target: pd.Series,
    *,
    min_ic: float = 0.02,
    min_hit_rate: float = 0.52,
    min_signals: int = 30,
) -> Dict[str, Any]:
    """Validate a raw signal against future returns.

    Args:
        signal: prediction signal (higher = expect positive return)
        target: actual future returns
        min_ic: minimum absolute IC (information coefficient)
        min_hit_rate: minimum directional hit rate
        min_signals: minimum non-zero signals

    Returns:
        {passed, ic, hit_rate, n_signals, failures}
    """
    # Align and clean
    combined = pd.DataFrame({"signal": signal, "target": target}).dropna()
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    if len(combined) < min_signals:
        return {"passed": False, "ic": 0, "hit_rate": 0, "n_signals": len(combined),
                "failures": [f"only {len(combined)} samples (need {min_signals})"]}

    sig = combined["signal"]
    tgt = combined["target"]

    # Only count non-zero signals
    active = combined[sig.abs() > 1e-10]
    n_signals = len(active)
    if n_signals < min_signals:
        return {"passed": False, "ic": 0, "hit_rate": 0, "n_signals": n_signals,
                "failures": [f"only {n_signals} active signals (need {min_signals})"]}

    # IC = correlation between signal and future return
    ic = float(sig.corr(tgt))
    if not np.isfinite(ic):
        ic = 0.0

    # Hit rate = % of times sign(signal) == sign(target)
    correct = ((active["signal"] > 0) & (active["target"] > 0)) | \
              ((active["signal"] < 0) & (active["target"] < 0))
    hit_rate = float(correct.sum() / len(active)) if len(active) > 0 else 0.0

    # Rolling IC stability
    rolling_ic = sig.rolling(200).corr(tgt).dropna()
    ic_stability = float(rolling_ic.mean() / (rolling_ic.std() + 1e-10)) if len(rolling_ic) > 20 else 0

    failures = []
    if abs(ic) < min_ic:
        failures.append(f"IC={ic:.4f} < {min_ic}")
    if hit_rate < min_hit_rate:
        failures.append(f"hit_rate={hit_rate:.2%} < {min_hit_rate:.0%}")

    return {
        "passed": len(failures) == 0,
        "ic": round(ic, 4),
        "hit_rate": round(hit_rate, 4),
        "ic_stability": round(ic_stability, 4),
        "n_signals": n_signals,
        "n_samples": len(combined),
        "failures": failures,
    }


def validate_pairs_signal(
    pair_a: str,
    pair_b: str,
    *,
    exchange: str = "gate",
    lookback: int = 80,
    entry_z: float = 2.5,
    exit_z: float = 0.7,
    forward_bars: int = 24,
) -> Dict[str, Any]:
    """Gate 1 check for a relative-value pairs signal (spot, long-only).

    Computes the log-spread z-score between two correlated assets.
    Checks if extreme z-scores predict spread convergence over the
    next N bars. NOTE: this validates the SIGNAL, not execution.
    In spot mode, we can only go long one asset at a time (no hedge).
    """
    from workspace.pairs_engine import PairsEngine

    pe = PairsEngine(pair_a, pair_b, exchange=exchange)
    df = pe.compute_spread(lookback=lookback)
    df = df.dropna(subset=["zscore"])

    if len(df) < lookback + forward_bars + 50:
        return {"passed": False, "failures": ["insufficient data"]}

    # Signal: -zscore (negative z = expect spread to increase = buy A)
    signal = -df["zscore"]

    # Target: future spread change over forward_bars
    target = df["spread"].shift(-forward_bars) - df["spread"]
    target = target.iloc[:-forward_bars]
    signal = signal.iloc[:-forward_bars]

    result = validate_signal(signal, target)
    result["pair"] = f"{pair_a}/{pair_b}"
    result["params"] = {"lookback": lookback, "entry_z": entry_z, "forward_bars": forward_bars}
    return result


__all__ = ["validate_signal", "validate_pairs_signal"]
