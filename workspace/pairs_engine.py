"""Pairs Relative-Value Engine — spot-mode directional pairs trading.

IMPORTANT: This is NOT true relative-value pairs trading (which requires
simultaneous long+short). In spot mode we can only buy one asset at a time.
The strategy exploits mean-reverting spread between two correlated assets:
- When spread is low (z < -entry_z): buy asset A (expecting A to outperform)
- When spread is high (z > +entry_z): buy asset B (expecting B to outperform)
This is a RELATIVE VALUE play, not a hedged position.

Usage:
    from workspace.pairs_engine import PairsEngine
    pe = PairsEngine("BTC/USDT", "ETH/USDT", exchange="gate")
    signals = pe.generate_signals(lookback=100, entry_z=2.0, exit_z=0.5)
    backtest = pe.backtest(signals, maker_fee_bps=1.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PairsSignal:
    """Signal for a pairs trade."""
    timestamp: str
    action: str        # "enter_long_spread", "enter_short_spread", "exit", "hold"
    zscore: float
    spread: float
    hedge_ratio: float


@dataclass
class PairsBacktestResult:
    """Result of a pairs backtest."""
    trades: int
    profit_pct: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    avg_hold_hours: float
    total_cost_bps: float
    profit_after_cost_pct: float
    per_trade: List[Dict[str, Any]]


class PairsEngine:
    """Spot relative-value pairs engine (NOT market-neutral hedged)."""

    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        *,
        exchange: str = "gate",
        timeframe: str = "1h",
    ):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.exchange = exchange
        self.timeframe = timeframe
        self._df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load and merge price data for both assets."""
        if self._df is not None:
            return self._df

        slug_a = self.asset_a.replace("/", "_")
        slug_b = self.asset_b.replace("/", "_")
        path_a = ROOT / "user_data" / "data" / self.exchange / f"{slug_a}-{self.timeframe}.feather"
        path_b = ROOT / "user_data" / "data" / self.exchange / f"{slug_b}-{self.timeframe}.feather"

        df_a = pd.read_feather(path_a)
        df_b = pd.read_feather(path_b)
        df_a["date"] = pd.to_datetime(df_a["date"])
        df_b["date"] = pd.to_datetime(df_b["date"])

        # Merge on date
        df = df_a[["date", "close", "volume"]].rename(columns={"close": "price_a", "volume": "vol_a"})
        df = df.merge(
            df_b[["date", "close", "volume"]].rename(columns={"close": "price_b", "volume": "vol_b"}),
            on="date", how="inner",
        ).sort_values("date").reset_index(drop=True)

        self._df = df
        return df

    def cointegration_test(self, lookback: Optional[int] = None) -> Dict[str, Any]:
        """Test if the pair is cointegrated (mean-reverting spread)."""
        df = self.load_data()
        if lookback:
            df = df.tail(lookback)

        log_a = np.log(df["price_a"].values)
        log_b = np.log(df["price_b"].values)

        # OLS regression: log_a = beta * log_b + alpha + epsilon
        X = np.column_stack([log_b, np.ones(len(log_b))])
        beta, alpha = np.linalg.lstsq(X, log_a, rcond=None)[0]
        residuals = log_a - beta * log_b - alpha

        # ADF test (simplified: check if residuals are stationary)
        # Use Dickey-Fuller: delta_r = gamma * r_{t-1} + noise
        dr = np.diff(residuals)
        r_lag = residuals[:-1]
        if len(r_lag) < 10:
            return {"cointegrated": False, "reason": "insufficient data"}

        X_df = np.column_stack([r_lag, np.ones(len(r_lag))])
        try:
            gamma_hat, _ = np.linalg.lstsq(X_df, dr, rcond=None)[0]
        except Exception:
            return {"cointegrated": False, "reason": "lstsq failed"}

        # t-statistic for gamma
        predicted = X_df @ np.array([gamma_hat, _])
        sse = np.sum((dr - predicted) ** 2)
        mse = sse / (len(dr) - 2)
        var_gamma = mse / np.sum((r_lag - r_lag.mean()) ** 2)
        t_stat = gamma_hat / (np.sqrt(var_gamma) + 1e-10)

        # ADF critical values (approximate, 5% level)
        # For n>250: -2.87, for n>100: -2.88
        critical = -2.87
        is_coint = t_stat < critical

        half_life = -np.log(2) / gamma_hat if gamma_hat < 0 else float("inf")

        return {
            "cointegrated": bool(is_coint),
            "t_stat": float(t_stat),
            "critical_5pct": critical,
            "hedge_ratio": float(beta),
            "half_life_bars": float(half_life),
            "residual_std": float(np.std(residuals)),
            "correlation": float(np.corrcoef(log_a, log_b)[0, 1]),
            "n_samples": len(df),
        }

    def compute_spread(self, lookback: int = 100) -> pd.DataFrame:
        """Compute the spread and z-score."""
        df = self.load_data().copy()

        # Rolling hedge ratio (OLS over lookback window)
        log_a = np.log(df["price_a"])
        log_b = np.log(df["price_b"])

        # Simple approach: use expanding or rolling beta
        ratios = []
        for i in range(len(df)):
            start = max(0, i - lookback)
            if i - start < 20:
                ratios.append(np.nan)
                continue
            la = log_a.iloc[start:i+1].values
            lb = log_b.iloc[start:i+1].values
            X = np.column_stack([lb, np.ones(len(lb))])
            try:
                beta, _ = np.linalg.lstsq(X, la, rcond=None)[0]
            except Exception:
                beta = np.nan
            ratios.append(beta)

        df["hedge_ratio"] = ratios
        df["spread"] = log_a - df["hedge_ratio"] * log_b

        # Z-score of spread
        df["spread_mean"] = df["spread"].rolling(lookback).mean()
        df["spread_std"] = df["spread"].rolling(lookback).std()
        df["zscore"] = (df["spread"] - df["spread_mean"]) / (df["spread_std"] + 1e-10)

        return df

    def generate_signals(
        self,
        lookback: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> pd.DataFrame:
        """Generate pairs trading signals based on z-score.

        - Enter long spread (buy A, sell B) when zscore < -entry_z
        - Enter short spread (sell A, buy B) when zscore > +entry_z
        - Exit when |zscore| < exit_z
        """
        df = self.compute_spread(lookback)

        # In spot mode (no shorting), we can only:
        # - Buy A when spread is too low (expecting A to outperform B)
        # - Buy B when spread is too high (expecting B to outperform A)
        # This is a "relative value" play, not a true hedge

        signals = []
        position = 0  # 0=flat, 1=long_spread(buy A), -1=short_spread(buy B)

        for i, row in df.iterrows():
            z = row.get("zscore", np.nan)
            if np.isnan(z):
                signals.append("hold")
                continue

            if position == 0:
                if z < -entry_z:
                    signals.append("enter_long_spread")  # buy A (undervalued vs B)
                    position = 1
                elif z > entry_z:
                    signals.append("enter_short_spread")  # buy B (undervalued vs A)
                    position = -1
                else:
                    signals.append("hold")
            else:
                if abs(z) < exit_z:
                    signals.append("exit")
                    position = 0
                elif position == 1 and z > entry_z:  # spread reversed
                    signals.append("exit")
                    position = 0
                elif position == -1 and z < -entry_z:
                    signals.append("exit")
                    position = 0
                else:
                    signals.append("hold")

        df["signal"] = signals
        return df

    def backtest(
        self,
        signals_df: pd.DataFrame,
        *,
        capital: float = 1000.0,
        position_pct: float = 0.5,
        maker_fee_bps: float = 1.0,
    ) -> PairsBacktestResult:
        """Backtest the pairs signals.

        In spot mode:
        - Long spread = buy asset_a
        - Short spread = buy asset_b
        Each position uses position_pct of capital.
        """
        equity = capital
        position = None  # {"side": "long_a"/"long_b", "entry_price": float, "entry_idx": int}
        trades: List[Dict[str, Any]] = []
        equity_curve = [capital]
        fee_rate = maker_fee_bps / 10000.0

        for idx, row in signals_df.iterrows():
            signal = row.get("signal", "hold")
            price_a = row["price_a"]
            price_b = row["price_b"]

            if signal == "enter_long_spread" and position is None:
                # Buy asset A
                size = equity * position_pct
                cost = size * fee_rate
                position = {
                    "side": "long_a",
                    "entry_price": price_a,
                    "entry_idx": idx,
                    "size": size,
                    "entry_cost": cost,
                    "entry_z": row.get("zscore", 0),
                }
                equity -= cost

            elif signal == "enter_short_spread" and position is None:
                # Buy asset B
                size = equity * position_pct
                cost = size * fee_rate
                position = {
                    "side": "long_b",
                    "entry_price": price_b,
                    "entry_idx": idx,
                    "size": size,
                    "entry_cost": cost,
                    "entry_z": row.get("zscore", 0),
                }
                equity -= cost

            elif signal == "exit" and position is not None:
                exit_cost = position["size"] * fee_rate
                if position["side"] == "long_a":
                    pnl = position["size"] * (price_a / position["entry_price"] - 1)
                else:
                    pnl = position["size"] * (price_b / position["entry_price"] - 1)

                total_cost = position["entry_cost"] + exit_cost
                net_pnl = pnl - total_cost
                equity += net_pnl

                trades.append({
                    "side": position["side"],
                    "entry_z": position["entry_z"],
                    "exit_z": row.get("zscore", 0),
                    "pnl_pct": float(pnl / position["size"] * 100),
                    "cost_pct": float(total_cost / position["size"] * 100),
                    "net_pnl_pct": float(net_pnl / position["size"] * 100),
                    "hold_bars": int(idx - position["entry_idx"]),
                })
                position = None

            equity_curve.append(equity)

        # Metrics
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(24 * 365))

        max_dd = 0.0
        if len(eq) > 1:
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
            max_dd = float(abs(dd.min()) * 100)

        win_rate = 0.0
        if trades:
            win_rate = sum(1 for t in trades if t["net_pnl_pct"] > 0) / len(trades)

        avg_hold = np.mean([t["hold_bars"] for t in trades]) if trades else 0
        total_cost = sum(t["cost_pct"] for t in trades) if trades else 0

        return PairsBacktestResult(
            trades=len(trades),
            profit_pct=float((equity / capital - 1) * 100),
            sharpe=sharpe,
            max_dd_pct=max_dd,
            win_rate=win_rate,
            avg_hold_hours=float(avg_hold),
            total_cost_bps=float(total_cost * 100),
            profit_after_cost_pct=float((equity / capital - 1) * 100),
            per_trade=trades,
        )


def scan_pairs(
    exchange: str = "gate",
    timeframe: str = "1h",
    min_correlation: float = 0.7,
) -> List[Dict[str, Any]]:
    """Scan all available pairs and find cointegrated ones."""
    data_dir = ROOT / "user_data" / "data" / exchange
    files = sorted(data_dir.glob(f"*-{timeframe}.feather"))
    pairs = [f.stem.replace(f"-{timeframe}", "").replace("_", "/") for f in files]

    results = []
    for i, a in enumerate(pairs):
        for b in pairs[i+1:]:
            try:
                pe = PairsEngine(a, b, exchange=exchange, timeframe=timeframe)
                coint = pe.cointegration_test()
                if coint["correlation"] >= min_correlation:
                    results.append({
                        "pair_a": a,
                        "pair_b": b,
                        "correlation": coint["correlation"],
                        "cointegrated": coint["cointegrated"],
                        "half_life": coint["half_life_bars"],
                        "hedge_ratio": coint["hedge_ratio"],
                        "t_stat": coint["t_stat"],
                    })
            except Exception as _exc:
                import logging; logging.getLogger(__name__).debug("Skipped: %s", _exc)
                continue

    results.sort(key=lambda x: x["correlation"], reverse=True)
    return results


__all__ = ["PairsEngine", "PairsBacktestResult", "PairsSignal", "scan_pairs"]
