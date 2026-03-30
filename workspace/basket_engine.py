"""Multi-Asset Basket Trading Engine — PCA-based statistical arbitrage.

Instead of A vs B pairs, decomposes N assets into principal components.
Trades the residual: buy assets that are cheap relative to their
factor-model fair value, sell (or avoid) expensive ones.

Usage:
    from workspace.basket_engine import BasketEngine
    be = BasketEngine(exchange="gate")
    report = be.backtest_residual_strategy(lookback=200, entry_z=1.5)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BasketBacktestResult:
    trades: int
    profit_pct: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    per_asset_pnl: Dict[str, float]
    equity_curve: List[float]


class BasketEngine:
    """PCA-based multi-asset statistical arbitrage."""

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        *,
        exchange: str = "gate",
        timeframe: str = "1h",
    ):
        self.exchange = exchange
        self.timeframe = timeframe
        self.assets = assets or ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "ADA", "DOT", "LINK"]
        self._price_df: Optional[pd.DataFrame] = None

    def load_prices(self) -> pd.DataFrame:
        if self._price_df is not None:
            return self._price_df
        prices = {}
        for a in self.assets:
            p = ROOT / "user_data" / "data" / self.exchange / f"{a}_USDT-{self.timeframe}.feather"
            if p.exists():
                df = pd.read_feather(p)
                df["date"] = pd.to_datetime(df["date"])
                prices[a] = df.set_index("date")["close"]
        self._price_df = pd.DataFrame(prices).dropna()
        return self._price_df

    def compute_residuals(self, lookback: int = 200, n_factors: int = 1) -> pd.DataFrame:
        """Compute residuals from PCA factor model.

        For each bar, regress asset returns onto the top N PCA factors.
        The residual = asset-specific return (alpha opportunity).
        """
        prices = self.load_prices()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        assets = list(log_returns.columns)
        n_assets = len(assets)

        residuals_list = []
        zscores_list = []

        for i in range(lookback, len(log_returns)):
            window = log_returns.iloc[i - lookback:i].values  # (lookback, n_assets)

            # PCA on window
            cov = np.cov(window.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            factors = eigenvectors[:, idx[:n_factors]]  # (n_assets, n_factors)

            # Project returns onto factors
            current_return = log_returns.iloc[i].values  # (n_assets,)
            factor_scores = factors.T @ current_return  # (n_factors,)
            predicted = factors @ factor_scores  # (n_assets,)
            residual = current_return - predicted  # (n_assets,)

            # Z-score of residuals (using rolling std)
            hist_residuals = []
            for j in range(max(0, i - lookback), i):
                r = log_returns.iloc[j].values
                fs = factors.T @ r
                pred = factors @ fs
                hist_residuals.append(r - pred)

            hist_arr = np.array(hist_residuals)
            res_mean = hist_arr.mean(axis=0)
            res_std = hist_arr.std(axis=0) + 1e-10
            zscore = (residual - res_mean) / res_std

            residuals_list.append(residual)
            zscores_list.append(zscore)

        dates = log_returns.index[lookback:]
        residuals_df = pd.DataFrame(residuals_list, index=dates, columns=assets)
        zscores_df = pd.DataFrame(zscores_list, index=dates, columns=assets)

        return zscores_df

    def generate_signals(
        self,
        lookback: int = 200,
        n_factors: int = 1,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        max_positions: int = 3,
    ) -> pd.DataFrame:
        """Generate basket trading signals.

        Buy assets with zscore < -entry_z (cheap vs factor model).
        Sell/exit when zscore > -exit_z (reverted to fair value).
        """
        zscores = self.compute_residuals(lookback, n_factors)
        prices = self.load_prices().loc[zscores.index]
        assets = list(zscores.columns)

        signals = pd.DataFrame(0, index=zscores.index, columns=assets)

        for i in range(len(zscores)):
            z_row = zscores.iloc[i]

            # Find assets that are cheap (z < -entry_z)
            cheap = z_row[z_row < -entry_z].sort_values()
            # Find assets that reverted (|z| < exit_z)
            reverted = z_row[abs(z_row) < exit_z]

            # Buy cheapest N assets
            for asset in cheap.index[:max_positions]:
                signals.iloc[i][asset] = 1  # buy

            # Mark reverted as exit
            for asset in reverted.index:
                if i > 0 and signals.iloc[i - 1][asset] == 1:
                    signals.iloc[i][asset] = -1  # exit

        return signals

    def backtest(
        self,
        signals: pd.DataFrame,
        *,
        capital: float = 1000.0,
        per_position_pct: float = 0.1,
        maker_fee_bps: float = 1.0,
    ) -> BasketBacktestResult:
        """Backtest basket signals."""
        prices = self.load_prices().loc[signals.index]
        assets = list(signals.columns)
        fee = maker_fee_bps / 10000.0

        equity = capital
        positions = {a: 0.0 for a in assets}  # qty held
        entry_prices = {a: 0.0 for a in assets}
        per_asset_pnl = {a: 0.0 for a in assets}
        trades = 0
        wins = 0
        equity_curve = [capital]

        for i in range(len(signals)):
            for asset in assets:
                sig = signals.iloc[i][asset]
                price = prices.iloc[i][asset]

                if sig == 1 and positions[asset] == 0:
                    # Buy
                    size = equity * per_position_pct
                    qty = size / price
                    cost = size * fee
                    positions[asset] = qty
                    entry_prices[asset] = price
                    equity -= cost

                elif sig == -1 and positions[asset] > 0:
                    # Sell
                    pnl = positions[asset] * (price - entry_prices[asset])
                    cost = positions[asset] * price * fee
                    equity += pnl - cost
                    per_asset_pnl[asset] += pnl - cost
                    trades += 1
                    if pnl > cost:
                        wins += 1
                    positions[asset] = 0

            # Mark-to-market
            mtm = sum(positions[a] * prices.iloc[i][a] for a in assets if positions[a] > 0)
            equity_curve.append(equity + mtm - sum(positions[a] * entry_prices[a] for a in assets if positions[a] > 0))

        eq = np.array(equity_curve)
        returns = np.diff(eq) / (eq[:-1] + 1e-10)
        returns = returns[np.isfinite(returns)]
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(24 * 365)) if len(returns) > 1 else 0

        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / (peak + 1e-10)
        max_dd = float(abs(dd.min()) * 100) if len(dd) > 0 else 0

        return BasketBacktestResult(
            trades=trades,
            profit_pct=float((eq[-1] / capital - 1) * 100),
            sharpe=sharpe,
            max_dd_pct=max_dd,
            win_rate=wins / trades if trades > 0 else 0,
            per_asset_pnl={a: round(v, 2) for a, v in per_asset_pnl.items() if abs(v) > 0.01},
            equity_curve=equity_curve[::24],  # hourly → daily for storage
        )

    def walk_forward(
        self,
        *,
        n_windows: int = 4,
        lookback: int = 200,
        n_factors: int = 1,
        entry_z: float = 1.5,
        maker_fee_bps: float = 1.0,
    ) -> Dict[str, Any]:
        """Walk-forward validation of basket strategy."""
        prices = self.load_prices()
        n = len(prices)
        window_size = n // (n_windows + 1)

        results = []
        for wi in range(n_windows):
            test_start = (wi + 1) * window_size
            test_end = min(test_start + window_size, n)

            be_window = BasketEngine(self.assets, exchange=self.exchange, timeframe=self.timeframe)
            # Include lookback before test window for warm-up
            warm_start = max(0, test_start - lookback - 50)
            be_window._price_df = prices.iloc[warm_start:test_end].copy()

            signals = be_window.generate_signals(lookback=lookback, n_factors=n_factors, entry_z=entry_z)
            bt = be_window.backtest(signals, maker_fee_bps=maker_fee_bps)
            results.append(bt)

        profits = [r.profit_pct for r in results]
        sharpes = [r.sharpe for r in results]
        return {
            "n_windows": n_windows,
            "profits": profits,
            "mean_profit": float(np.mean(profits)),
            "mean_sharpe": float(np.mean(sharpes)),
            "profitable_windows": sum(1 for p in profits if p > 0),
            "total_trades": sum(r.trades for r in results),
            "per_window": [
                {"profit": r.profit_pct, "sharpe": r.sharpe, "trades": r.trades, "wr": r.win_rate}
                for r in results
            ],
        }


__all__ = ["BasketEngine", "BasketBacktestResult"]
