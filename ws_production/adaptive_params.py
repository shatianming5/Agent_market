"""Adaptive Parameter Engine — auto-tunes strategy params based on recent data.

Instead of fixed parameters, recalibrate using a rolling window of recent
market data before each trading period.

Usage:
    from workspace.adaptive_params import AdaptiveEngine
    ae = AdaptiveEngine()
    best = ae.optimize_pairs("DOGE/USDT", "SOL/USDT", recent_days=60)
    # best = {"lookback": 80, "entry_z": 3.0, "exit_z": 0.5, "sharpe": 0.72}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveEngine:
    """Recalibrates strategy parameters using recent data."""

    def __init__(self, exchange: str = "gate"):
        self.exchange = exchange

    def optimize_pairs(
        self,
        pair_a: str,
        pair_b: str,
        *,
        recent_bars: int = 2000,
        param_grid: Optional[List[Tuple[int, float, float]]] = None,
        cost_bps: float = 10.0,
    ) -> Dict[str, Any]:
        """Find optimal pairs params on recent data via grid search.

        Uses the most recent `recent_bars` for in-sample optimization,
        then validates on the last 20% as mini-OOS.
        """
        from workspace.pairs_engine import PairsEngine

        pe = PairsEngine(pair_a, pair_b, exchange=self.exchange)
        df = pe.load_data()

        # Use most recent data only
        if len(df) > recent_bars:
            df = df.tail(recent_bars).reset_index(drop=True)

        # Split: 80% optimize, 20% validate
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        valid_df = df.iloc[split:]

        if param_grid is None:
            param_grid = [
                (60, 2.0, 0.5), (60, 2.5, 0.7), (60, 3.0, 0.5),
                (80, 2.0, 0.5), (80, 2.5, 0.7), (80, 3.0, 0.5), (80, 3.5, 0.7),
                (120, 2.0, 0.5), (120, 2.5, 0.7), (120, 3.0, 0.7),
            ]

        # Optimize on train
        best_train = None
        best_score = -999.0
        for lb, ez, xz in param_grid:
            pe_t = PairsEngine(pair_a, pair_b, exchange=self.exchange)
            pe_t._df = train_df.copy()
            sig = pe_t.generate_signals(lookback=lb, entry_z=ez, exit_z=xz)
            bt = pe_t.backtest(sig, maker_fee_bps=cost_bps)
            # Score: Sharpe (primary) + profit/100 (tiebreaker)
            score = bt.sharpe + bt.profit_pct / 100.0
            if score > best_score and bt.trades >= 5:
                best_score = score
                best_train = {"lookback": lb, "entry_z": ez, "exit_z": xz,
                              "train_sharpe": bt.sharpe, "train_profit": bt.profit_pct, "train_trades": bt.trades}

        if best_train is None:
            return {"error": "no valid params found", "pair": f"{pair_a}/{pair_b}"}

        # Validate best params on held-out data
        pe_v = PairsEngine(pair_a, pair_b, exchange=self.exchange)
        pe_v._df = valid_df.copy()
        sig_v = pe_v.generate_signals(
            lookback=best_train["lookback"],
            entry_z=best_train["entry_z"],
            exit_z=best_train["exit_z"],
        )
        bt_v = pe_v.backtest(sig_v, maker_fee_bps=cost_bps)

        return {
            "pair_a": pair_a, "pair_b": pair_b,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "recent_bars": recent_bars,
            "cost_bps": cost_bps,
            "best_params": {
                "lookback": best_train["lookback"],
                "entry_z": best_train["entry_z"],
                "exit_z": best_train["exit_z"],
            },
            "train": {
                "sharpe": best_train["train_sharpe"],
                "profit_pct": best_train["train_profit"],
                "trades": best_train["train_trades"],
            },
            "validation": {
                "sharpe": bt_v.sharpe,
                "profit_pct": bt_v.profit_pct,
                "trades": bt_v.trades,
                "win_rate": bt_v.win_rate,
            },
            "overfit_ratio": abs(best_train["train_sharpe"]) / (abs(bt_v.sharpe) + 0.01),
        }

    def optimize_directional(
        self,
        pair: str = "BTC/USDT",
        *,
        recent_bars: int = 2000,
        indicator_params: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """Optimize directional strategy params (RSI period, BB width, etc.)."""
        data_path = ROOT / "user_data" / "data" / self.exchange / f"{pair.replace('/', '_')}-1h.feather"
        if not data_path.exists():
            return {"error": f"data not found: {data_path}"}

        df = pd.read_feather(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if len(df) > recent_bars:
            df = df.tail(recent_bars).reset_index(drop=True)

        if indicator_params is None:
            indicator_params = {
                "rsi_period": [7, 14, 21],
                "bb_period": [15, 20, 30],
                "bb_std": [1.5, 2.0, 2.5],
                "rsi_entry": [25, 30, 35],
            }

        # Simple grid search: RSI+BB mean reversion signal
        best = None
        best_ic = -1
        forward = 12  # predict 12-bar return

        for rsi_p in indicator_params["rsi_period"]:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

            for bb_p in indicator_params["bb_period"]:
                sma = df["close"].rolling(bb_p).mean()
                std = df["close"].rolling(bb_p).std()

                for bb_s in indicator_params["bb_std"]:
                    bb_lower = sma - bb_s * std
                    # Signal: how far below BB lower (bigger = more oversold)
                    signal = (bb_lower - df["close"]) / (std + 1e-10)
                    signal = signal.where(rsi < 50, 0)  # only when RSI is low-ish

                    target = df["close"].pct_change(forward).shift(-forward)
                    combined = pd.DataFrame({"signal": signal, "target": target}).dropna()
                    if len(combined) < 100:
                        continue

                    ic = float(combined["signal"].corr(combined["target"]))
                    if np.isfinite(ic) and ic > best_ic:
                        best_ic = ic
                        best = {"rsi_period": rsi_p, "bb_period": bb_p, "bb_std": bb_s, "ic": ic}

        if best is None:
            return {"error": "no valid params", "pair": pair}

        return {
            "pair": pair,
            "optimized_at": datetime.now(timezone.utc).isoformat(),
            "recent_bars": recent_bars,
            "best_params": best,
            "ic": best["ic"],
        }

    def recalibrate_all_pairs(
        self,
        pairs_list: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Recalibrate all active pairs strategies."""
        if pairs_list is None:
            pairs_list = [
                ("DOGE/USDT", "SOL/USDT"),
                ("ADA/USDT", "AVAX/USDT"),
                ("LINK/USDT", "SOL/USDT"),
                ("ADA/USDT", "DOGE/USDT"),
                ("DOGE/USDT", "XRP/USDT"),
            ]

        results = []
        for a, b in pairs_list:
            r = self.optimize_pairs(a, b, **kwargs)
            results.append(r)

        # Save calibration
        out = ROOT / "workspace" / "results" / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

        return results


__all__ = ["AdaptiveEngine"]
