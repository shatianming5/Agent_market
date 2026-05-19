"""Fast Strategy Miner — optimized version with caching + early termination.

10x faster than naive loop by:
  A. Dedup cache: skip already-tested (pair, lb, ez, xz) combos
  B. Data preload: load all pair data once at startup
  C. Early termination: skip remaining windows if first 2 both lose
  D. Spread cache: precompute spreads per (pair, lookback)

Usage:
    from workspace.fast_miner import FastMiner
    miner = FastMiner(exchange="gate")
    report = miner.run(n_cycles=30)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MineResult:
    pair: str
    lookback: int
    entry_z: float
    exit_z: float
    mean_profit: float
    sharpe: float
    profitable_windows: int
    total_windows: int
    per_window: List[float]
    ic: float = 0.0


class FastMiner:
    """Optimized strategy miner with caching and early termination."""

    def __init__(self, exchange: str = "gate", cost_bps: float = 10.0):
        self.exchange = exchange
        self.cost_bps = cost_bps
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._result_cache: Dict[str, MineResult] = {}
        self._spread_cache: Dict[str, pd.DataFrame] = {}

    def preload_all_data(self, pairs: List[Tuple[str, str]]) -> int:
        """Load all pair data into memory once."""
        loaded = 0
        for a, b in pairs:
            key = f"{a}|{b}"
            if key in self._data_cache:
                continue
            try:
                from workspace.pairs_engine import PairsEngine
                pe = PairsEngine(a, b, exchange=self.exchange)
                df = pe.load_data()
                if len(df) > 500:
                    self._data_cache[key] = df
                    loaded += 1
            except Exception:
                pass
        return loaded

    def _get_spread(self, pair_key: str, lookback: int) -> Optional[pd.DataFrame]:
        """Get cached spread computation."""
        cache_key = f"{pair_key}|{lookback}"
        if cache_key in self._spread_cache:
            return self._spread_cache[cache_key]

        df = self._data_cache.get(pair_key)
        if df is None:
            return None

        # Compute spread inline (avoid PairsEngine overhead)
        log_a = np.log(df["price_a"].values.astype(float))
        log_b = np.log(df["price_b"].values.astype(float))

        # Rolling OLS hedge ratio
        n = len(df)
        ratios = np.full(n, np.nan)
        for i in range(lookback, n):
            la = log_a[i - lookback:i + 1]
            lb = log_b[i - lookback:i + 1]
            X = np.column_stack([lb, np.ones(len(lb))])
            try:
                beta = np.linalg.lstsq(X, la, rcond=None)[0][0]
            except Exception:
                beta = np.nan
            ratios[i] = beta

        spread = log_a - ratios * log_b
        spread_mean = pd.Series(spread).rolling(lookback).mean().values
        spread_std = pd.Series(spread).rolling(lookback).std().values + 1e-10
        zscore = (spread - spread_mean) / spread_std

        result = df.copy()
        result["spread"] = spread
        result["zscore"] = zscore
        result["hedge_ratio"] = ratios

        self._spread_cache[cache_key] = result
        return result

    def _backtest_window(self, df: pd.DataFrame, entry_z: float, exit_z: float) -> float:
        """Fast backtest on a single window — minimal overhead."""
        zscore = df["zscore"].values
        price_a = df["price_a"].values
        price_b = df["price_b"].values
        fee = self.cost_bps / 10000.0

        equity = 1000.0
        position = 0  # 0=flat, 1=long_a, -1=long_b
        entry_price = 0.0

        for i in range(len(zscore)):
            z = zscore[i]
            if np.isnan(z):
                continue

            if position == 0:
                if z < -entry_z:
                    position = 1
                    entry_price = price_a[i]
                    equity -= equity * 0.5 * fee
                elif z > entry_z:
                    position = -1
                    entry_price = price_b[i]
                    equity -= equity * 0.5 * fee
            else:
                if abs(z) < exit_z or (position == 1 and z > entry_z) or (position == -1 and z < -entry_z):
                    if position == 1:
                        pnl = (price_a[i] / entry_price - 1) * equity * 0.5
                    else:
                        pnl = (price_b[i] / entry_price - 1) * equity * 0.5
                    equity += pnl - equity * 0.5 * fee
                    position = 0

        return (equity / 1000.0 - 1) * 100

    def mine_single(self, pair_key: str, lookback: int, entry_z: float, exit_z: float,
                    *, early_terminate: bool = True) -> Optional[MineResult]:
        """Mine a single (pair, params) combo with caching + early termination."""
        cache_key = f"{pair_key}|{lookback}|{entry_z}|{exit_z}"
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        spread_df = self._get_spread(pair_key, lookback)
        if spread_df is None:
            return None

        n = len(spread_df)
        ws = n // 5
        if ws < 200:
            return None

        profits = []
        for wi in range(4):
            test_start = (wi + 1) * ws
            test_end = min(test_start + ws, n)
            window = spread_df.iloc[test_start:test_end]
            pnl = self._backtest_window(window, entry_z, exit_z)
            profits.append(pnl)

            # Early termination: if first 2 windows both lose badly, skip rest
            if early_terminate and wi == 1 and all(p < -3.0 for p in profits):
                self._result_cache[cache_key] = None
                return None

        mean_p = float(np.mean(profits))
        std_p = float(np.std(profits)) + 0.01
        sharpe = mean_p / std_p * 2
        pw = sum(1 for x in profits if x > 0)

        if mean_p <= 0 or pw < 2:
            self._result_cache[cache_key] = None
            return None

        result = MineResult(
            pair=pair_key.replace("|", "/"),
            lookback=lookback, entry_z=entry_z, exit_z=exit_z,
            mean_profit=mean_p, sharpe=sharpe,
            profitable_windows=pw, total_windows=4,
            per_window=profits,
        )
        self._result_cache[cache_key] = result
        return result

    def run(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        param_grid: Optional[List[Tuple[int, float, float]]] = None,
        n_cycles: int = 30,
    ) -> Dict[str, Any]:
        """Run the optimized mining loop."""
        if pairs is None:
            pairs = [
                ("BTC/USDT","ETH/USDT"),("BTC/USDT","SOL/USDT"),("ETH/USDT","SOL/USDT"),
                ("SOL/USDT","DOGE/USDT"),("DOGE/USDT","XRP/USDT"),("AVAX/USDT","DOGE/USDT"),
                ("ADA/USDT","AVAX/USDT"),("ADA/USDT","DOT/USDT"),("LINK/USDT","SOL/USDT"),
                ("DOGE/USDT","SOL/USDT"),("ADA/USDT","DOGE/USDT"),("LINK/USDT","XRP/USDT"),
                ("DOT/USDT","LINK/USDT"),("BTC/USDT","DOGE/USDT"),("ETH/USDT","LINK/USDT"),
            ]
        if param_grid is None:
            param_grid = [
                (40,1.8,0.3),(40,2.5,0.5),(60,2.0,0.5),(60,2.5,0.7),(60,3.0,0.5),
                (60,3.5,0.7),(60,4.0,1.0),(80,2.0,0.5),(80,2.5,0.7),(80,3.0,0.5),
                (80,3.5,0.7),(80,4.0,1.0),(100,2.5,0.7),(100,3.0,0.5),(120,2.5,0.7),
                (120,3.0,0.7),(120,3.5,1.0),(150,3.0,1.0),
            ]

        t_start = time.time()

        # Phase 1: Preload all data
        print(f"Preloading {len(pairs)} pairs...", end=" ", flush=True)
        loaded = self.preload_all_data(pairs)
        t_preload = time.time() - t_start
        print(f"{loaded} loaded ({t_preload:.1f}s)")

        # Phase 2: Mine all unique combos (no cycles needed — just test everything once)
        total_combos = len(pairs) * len(param_grid)
        print(f"Mining {total_combos} unique combos ({len(pairs)} pairs × {len(param_grid)} params)...", flush=True)

        all_results: List[MineResult] = []
        tested = 0
        cached = 0
        early_stopped = 0

        for a, b in pairs:
            pair_key = f"{a}|{b}"
            if pair_key not in self._data_cache:
                continue
            for lb, ez, xz in param_grid:
                tested += 1
                cache_key = f"{pair_key}|{lb}|{ez}|{xz}"
                if cache_key in self._result_cache:
                    cached += 1
                    r = self._result_cache[cache_key]
                    if r is not None:
                        all_results.append(r)
                    continue

                r = self.mine_single(pair_key, lb, ez, xz)
                if r is not None:
                    all_results.append(r)

        t_mine = time.time() - t_start - t_preload
        print(f"  Tested: {tested}, Cached: {cached}, Found: {len(all_results)}", flush=True)
        print(f"  Mine time: {t_mine:.1f}s ({tested/max(t_mine,0.1):.0f} combos/s)", flush=True)

        # Phase 3: Adaptive refinement — search near best results
        if all_results:
            print(f"Adaptive refinement on top {min(5, len(all_results))} results...", flush=True)
            top_results = sorted(all_results, key=lambda r: r.mean_profit, reverse=True)[:5]
            refined = 0
            for r in top_results:
                pair_key = r.pair.replace("/", "|")
                base_lb, base_ez, base_xz = r.lookback, r.entry_z, r.exit_z
                # Search ±20% around best params
                for lb_delta in [-20, -10, 0, 10, 20]:
                    for ez_delta in [-0.3, -0.15, 0, 0.15, 0.3]:
                        lb = max(20, base_lb + lb_delta)
                        ez = max(1.0, base_ez + ez_delta)
                        xz = base_xz
                        new_r = self.mine_single(pair_key, lb, ez, xz)
                        if new_r is not None:
                            all_results.append(new_r)
                            refined += 1
            t_refine = time.time() - t_start - t_preload - t_mine
            print(f"  Refined: +{refined} new results ({t_refine:.1f}s)", flush=True)

        total_time = time.time() - t_start
        all_results.sort(key=lambda r: r.mean_profit, reverse=True)

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            key = f"{r.pair}_{r.lookback}_{r.entry_z}_{r.exit_z}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_s": round(total_time, 1),
            "preload_time_s": round(t_preload, 1),
            "combos_tested": tested,
            "cache_hits": cached,
            "results_found": len(unique_results),
            "top_10": [
                {"pair": r.pair, "lb": r.lookback, "ez": r.entry_z, "xz": r.exit_z,
                 "profit": round(r.mean_profit, 2), "sharpe": round(r.sharpe, 2),
                 "pw": r.profitable_windows, "windows": [round(p, 2) for p in r.per_window]}
                for r in unique_results[:10]
            ],
            "best": {
                "pair": unique_results[0].pair if unique_results else "",
                "profit": round(unique_results[0].mean_profit, 2) if unique_results else 0,
                "sharpe": round(unique_results[0].sharpe, 2) if unique_results else 0,
            } if unique_results else None,
        }

        # Save
        out = ROOT / "workspace" / "results" / f"fast_mine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))

        return report


__all__ = ["FastMiner", "MineResult"]
