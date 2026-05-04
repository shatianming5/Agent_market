"""4-layer anti-overfitting detector for factor backtests.

Port of QuantGPT anti_overfit.py (MIT). Slightly simplified labels (English
recommendations) and tightened numerics for our wq_brain workflow.

Tests:
1. IC Stability        — yearly Spearman IC consistency
2. Sub-sample Stress   — IC across bull/bear/sideways + high/low vol regimes
3. Placebo Test        — random permutation + time-shift decay check
4. Half-life Estimation — exponential decay fit on multi-horizon IC
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _np():
    import numpy as np
    return np


def _pd():
    import pandas as pd
    return pd


@dataclass
class TestResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AntiOverfitResult:
    score: float           # 0-100 (passed/total × 100)
    recommendation: str    # "RECOMMEND" / "CAUTION" / "NEEDS_WORK" / "REJECT"
    tests: list[TestResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "recommendation": self.recommendation,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "tests": [
                {"name": t.name, "passed": t.passed, "details": t.details}
                for t in self.tests
            ],
        }


class AntiOverfitDetector:
    """Run 4 anti-overfitting tests on a factor backtest.

    Args:
        factor_df: DataFrame with columns trade_date, stock_code, factor_value, daily_ret.
        holding_period: forward-return horizon in trading days for the primary IC.
    """

    def __init__(self, factor_df, holding_period: int = 5) -> None:
        pd = _pd()
        self.df = factor_df.copy()
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.holding_period = holding_period
        self._prepare_forward_returns()

    def _prepare_forward_returns(self) -> None:
        self.df = self.df.sort_values(["stock_code", "trade_date"])
        self.df["fwd_ret"] = (
            self.df.groupby("stock_code")["daily_ret"]
            .transform(
                lambda s: s.shift(-1)
                .rolling(self.holding_period, min_periods=self.holding_period)
                .sum()
                .shift(-(self.holding_period - 1))
            )
        )

    def _calc_daily_ic(self, df=None):
        np = _np()
        pd = _pd()
        from scipy import stats as sp_stats
        data = df if df is not None else self.df
        valid = data.dropna(subset=["factor_value", "fwd_ret"])
        if valid.empty:
            return pd.Series(dtype=float)

        def _spearman(g):
            if len(g) < 5 or g["factor_value"].nunique() < 2:
                return np.nan
            corr, _ = sp_stats.spearmanr(g["factor_value"], g["fwd_ret"])
            return corr if not np.isnan(corr) else 0.0

        return valid.groupby("trade_date").apply(_spearman).dropna()

    def run_all(self) -> AntiOverfitResult:
        tests = [
            self.test_ic_stability(),
            self.test_subsample_stress(),
            self.test_placebo(),
            self.test_half_life(),
        ]
        passed = sum(1 for t in tests if t.passed)
        score = passed / len(tests) * 100
        if score >= 75:
            rec = "RECOMMEND"
        elif score >= 50:
            rec = "CAUTION"
        elif score >= 25:
            rec = "NEEDS_WORK"
        else:
            rec = "REJECT"
        return AntiOverfitResult(
            score=score, recommendation=rec, tests=tests,
            passed_count=passed, total_count=len(tests),
        )

    # ── Test 1: IC stability ─────────────────────────────────────────────
    def test_ic_stability(self) -> TestResult:
        np = _np()
        ic = self._calc_daily_ic()
        if len(ic) < 20:
            return TestResult("IC stability", False, {"error": "insufficient IC data"})
        ic_mean = float(ic.mean())
        positive_rate = float((ic > 0).sum() / len(ic))
        yearly_ic = ic.groupby(ic.index.year).mean()
        overall_sign = np.sign(ic_mean)
        yearly_signs = np.sign(yearly_ic.values)
        has_reversal = bool((yearly_signs != overall_sign).any()) if overall_sign != 0 else True
        passed = (positive_rate >= 0.55) and (abs(ic_mean) >= 0.02) and (not has_reversal)
        return TestResult("IC stability", passed, {
            "ic_mean": round(ic_mean, 4),
            "positive_rate": round(positive_rate, 4),
            "yearly_ic": {str(y): round(float(v), 4) for y, v in yearly_ic.items()},
            "has_reversal": has_reversal,
        })

    # ── Test 2: sub-sample stress ────────────────────────────────────────
    def test_subsample_stress(self) -> TestResult:
        np = _np()
        ic = self._calc_daily_ic()
        if len(ic) < 40:
            return TestResult("Sub-sample stress", False, {"error": "insufficient data"})
        overall_sign = np.sign(ic.mean())
        if overall_sign == 0:
            return TestResult("Sub-sample stress", False, {"error": "overall IC = 0"})

        market_ret = self.df.groupby("trade_date")["daily_ret"].mean()
        market_ret = market_ret.reindex(ic.index).fillna(0)
        cum60 = market_ret.rolling(60, min_periods=30).sum()
        vol60 = market_ret.rolling(60, min_periods=30).std()

        sub_ics: dict[str, float] = {}
        for name, mask in [
            ("bull", cum60 > 0.05),
            ("bear", cum60 < -0.05),
            ("sideways", (cum60 >= -0.05) & (cum60 <= 0.05)),
        ]:
            aligned = mask.reindex(ic.index).fillna(False)
            sub = ic[aligned]
            if len(sub) >= 10:
                sub_ics[name] = float(sub.mean())
        median_vol = vol60.median()
        for name, mask in [("high_vol", vol60 > median_vol), ("low_vol", vol60 <= median_vol)]:
            aligned = mask.reindex(ic.index).fillna(False)
            sub = ic[aligned]
            if len(sub) >= 10:
                sub_ics[name] = float(sub.mean())

        if not sub_ics:
            return TestResult("Sub-sample stress", False, {"error": "subsample classification failed"})

        same_sign = sum(1 for v in sub_ics.values() if np.sign(v) == overall_sign)
        consistency = same_sign / len(sub_ics)
        passed = consistency >= 0.6

        return TestResult("Sub-sample stress", passed, {
            "overall_ic_sign": int(overall_sign),
            "sub_sample_ics": {k: round(v, 4) for k, v in sub_ics.items()},
            "consistency": round(consistency, 4),
        })

    # ── Test 3: placebo ──────────────────────────────────────────────────
    def test_placebo(self, *, n_permutations: int = 20) -> TestResult:
        np = _np()
        ic = self._calc_daily_ic()
        if len(ic) < 20:
            return TestResult("Placebo", False, {"error": "insufficient IC data"})
        real_ic = float(ic.mean())

        rng = np.random.RandomState(42)
        valid = self.df.dropna(subset=["factor_value", "fwd_ret"])
        sampled_dates = sorted(valid["trade_date"].unique())[::5]  # sample 1 in 5
        valid_sampled = valid[valid["trade_date"].isin(sampled_dates)]

        perm_ics: list[float] = []
        for _ in range(n_permutations):
            shuffled = valid_sampled.copy()
            shuffled["factor_value"] = (
                shuffled.groupby("trade_date")["factor_value"]
                .transform(lambda s: s.sample(frac=1, random_state=rng).values)
            )
            perm = self._calc_daily_ic(shuffled)
            if len(perm) > 0:
                perm_ics.append(float(perm.mean()))

        if len(perm_ics) < 10:
            return TestResult("Placebo", False, {"error": "permutation data insufficient"})

        perm_95 = float(np.percentile(perm_ics, 95))
        perm_pass = abs(real_ic) > abs(perm_95)

        # Time shift decay
        shift_ics: dict[int, float] = {}
        for shift in (5, 10, 20):
            shifted = self.df.copy()
            shifted["factor_value"] = shifted.groupby("stock_code")["factor_value"].shift(shift)
            sic = self._calc_daily_ic(shifted)
            if len(sic) > 0:
                shift_ics[shift] = float(sic.mean())
        decay_ok = True
        for v in shift_ics.values():
            if abs(v) >= abs(real_ic):
                decay_ok = False
                break

        passed = perm_pass and decay_ok
        return TestResult("Placebo", passed, {
            "real_ic": round(real_ic, 4),
            "perm_95th": round(perm_95, 4),
            "perm_pass": perm_pass,
            "shift_ics": {str(k): round(v, 4) for k, v in shift_ics.items()},
            "decay_ok": decay_ok,
        })

    # ── Test 4: half-life estimation ─────────────────────────────────────
    def test_half_life(self) -> TestResult:
        np = _np()
        from scipy.optimize import curve_fit
        from scipy import stats as sp_stats
        periods = [1, 2, 5, 10, 20, 40]
        period_ics: dict[int, float] = {}

        valid = self.df.dropna(subset=["factor_value"]).copy()
        valid = valid.sort_values(["stock_code", "trade_date"])
        sampled_dates = sorted(valid["trade_date"].unique())[::3]
        valid = valid[valid["trade_date"].isin(sampled_dates)]

        for p in periods:
            valid[f"fwd_ret_{p}"] = (
                valid.groupby("stock_code")["daily_ret"]
                .transform(
                    lambda s: s.shift(-1)
                    .rolling(p, min_periods=p)
                    .sum()
                    .shift(-(p - 1))
                )
            )
            sub = valid.dropna(subset=["factor_value", f"fwd_ret_{p}"])
            if sub.empty:
                continue

            def _spearman_p(g, col=f"fwd_ret_{p}"):
                if len(g) < 5 or g["factor_value"].nunique() < 2:
                    return np.nan
                corr, _ = sp_stats.spearmanr(g["factor_value"], g[col])
                return corr if not np.isnan(corr) else 0.0

            ic_s = sub.groupby("trade_date").apply(_spearman_p).dropna()
            if len(ic_s) > 0:
                period_ics[p] = abs(float(ic_s.mean()))

        if len(period_ics) < 3:
            return TestResult("Half-life", False, {"error": "insufficient multi-period IC"})

        x = np.array(list(period_ics.keys()), dtype=float)
        y = np.array(list(period_ics.values()), dtype=float)

        half_life = 999.0
        try:
            def _exp_decay(t, a, b):
                return a * np.exp(-b * t)
            popt, _ = curve_fit(_exp_decay, x, y, p0=[y[0], 0.05], maxfev=5000)
            _, b = popt
            half_life = float(np.log(2) / b) if b > 0 else 999.0
        except Exception:
            sorted_p = sorted(period_ics.items())
            if len(sorted_p) >= 2:
                ic_first, ic_last = sorted_p[0][1], sorted_p[-1][1]
                t_span = sorted_p[-1][0] - sorted_p[0][0]
                if ic_first > 0 and ic_last > 0 and ic_last < ic_first:
                    b_est = np.log(ic_first / ic_last) / t_span
                    half_life = float(np.log(2) / b_est) if b_est > 0 else 999.0

        passed = half_life > 5.0
        return TestResult("Half-life", passed, {
            "half_life_days": round(half_life, 1),
            "period_ics": {str(k): round(v, 4) for k, v in period_ics.items()},
        })


def run_anti_overfit(factor_df, *, holding_period: int = 5) -> dict[str, Any]:
    """Convenience: run 4 tests, return serializable dict."""
    return AntiOverfitDetector(factor_df, holding_period=holding_period).run_all().to_dict()


def run_anti_overfit_for_expression(
    expr: str,
    ohlcv: Optional[Any] = None,
    *,
    holding_period: int = 5,
) -> dict[str, Any]:
    """High-level: evaluate `expr` against cached OHLCV, then run anti-overfit."""
    from .local_sim import evaluate_expression
    from .data_loader import load_cached_ohlcv
    if ohlcv is None:
        ohlcv = load_cached_ohlcv()
    if ohlcv is None or len(ohlcv) == 0:
        raise RuntimeError("no cached OHLCV; run `wq_brain fetch-data` first")
    work = evaluate_expression(expr, ohlcv)
    if len(work) < 100:
        return {
            "score": 0.0,
            "recommendation": "REJECT",
            "tests": [],
            "error": "insufficient evaluated rows",
        }
    return run_anti_overfit(work, holding_period=holding_period)
