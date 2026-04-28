"""Factor validation: sub-period IC stability + random baseline + correlation."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .paths import KUCOIN_DIR, FEATURE_FILE, DEFAULT_PAIRS, DEFAULT_REAL_TEST3

_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path: sys.path.insert(0, _SRC)
from agent_market.freqai.expression_engine import safe_eval_expression
from agent_market.freqai.features import apply_configured_features


def _spearman(x, y):
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 100: return float("nan")
    xa, ya = x[mask], y[mask]
    if xa.std() == 0 or ya.std() == 0: return float("nan")
    return float(xa.rank(method="average").corr(ya.rank(method="average")))


def _load_big():
    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig"))
    frames = []
    for pair in DEFAULT_PAIRS:
        f = KUCOIN_DIR / f"{pair.replace('/', '_')}-1h.feather"
        if not f.exists(): continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.reset_index(drop=True)
        df = apply_configured_features(df, feat_cfg)
        df["__pair__"] = pair
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"], utc=True)
    big["__lbl__"] = (big.groupby("__pair__")["close"].shift(-12) / big["close"]) - 1.0
    return big


def _eval_period(big: pd.DataFrame, expr: str, start: str, end: str):
    ics = []
    for pair in DEFAULT_PAIRS:
        sub = big.loc[big["__pair__"] == pair]
        try: series = safe_eval_expression(expr, sub)
        except Exception: continue
        m = ((sub["date"] >= pd.Timestamp(start, tz="UTC"))
             & (sub["date"] < pd.Timestamp(end, tz="UTC")))
        ic = _spearman(series[m.values], sub.loc[m, "__lbl__"])
        if not np.isnan(ic): ics.append(ic)
    return (float(np.mean(ics)) if ics else float("nan"),
            sum(1 for ic in ics if ic * (np.mean(ics) if ics else 0) > 0),
            len(ics))


def validate(expressions_file: Path) -> Dict:
    """Full validation: sub-period stability + random baseline."""
    d = json.loads(Path(expressions_file).read_text(encoding="utf-8-sig"))
    exprs = d.get("expressions", [])
    if not exprs:
        return {"error": "no expressions"}

    print(f"[validate] loading features...")
    big = _load_big()

    # Sub-periods over REAL_TEST3 — the region that no miner (pre-v6) should
    # have touched during selection. Split into 3 chronological slices to
    # measure sign-consistency without re-testing on training data.
    rt_start, rt_end = DEFAULT_REAL_TEST3  # 2025-12-01 → 2026-04-12
    import datetime as _dt
    s = _dt.datetime.fromisoformat(rt_start)
    e = _dt.datetime.fromisoformat(rt_end)
    total_days = (e - s).days
    cut1 = s + _dt.timedelta(days=total_days // 3)
    cut2 = s + _dt.timedelta(days=2 * total_days // 3)
    _fmt = lambda d: d.strftime("%Y-%m-%d")
    PERIODS = {
        f"Slice-1 {_fmt(s)[2:]}": (_fmt(s), _fmt(cut1)),
        f"Slice-2 {_fmt(cut1)[2:]}": (_fmt(cut1), _fmt(cut2)),
        f"Slice-3 {_fmt(cut2)[2:]}": (_fmt(cut2), _fmt(e)),
    }

    print(f"\n[validate] sub-period IC stability across {len(exprs)} factors")
    print(f"{'name':<6} ", end="")
    for name in PERIODS: print(f"{name:<14}", end="")
    print("consistent?")
    print("-" * 90)
    stable = 0; flipping = 0
    per_factor = {}
    for e in exprs:
        name = e.get("name", "?")
        signs = []
        row = f"{name:<6} "
        for label, (s, eend) in PERIODS.items():
            ic, _, _ = _eval_period(big, e["expression"], s, eend)
            per_factor.setdefault(name, {})[label] = ic
            if not np.isnan(ic):
                signs.append(1 if ic > 0 else -1)
            row += f"IC={ic:+.3f}    "
        if len(signs) == 3 and len(set(signs)) == 1:
            stable += 1; row += "✓ YES"
        else:
            flipping += 1; row += "✗ FLIPS"
        print(row)

    # Random baseline
    print(f"\n[validate] random baseline ({len(exprs)} random factors)")
    rng = random.Random(42)
    base_feats = ["rsi_14","adx_14","ema_pct_12","macd_diff","atr_norm_14",
                  "ema_spread","di_spread","momentum_12","volume_ratio_20"]
    wrappers = ["z","sign","ema","tanh"]
    ops = ["+","-","*"]
    rnd_ics = []
    for _ in range(len(exprs)):
        n = rng.randint(2, 3)
        feats = rng.sample(base_feats, n)
        terms = []
        for f in feats:
            w = rng.choice(wrappers + [None])
            if w == "ema": terms.append(f"ema({f}, 12)")
            elif w is None: terms.append(f)
            else: terms.append(f"{w}({f})")
        expr = terms[0]
        for t in terms[1:]:
            expr = f"({expr}) {rng.choice(ops)} ({t})"
        ic, _, _ = _eval_period(big, expr, rt_start, rt_end)
        if not np.isnan(ic): rnd_ics.append(abs(ic))

    # Use the FULL REAL_TEST3 region as the reference IC for each factor
    # (avoids measuring only the first slice — fairer comparison with random).
    factor_ics = []
    for name in per_factor:
        ic_full, _, _ = _eval_period(big, next(e["expression"] for e in exprs if e["name"] == name), rt_start, rt_end)
        if not np.isnan(ic_full): factor_ics.append(abs(ic_full))
    factor_ics = [x for x in factor_ics if not np.isnan(x)]

    summary = {
        "n_factors": len(exprs),
        "sign_stable": stable,
        "sign_flipping": flipping,
        "mean_abs_ic": float(np.mean(factor_ics)) if factor_ics else 0,
        "random_mean_ic": float(np.mean(rnd_ics)) if rnd_ics else 0,
        "ic_lift_vs_random": float(np.mean(factor_ics) / (np.mean(rnd_ics) + 1e-9)) if rnd_ics and factor_ics else 0,
    }
    print(f"\nSummary: {stable}/{len(exprs)} stable | mean |IC|={summary['mean_abs_ic']:.3f} "
          f"| random={summary['random_mean_ic']:.3f} | lift={summary['ic_lift_vs_random']:.2f}x")
    return summary
