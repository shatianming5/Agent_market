#!/usr/bin/env python3
"""Clean remine — strict NO-OOS filtering.

Supersedes an earlier remine script that filtered on OOS IC (data snooping).
This version splits TRAIN3 (2023-05-15 → 2025-07-01) into a
chronological fit/holdout (80/20) and selects factors by holdout stats:

    - |fit_ic_mean|    >= 0.03       (signal strength on fit slice)
    - |hold_ic_mean|   >= 0.02       (carries to internal holdout)
    - sign_pairs_hold  >= 7/10       (hold IC agrees with hold-mean direction)
    - decay_ratio      >= 0.30       (|hold|/|fit| — no catastrophic decay)
    - fit_ic_std       <= 0.10       (cross-pair stability on fit slice)

REAL_TEST3 (2025-12-01 → 2026-04-12) is NEVER touched. Validate/backtest
run independently afterwards.

Output: user_data/freqai_expressions_gclean.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_market.freqai.features import apply_configured_features
from agent_market.freqai.expression_engine import safe_eval_expression


PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
         "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT"]
DATA_DIR = ROOT / "user_data" / "data" / "kucoin"
FEATURE_FILE = ROOT / "user_data" / "freqai_features_real.json"
OUT_FILE = ROOT / "user_data" / "freqai_expressions_gclean.json"

LABEL_PERIOD = 12
TRAIN3 = ("2023-05-15", "2025-07-01")
FIT_FRAC = 0.8   # chronological split inside TRAIN3

GATE = {
    "min_fit_ic":   0.03,
    "min_hold_ic":  0.02,
    "min_sign_pairs": 7,
    "min_decay_ratio": 0.30,
    "max_fit_ic_std": 0.10,
}


def spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 200: return float("nan")
    xa, ya = x[mask], y[mask]
    if xa.std() == 0 or ya.std() == 0: return float("nan")
    return float(xa.rank(method="average").corr(ya.rank(method="average")))


def build_features() -> pd.DataFrame:
    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig"))
    frames = []
    for pair in PAIRS:
        f = DATA_DIR / f"{pair.replace('/', '_')}-1h.feather"
        if not f.exists(): continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = apply_configured_features(df, feat_cfg)
        df["__pair__"] = pair
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"], utc=True)
    big["__lbl__"] = (big.groupby("__pair__")["close"].shift(-LABEL_PERIOD) / big["close"]) - 1.0
    return big


def load_clean_candidates() -> List[Dict]:
    """Pull every factor whose `snoop_level='clean'` from Hub, PLUS the
    hand-curated g-factors-13 and v6 exports as expression-text pool so
    the clean remine has a decent starting pool even before new mining."""
    from agent_market.factor_hub import Client
    c = Client()
    seen = set()
    out = []
    with c.connect() as conn:
        rows = conn.execute("""
            SELECT id, name, expression, origin, source_lib
            FROM factors
            WHERE json_extract(metadata, '$.snoop_level') = 'clean'
        """).fetchall()
    for r in rows:
        e = (r["expression"] or "").strip()
        if not e or e in seen: continue
        seen.add(e)
        out.append({"name": r["name"], "expression": e,
                    "origin": r["origin"] or "hub:clean",
                    "source": "hub_clean"})
    # Also pull the broad v6 / composite candidates as backup (already clean)
    return out


def evaluate_candidate(big: pd.DataFrame, expr: str) -> Dict[str, Any]:
    try:
        tr_start = pd.Timestamp(TRAIN3[0], tz="UTC")
        tr_end = pd.Timestamp(TRAIN3[1], tz="UTC")
        split_ts = tr_start + (tr_end - tr_start) * FIT_FRAC
        fit_ics, hold_ics = [], []
        fit_ic_by_pair: Dict[str, float] = {}
        hold_ic_by_pair: Dict[str, float] = {}
        for pair in PAIRS:
            sub = big.loc[big["__pair__"] == pair].copy()
            series = safe_eval_expression(expr, sub)
            sub["__f__"] = series
            m_fit = (sub["date"] >= tr_start) & (sub["date"] < split_ts)
            m_hold = (sub["date"] >= split_ts) & (sub["date"] < tr_end)
            ic_f = spearman(sub.loc[m_fit, "__f__"], sub.loc[m_fit, "__lbl__"])
            ic_h = spearman(sub.loc[m_hold, "__f__"], sub.loc[m_hold, "__lbl__"])
            fit_ic_by_pair[pair] = ic_f
            hold_ic_by_pair[pair] = ic_h
            if not np.isnan(ic_f): fit_ics.append(ic_f)
            if not np.isnan(ic_h): hold_ics.append(ic_h)
        if not fit_ics or not hold_ics:
            return {"status": "insufficient"}
        fit_mean = float(np.mean(fit_ics))
        hold_mean = float(np.mean(hold_ics))
        fit_std = float(np.std(fit_ics))
        sign_agree = sum(
            1 for p in PAIRS
            if not np.isnan(hold_ic_by_pair[p])
            and (hold_ic_by_pair[p] * hold_mean > 0)
        )
        decay_ratio = abs(hold_mean) / (abs(fit_mean) + 1e-9)
        return {
            "status": "ok",
            "fit_ic": fit_mean, "hold_ic": hold_mean,
            "fit_ic_std": fit_std,
            "sign_pairs_hold": sign_agree,
            "decay_ratio": decay_ratio,
            "fit_per_pair": fit_ic_by_pair,
            "hold_per_pair": hold_ic_by_pair,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:120]}


def passes_gate(m: Dict) -> bool:
    if m.get("status") != "ok": return False
    return (abs(m["fit_ic"]) >= GATE["min_fit_ic"]
            and abs(m["hold_ic"]) >= GATE["min_hold_ic"]
            and m["sign_pairs_hold"] >= GATE["min_sign_pairs"]
            and m["decay_ratio"] >= GATE["min_decay_ratio"]
            and m["fit_ic_std"] <= GATE["max_fit_ic_std"])


def main() -> int:
    print(f"[clean-remine] TRAIN3={TRAIN3}  fit_frac={FIT_FRAC}  gate={GATE}")
    print(f"[clean-remine] loading feature matrix ({len(PAIRS)} pairs)...")
    big = build_features()
    total = len(big)
    n_fit = int(((big["date"] >= pd.Timestamp(TRAIN3[0], tz="UTC"))
                  & (big["date"] < pd.Timestamp(TRAIN3[1], tz="UTC"))).sum())
    print(f"  total_rows={total:,}  train3_rows={n_fit:,}")

    candidates = load_clean_candidates()
    print(f"[clean-remine] candidate pool: {len(candidates)} clean expressions")
    if not candidates:
        print("[clean-remine] no clean candidates in Hub — run "
              "scripts/tag_factor_snoop_level.py first, then mine with "
              "eval_mode=composite to populate")
        return 1

    results = []
    n_errors = 0
    for i, c in enumerate(candidates):
        if i % 50 == 0:
            print(f"  [{i}/{len(candidates)}] passed so far: "
                  f"{sum(1 for r in results if r.get('passed'))}", flush=True)
        m = evaluate_candidate(big, c["expression"])
        if m.get("status") == "error": n_errors += 1; continue
        if m.get("status") != "ok": continue
        passed = passes_gate(m)
        results.append({"name": c["name"], "expression": c["expression"],
                        "origin": c.get("origin", ""),
                        "fit_ic": m["fit_ic"], "hold_ic": m["hold_ic"],
                        "fit_ic_std": m["fit_ic_std"],
                        "sign_pairs_hold": m["sign_pairs_hold"],
                        "decay_ratio": m["decay_ratio"],
                        "passed": passed})

    survivors = sorted([r for r in results if r["passed"]],
                       key=lambda r: abs(r["hold_ic"]), reverse=True)

    print(f"\nEvaluated: {len(results)}  Errors: {n_errors}  Survivors: {len(survivors)}")
    print(f"{'Name':<10} {'Fit.IC':>8} {'Hld.IC':>8} {'Decay':>6} {'Sign':>5} {'Std':>6}   Expr")
    print("-" * 120)
    for r in survivors[:40]:
        exshort = r["expression"][:70] + ("…" if len(r["expression"]) > 70 else "")
        print(f"{r['name']:<10} {r['fit_ic']:>+8.3f} {r['hold_ic']:>+8.3f} "
              f"{r['decay_ratio']:>6.2f} {r['sign_pairs_hold']:>3}/10 "
              f"{r['fit_ic_std']:>6.3f}   {exshort}")

    # Write output library (name factors gc001, gc002… to distinguish from g*)
    out = {
        "version": 1,
        "generated_at": "remine_clean_train3_holdout",
        "train3": TRAIN3,
        "fit_frac": FIT_FRAC,
        "label_period": LABEL_PERIOD,
        "gate": GATE,
        "note": "NO OOS leakage — selection only used TRAIN3 chronological 80/20 split.",
        "expressions": [
            {
                "name": f"gc{i+1:03d}",
                "expression": r["expression"],
                "origin": r.get("origin", ""),
                "fit_ic": r["fit_ic"],
                "hold_ic": r["hold_ic"],
                "decay_ratio": r["decay_ratio"],
                "category": "clean",
            } for i, r in enumerate(survivors)
        ],
    }
    OUT_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {len(survivors)} clean survivors → {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
