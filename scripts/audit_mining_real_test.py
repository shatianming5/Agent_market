#!/usr/bin/env python3
"""Post-mining REAL_TEST3 audit — one-shot honest OOS measurement.

Mining selects on VAL3 (2025-07 → 2025-12). This script takes the exported
survivor library (user_data/freqai_expressions_<tag>.json) and reports
per-factor IC on three sections: TRAIN3 / VAL3 / REAL_TEST3. The goal is to
quantify selection-bias decay VAL→REAL without ever using REAL_TEST3 as a
gate.

Usage:
    python scripts/audit_mining_real_test.py --tag v7_1h_funding --timeframe 1h
    python scripts/audit_mining_real_test.py --tag v7_4h --timeframe 4h

Emits a table to stdout and writes
    user_data/freqai_expressions_<tag>_realtest_audit.json
with per-factor per-section IC so this one-shot audit isn't re-run
repeatedly (which would itself induce snooping).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_market.factor_lab.mining import build_big, spearman  # noqa: E402
from agent_market.factor_lab.paths import (  # noqa: E402
    DEFAULT_PAIRS, DEFAULT_TRAIN3, DEFAULT_VAL3, DEFAULT_REAL_TEST3,
)
from agent_market.freqai.expression_engine import safe_eval_expression  # noqa: E402


def section_ic(big: pd.DataFrame, expr: str, section: tuple) -> tuple[float, int]:
    """Return (mean IC across pairs, sign_agree) for a factor on one date section."""
    ts = pd.Timestamp(section[0], tz="UTC")
    te = pd.Timestamp(section[1], tz="UTC")
    ics: List[float] = []
    for pair in DEFAULT_PAIRS:
        sub = big.loc[big["__pair__"] == pair]
        if len(sub) < 200:
            continue
        series = safe_eval_expression(expr, sub)
        m = (sub["date"] >= ts) & (sub["date"] < te)
        if m.sum() < 200:
            continue
        ic = spearman(series[m.values], sub.loc[m, "__fwd_ret__"])
        if np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return float("nan"), 0
    mean_ic = float(np.mean(ics))
    sig = sum(1 for ic in ics if ic * mean_ic > 0)
    return mean_ic, sig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="mining tag, e.g. v7_1h_funding")
    ap.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--expr-file", default=None,
                    help="override path; default is user_data/freqai_expressions_<tag>.json")
    args = ap.parse_args()

    expr_file = Path(args.expr_file) if args.expr_file else (
        ROOT / "user_data" / f"freqai_expressions_{args.tag}.json")
    if not expr_file.exists():
        print(f"[audit] expression file not found: {expr_file}")
        print(f"[audit] hint: run `factor_lab export --tag {args.tag}` first"
              " or check scripts/factor_lab.py for export logic")
        return 1

    data = json.loads(expr_file.read_text(encoding="utf-8-sig"))
    exprs = data.get("expressions", [])[:args.top_n]
    if not exprs:
        print(f"[audit] no expressions in {expr_file}")
        return 1

    print(f"[audit] tag={args.tag} tf={args.timeframe} factors={len(exprs)}")
    print(f"[audit] loading {args.timeframe} feature matrix...")
    big, _ = build_big(timeframe=args.timeframe)
    print(f"  rows={len(big):,}")

    rows = []
    print()
    print(f"{'Name':<8} {'TR3.IC':>8} {'VAL3.IC':>8} {'RT3.IC':>8} "
          f"{'Decay':>6} {'TR.sig':>6} {'VL.sig':>6} {'RT.sig':>6}")
    print("-" * 75)
    for e in exprs:
        name = e.get("name", "?")
        expr = (e.get("expression") or "").strip()
        if not expr:
            continue
        try:
            t_ic, t_sig = section_ic(big, expr, DEFAULT_TRAIN3)
            v_ic, v_sig = section_ic(big, expr, DEFAULT_VAL3)
            r_ic, r_sig = section_ic(big, expr, DEFAULT_REAL_TEST3)
        except Exception as exc:
            print(f"  {name}: EVAL ERROR {exc!s:.80}")
            continue
        decay = abs(r_ic) / (abs(v_ic) + 1e-9) if np.isfinite(r_ic) and np.isfinite(v_ic) else float("nan")
        row = {
            "name": name, "expression": expr,
            "train3_ic": t_ic, "val3_ic": v_ic, "real_test3_ic": r_ic,
            "train3_sign": t_sig, "val3_sign": v_sig, "real_test3_sign": r_sig,
            "decay_val_to_rt": decay,
        }
        rows.append(row)
        print(f"{name:<8} {t_ic:>+8.3f} {v_ic:>+8.3f} {r_ic:>+8.3f} "
              f"{decay:>6.2f} {t_sig:>3}/10 {v_sig:>3}/10 {r_sig:>3}/10")

    # Summaries
    val_abs = [abs(r["val3_ic"]) for r in rows if np.isfinite(r["val3_ic"])]
    rt_abs = [abs(r["real_test3_ic"]) for r in rows if np.isfinite(r["real_test3_ic"])]
    decays = [r["decay_val_to_rt"] for r in rows if np.isfinite(r["decay_val_to_rt"])]
    print()
    print(f"[audit] VAL3  median |IC|: {np.median(val_abs) if val_abs else 0:.3f}")
    print(f"[audit] RT3   median |IC|: {np.median(rt_abs) if rt_abs else 0:.3f}")
    print(f"[audit] VAL→RT median decay: {np.median(decays) if decays else 0:.3f}"
          " (>0.5 = generalizes well, <0.2 = overfit)")
    sign_sane = sum(1 for r in rows
                    if np.sign(r["val3_ic"]) == np.sign(r["real_test3_ic"])
                    and r["val3_ic"] != 0)
    print(f"[audit] sign(VAL)==sign(RT): {sign_sane}/{len(rows)}")

    out = ROOT / "user_data" / f"freqai_expressions_{args.tag}_realtest_audit.json"
    out.write_text(json.dumps({
        "tag": args.tag, "timeframe": args.timeframe,
        "train3": DEFAULT_TRAIN3, "val3": DEFAULT_VAL3,
        "real_test3": DEFAULT_REAL_TEST3,
        "per_factor": rows,
        "summary": {
            "val3_median_abs_ic": float(np.median(val_abs)) if val_abs else 0.0,
            "real_test3_median_abs_ic": float(np.median(rt_abs)) if rt_abs else 0.0,
            "val_to_rt_median_decay": float(np.median(decays)) if decays else 0.0,
            "sign_consistent": sign_sane, "n": len(rows),
        },
    }, indent=2), encoding="utf-8")
    print(f"\n[audit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
