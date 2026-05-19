#!/usr/bin/env python3
"""GP factor miner v2 — multi-window fitness to prevent VAL3 memorization.

KEY CHANGES vs v1:
  1. fitness = mean rank-IC across 3 non-overlapping 6-month windows in TRAIN3
     (all windows must show SAME sign IC, else fitness = -1.0)
  2. VAL3 is used ONLY as final output gate (not during evolution)
  3. max_depth 3 for random trees, crossover max_d=5 (was 6)
  4. Novelty filter threshold 0.80 vs v7

Usage:
  python scripts/gp_factor_mine_v2.py --n-gen 80 --pop-size 200 --top-k 20
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_market.factor_lab.paths import DEFAULT_PAIRS
from agent_market.freqai.expression_engine import (
    load_expression_file, apply_expressions, safe_eval_expression,
)
from agent_market.freqai.features import apply_configured_features

Node = Dict[str, Any]

_UNARY  = ["z", "tanh", "sign", "neg"]
_BINARY = ["sub", "add"]
_WIND   = [
    ("ema",         [6, 12, 24, 48]),
    ("roll_mean",   [6, 12, 24]),
    ("roll_std",    [12, 24, 48]),
    ("rolling_max", [12, 24]),
    ("rolling_min", [12, 24]),
]
_REGIMES = [
    "adx_14 > 25",       "ema_pct_48 > 0",
    "macd_diff > 0",     "realized_vol_24 > realized_vol_72",
    "funding_z_200 > 1", "funding_z_200 < -1",
    "rsi_14 > 70",       "rsi_14 < 30",
]

# 3 non-overlapping 6-month windows inside TRAIN3
TRAIN_WINDOWS = [
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
]


# ─── Tree serialisation ───────────────────────────────────────────────────────

def node_to_str(n: Node) -> str:
    t = n["t"]
    if t == "leaf":
        return n["col"]
    if t == "unary":
        s = node_to_str(n["a"])
        return f"-({s})" if n["op"] == "neg" else f"{n['op']}({s})"
    if t == "binary":
        sym = {"sub": "-", "add": "+"}[n["op"]]
        return f"({node_to_str(n['a'])}) {sym} ({node_to_str(n['b'])})"
    if t == "wind":
        return f"{n['op']}({node_to_str(n['a'])}, {n['period']})"
    if t == "if":
        return f"ifelse({n['cond']}, {node_to_str(n['a'])}, {node_to_str(n['b'])})"
    raise ValueError(f"unknown node type: {t!r}")


def _depth(n: Node) -> int:
    t = n["t"]
    if t == "leaf": return 0
    if t in ("unary", "wind"): return 1 + _depth(n["a"])
    return 1 + max(_depth(n["a"]), _depth(n["b"]))


def _all_paths(n: Node, path: tuple = ()) -> Generator[tuple, None, None]:
    yield path
    t = n["t"]
    if t in ("unary", "wind"):
        yield from _all_paths(n["a"], path + ("a",))
    elif t in ("binary", "if"):
        yield from _all_paths(n["a"], path + ("a",))
        yield from _all_paths(n["b"], path + ("b",))


def _get(n: Node, path: tuple) -> Node:
    for k in path:
        n = n[k]
    return n


def _set(n: Node, path: tuple, val: Node) -> Node:
    if not path:
        return copy.deepcopy(val)
    n = dict(n)
    n[path[0]] = _set(n[path[0]], path[1:], val)
    return n


# ─── Random tree generation (max depth 3) ────────────────────────────────────

def _rand_tree(cols: List[str], regimes: List[str], rng: random.Random,
               D: int = 3) -> Node:
    """Random expression tree with depth ≤ D=3 — prevents deep memorisation."""
    if D <= 0 or (D <= 1 and rng.random() < 0.70):
        return {"t": "leaf", "col": rng.choice(cols)}

    r = rng.random()
    if r < 0.25:
        return {"t": "unary", "op": rng.choice(_UNARY),
                "a": _rand_tree(cols, regimes, rng, D - 1)}
    if r < 0.50:
        op = rng.choice(_BINARY)
        return {"t": "binary", "op": op,
                "a": _rand_tree(cols, regimes, rng, D - 1),
                "b": _rand_tree(cols, regimes, rng, D - 1)}
    if r < 0.75:
        wop, periods = rng.choice(_WIND)
        return {"t": "wind", "op": wop, "period": rng.choice(periods),
                "a": _rand_tree(cols, regimes, rng, D - 1)}
    if regimes and D >= 2:
        # Only allow 1 level of ifelse to prevent deep nesting
        cond = rng.choice(regimes)
        return {"t": "if", "cond": cond,
                "a": _rand_tree(cols, regimes, rng, min(D - 1, 1)),
                "b": _rand_tree(cols, regimes, rng, min(D - 1, 1))}
    return {"t": "unary", "op": rng.choice(["z", "tanh"]),
            "a": _rand_tree(cols, regimes, rng, D - 1)}


# ─── Genetic operators ────────────────────────────────────────────────────────

def crossover(p1: Node, p2: Node, rng: random.Random,
              max_d: int = 5) -> Tuple[Node, Node]:
    paths1 = list(_all_paths(p1))
    paths2 = list(_all_paths(p2))
    for _ in range(10):
        pt1 = rng.choice(paths1)
        pt2 = rng.choice(paths2)
        s1, s2 = _get(p1, pt1), _get(p2, pt2)
        if (_depth(p1) - len(pt1) + _depth(s2) <= max_d and
                _depth(p2) - len(pt2) + _depth(s1) <= max_d):
            return _set(p1, pt1, s2), _set(p2, pt2, s1)
    return copy.deepcopy(p1), copy.deepcopy(p2)


def mutate(n: Node, cols: List[str], regimes: List[str],
           rng: random.Random, rate: float = 0.15, max_d: int = 4) -> Node:
    if rng.random() < rate:
        return _rand_tree(cols, regimes, rng, max(1, max_d - _depth(n)))
    n = dict(n)
    t = n["t"]
    if t in ("unary", "wind"):
        n["a"] = mutate(n["a"], cols, regimes, rng, rate, max_d)
    elif t in ("binary", "if"):
        n["a"] = mutate(n["a"], cols, regimes, rng, rate, max_d)
        n["b"] = mutate(n["b"], cols, regimes, rng, rate, max_d)
    return n


# ─── Multi-window fitness ─────────────────────────────────────────────────────

def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def eval_fitness(expr: str, big: pd.DataFrame,
                 window_masks: List[np.ndarray],
                 ret_col: str = "__ret__") -> float:
    """Multi-window fitness: mean |rank-IC| across windows IF all same sign.

    Returns -1.0 if:
      - expression fails to evaluate
      - fewer than 2 windows have enough data
      - windows show mixed signs (factor is unstable)
    """
    if not (3 < len(expr) < 500):
        return -1.0
    try:
        series = safe_eval_expression(expr, big)
    except Exception:
        return -1.0
    series = series.replace([np.inf, -np.inf], np.nan)
    sv = series.values
    rv = big[ret_col].values

    ics: List[float] = []
    for mask in window_masks:
        ok = mask & np.isfinite(sv) & np.isfinite(rv)
        if ok.sum() < 80:
            continue
        ic = _rank_ic(sv[ok], rv[ok])
        if not np.isnan(ic):
            ics.append(ic)

    if len(ics) < 2:
        return -1.0

    # Require all windows to agree in sign — no sign-flip factors
    pos = sum(1 for v in ics if v > 0)
    neg = sum(1 for v in ics if v < 0)
    if pos > 0 and neg > 0:
        return -1.0  # unstable across time → reject

    mean_abs = float(np.mean(np.abs(ics)))
    # Consistency bonus: reward low variance across windows
    std_ic = float(np.std(ics)) if len(ics) > 1 else 0.0
    consistency = 1.0 / (1.0 + 5.0 * std_ic / (mean_abs + 1e-6))
    return mean_abs * (1.0 + 0.3 * consistency)


def _tourn(pop_fit: List[Tuple[float, Node]], rng: random.Random, k: int = 5) -> Node:
    return max(rng.choices(pop_fit, k=k), key=lambda x: x[0])[1]


# ─── Data construction ────────────────────────────────────────────────────────

def build_big(pairs: List[str], feat_cfg_path: Path, v7_path: Path,
              data_start: str, data_end: str,
              label_period: int = 12) -> Tuple[pd.DataFrame, List[np.ndarray], np.ndarray, List[str]]:
    """Build combined DataFrame, return per-window masks (for fitness) and val3 mask (for gate)."""
    feat_cfg = json.loads(feat_cfg_path.read_text(encoding="utf-8-sig"))
    v7_specs = load_expression_file(v7_path)

    ts = pd.Timestamp(data_start, tz="UTC")
    te = pd.Timestamp(data_end,   tz="UTC")
    # VAL3 gate mask: 2025-07-01 → 2025-12-01
    val3_s = pd.Timestamp("2025-07-01", tz="UTC")
    val3_e = pd.Timestamp("2025-12-01", tz="UTC")

    dfs = []
    for pair in pairs:
        f = ROOT / "user_data" / "data" / "kucoin" / f"{pair.replace('/','_')}-1h.feather"
        if not f.exists():
            continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = apply_configured_features(df, feat_cfg)
        df, _ = apply_expressions(df, v7_specs, on_error="skip")
        df["__ret__"] = df["close"].pct_change(label_period).shift(-label_period)
        ok = (df["date"] >= ts) & (df["date"] < te) & df["close"].notna() & df["__ret__"].notna()
        if ok.sum() > 0:
            dfs.append(df.loc[ok].reset_index(drop=True))

    if not dfs:
        raise RuntimeError("no data loaded")
    big = pd.concat(dfs, ignore_index=True).copy()

    # Build per-window masks (TRAIN3 sub-windows for fitness)
    window_masks = []
    for ws, we in TRAIN_WINDOWS:
        wts = pd.Timestamp(ws, tz="UTC"); wte = pd.Timestamp(we, tz="UTC")
        m = ((big["date"] >= wts) & (big["date"] < wte)).values
        if m.sum() > 200:
            window_masks.append(m)

    val3_mask = ((big["date"] >= val3_s) & (big["date"] < val3_e)).values

    excl = {"date", "open", "high", "low", "close", "volume", "__ret__"}
    cols = [c for c in big.columns if c not in excl]
    w_sizes = [m.sum() for m in window_masks]
    print(f"[data] {len(big):,} rows | {len(cols)} cols | "
          f"train windows={w_sizes} | val3={val3_mask.sum():,}")
    return big, window_masks, val3_mask, cols


# ─── Novelty filter ───────────────────────────────────────────────────────────

def novelty_filter(candidates: List[Tuple[float, str]],
                   big: pd.DataFrame,
                   mask: np.ndarray,
                   v7_cols: List[str],
                   max_corr: float = 0.80) -> List[Tuple[float, str]]:
    present = [c for c in v7_cols if c in big.columns]
    v7_mat = big.loc[mask, present].values.astype(float)
    sample_idx = list(range(0, len(present), max(1, len(present) // 12)))

    kept = []
    for fit, expr in candidates:
        try:
            s = safe_eval_expression(expr, big).replace([np.inf, -np.inf], np.nan)
            vals = s.values[mask].astype(float)
        except Exception:
            kept.append((fit, expr))
            continue
        ok = np.isfinite(vals)
        if ok.sum() < 50:
            kept.append((fit, expr))
            continue
        max_c = 0.0
        for j in sample_idx:
            vc = v7_mat[:, j]
            both = ok & np.isfinite(vc)
            if both.sum() < 30:
                continue
            c = np.corrcoef(vals[both], vc[both])[0, 1]
            if np.isfinite(c):
                max_c = max(max_c, abs(c))
        if max_c <= max_corr:
            kept.append((fit, expr))
    return kept


# ─── VAL3 gate ────────────────────────────────────────────────────────────────

def val3_gate(candidates: List[Tuple[float, str]],
              big: pd.DataFrame,
              val3_mask: np.ndarray,
              min_ic: float = 0.010) -> List[Tuple[float, str, float]]:
    """Apply VAL3 IC gate: keep factors where val3 IC > min_ic (any sign)."""
    ret_col = "__ret__"
    rv = big[ret_col].values
    kept = []
    for fit, expr in candidates:
        try:
            s = safe_eval_expression(expr, big).replace([np.inf, -np.inf], np.nan)
            sv = s.values
        except Exception:
            continue
        ok = val3_mask & np.isfinite(sv) & np.isfinite(rv)
        if ok.sum() < 50:
            continue
        val3_ic = _rank_ic(sv[ok], rv[ok])
        if np.isfinite(val3_ic) and abs(val3_ic) >= min_ic:
            kept.append((fit, expr, float(val3_ic)))
    return kept


# ─── Main GP loop ─────────────────────────────────────────────────────────────

def run_gp(big: pd.DataFrame,
           window_masks: List[np.ndarray],
           val3_mask: np.ndarray,
           feat_cols: List[str],
           v7_cols: List[str],
           regimes: List[str],
           n_gen: int, pop_size: int, elite_k: int,
           cx_rate: float, mut_rate: float, rng_seed: int) -> List[Tuple[float, str]]:

    rng = random.Random(rng_seed)
    all_terms = feat_cols + v7_cols

    # Seed population: 40% raw-feature trees, 60% v7-wrapped combos
    pop: List[Node] = []
    for col in v7_cols:
        pop.append({"t": "unary", "op": "z",    "a": {"t": "leaf", "col": col}})
        pop.append({"t": "unary", "op": "tanh",  "a": {"t": "leaf", "col": col}})
        if len(pop) >= int(pop_size * 0.60):
            break
    while len(pop) < pop_size:
        pop.append(_rand_tree(feat_cols, regimes, rng, D=3))
    pop = pop[:pop_size]

    seen: set = {node_to_str(n) for n in pop}
    hall_of_fame: List[Tuple[float, str]] = []

    print(f"[GP-v2] {n_gen} gen × {pop_size} pop | elite={elite_k} | "
          f"fitness=multi-window(TRAIN3) depth≤3")

    for g in range(n_gen):
        t0 = time.time()
        scored: List[Tuple[float, Node]] = []
        for ind in pop:
            expr = node_to_str(ind)
            f = eval_fitness(expr, big, window_masks)
            scored.append((f, ind))
        scored.sort(key=lambda x: -x[0])

        n_valid = sum(1 for f, _ in scored if f > 0)
        elapsed = time.time() - t0
        print(f"  gen {g+1:3d}/{n_gen}  best={scored[0][0]:.4f}  "
              f"valid={n_valid}/{pop_size}  {elapsed:.0f}s", flush=True)

        for f, n in scored[:elite_k]:
            if f > 0:
                hall_of_fame.append((f, node_to_str(n)))

        elite = [n for _, n in scored[:elite_k]]
        scored_pf = [(f, n) for f, n in scored if f > -0.5] or scored
        offspring: List[Node] = []
        tries = 0
        while len(offspring) < pop_size - elite_k and tries < 8000:
            tries += 1
            if rng.random() < cx_rate:
                p1 = _tourn(scored_pf, rng)
                p2 = _tourn(scored_pf, rng)
                c1, c2 = crossover(p1, p2, rng, max_d=5)
                for c in (mutate(c1, all_terms, regimes, rng, mut_rate),
                           mutate(c2, all_terms, regimes, rng, mut_rate)):
                    s = node_to_str(c)
                    if 3 < len(s) < 500 and s not in seen:
                        seen.add(s)
                        offspring.append(c)
                        if len(offspring) >= pop_size - elite_k:
                            break
            else:
                t = _rand_tree(all_terms, regimes, rng, D=3)
                s = node_to_str(t)
                if s not in seen and 3 < len(s) < 500:
                    seen.add(s)
                    offspring.append(t)

        pop = elite + offspring

    # Deduplicate hall-of-fame by expression string
    seen2: set = set()
    results = []
    for f, s in sorted(hall_of_fame, key=lambda x: -x[0]):
        if s not in seen2 and f > 0:
            seen2.add(s)
            results.append((f, s))
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="GP factor miner v2 (multi-window fitness)")
    ap.add_argument("--v7-file",     default="user_data/freqai_expressions_honest_v7.json")
    ap.add_argument("--feature-cfg", default="user_data/freqai_features_real.json")
    ap.add_argument("--data-start",  default="2024-01-01",
                    help="start of data window (covers all 3 fitness windows + val3)")
    ap.add_argument("--data-end",    default="2025-12-01",
                    help="end of data (includes VAL3 for gate check)")
    ap.add_argument("--n-gen",       type=int,   default=80)
    ap.add_argument("--pop-size",    type=int,   default=200)
    ap.add_argument("--elite-k",     type=int,   default=30)
    ap.add_argument("--cx-rate",     type=float, default=0.70)
    ap.add_argument("--mut-rate",    type=float, default=0.15)
    ap.add_argument("--top-k",       type=int,   default=20)
    ap.add_argument("--novelty-thr", type=float, default=0.80)
    ap.add_argument("--val3-min-ic", type=float, default=0.010,
                    help="min |IC| on VAL3 for final output gate")
    ap.add_argument("--out", default="user_data/freqai_expressions_gp_v2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v7_path  = ROOT / args.v7_file
    feat_path = ROOT / args.feature_cfg

    print("[GP-v2] Building data…")
    big, window_masks, val3_mask, all_cols = build_big(
        DEFAULT_PAIRS, feat_path, v7_path, args.data_start, args.data_end,
    )

    v7_names  = {e["name"] for e in json.loads(v7_path.read_text(encoding="utf-8-sig"))["expressions"]}
    v7_cols   = [c for c in all_cols if c in v7_names]
    feat_cols = [c for c in all_cols if c not in v7_names and c != "__ret__"]
    regimes   = [r for r in _REGIMES if r.split()[0] in big.columns]

    print(f"[GP-v2] terminals: {len(feat_cols)} raw + {len(v7_cols)} v7-cols | "
          f"regimes: {len(regimes)} | fitness windows: {len(window_masks)}")

    results = run_gp(
        big=big, window_masks=window_masks, val3_mask=val3_mask,
        feat_cols=feat_cols, v7_cols=v7_cols, regimes=regimes,
        n_gen=args.n_gen, pop_size=args.pop_size, elite_k=args.elite_k,
        cx_rate=args.cx_rate, mut_rate=args.mut_rate, rng_seed=args.seed,
    )
    print(f"[GP-v2] {len(results)} unique candidates before novelty filter")

    # Novelty filter on VAL3 period (use val3 for diversity check)
    if args.novelty_thr > 0 and v7_cols:
        print(f"[GP-v2] novelty filter (max_corr={args.novelty_thr}, val3 period)…")
        before = len(results)
        results = novelty_filter(results, big, val3_mask, v7_cols, args.novelty_thr)
        print(f"        {before} → {len(results)} after novelty filter")

    # VAL3 honest gate
    print(f"[GP-v2] applying VAL3 gate (|IC| ≥ {args.val3_min_ic})…")
    gated = val3_gate(results, big, val3_mask, args.val3_min_ic)
    print(f"        {len(results)} → {len(gated)} after VAL3 gate")

    # Sort by multi-window fitness (not val3 IC)
    gated.sort(key=lambda x: -x[0])
    top = gated[:args.top_k]

    print(f"\n[GP-v2] Top {len(top)} factors (fitness | val3_IC):")
    print(f"  {'rank':>4}  {'fitness':>8}  {'val3_ic':>8}  expression")
    for i, (fit, expr, val3_ic) in enumerate(top):
        snippet = expr[:80] + ("…" if len(expr) > 80 else "")
        print(f"  {i+1:4d}  {fit:8.5f}  {val3_ic:+8.4f}  {snippet}")

    out_exprs = [
        {"name": f"gp2_{i+1:03d}", "expression": expr,
         "_gp_fitness": round(fit, 6), "_val3_ic": round(val3_ic, 4)}
        for i, (fit, expr, val3_ic) in enumerate(top)
    ]
    out = ROOT / args.out
    out.write_text(json.dumps({
        "version": "gp_v2",
        "gp_config": {
            "n_gen": args.n_gen, "pop_size": args.pop_size, "elite_k": args.elite_k,
            "cx_rate": args.cx_rate, "mut_rate": args.mut_rate, "seed": args.seed,
            "max_depth": 3, "fitness": "multi-window-TRAIN3",
        },
        "fitness_windows": TRAIN_WINDOWS,
        "val3_gate": {"min_ic": args.val3_min_ic},
        "novelty_thr": args.novelty_thr,
        "n_candidates_total": len(results),
        "n_val3_passed": len(gated),
        "expressions": out_exprs,
    }, indent=2), encoding="utf-8")
    print(f"\n[GP-v2] → {out}  ({len(out_exprs)} factors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
