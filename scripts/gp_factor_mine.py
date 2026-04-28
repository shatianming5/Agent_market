#!/usr/bin/env python3
"""Genetic Programming factor miner.

Generation-0 seeds: v7 factor columns (hv7_*) + raw feature terminals.
GP operators:       subtree crossover + point mutation.
Fitness:            abs(rank-IC on VAL3) + TRAIN3 sign-consistency bonus.
Novelty filter:     post-evolution, drop |corr| > 0.80 with any v7 factor.
Output:             freqai_expressions_gp_v1.json (top-K survivors)

Usage:
  python scripts/gp_factor_mine.py --n-gen 100 --pop-size 200 --top-k 25
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

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

# ─── Expression tree representation ─────────────────────────────────────────
# Node is a plain dict (JSON-serializable, easy deepcopy).
# t="leaf"  : {"t":"leaf", "col": str}
# t="unary" : {"t":"unary", "op": str, "a": Node}
# t="binary": {"t":"binary","op": str, "a": Node, "b": Node}
# t="wind"  : {"t":"wind",  "op": str, "period": int, "a": Node}
# t="if"    : {"t":"if",    "cond": str, "a": Node, "b": Node}

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
    "mtf4h_rsi_14 > 60", "mtf4h_rsi_14 < 40",
    "adx_14 > 25",        "ema_pct_48 > 0",
    "macd_diff > 0",      "realized_vol_24 > realized_vol_72",
    "funding_z_200 > 1",  "funding_z_200 < -1",
    "rsi_14 > 70",        "rsi_14 < 30",
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


# ─── Tree traversal (for crossover point selection) ───────────────────────────

def _all_paths(n: Node, path: tuple = ()) -> Generator[tuple, None, None]:
    """Yield path tuples to every node in the tree."""
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
    """Return a new tree with node at *path* replaced by *val*."""
    if not path:
        return copy.deepcopy(val)
    n = dict(n)
    n[path[0]] = _set(n[path[0]], path[1:], val)
    return n


# ─── Random tree generation ───────────────────────────────────────────────────

def _rand_tree(cols: List[str], regimes: List[str], rng: random.Random,
               D: int = 4) -> Node:
    """Generate a random expression tree of depth ≤ D."""
    if D <= 0 or (D <= 2 and rng.random() < 0.55):
        return {"t": "leaf", "col": rng.choice(cols)}

    r = rng.random()
    if r < 0.22:
        return {"t": "unary", "op": rng.choice(_UNARY),
                "a": _rand_tree(cols, regimes, rng, D - 1)}
    if r < 0.47:
        op = rng.choice(_BINARY)
        return {"t": "binary", "op": op,
                "a": _rand_tree(cols, regimes, rng, D - 1),
                "b": _rand_tree(cols, regimes, rng, D - 1)}
    if r < 0.72:
        wop, periods = rng.choice(_WIND)
        return {"t": "wind", "op": wop, "period": rng.choice(periods),
                "a": _rand_tree(cols, regimes, rng, D - 1)}
    if regimes:
        cond = rng.choice(regimes)
        return {"t": "if", "cond": cond,
                "a": _rand_tree(cols, regimes, rng, D - 1),
                "b": _rand_tree(cols, regimes, rng, D - 1)}
    return {"t": "unary", "op": rng.choice(["z", "tanh"]),
            "a": _rand_tree(cols, regimes, rng, D - 1)}


# ─── Genetic operators ────────────────────────────────────────────────────────

def crossover(p1: Node, p2: Node, rng: random.Random,
              max_d: int = 6) -> Tuple[Node, Node]:
    """Swap random subtrees between two parents (up to 10 retries for depth)."""
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
           rng: random.Random, rate: float = 0.15, max_d: int = 5) -> Node:
    """Randomly replace subtrees at probability *rate*."""
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


# ─── Fitness ──────────────────────────────────────────────────────────────────

def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank-IC between x and y (both 1-D, no NaN assumed)."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def eval_fitness(expr: str, big: pd.DataFrame,
                 vl_mask: np.ndarray, tr_mask: np.ndarray,
                 ret_col: str = "__ret__") -> float:
    """Return fitness score (≥0 meaningful, <0 invalid)."""
    if not (4 < len(expr) < 700):
        return -1.0
    try:
        series = safe_eval_expression(expr, big)
    except Exception:
        return -1.0
    series = series.replace([np.inf, -np.inf], np.nan)

    # VAL3 rank-IC
    ok_v = vl_mask & series.notna().values & big[ret_col].notna().values
    if ok_v.sum() < 200:
        return -1.0
    ic_val = _rank_ic(series.values[ok_v], big[ret_col].values[ok_v])
    if np.isnan(ic_val):
        return -1.0

    # TRAIN3 stability bonus (+10% if same sign, -5% if opposite)
    ok_t = tr_mask & series.notna().values & big[ret_col].notna().values
    ic_tr = 0.0
    if ok_t.sum() > 200:
        ic_tr_raw = _rank_ic(series.values[ok_t], big[ret_col].values[ok_t])
        ic_tr = 0.0 if np.isnan(ic_tr_raw) else ic_tr_raw

    sign_bonus = (0.10 if ic_tr * ic_val > 0 else -0.05) * abs(ic_val)
    return abs(ic_val) + sign_bonus


def _tourn(pop_fit: List[Tuple[float, Node]], rng: random.Random, k: int = 5) -> Node:
    return max(rng.choices(pop_fit, k=k), key=lambda x: x[0])[1]


# ─── Data construction ────────────────────────────────────────────────────────

def build_big(pairs: List[str], feat_cfg_path: Path, v7_path: Path,
              tr3_start: str, tr3_end: str, v3_start: str, v3_end: str,
              label_period: int = 12) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    feat_cfg = json.loads(feat_cfg_path.read_text(encoding="utf-8-sig"))
    v7_specs = load_expression_file(v7_path)

    ts = pd.Timestamp(tr3_start, tz="UTC"); te = pd.Timestamp(tr3_end, tz="UTC")
    vs = pd.Timestamp(v3_start, tz="UTC");  ve = pd.Timestamp(v3_end, tz="UTC")

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
        ok = (df["date"] >= ts) & (df["date"] < ve) & df["close"].notna() & df["__ret__"].notna()
        if ok.sum() > 0:
            dfs.append(df.loc[ok].reset_index(drop=True))

    if not dfs:
        raise RuntimeError("no data loaded")

    big = pd.concat(dfs, ignore_index=True).copy()  # defragment
    tr_mask = ((big["date"] >= ts) & (big["date"] < te)).values
    vl_mask = ((big["date"] >= vs) & (big["date"] < ve)).values
    excl = {"date", "open", "high", "low", "close", "volume", "__ret__"}
    cols = [c for c in big.columns if c not in excl]
    print(f"[data] {len(big):,} rows | {len(cols)} cols | "
          f"train3={tr_mask.sum():,} val3={vl_mask.sum():,}")
    return big, tr_mask, vl_mask, cols


# ─── Novelty filter ───────────────────────────────────────────────────────────

def novelty_filter(candidates: List[Tuple[float, str]],
                   big: pd.DataFrame, vl_mask: np.ndarray,
                   v7_cols: List[str],
                   max_corr: float = 0.85) -> List[Tuple[float, str]]:
    """Remove candidates with |Pearson corr| > max_corr with any v7 factor on VAL3.

    Uses np.corrcoef per column (numerically stable vs matrix normalisation).
    Samples every 3rd v7 column for speed (~12 checks instead of 36).
    """
    if not v7_cols:
        return candidates

    present = [c for c in v7_cols if c in big.columns]
    v7_mat = big.loc[vl_mask, present].values.astype(float)  # (n_val3, n_v7)
    sample_idx = list(range(0, len(present), max(1, len(present) // 12)))  # ≤12 cols

    kept = []
    for fit, expr in candidates:
        try:
            s = safe_eval_expression(expr, big).replace([np.inf, -np.inf], np.nan)
            vals = s.values[vl_mask].astype(float)
        except Exception:
            kept.append((fit, expr))
            continue
        ok = np.isfinite(vals)
        if ok.sum() < 100:
            kept.append((fit, expr))
            continue
        max_c = 0.0
        for j in sample_idx:
            vc = v7_mat[:, j]
            both = ok & np.isfinite(vc)
            if both.sum() < 50:
                continue
            c = np.corrcoef(vals[both], vc[both])[0, 1]
            if np.isfinite(c):
                max_c = max(max_c, abs(c))
        if max_c <= max_corr:
            kept.append((fit, expr))

    return kept


# ─── Main GP loop ─────────────────────────────────────────────────────────────

def run_gp(big: pd.DataFrame, tr_mask: np.ndarray, vl_mask: np.ndarray,
           feat_cols: List[str], v7_cols: List[str], regimes: List[str],
           n_gen: int, pop_size: int, elite_k: int,
           cx_rate: float, mut_rate: float, rng_seed: int) -> List[Tuple[float, str]]:

    rng = random.Random(rng_seed)
    all_terms = feat_cols + v7_cols

    # --- Initialise population ---
    # Seed with combinations that USE v7 cols but aren't direct copies:
    #   50% raw-feature trees (high novelty from the start)
    #   50% trees built from v7 cols wrapped in operations (not bare leaves)
    pop: List[Node] = []
    # Group 1: second-generation v7 combos (z, sub, ema wraps)
    for col in v7_cols:
        pop.append({"t": "unary", "op": "z",   "a": {"t": "leaf", "col": col}})
        pop.append({"t": "unary", "op": "tanh", "a": {"t": "leaf", "col": col}})
        if len(pop) >= pop_size // 2:
            break
    # Group 2: random trees from raw feature cols only (high novelty baseline)
    while len(pop) < pop_size:
        pop.append(_rand_tree(feat_cols, regimes, rng, D=4))
    pop = pop[:pop_size]

    seen: set = {node_to_str(n) for n in pop}
    hall_of_fame: List[Tuple[float, str]] = []

    print(f"[GP] {n_gen} gen × {pop_size} pop | elite={elite_k} | "
          f"cx={cx_rate:.0%} mut={mut_rate:.0%}")

    for g in range(n_gen):
        t0 = time.time()
        scored: List[Tuple[float, Node]] = []
        for ind in pop:
            expr = node_to_str(ind)
            f = eval_fitness(expr, big, vl_mask, tr_mask)
            scored.append((f, ind))
        scored.sort(key=lambda x: -x[0])

        n_valid = sum(1 for f, _ in scored if f > 0)
        elapsed = time.time() - t0
        print(f"  gen {g+1:3d}/{n_gen}  best={scored[0][0]:.4f}  "
              f"valid={n_valid}/{pop_size}  {elapsed:.0f}s", flush=True)

        # Accumulate hall-of-fame
        for f, n in scored[:elite_k]:
            if f > 0:
                hall_of_fame.append((f, node_to_str(n)))

        # Elite survives unchanged
        elite = [n for _, n in scored[:elite_k]]

        # Generate offspring via crossover + mutation
        scored_pf = [(f, n) for f, n in scored if f > -0.5] or scored
        offspring: List[Node] = []
        tries = 0
        while len(offspring) < pop_size - elite_k and tries < 8000:
            tries += 1
            if rng.random() < cx_rate:
                p1 = _tourn(scored_pf, rng)
                p2 = _tourn(scored_pf, rng)
                c1, c2 = crossover(p1, p2, rng)
                for c in (mutate(c1, all_terms, regimes, rng, mut_rate),
                           mutate(c2, all_terms, regimes, rng, mut_rate)):
                    s = node_to_str(c)
                    if 4 < len(s) < 700 and s not in seen:
                        seen.add(s)
                        offspring.append(c)
                        if len(offspring) >= pop_size - elite_k:
                            break
            else:
                t = _rand_tree(all_terms, regimes, rng)
                s = node_to_str(t)
                if s not in seen and 4 < len(s) < 700:
                    seen.add(s)
                    offspring.append(t)

        pop = elite + offspring

    # Deduplicate hall-of-fame
    seen2: set = set()
    results = []
    for f, s in sorted(hall_of_fame, key=lambda x: -x[0]):
        if s not in seen2 and f > 0:
            seen2.add(s)
            results.append((f, s))
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="GP factor miner")
    ap.add_argument("--v7-file",     default="user_data/freqai_expressions_honest_v7.json")
    ap.add_argument("--feature-cfg", default="user_data/freqai_features_real.json")
    ap.add_argument("--tr3-start",   default="2023-05-15")
    ap.add_argument("--tr3-end",     default="2025-07-01")
    ap.add_argument("--v3-start",    default="2025-07-01")
    ap.add_argument("--v3-end",      default="2025-12-01")
    ap.add_argument("--n-gen",       type=int,   default=100)
    ap.add_argument("--pop-size",    type=int,   default=200)
    ap.add_argument("--elite-k",     type=int,   default=30)
    ap.add_argument("--cx-rate",     type=float, default=0.70)
    ap.add_argument("--mut-rate",    type=float, default=0.15)
    ap.add_argument("--top-k",       type=int,   default=25)
    ap.add_argument("--novelty-thr", type=float, default=0.80,
                    help="max Pearson corr with any v7 factor (0=off)")
    ap.add_argument("--out", default="user_data/freqai_expressions_gp_v1.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    v7_path = ROOT / args.v7_file
    feat_path = ROOT / args.feature_cfg

    print("[GP] Building data…")
    big, tr_mask, vl_mask, all_cols = build_big(
        DEFAULT_PAIRS, feat_path, v7_path,
        args.tr3_start, args.tr3_end, args.v3_start, args.v3_end,
    )

    v7_names = {e["name"] for e in
                json.loads(v7_path.read_text(encoding="utf-8-sig"))["expressions"]}
    v7_cols  = [c for c in all_cols if c in v7_names]
    feat_cols = [c for c in all_cols if c not in v7_names and c != "__ret__"]
    regimes  = [r for r in _REGIMES if r.split()[0] in big.columns]

    print(f"[GP] terminals: {len(feat_cols)} raw + {len(v7_cols)} v7-cols | "
          f"regimes: {len(regimes)}")

    results = run_gp(
        big=big, tr_mask=tr_mask, vl_mask=vl_mask,
        feat_cols=feat_cols, v7_cols=v7_cols, regimes=regimes,
        n_gen=args.n_gen, pop_size=args.pop_size, elite_k=args.elite_k,
        cx_rate=args.cx_rate, mut_rate=args.mut_rate, rng_seed=args.seed,
    )

    # Novelty filter
    if args.novelty_thr > 0 and v7_cols:
        print(f"[GP] novelty filter (max_corr={args.novelty_thr})…")
        before = len(results)
        results = novelty_filter(results, big, vl_mask, v7_cols, args.novelty_thr)
        print(f"     {before} → {len(results)} after novelty filter")

    top = results[:args.top_k]

    print(f"\n[GP] Top {len(top)} evolved factors (val3_IC):")
    print(f"  {'rank':>4}  {'fitness':>8}  expression")
    for i, (fit, expr) in enumerate(top):
        snippet = expr[:90] + ("…" if len(expr) > 90 else "")
        print(f"  {i+1:4d}  {fit:8.5f}  {snippet}")

    out_exprs = [
        {"name": f"gp_{i+1:03d}", "expression": expr, "_gp_fitness": round(fit, 6)}
        for i, (fit, expr) in enumerate(top)
    ]
    out = Path(args.out)
    out.write_text(json.dumps({
        "version": "gp_v1",
        "gp_config": {
            "n_gen": args.n_gen, "pop_size": args.pop_size,
            "elite_k": args.elite_k, "cx_rate": args.cx_rate,
            "mut_rate": args.mut_rate, "seed": args.seed,
        },
        "fitness_metric": "abs_rank_IC_val3 + sign_consistency_bonus",
        "novelty_thr": args.novelty_thr,
        "seed_source": args.v7_file,
        "n_candidates_before_filter": len(results) + (0 if args.novelty_thr <= 0 else 0),
        "expressions": out_exprs,
    }, indent=2))
    print(f"\n[GP] → {out}  ({len(out_exprs)} factors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
