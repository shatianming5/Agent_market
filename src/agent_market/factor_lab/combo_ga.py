"""Combinatorial factor-library discovery via genetic algorithm.

Search target: find a K-factor subset whose Ridge-regression-fitted linear
combination maximizes walk-forward-lite OOS Spearman IC, averaged across
all training pairs.

Pipeline:
  1. Pull a candidate pool from Factor Hub (|IC| >= min_ic).
  2. Pre-compute per-pair rank series on the full feature matrix (one time).
  3. Run a fixed-size-K genetic algorithm (elitism + tournament + crossover +
     swap mutation + Jaccard-novelty rejection).
  4. Export the top-N combos to JSON libraries for full walk-forward backtest.

Usage:
  from agent_market.factor_lab import combo_ga
  summary = combo_ga.run(combo_size=13, pool_size=300, population=30,
                         generations=50, novelty_gate=0.7)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .paths import (KUCOIN_DIR, FEATURE_FILE, USER_DATA,
                    DEFAULT_PAIRS, DEFAULT_TRAIN, DEFAULT_OOS,
                    DEFAULT_TRAIN3, DEFAULT_VAL3, DEFAULT_REAL_TEST3,
                    DEFAULT_LABEL_PERIOD, DEFAULT_TAKER_FEE, DEFAULT_SLIPPAGE,
                    LAB_STATE, TIMEFRAME_LABEL_BARS, kucoin_feather)
from . import fitness as F


# ============================================================
# Expression evaluation: reuse mining's engine
# ============================================================

def _load_big(timeframe: str = "1h", label_bars: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load the same feature matrix used by mining, at the chosen timeframe."""
    import sys
    _SRC = str(Path(__file__).resolve().parents[2])
    if _SRC not in sys.path: sys.path.insert(0, _SRC)
    from agent_market.freqai.features import apply_configured_features

    feat_cfg = json.loads(FEATURE_FILE.read_text(encoding="utf-8-sig"))
    if label_bars is None:
        label_bars = TIMEFRAME_LABEL_BARS.get(timeframe, DEFAULT_LABEL_PERIOD)
    frames = []
    for pair in DEFAULT_PAIRS:
        f = kucoin_feather(pair, timeframe)
        if not f.exists(): continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = apply_configured_features(df, feat_cfg).reset_index(drop=True)
        df["__pair__"] = pair
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {timeframe} feather data under {KUCOIN_DIR}")
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"], utc=True)
    fwd = (big.groupby("__pair__")["close"].shift(-int(label_bars)) / big["close"]) - 1.0
    big["__fwd_ret__"] = fwd
    exclude = {"date", "open", "high", "low", "close", "volume", "__pair__", "__fwd_ret__"}
    base_cols = [c for c in big.columns if c not in exclude]
    return big, base_cols


# ============================================================
# Candidate pool from Factor Hub
# ============================================================

@dataclass
class Candidate:
    factor_id: int
    name: str
    expression: str
    oos_ic: float
    origin: str


def pull_candidate_pool(min_abs_ic: float = 0.08, limit: int = 500,
                         require_clean: bool = True) -> List[Candidate]:
    """Pull the top-|IC| factors from Factor Hub.

    require_clean=True (default) enforces the `snoop_level=clean` tag so
    pre-v6 factors (run500 / g-factors / remine output) whose metadata was
    leaked by OOS-IC gates never contaminate GA search again. Set False to
    run legacy diagnostic mode.
    """
    from agent_market.factor_hub import Client
    c = Client()
    clean_clause = (
        "AND json_extract(f.metadata, '$.snoop_level') = 'clean'"
        if require_clean else ""
    )
    with c.connect() as conn:
        rows = conn.execute(f"""
            SELECT f.id, f.name, f.expression, f.origin,
                   (SELECT e.metric_value FROM evaluations e
                    WHERE e.factor_id = f.id AND e.metric_name = 'oos_ic'
                    ORDER BY ABS(e.metric_value) DESC LIMIT 1) AS ic
            FROM factors f
            WHERE EXISTS (SELECT 1 FROM evaluations e
                          WHERE e.factor_id = f.id AND e.metric_name = 'oos_ic'
                            AND ABS(e.metric_value) >= ?)
              {clean_clause}
            ORDER BY ABS((SELECT e.metric_value FROM evaluations e
                          WHERE e.factor_id = f.id AND e.metric_name='oos_ic'
                          ORDER BY ABS(e.metric_value) DESC LIMIT 1)) DESC
            LIMIT ?
        """, (min_abs_ic, limit)).fetchall()
    out: List[Candidate] = []
    for r in rows:
        if r["ic"] is None: continue
        out.append(Candidate(
            factor_id=int(r["id"]), name=str(r["name"]),
            expression=str(r["expression"]), origin=str(r["origin"] or ""),
            oos_ic=float(r["ic"]),
        ))
    return out


# ============================================================
# Pre-computed per-pair rank / forward-return cache
# ============================================================

@dataclass
class PairData:
    ranks: np.ndarray              # [T, N] rank-transformed factor values (NaN preserved)
    fwd: np.ndarray                # [T]
    valid: np.ndarray              # [T] forward-return finite mask
    fit_mask: np.ndarray           # [T] TRAIN3 region  (GA fits on this)
    holdout_mask: np.ndarray       # [T] VAL3 region    (GA scores on this — OOS surrogate)
    real_oos_mask: np.ndarray      # [T] REAL_TEST3 — NEVER touched by GA, only by final check


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """Per-column rank with average ties; NaN preserved."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isfinite(col)
        if mask.sum() < 50: continue
        order = np.argsort(col[mask], kind="mergesort")
        ranks = np.empty(mask.sum(), dtype=np.float64)
        ranks[order] = np.arange(1, mask.sum() + 1, dtype=np.float64)
        out[mask, j] = ranks
    return out


def build_pair_data(big: pd.DataFrame, candidates: List[Candidate],
                    train3: Tuple[str, str] = DEFAULT_TRAIN3,
                    val3: Tuple[str, str] = DEFAULT_VAL3,
                    real_test3: Tuple[str, str] = DEFAULT_REAL_TEST3) -> Dict[str, PairData]:
    """Evaluate every candidate expression per pair, rank-transform once.

    Uses the 3-section split so REAL_TEST3 stays isolated. GA only sees
    TRAIN3 (fit) and VAL3 (holdout). Final export uses REAL_TEST3 for a
    one-shot integrity check.
    """
    from agent_market.freqai.expression_engine import safe_eval_expression

    out: Dict[str, PairData] = {}
    tr_start, tr_end = [pd.Timestamp(s, tz="UTC") for s in train3]
    va_start, va_end = [pd.Timestamp(s, tz="UTC") for s in val3]
    rt_start, rt_end = [pd.Timestamp(s, tz="UTC") for s in real_test3]

    for pair in DEFAULT_PAIRS:
        sub = big.loc[big["__pair__"] == pair].reset_index(drop=True)
        if len(sub) < 500: continue
        T = len(sub)
        X = np.full((T, len(candidates)), np.nan, dtype=np.float64)
        for j, cand in enumerate(candidates):
            try:
                s = safe_eval_expression(cand.expression, sub)
                X[:, j] = np.asarray(s, dtype=np.float64)
            except Exception:
                pass
        ranks = _rank_transform(X)
        fwd = np.asarray(sub["__fwd_ret__"].values, dtype=np.float64)
        valid = np.isfinite(fwd)
        fit_mask = ((sub["date"] >= tr_start) & (sub["date"] < tr_end)).values
        holdout_mask = ((sub["date"] >= va_start) & (sub["date"] < va_end)).values
        real_oos_mask = ((sub["date"] >= rt_start) & (sub["date"] < rt_end)).values
        out[pair] = PairData(
            ranks=ranks, fwd=fwd, valid=valid,
            fit_mask=fit_mask, holdout_mask=holdout_mask,
            real_oos_mask=real_oos_mask,
        )
    return out


def real_test_ic(combo_idx: Sequence[int],
                 pair_data: Dict[str, PairData],
                 ridge_lambda: float = 1.0) -> Dict[str, float]:
    """One-shot integrity score on REAL_TEST3 — *only call once per run, never
    use the result to tune GA hyper-parameters*. Mirrors combo_fitness but on
    real_oos_mask."""
    ics: List[float] = []
    idx = np.asarray(combo_idx, dtype=np.int64)
    K = len(idx)
    for pair, pd_ in pair_data.items():
        X = pd_.ranks[:, idx]
        y = pd_.fwd
        row_ok = np.isfinite(X).all(axis=1) & pd_.valid
        mask_fit = row_ok & pd_.fit_mask
        mask_rt = row_ok & pd_.real_oos_mask
        if mask_fit.sum() < 500 or mask_rt.sum() < 100:
            continue
        X_fit, y_fit = X[mask_fit], y[mask_fit]
        X_rt, y_rt = X[mask_rt], y[mask_rt]
        mu = X_fit.mean(axis=0); sd = X_fit.std(axis=0) + 1e-9
        X_fit = (X_fit - mu) / sd
        X_rt = (X_rt - mu) / sd
        try:
            beta = np.linalg.solve(X_fit.T @ X_fit + ridge_lambda * np.eye(K),
                                    X_fit.T @ y_fit)
        except np.linalg.LinAlgError:
            continue
        pred = X_rt @ beta
        ics.append(_spearman(pred, y_rt))
    if not ics:
        return {"real_ic": 0.0, "pairs": 0, "sign_agree": 0}
    m = float(np.mean(ics))
    sig = sum(1 for ic in ics if ic * m > 0)
    return {"real_ic": m, "pairs": len(ics), "sign_agree": sig}


# ============================================================
# Fitness: walk-forward-lite Ridge regression per pair
# ============================================================

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 100: return 0.0
    if a.std() == 0 or b.std() == 0: return 0.0
    ar = pd.Series(a).rank(method="average").values
    br = pd.Series(b).rank(method="average").values
    return float(np.corrcoef(ar, br)[0, 1])


def dedupe_by_rank_corr(candidates: List[Candidate],
                         pair_data: Dict[str, PairData],
                         corr_gate: float = 0.9,
                         ref_pair: str = "BTC/USDT") -> Tuple[List[Candidate], Dict[str, PairData]]:
    """Remove rank-redundant candidates (Spearman ≥ corr_gate vs already kept).

    Greedy by |candidate.oos_ic| descending. Uses the reference pair's rank
    matrix (BTC/USDT by default) for the corr estimate — fast and a good
    proxy since genuine monotone-variants collide on every pair.
    """
    if ref_pair not in pair_data:
        ref_pair = next(iter(pair_data))
    ranks = pair_data[ref_pair].ranks
    valid = np.isfinite(ranks).all(axis=1)
    if valid.sum() < 500:
        # Column-wise standardization ignoring NaN
        R = np.where(np.isfinite(ranks), ranks, 0.0)
        mu = R.mean(axis=0)
        sd = R.std(axis=0) + 1e-9
        Z = (R - mu) / sd
    else:
        R = ranks[valid]
        mu = R.mean(axis=0)
        sd = R.std(axis=0) + 1e-9
        Z = (R - mu) / sd
    corr = (Z.T @ Z) / len(Z)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    order = sorted(range(len(candidates)), key=lambda i: -abs(candidates[i].oos_ic))
    keep_set: set = set()
    for i in order:
        if all(abs_corr[i, j] < corr_gate for j in keep_set):
            keep_set.add(i)
    keep = sorted(keep_set)
    new_cands = [candidates[i] for i in keep]
    new_pair_data = {
        p: PairData(
            ranks=pd_.ranks[:, keep], fwd=pd_.fwd, valid=pd_.valid,
            fit_mask=pd_.fit_mask, holdout_mask=pd_.holdout_mask,
            real_oos_mask=pd_.real_oos_mask,
        ) for p, pd_ in pair_data.items()
    }
    return new_cands, new_pair_data


def combo_fitness(combo_idx: Sequence[int],
                  pair_data: Dict[str, PairData],
                  ridge_lambda: float = 1.0) -> Dict[str, float]:
    """GA scores on a chronological holdout *inside* train (never touches real OOS).

    Per pair: fit Ridge on `fit_mask`, score Spearman IC on `holdout_mask`.
    Returns IC averaged across pairs and sign-consistency bonus.
    """
    ics_hold: List[float] = []
    ics_fit: List[float] = []
    idx = np.asarray(combo_idx, dtype=np.int64)
    K = len(idx)
    for pair, pd_ in pair_data.items():
        X = pd_.ranks[:, idx]  # [T, K]
        y = pd_.fwd
        row_ok = np.isfinite(X).all(axis=1) & pd_.valid
        mask_fit = row_ok & pd_.fit_mask
        mask_hold = row_ok & pd_.holdout_mask
        if mask_fit.sum() < 500 or mask_hold.sum() < 200:
            continue
        X_fit, y_fit = X[mask_fit], y[mask_fit]
        X_hold, y_hold = X[mask_hold], y[mask_hold]
        # Standardize by fit stats
        mu = X_fit.mean(axis=0)
        sd = X_fit.std(axis=0) + 1e-9
        X_fit = (X_fit - mu) / sd
        X_hold = (X_hold - mu) / sd
        # Ridge: beta = (X'X + lambda*I)^-1 X'y
        XtX = X_fit.T @ X_fit
        A = XtX + ridge_lambda * np.eye(K)
        try:
            beta = np.linalg.solve(A, X_fit.T @ y_fit)
        except np.linalg.LinAlgError:
            continue
        pred_fit = X_fit @ beta
        pred_hold = X_hold @ beta
        ics_fit.append(_spearman(pred_fit, y_fit))
        ics_hold.append(_spearman(pred_hold, y_hold))
    if not ics_hold:
        return {"hold_ic": 0.0, "fit_ic": 0.0, "pairs": 0,
                "sign_agree": 0, "fitness": 0.0}
    mean_hold = float(np.mean(ics_hold))
    mean_fit = float(np.mean(ics_fit)) if ics_fit else 0.0
    sig = sum(1 for ic in ics_hold if ic * mean_hold > 0)
    # Reward absolute holdout IC × fraction of pairs agreeing on sign
    fitness = abs(mean_hold) * (sig / max(len(ics_hold), 1))
    return {"hold_ic": mean_hold, "fit_ic": mean_fit,
            "sign_agree": sig, "pairs": len(ics_hold), "fitness": fitness}


# ============================================================
# Genetic algorithm
# ============================================================

@dataclass
class Individual:
    combo: Tuple[int, ...]    # sorted tuple of candidate indices
    hold_ic: float = 0.0
    fit_ic: float = 0.0
    sign_agree: int = 0
    pairs: int = 0
    fitness: float = 0.0


def _random_combo(pool_size: int, K: int, rng: random.Random) -> Tuple[int, ...]:
    return tuple(sorted(rng.sample(range(pool_size), K)))


def _tournament_select(pop: List[Individual], k: int, rng: random.Random) -> Individual:
    contenders = rng.sample(pop, k)
    return max(contenders, key=lambda x: x.fitness)


def _crossover(a: Tuple[int, ...], b: Tuple[int, ...], K: int,
               rng: random.Random) -> Tuple[int, ...]:
    union = list(set(a) | set(b))
    rng.shuffle(union)
    chosen = union[:K]
    if len(chosen) < K:
        pool_size_needed = K - len(chosen)
        # fill from parents' union complement — can't without pool_size; just return a
        return a
    return tuple(sorted(chosen))


def _mutate(combo: Tuple[int, ...], pool_size: int, rng: random.Random,
            n_swap: int = 1) -> Tuple[int, ...]:
    new = list(combo)
    available = set(range(pool_size)) - set(combo)
    if not available: return combo
    for _ in range(n_swap):
        idx_out = rng.randrange(len(new))
        new[idx_out] = rng.choice(list(available))
        available = set(range(pool_size)) - set(new)
        if not available: break
    return tuple(sorted(new))


def _jaccard(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u > 0 else 0.0


def _novel_enough(cand: Tuple[int, ...], pool: List[Individual],
                  gate: float) -> bool:
    for p in pool:
        if _jaccard(cand, p.combo) >= gate:
            return False
    return True


def run_ga(pair_data: Dict[str, PairData], pool_size: int, *,
           combo_size: int = 13, population: int = 30, generations: int = 50,
           elitism: int = 5, tournament: int = 3,
           crossover_rate: float = 0.7, mutation_rate: float = 0.3,
           novelty_gate: float = 0.7, seed: int = 42,
           hub_event=None) -> List[Individual]:
    rng = random.Random(seed)

    # Seed population — bias toward high-|IC| singletons by picking cherry indices
    pop: List[Individual] = []
    seen: set = set()
    while len(pop) < population:
        combo = _random_combo(pool_size, combo_size, rng)
        if combo in seen: continue
        seen.add(combo)
        m = combo_fitness(combo, pair_data)
        pop.append(Individual(combo=combo, **m))

    pop.sort(key=lambda x: x.fitness, reverse=True)
    print(f"[ga] gen 0: top fitness={pop[0].fitness:.4f} "
          f"hold_ic={pop[0].hold_ic:+.4f} sign={pop[0].sign_agree}/{pop[0].pairs}",
          flush=True)
    if hub_event is not None:
        hub_event("ga.generation", generation=0, top_fitness=pop[0].fitness,
                  top_hold_ic=pop[0].hold_ic, population=len(pop))

    for gen in range(1, generations + 1):
        new_pop: List[Individual] = []
        # Elitism
        new_pop.extend(pop[:elitism])
        # Reproduce
        attempts = 0
        while len(new_pop) < population and attempts < population * 10:
            attempts += 1
            p1 = _tournament_select(pop, tournament, rng)
            combo = p1.combo
            if rng.random() < crossover_rate:
                p2 = _tournament_select(pop, tournament, rng)
                combo = _crossover(combo, p2.combo, combo_size, rng)
            if rng.random() < mutation_rate:
                combo = _mutate(combo, pool_size, rng, n_swap=rng.randint(1, 2))
            if combo in seen: continue
            if not _novel_enough(combo, new_pop, novelty_gate): continue
            seen.add(combo)
            m = combo_fitness(combo, pair_data)
            new_pop.append(Individual(combo=combo, **m))
        pop = sorted(new_pop, key=lambda x: x.fitness, reverse=True)[:population]
        print(f"[ga] gen {gen:>3}/{generations}: top fitness={pop[0].fitness:.4f} "
              f"hold_ic={pop[0].hold_ic:+.4f} sign={pop[0].sign_agree}/{pop[0].pairs} "
              f"seen={len(seen)}", flush=True)
        if hub_event is not None:
            hub_event("ga.generation", generation=gen, top_fitness=pop[0].fitness,
                      top_hold_ic=pop[0].hold_ic, seen=len(seen), population=len(pop))
    return pop


# ============================================================
# Driver
# ============================================================

def run(*, combo_size: int = 13, pool_size: int = 300, min_abs_ic: float = 0.08,
        population: int = 30, generations: int = 50, novelty_gate: float = 0.7,
        dedupe_gate: float = 0.9, timeframe: str = "1h",
        seed: int = 42, tag: str = "combo_ga",
        export_top_n: int = 3, require_clean: bool = True) -> Dict:
    t0 = time.time()
    clean_tag = "clean-only" if require_clean else "all-factors (LEGACY)"
    print(f"[combo_ga] pulling candidate pool (|IC| >= {min_abs_ic}, "
          f"limit {pool_size}, {clean_tag})")
    cands = pull_candidate_pool(min_abs_ic=min_abs_ic, limit=pool_size,
                                 require_clean=require_clean)
    print(f"  pool: {len(cands)} factors")
    if len(cands) < combo_size * 2:
        raise RuntimeError(f"pool too small ({len(cands)}) for combo_size={combo_size}")

    print(f"[combo_ga] loading {timeframe} feature matrix")
    big, _ = _load_big(timeframe=timeframe)
    print(f"  rows={len(big):,}")

    print("[combo_ga] evaluating all candidates per-pair and rank-transforming")
    t_pre = time.time()
    pair_data = build_pair_data(big, cands)
    print(f"  prepared {len(pair_data)} pairs  elapsed={time.time()-t_pre:.1f}s")

    print(f"[combo_ga] de-duplicating candidate pool by Spearman rank corr >= {dedupe_gate} (BTC reference)")
    before = len(cands)
    cands, pair_data = dedupe_by_rank_corr(cands, pair_data, corr_gate=dedupe_gate)
    print(f"  pool {before} -> {len(cands)} after rank-corr dedupe")
    if len(cands) < combo_size * 2:
        raise RuntimeError(
            f"independent pool too small after dedupe ({len(cands)}); "
            f"lower --min-abs-ic or raise --dedupe-gate"
        )

    # Wire Hub event logging
    def hub_event(event_type, **payload):
        try:
            from agent_market.factor_hub import Client
            Client().log(event_type, payload={"tag": tag, **payload})
        except Exception:
            pass

    hub_event("ga.started", pool=len(cands), combo_size=combo_size,
              population=population, generations=generations,
              novelty_gate=novelty_gate, dedupe_gate=dedupe_gate)

    print(f"[combo_ga] GA: pop={population} gens={generations} K={combo_size} "
          f"jaccard_gate={novelty_gate}")
    pop = run_ga(pair_data, pool_size=len(cands),
                 combo_size=combo_size, population=population,
                 generations=generations, novelty_gate=novelty_gate,
                 seed=seed, hub_event=hub_event)

    # Export top-N combos + one-shot REAL_TEST3 integrity score
    out_dir = LAB_STATE / "combo_ga" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []
    print(f"\n[combo_ga] running one-shot REAL_TEST3 integrity check on top-{export_top_n}")
    for rank, ind in enumerate(pop[:export_top_n], start=1):
        selected = [cands[i] for i in ind.combo]
        rt_score = real_test_ic(ind.combo, pair_data)
        print(f"  rank {rank}: val_hold_ic={ind.hold_ic:+.4f}  "
              f"real_test_ic={rt_score['real_ic']:+.4f}  "
              f"sign={rt_score['sign_agree']}/{rt_score['pairs']}  "
              f"(generalization ratio = {abs(rt_score['real_ic'])/(abs(ind.hold_ic)+1e-9):.2f})")
        lib_name = f"freqai_expressions_{tag}_top{rank}.json"
        lib_path = USER_DATA / lib_name
        lib_path.write_text(json.dumps({
            "version": f"{tag}_rank{rank}",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ga": {
                "rank": rank, "combo_size": combo_size,
                "hold_ic": ind.hold_ic, "fit_ic": ind.fit_ic,
                "sign_agree": ind.sign_agree, "pairs": ind.pairs,
                "fitness": ind.fitness,
            },
            "expressions": [
                {"name": f"c{rank:02d}_{i+1:02d}",
                 "expression": s.expression,
                 "description": f"GA combo #{rank}, source factor {s.name}",
                 "category": "ga_combo",
                 "origin": f"combo_ga:{tag}:rank{rank}",
                 "source_factor_id": s.factor_id,
                 "source_oos_ic": s.oos_ic}
                for i, s in enumerate(selected)
            ],
        }, indent=2), encoding="utf-8")
        summaries.append({
            "rank": rank, "library": str(lib_path),
            "hold_ic": ind.hold_ic, "fit_ic": ind.fit_ic,
            "real_test_ic": rt_score["real_ic"],
            "real_test_sign_agree": rt_score["sign_agree"],
            "real_test_pairs": rt_score["pairs"],
            "fitness": ind.fitness, "sign_agree": ind.sign_agree,
            "pairs": ind.pairs, "factor_ids": list(ind.combo),
        })

    total = time.time() - t0
    hub_event("ga.finished", total_minutes=round(total / 60, 2),
              exported=len(summaries),
              top_hold_ic=pop[0].hold_ic if pop else 0.0)

    # Persist a machine-readable summary
    state = {
        "tag": tag,
        "elapsed_min": round(total / 60, 2),
        "pool_size": len(cands),
        "combo_size": combo_size, "population": population,
        "generations": generations, "novelty_gate": novelty_gate,
        "dedupe_gate": dedupe_gate, "seed": seed,
        "top_individuals": [
            {"rank": i + 1, **asdict(ind)} for i, ind in enumerate(pop[:export_top_n])
        ],
        "exported": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(state, indent=2))
    print(f"\n[combo_ga] done in {total/60:.1f}m. exported {len(summaries)} combos.")
    for s in summaries:
        print(f"  rank {s['rank']}: hold_ic={s['hold_ic']:+.4f} "
              f"fit_ic={s['fit_ic']:+.4f} sign={s['sign_agree']}/{s['pairs']}  → {s['library']}")
    return state
