#!/usr/bin/env python
from __future__ import annotations

import ast
import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from _lib import read_json, resolve_label_period, resolve_path, resolve_timeframe  # noqa: E402
from agent_market import paths  # noqa: E402


def _infer_feature_file(config_path: Path) -> Path:
    base = config_path.parent
    candidates = [
        base / "freqai_features_real.json",
        base / "freqai_features.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "feature-file not provided and no default found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _read_feedback(path: Optional[str], top_lines: int = 60) -> Optional[str]:
    if not path:
        return None
    p = resolve_path(path)
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return None
    if not text.strip():
        return None
    lines = text.splitlines()
    if len(lines) > top_lines:
        lines = lines[:top_lines] + [f"...(truncated {len(text.splitlines()) - top_lines} lines)"]
    return "\n".join(lines)


def _stable_slug(idx: int) -> str:
    return f"e{idx:02d}"


FEATURE_FILE = paths.resolve_repo_path("user_data/freqai_features.json")


from agent_market.freqai import llm as llm_utils  # noqa: E402
from agent_market.freqai.expression_engine import safe_eval_expression  # noqa: E402


def _safe_eval_expression(expr: str, df):  # noqa: ANN001
    return safe_eval_expression(expr, df)


def _compute_factor_series(expr: str, pairs: Sequence["_PairData"]) -> pd.Series:
    """Evaluate *expr* across all pairs and concatenate into one series (index-reset)."""
    parts: List[pd.Series] = []
    for pd_item in pairs:
        try:
            s = _safe_eval_expression(expr, pd_item.df)
            parts.append(s.reset_index(drop=True))
        except Exception:
            continue
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True)


def _factor_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Return lag-*lag* autocorrelation, NaN-safe."""
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < lag + 20:
        return 0.0
    ac = float(clean.autocorr(lag))
    return ac if np.isfinite(ac) else 0.0


def _select_diverse_topn(
    scored: List[Dict[str, Any]],
    pairs: Sequence["_PairData"],
    top_n: int,
    *,
    corr_threshold: float = 0.7,
    ac_penalty: float = 0.5,
    ac_threshold: float = 0.95,
    label_period: int = 1,
) -> List[Dict[str, Any]]:
    """Greedy diversity-aware top-N selection.

    1. Evaluate each candidate to get its factor-value series.
    2. Penalise score by autocorrelation if AC(1) > *ac_threshold*.
    3. Greedily pick the highest-scoring factor whose |corr| with
       every already-selected factor is below *corr_threshold*.
    """
    import warnings

    # Pre-compute factor value vectors (concatenated across pairs, non-overlapping)
    vec_cache: Dict[str, pd.Series] = {}
    adjusted: List[Tuple[float, int, Dict[str, Any]]] = []

    for idx, item in enumerate(scored):
        expr = str(item.get("expression") or "").strip()
        if not expr:
            continue

        parts: List[pd.Series] = []
        for pd_item in pairs:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                try:
                    s = _safe_eval_expression(expr, pd_item.df)
                    # non-overlapping sampling to reduce autocorrelation artefact
                    if label_period > 1:
                        s = s.iloc[::label_period]
                    parts.append(s.reset_index(drop=True))
                except Exception:
                    continue
        if not parts:
            continue
        vec = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        if len(vec) < 30:
            continue

        ac1 = float(vec.autocorr(1)) if len(vec) > 20 else 0.0
        if not np.isfinite(ac1):
            ac1 = 0.0
        ac_pen = ac_penalty * max(0.0, ac1 - ac_threshold)
        adj_score = float(item.get("score") or 0.0) - ac_pen

        vec_cache[expr] = vec
        adjusted.append((adj_score, idx, item))

    adjusted.sort(key=lambda t: t[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_vecs: List[pd.Series] = []

    for adj_score, _idx, item in adjusted:
        if len(selected) >= top_n:
            break
        expr = item["expression"]
        vec = vec_cache[expr]

        # Check correlation with every already-selected factor
        too_similar = False
        # Reject constant factors (zero variance)
        if vec.std() < 1e-12:
            continue
        for sv in selected_vecs:
            merged = pd.DataFrame({"a": vec, "b": sv}).dropna()
            if len(merged) < 20:
                continue
            corr = abs(float(merged["a"].corr(merged["b"])))
            if not np.isfinite(corr) or corr > corr_threshold:
                too_similar = True
                break
        if too_similar:
            continue

        item_copy = dict(item)
        item_copy["score_adjusted"] = adj_score
        ac1 = float(vec.autocorr(1)) if len(vec) > 20 else 0.0
        item_copy["autocorr_1"] = ac1 if np.isfinite(ac1) else 0.0
        selected.append(item_copy)
        selected_vecs.append(vec)

    return selected


def _score_expression(expr_series, target_series, method: str) -> float:  # noqa: ANN001
    aligned = (
        DataFrame({"x": expr_series, "y": target_series})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if aligned.empty or len(aligned) < 5:
        raise ValueError("not enough samples")
    if method == "correlation":
        return float(aligned["x"].corr(aligned["y"]))
    raise ValueError(f"Unsupported score_method: {method}")


@dataclass(slots=True)
class _PairData:
    pair: str
    df: DataFrame
    label: pd.Series


@dataclass(frozen=True, slots=True)
class _GPNode:
    op: str
    value: Any = None
    children: Tuple["_GPNode", ...] = ()


_WINDOW_CHOICES = (1, 2, 3, 4, 6, 8, 12, 24, 48, 72)
_SHIFT_CHOICES = (1, 2, 3)
_CONST_CHOICES = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)


# ── Fix 1: Expression string → _GPNode parser ──
def _parse_expr_to_gp(expr_str: str, feature_cols: Sequence[str]) -> Optional[_GPNode]:
    """Best-effort parse of an expression string back into a _GPNode tree.

    Returns None if parsing fails.  Handles the exact format produced by
    ``_gp_to_string`` plus the common LLM output patterns.
    """
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
    except SyntaxError:
        return None

    known_cols = set(feature_cols)

    def _walk(node: ast.expr) -> Optional[_GPNode]:
        # Constants
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return _GPNode("const", float(node.value))

        # Column reference
        if isinstance(node, ast.Name) and node.id in known_cols:
            return _GPNode("col", node.id)

        # Unary minus: -expr  →  neg(expr)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            child = _walk(node.operand)
            return _GPNode("neg", None, (child,)) if child else None

        # Binary ops: a + b, a - b, a * b
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            if left is None or right is None:
                return None
            op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
            op = op_map.get(type(node.op))
            if op:
                return _GPNode(op, None, (left, right))
            return None

        # Compare: x > 20, x < 18  →  treat as const 1/0 mask (can't represent well, skip)
        if isinstance(node, ast.Compare):
            return None

        # Function calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            args = node.args

            # Unary functions
            if fname in {"z", "tanh", "abs", "sign", "log1p"} and len(args) == 1:
                child = _walk(args[0])
                return _GPNode(fname, None, (child,)) if child else None

            # Rolling functions with window
            if fname in {"ema", "roll_mean", "roll_std", "rolling_max", "rolling_min", "ts_z"} and len(args) == 2:
                child = _walk(args[0])
                if child is None:
                    return None
                # Window param
                if isinstance(args[1], ast.Constant) and isinstance(args[1].value, (int, float)):
                    window = int(args[1].value)
                    return _GPNode(fname, window, (child,))
                return None

            # shift(expr, n)
            if fname == "shift" and len(args) == 2:
                child = _walk(args[0])
                if child is None:
                    return None
                if isinstance(args[1], ast.Constant):
                    return _GPNode("shift", int(args[1].value), (child,))
                return None

            # clip(expr, lo, hi)
            if fname == "clip" and len(args) == 3:
                child = _walk(args[0])
                if child is None:
                    return None
                if isinstance(args[1], ast.Constant) and isinstance(args[2], ast.Constant):
                    return _GPNode("clip", (float(args[1].value), float(args[2].value)), (child,))
                return None

            # pct_change(expr, n)
            if fname == "pct_change" and len(args) == 2:
                child = _walk(args[0])
                if child is None:
                    return None
                if isinstance(args[1], ast.Constant):
                    # Represent as shift-based: (x - shift(x, n)) / (abs(shift(x, n)) + 1e-9)
                    # Too complex, just wrap as a single terminal placeholder
                    return None
                return None

        return None

    try:
        result = _walk(tree.body)
        return result
    except Exception:
        return None


def _gp_depth(node: _GPNode) -> int:
    if not node.children:
        return 1
    return 1 + max(_gp_depth(child) for child in node.children)


def _gp_size(node: _GPNode) -> int:
    return 1 + sum(_gp_size(child) for child in node.children)


def _gp_random_terminal(rng: random.Random, feature_cols: Sequence[str]) -> _GPNode:
    if not feature_cols or rng.random() < 0.10:
        return _GPNode("const", float(rng.choice(_CONST_CHOICES)))
    return _GPNode("col", str(rng.choice(list(feature_cols))))


def _gp_random_expr(rng: random.Random, feature_cols: Sequence[str], max_depth: int) -> _GPNode:
    if max_depth <= 1 or rng.random() < 0.35:
        return _gp_random_terminal(rng, feature_cols)

    unary_ops = ("z", "tanh", "abs", "sign", "log1p", "neg")
    binary_ops = ("add", "sub", "mul", "div")
    roll_ops = ("ema", "roll_mean", "roll_std", "rolling_max", "rolling_min", "ts_z")
    special_ops = ("shift", "clip")

    choice = rng.random()
    if choice < 0.35:
        op = rng.choice(unary_ops)
        child = _gp_random_expr(rng, feature_cols, max_depth - 1)
        if op in {"z", "tanh", "abs", "sign"} and child.op == op:
            return child
        if op == "neg":
            return child.children[0] if child.op == "neg" else _GPNode("neg", None, (child,))
        return _GPNode(op, None, (child,))
    if choice < 0.75:
        op = rng.choice(binary_ops)
        return _GPNode(
            op,
            None,
            (
                _gp_random_expr(rng, feature_cols, max_depth - 1),
                _gp_random_expr(rng, feature_cols, max_depth - 1),
            ),
        )
    if choice < 0.95:
        op = rng.choice(roll_ops)
        window = int(rng.choice(_WINDOW_CHOICES))
        return _GPNode(op, window, (_gp_random_expr(rng, feature_cols, max_depth - 1),))
    op = rng.choice(special_ops)
    if op == "shift":
        n = int(rng.choice(_SHIFT_CHOICES))
        return _GPNode("shift", n, (_gp_random_expr(rng, feature_cols, max_depth - 1),))
    lo, hi = -3.0, 3.0
    return _GPNode("clip", (lo, hi), (_gp_random_expr(rng, feature_cols, max_depth - 1),))


def _gp_to_string(node: _GPNode) -> str:
    if node.op == "col":
        return str(node.value)
    if node.op == "const":
        return f"{float(node.value):.6g}"
    if node.op in {"z", "tanh", "abs", "sign", "log1p"}:
        return f"{node.op}({_gp_to_string(node.children[0])})"
    if node.op == "neg":
        return f"(-{_gp_to_string(node.children[0])})"
    if node.op == "shift":
        return f"shift({_gp_to_string(node.children[0])}, {int(node.value)})"
    if node.op in {"ema", "roll_mean", "roll_std", "rolling_max", "rolling_min", "ts_z"}:
        return f"{node.op}({_gp_to_string(node.children[0])}, {int(node.value)})"
    if node.op == "clip":
        lo, hi = node.value
        return f"clip({_gp_to_string(node.children[0])}, {float(lo)}, {float(hi)})"
    if node.op == "add":
        return f"({_gp_to_string(node.children[0])} + {_gp_to_string(node.children[1])})"
    if node.op == "sub":
        return f"({_gp_to_string(node.children[0])} - {_gp_to_string(node.children[1])})"
    if node.op == "mul":
        return f"({_gp_to_string(node.children[0])} * {_gp_to_string(node.children[1])})"
    if node.op == "div":
        a = _gp_to_string(node.children[0])
        b = _gp_to_string(node.children[1])
        return f"({a} / (1e-9 + abs({b})))"
    raise ValueError(f"Unknown GP op: {node.op}")


def _gp_all_paths(node: _GPNode, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
    paths = [prefix]
    for idx, child in enumerate(node.children):
        paths.extend(_gp_all_paths(child, prefix + (idx,)))
    return paths


def _gp_get(node: _GPNode, path: Tuple[int, ...]) -> _GPNode:
    cur = node
    for idx in path:
        cur = cur.children[idx]
    return cur


def _gp_replace(node: _GPNode, path: Tuple[int, ...], repl: _GPNode) -> _GPNode:
    if not path:
        return repl
    idx = path[0]
    children = list(node.children)
    children[idx] = _gp_replace(children[idx], path[1:], repl)
    return _GPNode(node.op, node.value, tuple(children))


def _gp_mutate(rng: random.Random, node: _GPNode, feature_cols: Sequence[str], max_depth: int) -> _GPNode:
    paths = _gp_all_paths(node)
    target_path = rng.choice(paths)
    subtree = _gp_get(node, target_path)

    if subtree.op in {"ema", "roll_mean", "roll_std", "rolling_max", "rolling_min"} and rng.random() < 0.5:
        new_window = int(rng.choice(_WINDOW_CHOICES))
        repl = _GPNode(subtree.op, new_window, subtree.children)
    elif subtree.op == "shift" and rng.random() < 0.5:
        repl = _GPNode("shift", int(rng.choice(_SHIFT_CHOICES)), subtree.children)
    else:
        repl = _gp_random_expr(rng, feature_cols, max(1, max_depth - len(target_path)))

    candidate = _gp_replace(node, target_path, repl)
    return candidate if _gp_depth(candidate) <= max_depth else _gp_random_expr(rng, feature_cols, max_depth)


def _gp_crossover(rng: random.Random, a: _GPNode, b: _GPNode, max_depth: int) -> _GPNode:
    a_path = rng.choice(_gp_all_paths(a))
    b_path = rng.choice(_gp_all_paths(b))
    swapped = _gp_replace(a, a_path, _gp_get(b, b_path))
    return swapped if _gp_depth(swapped) <= max_depth else a


def _score_expression_across_pairs(
    expr: str,
    pairs: Sequence[_PairData],
    *,
    method: str = "correlation",
    min_samples: int = 200,
    oos_fraction: float = 0.0,
    label_period: int = 1,
) -> Tuple[float, Dict[str, float], float]:
    """Score an expression across pairs.

    When *oos_fraction* > 0 the first ``1 - oos_fraction`` of each pair is
    used only for feature warm-up; the IC is computed exclusively on the
    remaining tail (out-of-sample).  *label_period* controls non-overlapping
    sampling for a more realistic IC.

    Returns ``(mean_abs_ic, per_pair_dict, turnover)``.
    """
    import warnings

    metrics: Dict[str, float] = {}
    values: List[float] = []
    turnover_vals: List[float] = []

    for pd_item in pairs:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            series = _safe_eval_expression(expr, pd_item.df)
            aligned = (
                DataFrame({"x": series, "y": pd_item.label})
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if aligned.empty or len(aligned) < min_samples:
                continue

            # OOS: only score on the tail fraction
            if oos_fraction > 0:
                split = int(len(aligned) * (1 - oos_fraction))
                aligned = aligned.iloc[split:]
                if len(aligned) < max(30, min_samples // 4):
                    continue

            # Non-overlapping sampling
            if label_period > 1:
                aligned = aligned.iloc[::label_period]
                if len(aligned) < 20:
                    continue

            if method == "correlation":
                metric = float(aligned["x"].corr(aligned["y"]))
            elif method == "spearman":
                metric = float(aligned["x"].corr(aligned["y"], method="spearman"))
            else:
                raise ValueError(f"Unsupported score_method: {method}")

        if not np.isfinite(metric):
            continue

        # Turnover: fraction of bars where signal truly flips sign (positive↔negative)
        # Only computed AFTER metric is confirmed finite
        signs = np.sign(aligned["x"].values)
        if len(signs) > 1:
            flips = float(np.sum((signs[1:] * signs[:-1]) < 0)) / (len(signs) - 1)
            turnover_vals.append(flips)

        metrics[pd_item.pair] = metric
        values.append(metric)

    if not values:
        raise ValueError("not enough samples")
    turnover = float(np.mean(turnover_vals)) if turnover_vals else 0.5
    return float(np.mean(np.abs(values))), metrics, turnover


def _evolve_candidates(  # noqa: PLR0913
    feature_cols: Sequence[str],
    pairs: Sequence[_PairData],
    *,
    population: int,
    generations: int,
    max_depth: int,
    mutation_rate: float,
    crossover_rate: float,
    elitism: int,
    method: str,
    min_samples: int,
    complexity_penalty: float,
    seed: Optional[int],
    oos_fraction: float = 0.0,
    label_period: int = 1,
    turnover_penalty: float = 0.0,
    llm_seeds: Optional[List[str]] = None,
    tournament_size: int = 4,
    n_islands: int = 4,
    migration_interval: int = 5,
    migration_count: int = 2,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cache: Dict[str, Tuple[float, Dict[str, float], int]] = {}

    def fitness(node: _GPNode) -> Tuple[float, Dict[str, float], int, str]:
        expr_str = _gp_to_string(node)
        if expr_str in cache:
            s, m, size = cache[expr_str]
            return s, m, size, expr_str
        try:
            abs_ic, per_pair, turnover = _score_expression_across_pairs(
                expr_str, pairs, method=method, min_samples=min_samples,
                oos_fraction=oos_fraction, label_period=label_period,
            )
        except Exception:
            abs_ic, per_pair, turnover = 0.0, {}, 0.5
        size = _gp_size(node)
        score = (float(abs_ic)
                 - float(complexity_penalty) * float(size)
                 - float(turnover_penalty) * float(turnover))
        cache[expr_str] = (score, per_pair, size)
        return score, per_pair, size, expr_str

    # ── Fix 3: Tournament selection ──
    def _tournament_select(
        scored_pop: List[Tuple[float, Dict[str, float], int, str, _GPNode]],
        k: int,
    ) -> _GPNode:
        """Pick *k* random individuals, return the best."""
        contestants = [scored_pop[rng.randrange(len(scored_pop))] for _ in range(k)]
        best = max(contestants, key=lambda t: float(t[0]))
        return best[-1]

    # ── Fix 4: Semantic crossover ──
    def _semantic_crossover(
        rng_: random.Random, a: _GPNode, b: _GPNode, depth_limit: int,
    ) -> _GPNode:
        """Crossover preferring swap points at matching operator types."""
        a_paths = _gp_all_paths(a)
        b_paths = _gp_all_paths(b)

        # Skip root path — replacing root discards the whole tree
        a_inner = [p for p in a_paths if p]
        if not a_inner:
            return _gp_crossover(rng_, a, b, depth_limit)

        # Group b paths by operator type
        b_by_op: Dict[str, List[Tuple[int, ...]]] = {}
        for bp in b_paths:
            op = _gp_get(b, bp).op
            b_by_op.setdefault(op, []).append(bp)

        # Try to find a semantically compatible swap (shuffle a copy)
        candidates = list(a_inner)
        rng_.shuffle(candidates)
        for ap in candidates[:8]:  # limit search for perf
            a_op = _gp_get(a, ap).op
            if a_op in b_by_op:
                bp = rng_.choice(b_by_op[a_op])
                swapped = _gp_replace(a, ap, _gp_get(b, bp))
                if _gp_depth(swapped) <= depth_limit:
                    return swapped

        # Fallback to random crossover
        return _gp_crossover(rng_, a, b, depth_limit)

    # Diversity-aware elite vectors
    vec_cache: Dict[str, pd.Series] = {}

    def _get_vec(expr_str: str) -> Optional[pd.Series]:
        if expr_str in vec_cache:
            return vec_cache[expr_str]
        parts: List[pd.Series] = []
        for pd_item in pairs:
            try:
                s = _safe_eval_expression(expr_str, pd_item.df)
                parts.append(s.reset_index(drop=True))
            except Exception:
                continue
        if not parts:
            vec_cache[expr_str] = None  # type: ignore[assignment]
            return None
        vec = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        vec_cache[expr_str] = vec if len(vec) >= 20 else None  # type: ignore[assignment]
        return vec_cache.get(expr_str)

    def _diverse_elites(
        scored_pop: List[Tuple[float, Dict[str, float], int, str, _GPNode]],
        n: int,
    ) -> List[_GPNode]:
        """Pick up to *n* elites ensuring pairwise |corr| < 0.85."""
        elites: List[_GPNode] = []
        elite_vecs: List[pd.Series] = []
        for _score, _pp, _sz, expr_str, node in scored_pop:
            if len(elites) >= n:
                break
            vec = _get_vec(expr_str)
            if vec is None or len(vec) < 20:
                continue
            if vec.std() < 1e-12:
                continue
            too_similar = False
            for ev in elite_vecs:
                merged = pd.DataFrame({"a": vec, "b": ev}).dropna()
                if len(merged) < 20:
                    continue
                corr_val = abs(float(merged["a"].corr(merged["b"])))
                if not np.isfinite(corr_val) or corr_val > 0.85:
                    too_similar = True
                    break
            if too_similar:
                continue
            elites.append(node)
            elite_vecs.append(vec)
        if len(elites) < n:
            seen = {id(e) for e in elites}
            for _, _, _, _, node in scored_pop:
                if id(node) not in seen:
                    elites.append(node)
                    seen.add(id(node))
                if len(elites) >= n:
                    break
        return elites

    # ── Fix 1: Seed population with parsed LLM expressions ──
    llm_nodes: List[_GPNode] = []
    if llm_seeds:
        for expr_str in llm_seeds:
            node = _parse_expr_to_gp(expr_str, feature_cols)
            if node is not None:
                llm_nodes.append(node)

    # ── Fix 2: Island Model ──
    actual_islands = max(1, min(n_islands, population // 10))
    island_size = (population + actual_islands - 1) // actual_islands  # ceiling division
    islands: List[List[_GPNode]] = []

    for i in range(actual_islands):
        island_pop: List[_GPNode] = []
        # Inject LLM seeds: distribute across islands
        llm_share = llm_nodes[i::actual_islands]
        island_pop.extend(llm_share[:island_size // 3])  # up to 1/3 from LLM
        # Fill the rest randomly
        while len(island_pop) < island_size:
            island_pop.append(_gp_random_expr(rng, feature_cols, max_depth))
        islands.append(island_pop)

    injected_llm = sum(
        min(len(llm_nodes[i::actual_islands]), island_size // 3)
        for i in range(actual_islands)
    )
    if llm_seeds:
        print(f"[evolve] parsed {len(llm_nodes)}/{len(llm_seeds)} LLM seeds → injected {injected_llm} into {actual_islands} islands")

    for gen in range(max(1, generations)):
        if gen % max(1, generations // 5) == 0:
            best_score = max((fitness(n)[0] for n in islands[0]), default=0.0)
            print(f"[evolve] gen {gen+1}/{generations} best_score={best_score:.4f} cache={len(cache)}")
        for isle_idx, isle_pop in enumerate(islands):
            scored = [fitness(node) + (node,) for node in isle_pop]
            scored.sort(key=lambda t: float(t[0]), reverse=True)

            elites_per_island = max(1, min(elitism // actual_islands + 1, island_size // 4))
            elites = _diverse_elites(scored, elites_per_island)
            next_isle: List[_GPNode] = list(elites)

            while len(next_isle) < island_size:
                parent = _tournament_select(scored, tournament_size)
                child = parent
                if rng.random() < crossover_rate:
                    other = _tournament_select(scored, tournament_size)
                    child = _semantic_crossover(rng, parent, other, max_depth)
                if rng.random() < mutation_rate:
                    child = _gp_mutate(rng, child, feature_cols, max_depth)
                next_isle.append(child)
            islands[isle_idx] = next_isle

        # ── Migration: replace WORST in destination with BEST from source ──
        if gen > 0 and gen % migration_interval == 0 and actual_islands > 1:
            for src in range(actual_islands):
                dst = (src + 1) % actual_islands
                src_scored = [fitness(n) + (n,) for n in islands[src]]
                src_scored.sort(key=lambda t: float(t[0]), reverse=True)
                migrants = [t[-1] for t in src_scored[:migration_count]]
                # Keep all but the worst migration_count in destination, append migrants
                dst_scored = [fitness(n) + (n,) for n in islands[dst]]
                dst_scored.sort(key=lambda t: float(t[0]), reverse=True)
                kept = [t[-1] for t in dst_scored[:len(dst_scored) - migration_count]]
                islands[dst] = kept + migrants

    # Merge all islands for final ranking
    all_pop: List[_GPNode] = []
    for isle in islands:
        all_pop.extend(isle)
    final = [fitness(node) + (node,) for node in all_pop]
    final.sort(key=lambda t: float(t[0]), reverse=True)

    def extract_terms(expr: str) -> List[str]:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return []
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return sorted(n for n in names if n not in called)

    def infer_category(expr: str, terms: Sequence[str]) -> str:
        blob = " ".join([expr.lower(), *[t.lower() for t in terms]])
        if any(k in blob for k in ("volume", "obv", "mfi")):
            return "volume"
        if any(k in blob for k in ("atr", "realized_vol", "bb_width", "donchian_width", "roll_std", "ts_z")):
            return "volatility"
        if any(k in blob for k in ("macd", "ema", "sma", "wma", "kama", "psar")):
            return "trend"
        if any(k in blob for k in ("momentum", "roc")):
            return "momentum"
        if any(k in blob for k in ("rsi", "stoch", "cci")):
            return "mean_reversion"
        return "other"

    def summarize_terms(terms: Sequence[str], *, depth: int, complexity: int) -> str:
        if not terms:
            return f"evolved composite factor (depth={depth}, complexity={complexity})"
        head = ", ".join(list(terms)[:3])
        suffix = ", …" if len(terms) > 3 else ""
        return f"evolved composite of {head}{suffix} (depth={depth}, complexity={complexity})"

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (score, per_pair, size, expr_str, node) in enumerate(final, start=1):
        if not expr_str or expr_str in seen:
            continue
        seen.add(expr_str)
        depth = _gp_depth(node)
        terms = extract_terms(expr_str)
        category = infer_category(expr_str, terms)
        out.append(
            {
                "name": f"g{rank:03d}",
                "expression": expr_str,
                "origin": "evolve",
                "score": float(score),
                "metric_abs_ic": float(score + float(complexity_penalty) * float(size)),
                "per_pair": per_pair,
                "complexity": int(size),
                "depth": int(depth),
                "terms": terms,
                "description": summarize_terms(terms, depth=depth, complexity=int(size)),
                "category": category,
            }
        )
        if len(out) >= max(20, population):
            break
    return out


def generate_llm_expressions(  # noqa: PLR0913
    df: DataFrame,
    feature_cols: Sequence[str],
    feature_cfg: Dict[str, Any],
    combos: Sequence[Dict[str, Any]],
    timeframe: str,
    label_period: int,
    score_method: str,
    complexity_penalty: float,
    backtest_weight: float,  # noqa: ARG001 - reserved
    periods_per_year: int,  # noqa: ARG001 - reserved
    stability_windows: int,  # noqa: ARG001 - reserved
    stability_min_samples: int,  # noqa: ARG001 - reserved
    llm_options: Any,
) -> List[Dict[str, Any]]:
    """
    Minimal API used by tests:
    - Calls LLM once and scores returned expressions via simple correlation.
    - Returns a list of scored candidates sorted by score desc.
    """

    llm_cfg = llm_utils.LLMConfig.from_args(llm_options)
    prompt = llm_utils.build_prompt(
        feature_cfg=feature_cfg,
        feature_cols=list(feature_cols),
        combos=list(combos),
        timeframe=timeframe,
        label_period=int(label_period),
        request_count=int(getattr(llm_options, "llm_count", 10) or 10),
    )
    content, usage = llm_utils.request_completion(prompt, llm_cfg)
    candidates = llm_utils.extract_candidates(content)
    if not candidates:
        raise ValueError("No candidates returned by LLM")

    labels = (df["close"].shift(-int(label_period)) / df["close"]) - 1
    scored: List[Dict[str, Any]] = []
    for item in candidates:
        expr_str = str(item.get("expression") or "").strip()
        if not expr_str:
            continue
        series = _safe_eval_expression(expr_str, df)
        try:
            metric = _score_expression(series, labels, score_method)
        except ValueError:
            continue
        score = abs(float(metric)) - float(complexity_penalty) * float(max(1, len(expr_str) // 10))
        scored.append(
            {
                "origin": "llm",
                "expression": expr_str,
                "score": float(score),
                "metric": float(metric),
                "description": item.get("description"),
                "reason": item.get("reason"),
                "category": item.get("category"),
                "usage": usage,
            }
        )
    if not scored:
        raise ValueError("No valid scored expressions")
    scored.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
    return scored


def _classic_candidates(feature_cols: Sequence[str], count: int) -> List[Dict[str, Any]]:
    cols = [c for c in feature_cols if isinstance(c, str) and c.strip()]
    if not cols:
        return []

    def has(name: str) -> bool:
        return name in cols

    preferred = [
        "ema_spread",
        "macd_diff",
        "rsi_14",
        "stoch_diff",
        "psar_ratio",
        "donchian_width_20",
        "kama_pct_20",
        "cci_20",
        "adx_14",
        "mfi_14",
    ]
    ordered = [c for c in preferred if has(c)] + [c for c in cols if c not in preferred]

    templates = [
        ("z({x})", "normalized signal", "other"),
        ("tanh(z({x}))", "squashed normalized signal", "other"),
        ("clip(z({x}), -3, 3)", "clipped z-score", "other"),
        ("-z({x})", "mean-reversion transform", "mean_reversion"),
        ("ts_z({x}, 48)", "time-series z-score (48)", "other"),
        ("tanh(ts_z({x}, 48))", "squashed time-series z-score (48)", "other"),
        ("z({x}) - z({y})", "spread between signals", "momentum"),
        ("z({x}) / (1 + abs(z({y})))", "risk-adjusted spread", "other"),
        ("ema(z({x}), 12)", "smoothed signal", "trend"),
        ("ema(z({x}), 12) - ema(z({x}), 48)", "multi-horizon EMA difference", "trend"),
        ("z({x}) / (1 + abs(roll_std(z({x}), 24)))", "volatility-adjusted signal", "volatility"),
        ("tanh(z({x}) / (1 + abs(roll_std(z({x}), 24))))", "squashed vol-adjusted signal", "volatility"),
        ("ema(z({x}) - z({y}), 12)", "smoothed spread", "trend"),
        ("(z({y}) > 0) * z({x})", "regime filter applied to x", "other"),
        ("(z({y}) < 0) * -z({x})", "mean-reversion in negative regime", "mean_reversion"),
        ("roll_mean(z({x}), 12)", "rolling mean", "trend"),
        ("roll_std(z({x}), 12)", "rolling std", "volatility"),
    ]

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    for x in ordered:
        for tpl, desc, cat in templates:
            expr = tpl.format(x=x, y=ordered[0] if ordered else x)
            if "{y}" in tpl and len(ordered) >= 2:
                for y in ordered:
                    if y == x:
                        continue
                    expr = tpl.format(x=x, y=y)
                    if expr in seen:
                        continue
                    idx += 1
                    out.append(
                        {
                            "name": _stable_slug(idx),
                            "expression": expr,
                            "description": f"{desc}: {x} vs {y}",
                            "category": cat,
                        }
                    )
                    seen.add(expr)
                    if len(out) >= count:
                        return out
            else:
                if expr in seen:
                    continue
                idx += 1
                out.append(
                    {
                        "name": _stable_slug(idx),
                        "expression": expr,
                        "description": f"{desc}: {x}",
                        "category": cat,
                    }
                )
                seen.add(expr)
                if len(out) >= count:
                    return out
    return out[:count]


def main(argv: Optional[List[str]] = None) -> int:
    from agent_market.config import FreqAISettings  # noqa: WPS433
    from agent_market.freqai.features import apply_configured_features  # noqa: WPS433

    parser = argparse.ArgumentParser(description="Generate expression candidates from engineered features.")
    parser.add_argument("--config", required=True, help="Freqtrade config JSON path.")
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Feature JSON path (defaults to user_data/freqai_features*.json).",
    )
    parser.add_argument(
        "--output",
        default="user_data/freqai_expressions.json",
        help="Output expressions JSON path.",
    )
    parser.add_argument("--timeframe", default=None, help="Timeframe used in prompt/metadata (e.g. 1h).")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional override pairs (e.g. BTC/USDT ETH/USDT).")

    parser.add_argument("--llm-enabled", action="store_true", help="Enable LLM-based expression generation.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based generation (force classic templates).")
    parser.add_argument("--llm-fallback", action="store_true", help="Fallback to classic templates if LLM fails.")
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["openai_compatible", "opencode", "auto"],
        help="LLM provider for factor mining. Use opencode for agentic generation.",
    )
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--llm-agent-url",
        default=None,
        help="Optional OpenCode server URL. Falls back to OPENCODE_URL or local opencode CLI.",
    )
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--llm-workspace",
        default=None,
        help="Repo workspace exposed to the OpenCode factor-mining agent.",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-count", type=int, default=20)
    parser.add_argument("--llm-loops", type=int, default=1)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=45)
    parser.add_argument("--llm-max-turns", type=int, default=12)
    parser.add_argument("--llm-stale-timeout", type=float, default=180.0)

    parser.add_argument("--feedback", default=None, help="Optional feedback file to guide the next batch.")
    parser.add_argument("--feedback-top", type=int, default=10, help="Reserved for orchestrator; kept for compatibility.")

    parser.add_argument("--mine", action="store_true", help="Run factor mining: score candidates and select top-N.")
    parser.add_argument("--top-n", type=int, default=30, help="Top N expressions to keep when --mine is enabled.")
    parser.add_argument(
        "--score-method",
        default="correlation",
        choices=["correlation", "spearman"],
        help="Scoring method for factor mining.",
    )
    parser.add_argument("--min-samples", type=int, default=200, help="Min samples required per pair for scoring.")
    parser.add_argument("--complexity-penalty", type=float, default=0.002, help="Penalty per complexity unit.")
    parser.add_argument("--oos-fraction", type=float, default=0.0,
                        help="Fraction of data reserved for OOS scoring (0=full-sample, 0.3=use last 30%%).")
    parser.add_argument("--turnover-penalty", type=float, default=0.0,
                        help="Penalty weight for factor signal turnover (higher = prefer stable signals).")

    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="Max |correlation| between selected factors (diversity filter).")
    parser.add_argument("--ac-penalty", type=float, default=0.5,
                        help="Penalty weight for factor autocorrelation above --ac-threshold.")
    parser.add_argument("--ac-threshold", type=float, default=0.95,
                        help="Autocorrelation(1) above this triggers penalty.")

    parser.add_argument("--llm-only", action="store_true", help="Only keep LLM-generated candidates; skip classic templates and evolve.")
    parser.add_argument("--evolve", action="store_true", help="Enable evolutionary search when --mine is enabled.")
    parser.add_argument("--no-evolve", action="store_true", help="Disable evolutionary search when --mine is enabled.")
    parser.add_argument("--evolve-population", type=int, default=60)
    parser.add_argument("--evolve-generations", type=int, default=8)
    parser.add_argument("--evolve-max-depth", type=int, default=4)
    parser.add_argument("--evolve-mutation-rate", type=float, default=0.45)
    parser.add_argument("--evolve-crossover-rate", type=float, default=0.65)
    parser.add_argument("--evolve-elitism", type=int, default=6)
    parser.add_argument("--evolve-seed", type=int, default=7)
    parser.add_argument("--refine-rounds", type=int, default=0,
                        help="Number of factor refinement rounds (each refines top-N factors via LLM).")
    parser.add_argument("--per-loop-refine", type=int, default=0,
                        help="Refinement rounds after EACH LLM loop (refines top-3 batch factors inline).")
    parser.add_argument(
        "--feature-cols-mode",
        default="auto",
        choices=["config", "auto", "data"],
        help=(
            "How to build the feature column pool used for classic templates + LLM prompt. "
            "'config' uses only the feature_file list. "
            "'auto' adds common engineered columns present in the OHLCV feather (mtf4h_/funding_/xs_/micro_/ret_1/rv_*/...). "
            "'data' uses all non-OHLCV columns present in the feather."
        ),
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    cfg = read_json(config_path)
    timeframe = resolve_timeframe(cfg, args.timeframe)

    feature_path = resolve_path(args.feature_file) if args.feature_file else _infer_feature_file(config_path)
    feature_cfg = read_json(feature_path)
    label_period = resolve_label_period(cfg, feature_cfg.get("label_period"))

    features = feature_cfg.get("features") or []
    combos = feature_cfg.get("feature_combos") or []
    feature_cols: List[str] = []
    for item in features:
        if isinstance(item, dict) and item.get("name"):
            feature_cols.append(str(item["name"]))
    for item in combos:
        if isinstance(item, dict) and item.get("name"):
            feature_cols.append(str(item["name"]))

    def _is_safe_name(name: str) -> bool:
        # Keep compatible with ExpressionEngine identifier rules (no leading underscore).
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$", str(name or ""))) and not str(name).startswith("_")

    def _discover_engineered_cols(cols: Sequence[str]) -> List[str]:
        prefixes = ("mtf4h_", "xs_", "funding_", "micro_")
        exact = {
            # OHLCV micro_feature names
            "ret_1", "logret_1", "range_pct", "body_pct", "wick_up_pct", "wick_down_pct",
            "rv_12", "rv_24", "rv_72",
            "vol_z_12", "vol_z_24", "vol_z_72",
            "amihud_12", "amihud_24", "amihud_72",
            # Microstructure names (only if you merged/aggregated them into the feather)
            "mid", "spread", "rel_spread", "microprice", "trade_sign",
            "expected_slippage_proxy", "fill_prob_proxy", "toxicity_proxy",
        }

        out: list[str] = []
        for c in cols:
            s = str(c)
            if not _is_safe_name(s):
                continue
            if s in exact or s.startswith(prefixes):
                out.append(s)
                continue
            if re.match(r"^(depth_bid|depth_ask|imbalance|slope_bid|convexity)_\\d+$", s):
                out.append(s); continue
            if re.match(r"^(vwap|ofi|arrival_intensity|buy_vol|sell_vol|l2_ofi)_\\d+$", s):
                out.append(s); continue
        return sorted(set(out))

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    want_llm = bool(args.llm_enabled) and not bool(args.no_llm)
    request_count = max(1, int(args.llm_count))

    feedback_text = _read_feedback(args.feedback)
    expressions: List[Dict[str, Any]] = []

    # ── Load existing factor memory for cross-run knowledge ──
    from agent_market.factor_memory import FactorMemoryStore  # noqa: WPS433
    from agent_market import paths as _paths  # noqa: WPS433

    global_memory_path = _paths.global_factor_memory_path()
    local_memory_dir = output_path.parent / "factor_memory"
    prior_expressions: List[str] = []
    prior_top_factors: List[Dict[str, Any]] = []
    prior_failure_hints: List[str] = []

    if global_memory_path.exists():
        try:
            gstore = FactorMemoryStore(global_memory_path)
            # Extract prior expressions to avoid
            for card in gstore.factor_cards:
                expr = str(card.get("expression") or "").strip()
                if expr:
                    prior_expressions.append(expr)
            # Extract top performers for feedback
            top_cards = sorted(
                [c for c in gstore.factor_cards if c.get("gate_pass")],
                key=lambda c: float((c.get("metrics") or {}).get("weighted_score") or 0),
                reverse=True,
            )[:15]
            for card in top_cards:
                prior_top_factors.append({
                    "expression": str(card.get("expression") or "")[:80],
                    "ic": float((card.get("metrics") or {}).get("ic") or 0),
                    "category": (card.get("category_tags") or ["other"])[0],
                })
            # Extract failure hints
            for fcard in gstore.failure_cards[-10:]:
                recipe = fcard.get("repair_recipe") or {}
                for hint in (recipe.get("avoid_rules") or []):
                    if hint not in prior_failure_hints:
                        prior_failure_hints.append(hint)
            if prior_expressions:
                print(f"[memory] loaded {len(prior_expressions)} prior factors, "
                      f"{len(prior_top_factors)} top performers from global memory")
        except Exception as exc:
            print(f"[memory] failed to load global memory: {exc}", file=sys.stderr)

    # Prepend memory feedback to feedback_text
    if prior_failure_hints:
        mem_feedback = "Prior run failure insights:\n" + "\n".join(f"- {h}" for h in prior_failure_hints[:5])
        feedback_text = (feedback_text + "\n\n" + mem_feedback) if feedback_text else mem_feedback
    llm_usage_totals: Dict[str, float] = {
        "total_tokens": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "reasoning_tokens": 0.0,
        "cache_read_tokens": 0.0,
        "cache_write_tokens": 0.0,
        "cost_usd": 0.0,
    }

    def _accumulate_llm_usage(usage: Optional[Dict[str, Any]]) -> None:
        if not usage:
            return
        for key in llm_usage_totals:
            try:
                llm_usage_totals[key] += float(usage.get(key) or 0.0)
            except Exception:
                continue

    if args.mine:
        settings = FreqAISettings.from_file(config_path, timeframe_override=timeframe)
        if args.pairs:
            settings = FreqAISettings.from_file(config_path, timeframe_override=timeframe, label_override=None)
            settings.pairs = [str(p) for p in args.pairs if str(p).strip()]
        settings.validate_dataset(settings.timeframe)

        pairs_data: List[_PairData] = []
        discovered_cols: set[str] = set()
        for pair in settings.pairs:
            sanitized = pair.replace("/", "_")
            data_path = settings.data_dir / f"{sanitized}-{settings.timeframe}.feather"
            df = pd.read_feather(data_path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)
            df = apply_configured_features(df, feature_cfg)
            if str(args.feature_cols_mode or "").strip().lower() in {"auto", "data"}:
                discovered_cols.update(map(str, df.columns))
            label = (df["close"].shift(-int(label_period)) / df["close"]) - 1
            pairs_data.append(_PairData(pair=pair, df=df, label=label))

        mode = str(args.feature_cols_mode or "auto").strip().lower()
        if mode == "data":
            exclude = {"date", "open", "high", "low", "close", "volume"}
            extra = sorted(
                {
                    c
                    for c in discovered_cols
                    if c not in exclude and _is_safe_name(c)
                }
            )
        elif mode == "auto":
            extra = _discover_engineered_cols(sorted(discovered_cols))
        else:
            extra = []

        if extra:
            seen = set(feature_cols)
            for c in extra:
                if c not in seen:
                    feature_cols.append(c)
                    seen.add(c)

        candidate_pool = max(int(args.top_n) * 10, request_count, 50, int(args.llm_loops) * int(args.llm_count) // 2)
        llm_only = bool(getattr(args, "llm_only", False))
        if not llm_only:
            classic_pool = min(candidate_pool, max(int(args.top_n) * 5, 50))
            expressions.extend(_classic_candidates(feature_cols, classic_pool))
        else:
            print("[llm-only] skipping classic templates")

        if want_llm:
            llm_cfg = llm_utils.LLMConfig.from_args(args)
            avoid: set[str] = {str(x.get("expression") or "").strip() for x in expressions}
            # Seed avoid with prior factor memory expressions
            avoid.update(prior_expressions)

            # Auto-review loop state
            base_temperature = float(llm_cfg.temperature)
            scored_so_far: List[Dict[str, Any]] = []
            # Seed scored_so_far with prior top performers from memory
            if prior_top_factors:
                scored_so_far.extend(prior_top_factors)
            # Global score cache: expr_str → (abs_ic, per_pair, turnover)
            _score_cache: Dict[str, Tuple[float, Dict[str, float], float]] = {}

            for loop in range(max(1, int(args.llm_loops))):
                # ── Adaptive temperature: ramp up over loops ──
                progress = loop / max(1, int(args.llm_loops) - 1)
                llm_cfg.temperature = min(1.0, base_temperature + progress * 0.4)

                # ── Build dynamic loop feedback ──
                loop_feedback: Optional[Dict[str, Any]] = None
                if scored_so_far and loop >= 5:
                    # Quick-score recent batch for feedback
                    cat_dist: Dict[str, int] = {}
                    for item in expressions:
                        cat = str(item.get("category") or "other")
                        cat_dist[cat] = cat_dist.get(cat, 0) + 1

                    all_categories = ["trend", "momentum", "volatility", "volume", "mean_reversion"]
                    total_cats = sum(cat_dist.values())
                    weak_cats = [c for c in all_categories if cat_dist.get(c, 0) / max(total_cats, 1) < 0.1]

                    top_scored = sorted(scored_so_far, key=lambda d: float(d.get("score", 0)), reverse=True)[:10]
                    top_factors = [
                        {"expression": d["expression"][:80], "ic": d.get("ic", 0), "category": d.get("category", "other")}
                        for d in top_scored
                    ]
                    loop_feedback = {
                        "category_distribution": cat_dist,
                        "top_factors": top_factors,
                        "weak_categories": weak_cats,
                    }

                prompt = llm_utils.build_prompt(
                    feature_cfg=feature_cfg,
                    feature_cols=feature_cols,
                    combos=combos,
                    timeframe=timeframe,
                    label_period=label_period,
                    request_count=request_count,
                    avoid_expressions=avoid,
                    feedback=feedback_text,
                    loop_feedback=loop_feedback,
                )
                # ── Layer 1: API call — network/auth failures skip this loop ──
                content, usage = None, None
                try:
                    content, usage = llm_utils.request_completion(prompt, llm_cfg)
                except Exception as exc:
                    print(f"[llm] API failed loop {loop+1} (skipping): {exc}", file=sys.stderr)
                    continue

                # ── Layer 2: JSON parse — malformed response skips this loop ──
                batch: List[Dict[str, Any]] = []
                try:
                    batch = llm_utils.extract_candidates(content)
                except Exception as exc:
                    print(f"[llm] parse failed loop {loop+1} (skipping): {exc}", file=sys.stderr)
                    if usage:
                        _accumulate_llm_usage(usage)
                    continue

                # ── Layer 3: Process batch + per-loop refine ──
                new_count = 0
                new_batch_items: List[Tuple[str, Dict[str, Any]]] = []
                for item in batch:
                    expr_str = str(item.get("expression") or "").strip()
                    if not expr_str or expr_str in avoid:
                        continue
                    avoid.add(expr_str)
                    expressions.append({"origin": "llm", **item})
                    new_count += 1

                    # Quick inline scoring for feedback to next loop (cached)
                    try:
                        if expr_str in _score_cache:
                            ic = _score_cache[expr_str][0]
                        else:
                            ic, pp, to = _score_expression_across_pairs(
                                expr_str, pairs_data,
                                method=str(args.score_method),
                                min_samples=max(50, int(args.min_samples) // 2),
                            )
                            _score_cache[expr_str] = (ic, pp, to)
                        scored_so_far.append({
                            "expression": expr_str, "score": ic, "ic": ic,
                            "category": item.get("category", "other"),
                        })
                        new_batch_items.append((expr_str, item))
                    except Exception:
                        pass

                    if len(expressions) >= candidate_pool:
                        break

                if usage:
                    _accumulate_llm_usage(usage)
                    print(
                        "[llm] usage",
                        json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                    )
                print(f"[llm] loop {loop+1}: total={len(expressions)} new={new_count} temp={llm_cfg.temperature:.2f}")

                # ── Per-loop inline refinement ──
                per_loop_refine = int(getattr(args, "per_loop_refine", 0))
                if per_loop_refine > 0 and new_batch_items:
                    top_to_refine = sorted(
                        new_batch_items,
                        key=lambda t: _score_cache.get(t[0], (0.0,))[0],
                        reverse=True,
                    )[:3]
                    for rnd in range(per_loop_refine):
                        rnd_new = 0
                        for r_expr, r_item in top_to_refine:
                            if r_expr not in _score_cache:
                                continue
                            r_ic, r_pp, r_to = _score_cache[r_expr]
                            refine_prompt = llm_utils.build_refine_prompt(
                                factor={
                                    "expression": r_expr,
                                    "metric_abs_ic": r_ic,
                                    "category": r_item.get("category", "other"),
                                },
                                feature_cols=feature_cols,
                                feature_cfg=feature_cfg,
                                per_pair_ic=r_pp,
                                timeframe=timeframe,
                                label_period=label_period,
                                turnover=r_to,
                            )
                            try:
                                rc, ru = llm_utils.request_completion(refine_prompt, llm_cfg)
                                if ru:
                                    _accumulate_llm_usage(ru)
                                for ritem in llm_utils.extract_candidates(rc):
                                    new_expr = str(ritem.get("expression") or "").strip()
                                    if not new_expr or new_expr in avoid:
                                        continue
                                    avoid.add(new_expr)
                                    ritem["origin"] = "refine"
                                    expressions.append(ritem)
                                    rnd_new += 1
                                    try:
                                        ric, rpp, rto = _score_expression_across_pairs(
                                            new_expr, pairs_data,
                                            method=str(args.score_method),
                                            min_samples=max(50, int(args.min_samples) // 2),
                                        )
                                        _score_cache[new_expr] = (ric, rpp, rto)
                                        scored_so_far.append({
                                            "expression": new_expr, "score": ric, "ic": ric,
                                            "category": ritem.get("category", "other"),
                                        })
                                    except Exception:
                                        pass
                            except Exception as exc:
                                print(f"[refine] loop {loop+1} rnd {rnd+1} failed: {exc}", file=sys.stderr)
                        print(f"[refine] loop {loop+1} rnd {rnd+1}/{per_loop_refine}: {rnd_new} new")

                if len(expressions) >= candidate_pool:
                    break

        evolve_enabled = bool(args.evolve) or not bool(args.no_evolve)
        if evolve_enabled:
            try:
                evolved = _evolve_candidates(
                    feature_cols=feature_cols,
                    pairs=pairs_data,
                    population=int(args.evolve_population),
                    generations=int(args.evolve_generations),
                    max_depth=int(args.evolve_max_depth),
                    mutation_rate=float(args.evolve_mutation_rate),
                    crossover_rate=float(args.evolve_crossover_rate),
                    elitism=int(args.evolve_elitism),
                    method=str(args.score_method),
                    min_samples=int(args.min_samples),
                    complexity_penalty=float(args.complexity_penalty),
                    seed=int(args.evolve_seed) if args.evolve_seed is not None else None,
                    oos_fraction=float(getattr(args, "oos_fraction", 0.0)),
                    label_period=int(label_period),
                    turnover_penalty=float(getattr(args, "turnover_penalty", 0.0)),
                    llm_seeds=[str(e.get("expression", "")) for e in expressions if e.get("origin") == "llm"],
                )
                expressions.extend(evolved)
                print(f"[evolve] candidates={len(evolved)}")
            except Exception as exc:
                print(f"[evolve] failed: {exc}", file=sys.stderr)

        dedup: Dict[str, Dict[str, Any]] = {}
        for item in expressions:
            expr_str = str(item.get("expression") or "").strip()
            if not expr_str:
                continue
            if expr_str not in dedup:
                dedup[expr_str] = item
        scored: List[Dict[str, Any]] = []
        oos_frac = float(getattr(args, "oos_fraction", 0.0))
        to_penalty = float(getattr(args, "turnover_penalty", 0.0))
        # Final scoring cache (shared with evolve's cache via expr string)
        _final_score_cache: Dict[str, Tuple[float, Dict[str, float], float]] = {}
        total_to_score = len(dedup)
        scored_count = 0
        for expr_str, item in dedup.items():
            scored_count += 1
            if scored_count % 500 == 0:
                print(f"[score] {scored_count}/{total_to_score}...")
            try:
                if expr_str in _final_score_cache:
                    abs_ic, per_pair, turnover = _final_score_cache[expr_str]
                else:
                    abs_ic, per_pair, turnover = _score_expression_across_pairs(
                        expr_str,
                        pairs_data,
                        method=str(args.score_method),
                        min_samples=int(args.min_samples),
                        oos_fraction=oos_frac,
                        label_period=int(label_period),
                    )
                    _final_score_cache[expr_str] = (abs_ic, per_pair, turnover)
            except Exception:
                continue
            complexity = int(item.get("complexity") or max(1, len(expr_str) // 10))
            score = (float(abs_ic)
                     - float(args.complexity_penalty) * float(complexity)
                     - to_penalty * float(turnover))
            scored.append(
                {
                    "origin": item.get("origin", "classic"),
                    "expression": expr_str,
                    "metric_abs_ic": float(abs_ic),
                    "per_pair": per_pair,
                    "complexity": int(complexity),
                    "turnover": float(turnover),
                    "score": float(score),
                    "description": item.get("description"),
                    "category": item.get("category"),
                }
            )
        scored.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        scored_path = output_path.with_name(output_path.stem + "_scored_all.json")
        scored_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "exchange": feature_cfg.get("exchange") or (cfg.get("exchange") or {}).get("name"),
                    "pairs": settings.pairs,
                    "timeframe": settings.timeframe,
                    "label_period": int(label_period),
                    "score_method": str(args.score_method),
                    "min_samples": int(args.min_samples),
                    "complexity_penalty": float(args.complexity_penalty),
                    "candidates": scored,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[mine] wrote scored candidates: {scored_path} total={len(scored)}")
        selected = _select_diverse_topn(
            scored,
            pairs_data,
            max(1, int(args.top_n)),
            corr_threshold=float(args.corr_threshold),
            ac_penalty=float(args.ac_penalty),
            ac_threshold=float(args.ac_threshold),
            label_period=int(label_period),
        )
        print(
            f"[diversity] selected {len(selected)} independent factors"
            f" (corr<{args.corr_threshold}, ac_penalty={args.ac_penalty})"
        )
        expressions = [
            {
                "name": f"f{idx:03d}",
                "expression": item["expression"],
                "description": item.get("description") or "mined factor",
                "category": item.get("category") or "other",
                "origin": item.get("origin"),
                "metric_abs_ic": item.get("metric_abs_ic"),
                "score": item.get("score"),
                "score_adjusted": item.get("score_adjusted"),
                "autocorr_1": item.get("autocorr_1"),
                "complexity": item.get("complexity"),
                "per_pair": item.get("per_pair"),
            }
            for idx, item in enumerate(selected, start=1)
        ]

        # ── Factor Refinement Loop ──
        refine_rounds = int(getattr(args, "refine_rounds", 0))
        if refine_rounds > 0 and want_llm and expressions:
            print(f"[refine] starting {refine_rounds} refinement rounds on {len(expressions)} factors")
            llm_cfg_refine = llm_utils.LLMConfig.from_args(args)
            llm_cfg_refine.temperature = 0.3  # lower temp for refinement (more precise)
            oos_frac = float(getattr(args, "oos_fraction", 0.0))

            all_refined: List[Dict[str, Any]] = []
            for rnd in range(refine_rounds):
                round_new = 0
                for factor in expressions:
                    expr_str = str(factor.get("expression") or "").strip()
                    per_pair = factor.get("per_pair") or {}
                    ac1 = float(factor.get("autocorr_1") or 0.0)

                    # Compute turnover for this factor
                    try:
                        _, _, turnover = _score_expression_across_pairs(
                            expr_str, pairs_data,
                            method=str(args.score_method),
                            min_samples=max(50, int(args.min_samples) // 2),
                        )
                    except Exception:
                        turnover = 0.5

                    prompt = llm_utils.build_refine_prompt(
                        factor=factor,
                        feature_cols=feature_cols,
                        feature_cfg=feature_cfg,
                        per_pair_ic=per_pair,
                        timeframe=timeframe,
                        label_period=label_period,
                        autocorr=ac1,
                        turnover=turnover,
                    )
                    try:
                        content, usage = llm_utils.request_completion(prompt, llm_cfg_refine)
                        if usage:
                            _accumulate_llm_usage(usage)
                        batch = llm_utils.extract_candidates(content)
                        for item in batch:
                            new_expr = str(item.get("expression") or "").strip()
                            if not new_expr or new_expr == expr_str:
                                continue
                            item["origin"] = "refine"
                            item["refined_from"] = expr_str
                            all_refined.append(item)
                            round_new += 1
                    except Exception as exc:
                        print(f"[refine] failed for {factor.get('name')}: {exc}", file=sys.stderr)
                        continue

                print(f"[refine] round {rnd+1}/{refine_rounds}: {round_new} new variants")

            # Score all refined candidates and merge with originals
            if all_refined:
                print(f"[refine] scoring {len(all_refined)} refined candidates...")
                refined_scored: List[Dict[str, Any]] = []
                for item in all_refined:
                    expr_str = str(item.get("expression") or "").strip()
                    try:
                        abs_ic, per_pair, turnover = _score_expression_across_pairs(
                            expr_str, pairs_data,
                            method=str(args.score_method),
                            min_samples=int(args.min_samples),
                            oos_fraction=oos_frac,
                            label_period=int(label_period),
                        )
                    except Exception:
                        continue
                    complexity = max(1, len(expr_str) // 10)
                    to_penalty = float(getattr(args, "turnover_penalty", 0.0))
                    score = (float(abs_ic)
                             - float(args.complexity_penalty) * float(complexity)
                             - to_penalty * float(turnover))
                    refined_scored.append({
                        "origin": "refine",
                        "expression": expr_str,
                        "metric_abs_ic": float(abs_ic),
                        "per_pair": per_pair,
                        "complexity": complexity,
                        "turnover": float(turnover),
                        "score": float(score),
                        "description": item.get("description"),
                        "category": item.get("category"),
                        "refined_from": item.get("refined_from"),
                    })

                # Merge original scored + refined, re-select diverse top-N
                combined = scored + refined_scored
                combined.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
                print(f"[refine] re-selecting from {len(combined)} (original {len(scored)} + refined {len(refined_scored)})")

                selected = _select_diverse_topn(
                    combined, pairs_data, max(1, int(args.top_n)),
                    corr_threshold=float(args.corr_threshold),
                    ac_penalty=float(args.ac_penalty),
                    ac_threshold=float(args.ac_threshold),
                    label_period=int(label_period),
                )
                n_refined_in = sum(1 for s in selected if s.get("origin") == "refine")
                print(f"[refine] final selection: {len(selected)} factors ({n_refined_in} from refinement)")

                expressions = [
                    {
                        "name": f"f{idx:03d}",
                        "expression": item["expression"],
                        "description": item.get("description") or "refined factor",
                        "category": item.get("category") or "other",
                        "origin": item.get("origin"),
                        "metric_abs_ic": item.get("metric_abs_ic"),
                        "score": item.get("score"),
                        "score_adjusted": item.get("score_adjusted"),
                        "autocorr_1": item.get("autocorr_1"),
                        "complexity": item.get("complexity"),
                        "per_pair": item.get("per_pair"),
                        "refined_from": item.get("refined_from"),
                    }
                    for idx, item in enumerate(selected, start=1)
                ]

    if not args.mine and want_llm:
        llm_cfg = llm_utils.LLMConfig.from_args(args)
        avoid: List[str] = []
        for loop in range(max(1, int(args.llm_loops))):
            prompt = llm_utils.build_prompt(
                feature_cfg=feature_cfg,
                feature_cols=feature_cols,
                combos=combos,
                timeframe=timeframe,
                label_period=label_period,
                request_count=request_count,
                avoid_expressions=avoid,
                feedback=feedback_text,
            )
            try:
                content, usage = llm_utils.request_completion(prompt, llm_cfg)
                batch = llm_utils.extract_candidates(content)
                for item in batch:
                    expr_str = str(item.get("expression") or "").strip()
                    if not expr_str or expr_str in avoid:
                        continue
                    avoid.append(expr_str)
                    expressions.append(item)
                    if len(expressions) >= request_count:
                        break
                if len(expressions) >= request_count:
                    break
                if usage:
                    _accumulate_llm_usage(usage)
                    print(
                        "[llm] usage",
                        json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                    )
                print(f"[llm] loop {loop+1}: total={len(expressions)}")
            except Exception as exc:
                print(f"[llm] failed: {exc}", file=sys.stderr)
                if not args.llm_fallback:
                    raise
                want_llm = False
                break

    if not args.mine and not want_llm:
        expressions = _classic_candidates(feature_cols, request_count)

    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exchange": feature_cfg.get("exchange") or (cfg.get("exchange") or {}).get("name"),
        "pairs": feature_cfg.get("pairs") or (cfg.get("exchange") or {}).get("pair_whitelist") or [],
        "timeframe": timeframe,
        "label_period": label_period,
        "feature_file": str(feature_path),
        "llm_used": bool(args.llm_enabled) and not bool(args.no_llm) and bool(args.llm_api_key or args.llm_model),
        "llm_usage": {
            key: (int(value) if key.endswith("tokens") else round(float(value), 6))
            for key, value in llm_usage_totals.items()
        },
        "mine_enabled": bool(args.mine),
        "score_method": str(args.score_method) if args.mine else None,
        "top_n": int(args.top_n) if args.mine else None,
        "expressions": expressions,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[expression_agent] wrote: {output_path} (count={len(expressions)})")

    # ── Save results to factor memory ──
    try:
        from agent_market.factor_memory import (  # noqa: WPS433
            build_factor_memory_artifacts_from_expression_output,
            merge_factor_memory_artifacts,
        )
        scored_path = output_path.with_name(output_path.stem + "_scored_all.json")
        mem = build_factor_memory_artifacts_from_expression_output(
            run_id=datetime.now().strftime("%Y%m%d-%H%M%S"),
            memory_dir=local_memory_dir.resolve(),
            expressions_path=output_path.resolve(),
            scored_candidates_path=scored_path.resolve() if scored_path.exists() else None,
        )
        print(f"[memory] saved local factor memory: {mem.factor_memory_json}")

        # Merge into global factor memory
        try:
            global_mem = merge_factor_memory_artifacts(
                source_memory_path=Path(mem.factor_memory_json),
                target_memory_dir=_paths.global_factor_memory_dir(),
            )
            gstore = FactorMemoryStore(Path(global_mem.factor_memory_json))
            print(f"[memory] merged into global: {len(gstore.factor_cards)} factor cards, "
                  f"{len(gstore.failure_cards)} failure cards")
        except Exception as exc:
            print(f"[memory] global merge failed (non-fatal): {exc}", file=sys.stderr)

    except Exception as exc:
        print(f"[memory] factor memory save failed (non-fatal): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
