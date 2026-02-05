#!/usr/bin/env python
from __future__ import annotations

import ast
import argparse
import json
import random
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

from agent_market import paths  # noqa: E402


def _resolve_path(path: str) -> Path:
    return paths.resolve_repo_path(path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_timeframe(cfg: Dict[str, Any], cli_timeframe: Optional[str]) -> str:
    if cli_timeframe:
        return cli_timeframe
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    tfs = fp.get("include_timeframes") or []
    if isinstance(tfs, list) and tfs and isinstance(tfs[0], str) and tfs[0]:
        return tfs[0]
    tf = cfg.get("timeframe")
    return str(tf) if tf else "1h"


def _resolve_label_period(cfg: Dict[str, Any], feature_cfg: Dict[str, Any]) -> int:
    if feature_cfg.get("label_period") is not None:
        return int(feature_cfg["label_period"])
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    return int(fp.get("label_period_candles") or 12)


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
    p = _resolve_path(path)
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


_WINDOW_CHOICES = (3, 6, 12, 24, 48, 72)
_SHIFT_CHOICES = (1, 2, 3)
_CONST_CHOICES = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)


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
) -> Tuple[float, Dict[str, float]]:
    import warnings

    metrics: Dict[str, float] = {}
    values: List[float] = []
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
            if method == "correlation":
                metric = float(aligned["x"].corr(aligned["y"]))
            elif method == "spearman":
                metric = float(aligned["x"].corr(aligned["y"], method="spearman"))
            else:
                raise ValueError(f"Unsupported score_method: {method}")
        if not np.isfinite(metric):
            continue
        metrics[pd_item.pair] = metric
        values.append(metric)
    if not values:
        raise ValueError("not enough samples")
    return float(np.mean(np.abs(values))), metrics


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
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cache: Dict[str, Tuple[float, Dict[str, float], int]] = {}

    def fitness(node: _GPNode) -> Tuple[float, Dict[str, float], int, str]:
        expr_str = _gp_to_string(node)
        if expr_str in cache:
            s, m, size = cache[expr_str]
            return s, m, size, expr_str
        try:
            abs_ic, per_pair = _score_expression_across_pairs(
                expr_str, pairs, method=method, min_samples=min_samples
            )
        except Exception:
            abs_ic, per_pair = 0.0, {}
        size = _gp_size(node)
        score = float(abs_ic) - float(complexity_penalty) * float(size)
        cache[expr_str] = (score, per_pair, size)
        return score, per_pair, size, expr_str

    pop: List[_GPNode] = [_gp_random_expr(rng, feature_cols, max_depth) for _ in range(population)]
    for _gen in range(max(1, generations)):
        scored = [fitness(node) + (node,) for node in pop]
        scored.sort(key=lambda t: float(t[0]), reverse=True)
        elites = [t[-1] for t in scored[: max(1, min(elitism, population))]]
        next_pop: List[_GPNode] = list(elites)
        while len(next_pop) < population:
            parent = scored[rng.randrange(0, max(1, population // 2))][-1]
            child = parent
            if rng.random() < crossover_rate:
                other = scored[rng.randrange(0, max(1, population // 2))][-1]
                child = _gp_crossover(rng, parent, other, max_depth)
            if rng.random() < mutation_rate:
                child = _gp_mutate(rng, child, feature_cols, max_depth)
            next_pop.append(child)
        pop = next_pop

    final = [fitness(node) + (node,) for node in pop]
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
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--llm-count", type=int, default=20)
    parser.add_argument("--llm-loops", type=int, default=1)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=45)

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

    parser.add_argument("--evolve", action="store_true", help="Enable evolutionary search when --mine is enabled.")
    parser.add_argument("--no-evolve", action="store_true", help="Disable evolutionary search when --mine is enabled.")
    parser.add_argument("--evolve-population", type=int, default=60)
    parser.add_argument("--evolve-generations", type=int, default=8)
    parser.add_argument("--evolve-max-depth", type=int, default=4)
    parser.add_argument("--evolve-mutation-rate", type=float, default=0.45)
    parser.add_argument("--evolve-crossover-rate", type=float, default=0.65)
    parser.add_argument("--evolve-elitism", type=int, default=6)
    parser.add_argument("--evolve-seed", type=int, default=7)
    args = parser.parse_args(argv)

    config_path = _resolve_path(args.config)
    cfg = _load_json(config_path)
    timeframe = _resolve_timeframe(cfg, args.timeframe)

    feature_path = _resolve_path(args.feature_file) if args.feature_file else _infer_feature_file(config_path)
    feature_cfg = _load_json(feature_path)
    label_period = _resolve_label_period(cfg, feature_cfg)

    features = feature_cfg.get("features") or []
    combos = feature_cfg.get("feature_combos") or []
    feature_cols: List[str] = []
    for item in features:
        if isinstance(item, dict) and item.get("name"):
            feature_cols.append(str(item["name"]))
    for item in combos:
        if isinstance(item, dict) and item.get("name"):
            feature_cols.append(str(item["name"]))

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    want_llm = bool(args.llm_enabled) and not bool(args.no_llm)
    request_count = max(1, int(args.llm_count))

    feedback_text = _read_feedback(args.feedback)
    expressions: List[Dict[str, Any]] = []

    if args.mine:
        settings = FreqAISettings.from_file(config_path, timeframe_override=timeframe)
        if args.pairs:
            settings = FreqAISettings.from_file(config_path, timeframe_override=timeframe, label_override=None)
            settings.pairs = [str(p) for p in args.pairs if str(p).strip()]
        settings.validate_dataset(settings.timeframe)

        pairs_data: List[_PairData] = []
        for pair in settings.pairs:
            sanitized = pair.replace("/", "_")
            data_path = settings.data_dir / f"{sanitized}-{settings.timeframe}.feather"
            df = pd.read_feather(data_path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)
            df = apply_configured_features(df, feature_cfg)
            label = (df["close"].shift(-int(label_period)) / df["close"]) - 1
            pairs_data.append(_PairData(pair=pair, df=df, label=label))

        candidate_pool = max(int(args.top_n) * 10, request_count, 50)
        classic_pool = min(candidate_pool, max(int(args.top_n) * 5, 50))
        expressions.extend(_classic_candidates(feature_cols, classic_pool))

        if want_llm:
            llm_cfg = llm_utils.LLMConfig.from_args(args)
            avoid: List[str] = [str(x.get("expression") or "").strip() for x in expressions]
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
                        expressions.append({"origin": "llm", **item})
                        if len(expressions) >= candidate_pool:
                            break
                    if usage:
                        print(
                            "[llm] usage",
                            json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                        )
                    print(f"[llm] loop {loop+1}: total={len(expressions)}")
                    if len(expressions) >= candidate_pool:
                        break
                except Exception as exc:
                    print(f"[llm] failed: {exc}", file=sys.stderr)
                    if not args.llm_fallback:
                        raise
                    want_llm = False
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
        for expr_str, item in dedup.items():
            try:
                abs_ic, per_pair = _score_expression_across_pairs(
                    expr_str,
                    pairs_data,
                    method=str(args.score_method),
                    min_samples=int(args.min_samples),
                )
            except Exception:
                continue
            complexity = int(item.get("complexity") or max(1, len(expr_str) // 10))
            score = float(abs_ic) - float(args.complexity_penalty) * float(complexity)
            scored.append(
                {
                    "origin": item.get("origin", "classic"),
                    "expression": expr_str,
                    "metric_abs_ic": float(abs_ic),
                    "per_pair": per_pair,
                    "complexity": int(complexity),
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
        selected = scored[: max(1, int(args.top_n))]
        expressions = [
            {
                "name": f"f{idx:03d}",
                "expression": item["expression"],
                "description": item.get("description") or "mined factor",
                "category": item.get("category") or "other",
                "origin": item.get("origin"),
                "metric_abs_ic": item.get("metric_abs_ic"),
                "score": item.get("score"),
                "complexity": item.get("complexity"),
                "per_pair": item.get("per_pair"),
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
        "mine_enabled": bool(args.mine),
        "score_method": str(args.score_method) if args.mine else None,
        "top_n": int(args.top_n) if args.mine else None,
        "expressions": expressions,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[expression_agent] wrote: {output_path} (count={len(expressions)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
