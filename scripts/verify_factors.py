#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_market.freqai.expression_engine import ExpressionSpec, apply_expressions, load_expression_file
from agent_market.freqai.features import apply_configured_features


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if x.size != y.size or x.size < 3:
        return float("nan")
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x) * np.nansum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.nansum(x * y) / denom)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    return _pearsonr(
        x.rank(method="average").to_numpy(dtype=np.float64),
        y.rank(method="average").to_numpy(dtype=np.float64),
    )


def _quantile_spread(x: pd.Series, y: pd.Series, *, q: int = 5) -> Optional[float]:
    if x.nunique(dropna=True) < q:
        return None
    try:
        buckets = pd.qcut(x, q=q, labels=False, duplicates="drop")
    except Exception:
        return None
    if buckets is None:
        return None
    df = pd.DataFrame({"bucket": buckets, "label": y}).dropna()
    if df.empty:
        return None
    means = df.groupby("bucket")["label"].mean()
    if means.empty:
        return None
    lo = means.iloc[0]
    hi = means.iloc[-1]
    return float(hi - lo)


@dataclass(slots=True)
class PairMetrics:
    n_total: int
    n_valid: int
    ic: float
    rank_ic: float
    spread_q5: Optional[float]


def _compute_pair_metrics(df: pd.DataFrame, factor: str, label_col: str) -> PairMetrics:
    subset = df[[factor, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    n_total = int(df[label_col].notna().sum())
    n_valid = int(len(subset))
    if n_valid < 3:
        return PairMetrics(n_total=n_total, n_valid=n_valid, ic=float("nan"), rank_ic=float("nan"), spread_q5=None)
    x = subset[factor]
    y = subset[label_col]
    ic = _pearsonr(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
    ric = _rank_ic(x, y)
    spread = _quantile_spread(x, y, q=5)
    return PairMetrics(n_total=n_total, n_valid=n_valid, ic=ic, rank_ic=ric, spread_q5=spread)


def _permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> Optional[float]:
    if permutations <= 0 or x.size != y.size or x.size < 10:
        return None
    rng = np.random.default_rng(seed)
    obs = abs(_pearsonr(x, y))
    if not np.isfinite(obs):
        return None
    count = 1
    total = permutations + 1
    y_copy = y.copy()
    for _ in range(permutations):
        rng.shuffle(y_copy)
        stat = abs(_pearsonr(x, y_copy))
        if np.isfinite(stat) and stat >= obs:
            count += 1
    return float(count / total)


def _resolve_data_paths(
    *,
    data_dir: Path,
    exchange: str,
    pairs: List[str],
    timeframe: str,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for pair in pairs:
        sanitized = pair.replace("/", "_")
        p = data_dir / exchange / f"{sanitized}-{timeframe}.feather"
        paths[pair] = p
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate mined factors out-of-sample (IC/RankIC/spread/p-value).")
    parser.add_argument("--config", default="user_data/config_freqai_kucoin.json", help="Freqtrade config JSON")
    parser.add_argument("--feature-file", default="user_data/freqai_features_real.json", help="Feature config JSON")
    parser.add_argument("--expressions-file", default="user_data/freqai_expressions_selected.json", help="Expressions JSON")
    parser.add_argument("--report-dir", default="user_data/reports", help="Directory to write reports")
    parser.add_argument("--test-days", type=int, default=None, help="Holdout days (default: freqai.backtest_period_days)")
    parser.add_argument("--label-period", type=int, default=None, help="Label period candles (default: from expressions file)")
    parser.add_argument("--permutations", type=int, default=50, help="Permutation count for p-value (0 to disable)")
    parser.add_argument("--seed", type=int, default=7, help="Seed for permutation test")
    parser.add_argument("--top", type=int, default=15, help="How many top factors to print")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    feature_path = (ROOT / args.feature_file).resolve() if not Path(args.feature_file).is_absolute() else Path(args.feature_file)
    expr_path = (ROOT / args.expressions_file).resolve() if not Path(args.expressions_file).is_absolute() else Path(args.expressions_file)

    cfg = _read_json(cfg_path)
    feature_cfg = _read_json(feature_path)
    expr_payload = _read_json(expr_path)

    exchange = str(expr_payload.get("exchange") or cfg.get("exchange", {}).get("name") or "unknown")
    timeframe = str(expr_payload.get("timeframe") or cfg.get("timeframe") or "1h")
    pairs = list(expr_payload.get("pairs") or cfg.get("exchange", {}).get("pair_whitelist") or [])
    if not pairs:
        raise SystemExit("No pairs found in expressions file or config.")

    label_period = args.label_period
    if label_period is None:
        label_period = int(expr_payload.get("label_period") or feature_cfg.get("label_period") or 12)
    test_days = args.test_days
    if test_days is None:
        test_days = int(cfg.get("freqai", {}).get("backtest_period_days") or 15)

    data_dir_raw = cfg.get("datadir") or "user_data/data"
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    expressions: List[ExpressionSpec] = load_expression_file(expr_path)
    factor_names = [s.name for s in expressions]
    if not factor_names:
        raise SystemExit("No expressions found in expressions file.")

    data_paths = _resolve_data_paths(data_dir=data_dir, exchange=exchange, pairs=pairs, timeframe=timeframe)
    missing = [pair for pair, p in data_paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing data files for pairs: {', '.join(missing)} (dir={data_dir/exchange})")

    cutoff_per_pair: Dict[str, str] = {}
    per_pair_rows: Dict[str, Dict[str, int]] = {}
    per_factor: Dict[str, Any] = {}

    # We'll also build aggregated test vectors for permutation p-values.
    test_vectors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pair, path in data_paths.items():
        df = pd.read_feather(path)
        if "date" not in df.columns:
            raise SystemExit(f"Missing 'date' column in {path}")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = apply_configured_features(df, feature_cfg)
        df, _added = apply_expressions(df, expressions, on_error="raise")

        label = (df["close"].shift(-label_period) / df["close"]) - 1.0
        df["__label__"] = label.astype(float).replace([np.inf, -np.inf], np.nan)

        cutoff = df["date"].max() - pd.Timedelta(days=int(test_days))
        cutoff_per_pair[pair] = cutoff.isoformat()
        per_pair_rows[pair] = {
            "rows": int(len(df)),
            "rows_with_label": int(df["__label__"].notna().sum()),
        }
        train_df = df[df["date"] < cutoff]
        test_df = df[df["date"] >= cutoff]

        for name in factor_names:
            per_factor.setdefault(name, {"expression": next((s.expression for s in expressions if s.name == name), "")})
            train_m = _compute_pair_metrics(train_df, name, "__label__")
            test_m = _compute_pair_metrics(test_df, name, "__label__")
            per_factor[name].setdefault("per_pair", {})[pair] = {
                "train": asdict(train_m),
                "test": asdict(test_m),
            }

            # Build aggregated test vectors for permutation p-values.
            subset = test_df[[name, "__label__"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) >= 10:
                x = subset[name].to_numpy(dtype=np.float64)
                y = subset["__label__"].to_numpy(dtype=np.float64)
                prev = test_vectors.get(name)
                if prev is None:
                    test_vectors[name] = (x, y)
                else:
                    test_vectors[name] = (np.concatenate([prev[0], x]), np.concatenate([prev[1], y]))

    # Aggregate across pairs and compute p-values
    ranking: List[Tuple[str, float]] = []
    for name in factor_names:
        entry = per_factor[name]
        per_pair = entry.get("per_pair") or {}

        test_ics = []
        test_ns = []
        train_ics = []
        train_ns = []
        for pair in pairs:
            metrics = per_pair.get(pair) or {}
            tr = metrics.get("train", {})
            te = metrics.get("test", {})
            if np.isfinite(tr.get("ic", float("nan"))):
                train_ics.append(float(tr["ic"]))
                train_ns.append(int(tr.get("n_valid") or 0))
            if np.isfinite(te.get("ic", float("nan"))):
                test_ics.append(float(te["ic"]))
                test_ns.append(int(te.get("n_valid") or 0))

        def _wavg(vals: List[float], ns: List[int]) -> float:
            total = sum(max(0, int(n)) for n in ns)
            if total <= 0:
                return float("nan")
            return float(sum(v * max(0, int(n)) for v, n in zip(vals, ns)) / total)

        train_ic = _wavg(train_ics, train_ns)
        test_ic = _wavg(test_ics, test_ns)

        x, y = test_vectors.get(name, (np.array([]), np.array([])))
        p = _permutation_p_value(x, y, permutations=int(args.permutations), seed=int(args.seed)) if args.permutations else None

        entry["aggregate"] = {
            "train_ic": train_ic,
            "test_ic": test_ic,
            "abs_test_ic": float(abs(test_ic)) if np.isfinite(test_ic) else float("nan"),
            "test_p_value": p,
            "test_samples": int(x.size),
        }
        ranking.append((name, entry["aggregate"]["abs_test_ic"]))

    ranking.sort(key=lambda t: (-(t[1] if np.isfinite(t[1]) else -1.0), t[0]))

    report = {
        "generated_at": _utcnow(),
        "config": str(cfg_path),
        "feature_file": str(feature_path),
        "expressions_file": str(expr_path),
        "exchange": exchange,
        "timeframe": timeframe,
        "pairs": pairs,
        "label_period": int(label_period),
        "test_days": int(test_days),
        "cutoff_per_pair": cutoff_per_pair,
        "per_pair_rows": per_pair_rows,
        "factors": per_factor,
        "ranking": [{"name": name, "abs_test_ic": abs_ic, **per_factor[name]["aggregate"]} for name, abs_ic in ranking],
    }

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = (ROOT / report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"factor_validation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    top = min(int(args.top), len(ranking))
    print(f"[verify_factors] wrote: {out_path}")
    print(f"[verify_factors] exchange={exchange} timeframe={timeframe} pairs={len(pairs)} label_period={label_period} test_days={test_days}")
    print("[verify_factors] TOP factors by |test IC|:")
    for name, _abs_ic in ranking[:top]:
        agg = per_factor[name]["aggregate"]
        print(
            f"  - {name}: test_ic={agg['test_ic']:.4f} |abs|={agg['abs_test_ic']:.4f} p={agg['test_p_value']} n={agg['test_samples']}"
        )


if __name__ == "__main__":
    main()
