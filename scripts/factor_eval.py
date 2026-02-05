#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))


def _make_synth_df(rows: int = 256, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rets = rng.normal(loc=0.0, scale=0.01, size=int(rows))
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close.astype(float)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a compiled factor expression (minimal offline mode)")
    parser.add_argument("--expression", default=None, help="ExpressionEngine expression string")
    parser.add_argument("--expression-path", default=None, help="Path to expression text file")
    parser.add_argument(
        "--features-parquet",
        default=None,
        help="Optional feature parquet to evaluate on (default: synthetic close series)",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.expression and not args.expression_path:
        raise SystemExit("Must provide --expression or --expression-path")

    expr = args.expression
    if not expr and args.expression_path:
        expr = Path(args.expression_path).read_text(encoding="utf-8").strip()
    expr = str(expr or "").strip()
    if not expr:
        raise SystemExit("Empty expression")

    from agent_market.factor_compiler.scoring import score_factors_to_artifacts
    from agent_market.freqai.expression_engine import safe_eval_expression

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.features_parquet:
        feat_path = Path(args.features_parquet).expanduser().resolve()
        if not feat_path.exists():
            raise SystemExit(f"features parquet not found: {feat_path}")
        df = pd.read_parquet(feat_path)
        if df.shape[0] > int(args.rows):
            df = df.head(int(args.rows))
        price_col = None
        for cand in ("close", "mid"):
            if cand in df.columns:
                price_col = cand
                break
        if not price_col:
            raise SystemExit("features parquet must contain 'close' or 'mid' column for target proxy")
        df["y"] = pd.Series(df[price_col]).astype("float64").pct_change().shift(-1)
    else:
        df = _make_synth_df(rows=int(args.rows), seed=int(args.seed))
        df["y"] = df["close"].pct_change().shift(-1)
    df["factor"] = safe_eval_expression(expr, df)

    from agent_market.factor_compiler.checks import (
        check_label_leakage_signature,
        check_permutation_leakage,
        check_shift_test,
    )

    leakage_checks = [
        check_permutation_leakage(df["factor"], df["y"]).to_dict(),
        check_shift_test(df["factor"], df["y"]).to_dict(),
        check_label_leakage_signature(df["factor"], df["y"]).to_dict(),
    ]

    values_path = out_dir / "factor_values.parquet"
    try:
        df.to_parquet(values_path, index=False)
    except Exception:
        values_path = None  # type: ignore[assignment]

    report = score_factors_to_artifacts(
        df,
        factor_cols=["factor"],
        target_col="y",
        out_dir=out_dir,
        max_nan_ratio=0.5,
        max_turnover=1e9,
        factor_exprs={"factor": expr},
    )

    meta_path = out_dir / "factor_eval_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "version": 1,
                "expression": expr,
                "rows": int(args.rows),
                "seed": int(args.seed),
                "leakage_checks": leakage_checks,
                "artifacts": {
                    **(report.get("artifacts") or {}),
                    "factor_values_parquet": str(values_path) if values_path is not None else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[factor_eval] wrote: {meta_path}")
    print(f"[factor_eval] wrote: {report['artifacts']['factor_scores_json']}")
    print(f"[factor_eval] wrote: {report['artifacts']['pareto_csv']}")


if __name__ == "__main__":
    main()
