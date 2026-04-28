#!/usr/bin/env python3
"""方案 D: Permutation Importance 筛选

在 train_window(2026-01-01) 已训好的 LGB 模型上，对每个因子：
1. 读 feature matrix (训练期 + valid holdout)
2. 找 valid holdout 段（训练的 20% 最后一段）
3. shuffle 该因子的值
4. 看 model accuracy / log-loss 下降多少
5. 按 importance 排序取 top-K

输出新的 expressions JSON，只含 top-K 因子。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agent_market.factor_lab.mining import build_big  # noqa: E402
from agent_market.factor_lab.paths import DEFAULT_PAIRS  # noqa: E402
from agent_market.freqai.expression_engine import load_expression_file, apply_expressions  # noqa: E402
from agent_market.freqai.features import apply_configured_features  # noqa: E402
from agent_market.freqai.training.labels import future_classify_3way  # noqa: E402

import lightgbm as lgb


def _build_training_data(expr_file: Path, feature_cfg_path: Path,
                          train_start: str, train_end: str,
                          label_period: int = 12,
                          class_threshold: float = 0.005):
    """Replicate pipeline.py's train data assembly but return X, y, dates per pair."""
    feat_cfg = json.loads(feature_cfg_path.read_text(encoding="utf-8-sig"))
    expr_specs = load_expression_file(expr_file)

    t_s = pd.Timestamp(train_start, tz="UTC")
    t_e = pd.Timestamp(train_end, tz="UTC")

    big_X_list = []
    big_y_list = []
    feature_names: list = []

    for pair in DEFAULT_PAIRS:
        f = ROOT / "user_data" / "data" / "kucoin" / f"{pair.replace('/', '_')}-1h.feather"
        if not f.exists(): continue
        df = pd.read_feather(f)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        df = apply_configured_features(df, feat_cfg)
        df, _ = apply_expressions(df, expr_specs, on_error="raise")

        # Filter to train window
        mask = (df["date"] >= t_s) & (df["date"] < t_e)
        df = df.loc[mask].reset_index(drop=True)
        if df.empty: continue

        labels = future_classify_3way(df["close"], label_period, class_threshold)
        exclude = {"date", "open", "high", "low", "close", "volume"}
        cols = [c for c in df.columns if c not in exclude]
        if not feature_names:
            feature_names = list(cols)

        # Drop last `label_period` rows (NaN labels)
        X = df[feature_names].iloc[:-label_period]
        y = labels.iloc[:-label_period]
        X = X.replace([np.inf, -np.inf], np.nan)
        mask2 = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask2]; y = y.loc[mask2]
        big_X_list.append(X)
        big_y_list.append(y)

    if not big_X_list:
        raise RuntimeError("no training data collected")
    X = pd.concat(big_X_list, axis=0, ignore_index=True)
    y = pd.concat(big_y_list, axis=0, ignore_index=True)
    # Validation split — last 20% as holdout for permutation
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_vl = X.iloc[:split], X.iloc[split:]
    y_tr, y_vl = y.iloc[:split], y.iloc[split:]
    return X_tr, y_tr, X_vl, y_vl, feature_names


def _train_lgb(X_tr, y_tr, X_vl, y_vl):
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "n_estimators": 200, "num_boost_round": 200, "learning_rate": 0.02,
        "num_leaves": 15, "min_child_samples": 150, "subsample": 0.6,
        "colsample_bytree": 0.4, "reg_alpha": 3.0, "reg_lambda": 3.0,
        "max_depth": 4, "verbosity": -1, "random_state": 42,
    }
    ds_tr = lgb.Dataset(X_tr.values, label=y_tr.values)
    ds_vl = lgb.Dataset(X_vl.values, label=y_vl.values, reference=ds_tr)
    booster = lgb.train(params, ds_tr, num_boost_round=200, valid_sets=[ds_vl])
    return booster


def _logloss(y_true: np.ndarray, proba: np.ndarray) -> float:
    proba = np.clip(proba, 1e-9, 1 - 1e-9)
    idx = np.asarray(y_true, dtype=int)
    n = len(idx)
    return float(-np.log(proba[np.arange(n), idx]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expressions", required=True, help="input expressions JSON")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--train-start", default="2025-07-01")
    ap.add_argument("--train-end",   default="2026-01-01")
    ap.add_argument("--feature-cfg", default="user_data/freqai_features_real.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-repeats", type=int, default=3)
    args = ap.parse_args()

    expr_path = Path(args.expressions)
    feat_path = ROOT / args.feature_cfg
    d = json.loads(expr_path.read_text(encoding="utf-8-sig"))
    all_exprs = d["expressions"]
    expr_names = [e["name"] for e in all_exprs]
    print(f"[perm-imp] input: {len(all_exprs)} factors from {expr_path.name}")
    print(f"[perm-imp] building training data...")
    X_tr, y_tr, X_vl, y_vl, feat_cols = _build_training_data(
        expr_path, feat_path, args.train_start, args.train_end)
    print(f"  X_tr={X_tr.shape} X_vl={X_vl.shape}  features={len(feat_cols)}")

    print(f"[perm-imp] training base LGB model...")
    booster = _train_lgb(X_tr, y_tr, X_vl, y_vl)
    base_proba = booster.predict(X_vl.values)
    base_logloss = _logloss(y_vl.values, base_proba)
    print(f"  base valid logloss: {base_logloss:.4f}")

    # Map expr name → column idx
    expr_col_idx = {name: feat_cols.index(name) for name in expr_names if name in feat_cols}
    print(f"[perm-imp] {len(expr_col_idx)}/{len(expr_names)} expr factors present in feature matrix")

    # Permutation on valid holdout
    rng = np.random.default_rng(args.seed)
    importances = {}
    for i, name in enumerate(expr_names):
        if name not in expr_col_idx: continue
        col = expr_col_idx[name]
        deltas = []
        for _ in range(args.n_repeats):
            Xp = X_vl.values.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, col] = Xp[perm, col]
            p = booster.predict(Xp)
            deltas.append(_logloss(y_vl.values, p) - base_logloss)
        importances[name] = float(np.mean(deltas))
        if (i + 1) % 10 == 0 or (i + 1) == len(expr_names):
            print(f"  scored {i+1}/{len(expr_names)}")

    # Rank — positive delta = factor helps (shuffling hurts logloss)
    ranked = sorted(importances.items(), key=lambda x: -x[1])

    print(f"\n[perm-imp] top-{args.top_k} factors by permutation importance:")
    print(f'{"rank":>4} {"name":<12} {"delta_logloss":>14}')
    for rk, (nm, v) in enumerate(ranked[:args.top_k]):
        print(f'{rk+1:>4} {nm:<12} {v:>14.5f}')
    print(f"\n[perm-imp] bottom 5 (least important — will be dropped):")
    for nm, v in ranked[-5:]:
        print(f'    {nm:<12} {v:>14.5f}')

    # Write filtered JSON
    keep = set(n for n, _ in ranked[:args.top_k])
    kept_exprs = [e for e in all_exprs if e["name"] in keep]
    out = Path(args.out)
    out.write_text(json.dumps({
        "version": f"perm_imp_top{args.top_k}",
        "source_file": str(expr_path),
        "selection": {"method": "permutation_importance",
                      "train_timerange": f"{args.train_start}..{args.train_end}",
                      "n_repeats": args.n_repeats, "seed": args.seed},
        "expressions": kept_exprs,
    }, indent=2), encoding="utf-8")
    print(f"\n[perm-imp] wrote {len(kept_exprs)} factors → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
