from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import math

import pandas as pd
import numpy as np

from agent_market.config import FreqAISettings
from agent_market.freqai.analysis import (
    build_feature_dataframe,
    compute_information_coefficient,
    evaluate_expressions,
    factor_correlation_matrix,
    factor_quantile_stats,
)
from agent_market.freqai.context import FeatureContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute information coefficients (IC) for FreqAI features and expressions."
    )
    parser.add_argument("--config", required=True, help="Path to freqtrade config JSON.")
    parser.add_argument("--feature-file", default="user_data/freqai_features.json")
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--label-period", type=int, default=None)
    parser.add_argument("--output", default="user_data/reports/factor_ic.json")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional pair override.")
    parser.add_argument("--update-expression-file", action="store_true", help="Write IC metrics back into expression file.")
    parser.add_argument("--quantiles", type=int, default=5, help="Quantile buckets for return analysis.")
    parser.add_argument("--corr-output", default=None, help="Optional path to write factor correlation CSV.")
    return parser.parse_args()


def load_dataframe(settings: FreqAISettings, pair: str) -> pd.DataFrame:
    sanitized = pair.replace("/", "_")
    file_path = settings.data_dir / f"{sanitized}-{settings.timeframe}.feather"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_feather(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    feature_file = Path(args.feature_file)
    expression_file = Path(args.expression_file) if args.expression_file else None
    output_path = Path(args.output)

    settings = FreqAISettings.from_file(
        config_path,
        timeframe_override=args.timeframe,
        label_override=args.label_period,
    )

    pairs = args.pairs or settings.pairs
    label_period = args.label_period or settings.label_period

    feature_cfg = json.loads(feature_file.read_text(encoding="utf-8"))
    expression_cfg: Optional[Dict] = None
    if expression_file and expression_file.exists():
        expression_cfg = json.loads(expression_file.read_text(encoding="utf-8"))

    base_context = FeatureContext(feature_file=feature_file, timeframe=settings.timeframe, settings=settings)

    feature_results: Dict[str, Dict] = {}
    expression_results: Dict[str, Dict] = {}
    corr_frames: List[pd.DataFrame] = []

    for pair in pairs:
        df_raw = load_dataframe(settings, pair)
        df_features = build_feature_dataframe(
            df_raw.copy(),
            feature_cfg,
            context=base_context,
            include_corr_pairs=True,
            base_pair=pair,
        )
        forward_returns = (
            df_features["close"].shift(-label_period) / (df_features["close"] + 1e-9) - 1.0
        )

        factor_names = [
            col for col in df_features.columns if col not in {"open", "high", "low", "close", "volume", "date"}
        ]
        if factor_names:
            corr_frames.append(df_features[factor_names].replace([np.inf, -np.inf], np.nan))
        ics = compute_information_coefficient(df_features, factor_names, forward_returns)
        for ic in ics:
            entry = feature_results.setdefault(
                ic.name,
                {
                    "pair_metrics": [],
                    "quantile_stats": [],
                    "ic_pearson": 0.0,
                    "ic_spearman": 0.0,
                    "samples": 0,
                    "mean": 0.0,
                    "std": 0.0,
                },
            )
            entry["pair_metrics"].append(
                {
                    "pair": pair,
                    "ic_pearson": ic.ic_pearson,
                    "ic_spearman": ic.ic_spearman,
                    "samples": ic.samples,
                    "mean": ic.mean,
                    "std": ic.std,
                    "positive_ratio": ic.positive_ratio,
                    "ic_std": ic.ic_std,
                    "ic_ir": ic.ic_ir,
                }
            )
            qs = factor_quantile_stats(df_features[ic.name], forward_returns, args.quantiles)
            if qs:
                entry["quantile_stats"].append({"pair": pair, **qs})

        if expression_cfg:
            expr_map = evaluate_expressions(df_features, expression_cfg.get("expressions", []))
            if expr_map:
                expr_df = df_features.assign(**expr_map)
                expr_ics = compute_information_coefficient(
                    expr_df, expr_map.keys(), forward_returns
                )
                for ic in expr_ics:
                    entry = expression_results.setdefault(
                        ic.name,
                        {
                            "expression": _lookup_expression(expression_cfg, ic.name),
                            "pair_metrics": [],
                            "quantile_stats": [],
                        },
                    )
                    entry["pair_metrics"].append(
                        {
                            "pair": pair,
                            "ic_pearson": ic.ic_pearson,
                            "ic_spearman": ic.ic_spearman,
                            "samples": ic.samples,
                            "mean": ic.mean,
                            "std": ic.std,
                            "positive_ratio": ic.positive_ratio,
                            "ic_std": ic.ic_std,
                            "ic_ir": ic.ic_ir,
                        }
                    )
                    qs = factor_quantile_stats(expr_df[ic.name], forward_returns, args.quantiles)
                    if qs:
                        entry["quantile_stats"].append({"pair": pair, **qs})

    if args.corr_output and corr_frames:
        combined = pd.concat(corr_frames, axis=0, ignore_index=True).dropna(how="any")
        if not combined.empty:
            corr_matrix = factor_correlation_matrix(combined, combined.columns)
            if not corr_matrix.empty:
                corr_path = Path(args.corr_output)
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                corr_matrix.to_csv(corr_path, index=True)
                print(f"Correlation matrix written to {corr_path}")

    summary = {
        "config": str(config_path),
        "feature_file": str(feature_file),
        "expression_file": str(expression_file) if expression_file else None,
        "label_period": label_period,
        "timeframe": settings.timeframe,
        "pairs": pairs,
        "features": _aggregate_metrics(feature_results),
        "expressions": _aggregate_metrics(expression_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"IC summary written to {output_path}")

    if args.update_expression_file and expression_file and expression_cfg:
        _write_back_expression_ic(expression_file, expression_cfg, summary["expressions"])


def _weighted_average(items: List[Dict], value_key: str) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for item in items:
        if value_key not in item:
            continue
        value = item.get(value_key)
        weight = float(item.get("samples") or 0)
        if value is None or weight <= 0:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total_value / total_weight


def _aggregate_quantile_stats(stats: List[Dict]) -> Optional[Dict[str, float]]:
    if not stats:
        return None
    fields = ("top_mean", "bottom_mean", "spread", "t_stat")
    totals = {field: 0.0 for field in fields}
    total_weight = 0.0
    for item in stats:
        samples = float(item.get("samples") or 0)
        if samples <= 0:
            continue
        total_weight += samples
        for field in fields:
            value = item.get(field)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            totals[field] += value * samples
    if total_weight <= 0:
        return None
    return {field: totals[field] / total_weight for field in fields}


def _aggregate_metrics(entries: Dict[str, Dict]) -> List[Dict]:
    aggregated: List[Dict] = []
    for name, entry in entries.items():
        samples = sum(item["samples"] for item in entry["pair_metrics"])
        if samples <= 0:
            continue
        weighted_pearson = _weighted_average(entry["pair_metrics"], "ic_pearson") or 0.0
        weighted_spearman = _weighted_average(entry["pair_metrics"], "ic_spearman") or 0.0
        weighted_positive = _weighted_average(entry["pair_metrics"], "positive_ratio")
        weighted_ic_std = _weighted_average(entry["pair_metrics"], "ic_std")
        weighted_ic_ir = _weighted_average(entry["pair_metrics"], "ic_ir")
        aggregated.append(
            {
                "name": name,
                "ic_pearson": weighted_pearson,
                "ic_spearman": weighted_spearman,
                "samples": samples,
                "pair_metrics": entry["pair_metrics"],
                "expression": entry.get("expression"),
                "positive_ratio": weighted_positive,
                "ic_std": weighted_ic_std,
                "ic_ir": weighted_ic_ir,
                "quantile_stats": entry.get("quantile_stats", []),
                "quantile_summary": _aggregate_quantile_stats(entry.get("quantile_stats", [])),
            }
        )
    aggregated.sort(key=lambda item: abs(item["ic_pearson"]), reverse=True)
    return aggregated


def _lookup_expression(expr_cfg: Dict, name: str) -> Optional[Dict]:
    for item in expr_cfg.get("expressions", []):
        item_name = item.get("name") or ""
        if item_name == name:
            return item
    return None


def _write_back_expression_ic(expression_file: Path, expression_cfg: Dict, summary: List[Dict]) -> None:
    metrics_map = {item["name"]: item for item in summary}
    updated: List[Dict] = []
    for expr in expression_cfg.get("expressions", []):
        name = expr.get("name")
        if name and name in metrics_map:
            expr = dict(expr)
            expr["ic_pearson"] = metrics_map[name]["ic_pearson"]
            expr["ic_spearman"] = metrics_map[name]["ic_spearman"]
            expr["ic_samples"] = metrics_map[name]["samples"]
        updated.append(expr)
    expression_cfg["expressions"] = updated
    expression_file.write_text(json.dumps(expression_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Expression file updated with IC metrics: {expression_file}")


if __name__ == '__main__':
    main()

