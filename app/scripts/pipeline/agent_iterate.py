from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative agent loop: generate expressions -> evaluate -> backtest.")
    parser.add_argument("--config", required=True, help="Path to freqtrade config JSON.")
    parser.add_argument("--feature-file", required=True, help="Path to feature JSON file.")
    parser.add_argument("--expression-file", required=True, help="Path to expression JSON file.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run.")
    parser.add_argument("--top-expressions", type=int, default=10, help="Keep only top-N expressions by IC.")
    parser.add_argument("--min-ic", type=float, default=None, help="Minimum ic_pearson threshold to keep expressions.")
    parser.add_argument("--quantiles", type=int, default=5, help="Quantile buckets for factor analysis.")
    parser.add_argument("--report-file", default="user_data/reports/agent_iterations.json", help="Where to store iteration summary.")
    parser.add_argument("--compute-script", default="scripts/compute_factor_ic.py", help="Factor analysis script path.")
    parser.add_argument("--compute-output", default="user_data/reports/factor_ic.json", help="Factor analysis output path.")
    parser.add_argument("--corr-output", default="user_data/reports/factor_corr.csv", help="Optional correlation matrix CSV.")
    parser.add_argument("--generate-command", default=None, help="Optional shell command to regenerate expressions each iteration.")
    parser.add_argument("--backtest-command", required=True, help="Freqtrade backtesting command (use placeholders if needed).")
    parser.add_argument("--backtest-export-prefix", default="user_data/backtest_results/agent_iter", help="Prefix for backtest exports.")
    parser.add_argument("--expression-backup-dir", default="user_data/reports/agent_expression_snapshots", help="Directory to store expression snapshots per iteration.")
    parser.add_argument("--reuse-compute", action="store_true", help="Skip recomputing IC if factor_ic.json already exists in iteration (debug use).")
    parser.add_argument("--timeframe", default=None, help="Optional timeframe override for analysis/backtest placeholders.")
    parser.add_argument("--label-period", type=int, default=None, help="Optional label period override for analysis.")
    return parser.parse_args()


def run_command(cmd, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> str:
    if isinstance(cmd, str):
        completed = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    else:
        completed = subprocess.run(
            cmd,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    return completed.stdout


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_top_expressions(summary: Dict, top_n: int, min_ic: Optional[float]) -> List[Dict]:
    expressions = summary.get("expressions", [])
    if min_ic is not None:
        expressions = [expr for expr in expressions if expr.get("ic_pearson", 0.0) >= min_ic]
    expressions.sort(key=lambda item: item.get("ic_pearson", 0.0), reverse=True)
    return expressions[:top_n] if top_n > 0 else expressions


def select_high_ic_features(summary: Dict, existing: set[str], min_ic: float = 0.05, max_count: int = 6) -> List[Dict]:
    features = []
    for item in summary.get("features", []):
        name = item.get("name")
        ic_val = item.get("ic_pearson")
        if (
            not name
            or name in existing
            or ic_val is None
            or float(ic_val) < min_ic
        ):
            continue
        if not (name.startswith("arb_") or "_USDT_" in name):
            continue
        features.append(
            {
                "name": name,
                "expression": name,
                "origin": "ic_top",
                "score_metric": "ic_pearson",
                "score": ic_val,
                "ic_pearson": ic_val,
                "ic_spearman": item.get("ic_spearman"),
                "ic_samples": item.get("samples"),
                "description": f"Auto-selected high IC feature {name}",
            }
        )
        if len(features) >= max_count:
            break
    return features


def update_expression_file(expression_path: Path, selected: List[Dict]) -> None:
    expr_data = load_json(expression_path)
    selected_names = {expr["name"] for expr in selected if expr.get("name")}
    expressions = list(expr_data.get("expressions", []))
    name_to_item = {item.get("name"): item for item in expressions if item.get("name")}

    # Append missing expressions from selection
    for entry in selected:
        name = entry.get("name")
        if not name or name in name_to_item:
            continue
        new_entry = dict(entry)
        new_entry.setdefault("expression", name)
        new_entry.setdefault("origin", "ic_top")
        expressions.append(new_entry)
        name_to_item[name] = new_entry

    updated = []
    for item in expressions:
        item = dict(item)
        name = item.get("name")
        item["enabled"] = bool(name and name in selected_names)
        updated.append(item)

    expr_data["expressions"] = updated
    save_json(expression_path, expr_data)


def backup_expression_file(expression_path: Path, backup_dir: Path, iteration: int) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = backup_dir / f"expressions_iter{iteration}.json"
    snapshot_path.write_text(expression_path.read_text(encoding="utf-8"), encoding="utf-8")


def find_latest_meta(result_dir: Path) -> Optional[Path]:
    candidates = list(result_dir.glob("backtest-result-*.meta.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_backtest_summary(meta_path: Path) -> Dict:
    try:
        meta = load_json(meta_path)
    except Exception:
        return {"meta_file": str(meta_path), "error": "failed to load meta json"}
    result: Dict[str, Optional[str]] = {
        "meta_file": str(meta_path),
        "run_id": meta.get("run_id"),
        "timeframe": meta.get("timeframe"),
        "stake_currency": meta.get("stake_currency"),
        "dry_run": meta.get("dry_run"),
    }
    strategy_map = meta.get("strategy") or {}
    if strategy_map:
        name, stats = next(iter(strategy_map.items()))
        result["strategy"] = name
        result["stats"] = stats
    zip_path = _infer_backtest_zip(meta_path)
    if zip_path:
        result["export_file"] = zip_path
        parsed = _extract_backtest_stats(Path(zip_path))
        if parsed:
            result.update(parsed)
    return result


def _infer_backtest_zip(meta_path: Path) -> Optional[str]:
    if meta_path.suffix != ".json":
        return None
    target_name = meta_path.name.replace(".meta.json", ".zip")
    candidate = meta_path.with_name(target_name)
    if candidate.exists():
        return str(candidate)
    return None


def _extract_backtest_stats(zip_path: Path) -> Dict[str, Optional[float]]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".json")]
            if not members:
                return {}
            with archive.open(members[0]) as handle:
                payload = json.load(handle)
    except Exception:
        return {"stats_error": f"failed to parse {zip_path.name}"}

    strat_map = payload.get("strategy") or {}
    if not strat_map:
        return {}
    name, stats = next(iter(strat_map.items()))
    trades = stats.get("total_trades")
    if trades is None and isinstance(stats.get("trades"), dict):
        trades = stats["trades"].get("count")

    summary = {
        "strategy": name,
        "profit_total_abs": stats.get("profit_total_abs"),
        "profit_total_percent": stats.get("profit_total_percent"),
        "total_trades": trades,
        "winrate": stats.get("winrate"),
        "max_drawdown_abs": stats.get("max_drawdown_abs"),
        "timerange": stats.get("timerange"),
        "freqaimodel": stats.get("freqaimodel"),
    }
    profit_factor = stats.get("profit_factor")
    if profit_factor is not None:
        summary["profit_factor"] = profit_factor
    return summary


def snapshot_backtest_meta(root: Path) -> Dict[str, float]:
    if not root.exists():
        return {}
    return {
        path.name: path.stat().st_mtime
        for path in root.glob("backtest-result-*.meta.json")
    }


def _read_last_result_meta(root: Path) -> Optional[Path]:
    last_file = root / ".last_result.json"
    if not last_file.exists():
        return None
    try:
        payload = json.loads(last_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    latest = payload.get("latest_backtest")
    if not latest:
        return None
    meta_name = latest.replace(".zip", ".meta.json")
    candidate = root / meta_name
    return candidate if candidate.exists() else None


def resolve_new_backtest_meta(root: Path, before_snapshot: Dict[str, float]) -> Optional[Path]:
    after_snapshot = snapshot_backtest_meta(root)
    new_candidates = [
        (mtime, name)
        for name, mtime in after_snapshot.items()
        if name not in before_snapshot or mtime > before_snapshot.get(name, 0.0)
    ]
    if new_candidates:
        _, newest_name = max(new_candidates, key=lambda item: item[0])
        return root / newest_name
    last_meta = _read_last_result_meta(root)
    if last_meta and last_meta.name in after_snapshot:
        return last_meta
    return last_meta


def main() -> None:
    args = parse_args()
    expression_path = Path(args.expression_file)
    feature_path = Path(args.feature_file)
    compute_output = Path(args.compute_output)
    corr_output = Path(args.corr_output) if args.corr_output else None
    report_path = Path(args.report_file)
    backtest_export_prefix = Path(args.backtest_export_prefix)
    backtest_results_root = backtest_export_prefix.parent
    expression_backup_dir = Path(args.expression_backup_dir)

    env = os.environ.copy()
    env["FREQAI_EXPRESSIONS_FILE"] = str(expression_path)
    env["FREQAI_FEATURES_FILE"] = str(feature_path)

    iterations_report: List[Dict] = []

    backtest_results_root.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.iterations + 1):
        print(f"[agent] iteration {iteration}/{args.iterations}")

        if args.generate_command:
            print(f"[agent] generating expressions via command: {args.generate_command}")
            run_command(args.generate_command, env=env)

        if not args.reuse_compute or not compute_output.exists():
            if compute_output.exists():
                compute_output.unlink()
            compute_parts = [
                "python",
                args.compute_script,
                "--config",
                args.config,
                "--feature-file",
                str(feature_path),
                "--expression-file",
                str(expression_path),
                "--output",
                str(compute_output),
                "--quantiles",
                str(args.quantiles),
                "--update-expression-file",
            ]
            if args.timeframe:
                compute_parts.extend(["--timeframe", args.timeframe])
            if args.label_period:
                compute_parts.extend(["--label-period", str(args.label_period)])
            if corr_output:
                compute_parts.extend(["--corr-output", str(corr_output)])
            print(f"[agent] computing factor metrics: {shlex.join(compute_parts)}")
            run_command(compute_parts, env=env)

        summary = load_json(compute_output)
        selected_exprs = select_top_expressions(summary, args.top_expressions, args.min_ic)
        existing_names = {expr.get("name") for expr in selected_exprs if expr.get("name")}
        ic_extras = select_high_ic_features(summary, existing_names)
        if ic_extras:
            selected_exprs.extend(ic_extras)
        backup_expression_file(expression_path, expression_backup_dir, iteration)
        update_expression_file(expression_path, selected_exprs)

        export_dir = backtest_export_prefix / f"iter_{iteration}"
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        export_prefix = export_dir / "results"
        backtest_cmd = args.backtest_command.format(
            config=args.config,
            expressions=str(expression_path),
            features=str(feature_path),
            export=str(export_prefix),
            iteration=iteration,
            timeframe=args.timeframe or "",
            label_period=args.label_period or "",
        )
        print(f"[agent] running backtest: {backtest_cmd}")
        meta_snapshot_before = snapshot_backtest_meta(backtest_results_root)
        backtest_error: Optional[subprocess.CalledProcessError] = None
        try:
            run_command(backtest_cmd, env=env)
        except subprocess.CalledProcessError as exc:
            backtest_error = exc

        meta_path = resolve_new_backtest_meta(backtest_results_root, meta_snapshot_before)
        if not meta_path:
            meta_path = find_latest_meta(export_dir)
        if not meta_path:
            meta_path = find_latest_meta(backtest_results_root)
        if not meta_path:
            result = {
                "iteration": iteration,
                "expressions": selected_exprs,
                "error": f"backtest failed: {backtest_error}" if backtest_error else "backtest meta file missing",
                "backtest_command": backtest_cmd,
            }
            iterations_report.append(result)
            continue
        backtest_stats = load_backtest_summary(meta_path) if meta_path else {"warning": "no meta file located"}

        iteration_summary = {
            "iteration": iteration,
            "selected_expressions": selected_exprs,
            "backtest": backtest_stats,
            "factor_summary": summary.get("features", [])[:5],
            "backtest_command": backtest_cmd,
        }
        if backtest_error:
            iteration_summary["backtest_returncode"] = backtest_error.returncode
            iteration_summary["backtest_warning"] = str(backtest_error)
        iterations_report.append(iteration_summary)

        compute_output.unlink(missing_ok=True)
        if corr_output and corr_output.exists():
            corr_output.unlink()

    save_json(report_path, {"iterations": iterations_report})
    print(f"[agent] iteration log written to {report_path}")


if __name__ == "__main__":
    main()
