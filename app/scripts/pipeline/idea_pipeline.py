#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from agent_market.freqai.analysis import ALLOWED_FUNCS  # type: ignore  # noqa: E402

try:
    from agent_market.freqai.rl.trainer import RLTrainer  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    RLTrainer = None  # type: ignore

from scripts import idea_agent as idea_mod  # noqa: E402
from scripts import refine_agent as refine_mod  # noqa: E402

THEME_CONTEXTS: Dict[str, Dict[str, str]] = {
    "arbitrage": {
        "idea": (
            "请围绕跨市场/跨品种套利构思："
            "关注现货-永续、不同交易所、不同时间框架之间的价差与资金费率结构，"
            "强调风险控制、滑点与资金占用，并给出可量化的触发条件。"
            "数据会提供带有 `_BTC_USDT`、`_ETH_USDT` 等后缀的跨币种特征，可直接用于表达式构建。"
        ),
        "refine": (
            "细化套利场景，包括：1) 价格/资金费率/成交量差异的检测方法；"
            "2) 如何对冲方向性风险；3) 需要监控的风险指标（如报价深度、借贷成本、强平风险）；"
            "4) 对表达式和因子提出跨市场价差、资金费率、同步性相关的约束或阈值。"
            "注意：因子列名会提供带有 `_BTC_USDT` 这类后缀的跨币种特征，可直接引用这些列。"
        ),
    }
}

MODEL_ALIASES: Dict[str, str] = {
    "catboostregressor": "CatboostRegressor",
    "catboost": "CatboostRegressor",
    "lightgbmregressor": "LightGBMRegressor",
    "lightgbm": "LightGBMRegressor",
    "xgboostregressor": "XGBoostRegressor",
    "xgboost": "XGBoostRegressor",
    "pytorchmlpregressor": "PyTorchMLPRegressor",
    "pytorchmlp": "PyTorchMLPRegressor",
    "pytorchtransformerregressor": "PyTorchTransformerRegressor",
    "pytorchtransformer": "PyTorchTransformerRegressor",
}


def _merge_context(user_context: Optional[str], theme_context: Optional[str]) -> Optional[str]:
    parts = [ctx.strip() for ctx in (user_context, theme_context) if ctx and ctx.strip()]
    if not parts:
        return None
    return "\n\n".join(parts)


def _env_with_src() -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    root_str = str(ROOT)
    root_src = str(ROOT / "src")
    freqtrade_src = str(ROOT / "freqtrade")
    ordered = [freqtrade_src, root_src, root_str]
    for item in parts:
        if item not in ordered:
            ordered.append(item)
    env["PYTHONPATH"] = os.pathsep.join(ordered)
    return env


def _run_cmd(cmd: list[str], cwd: Path | None = None, extra_env: Dict[str, str] | None = None) -> None:
    env = _env_with_src()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, cwd=str(cwd or ROOT), env=env)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_last_result(backtest_dir: Path) -> Optional[Path]:
    last_file = backtest_dir / ".last_result.json"
    if not last_file.exists():
        return None
    try:
        payload = json.loads(last_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    zip_name = payload.get("latest_backtest")
    if not zip_name:
        return None
    candidate = backtest_dir / zip_name
    return candidate if candidate.exists() else None


def _summarize_backtest(zip_path: Path) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            json_members = [name for name in zf.namelist() if name.endswith(".json")]
            if not json_members:
                return {"error": "no json payload in archive", "file": str(zip_path)}
            # Prefer main result file (without config suffix)
            preferred = next((name for name in json_members if "config" not in name), json_members[0])
            with zf.open(preferred) as handle:
                data = json.load(TextIOWrapper(handle, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed to parse backtest archive: {exc}", "file": str(zip_path)}

    strat = data.get("strategy", {}).get("ExpressionLongStrategy", {})
    trades = strat.get("trades", [])
    trade_count = len(trades) if isinstance(trades, list) else strat.get("trades", 0)
    return {
        "file": str(zip_path),
        "profit_total_abs": strat.get("profit_total_abs"),
        "profit_total_percent": strat.get("profit_total_percent"),
        "winrate": strat.get("winrate"),
        "max_drawdown_abs": strat.get("max_drawdown_abs"),
        "trades": trade_count,
    }


def _inject_high_ic_features(
    expressions_path: Path,
    factor_summary: Dict[str, Any],
    *,
    min_ic: float = 0.05,
    max_injections: int = 6,
) -> List[Dict[str, Any]]:
    """将高 IC 的跨币种特征写回表达式文件，确保后续回测可用。"""
    try:
        expr_payload = json.loads(expressions_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    existing = {item.get("name") for item in expr_payload.get("expressions", [])}
    injected: List[Dict[str, Any]] = []
    for feature in factor_summary.get("features", []):
        name = feature.get("name")
        ic_val = feature.get("ic_pearson")
        if (
            not name
            or name in existing
            or ic_val is None
            or float(ic_val) < min_ic
        ):
            continue
        if not (name.startswith("arb_") or "_USDT_" in name):
            continue
        entry = {
            "name": name,
            "expression": name,
            "origin": "ic_top",
            "score_metric": "ic_pearson",
            "score": ic_val,
            "description": f"Auto-injected high IC feature {name}",
            "ic_pearson": ic_val,
            "ic_spearman": feature.get("ic_spearman"),
            "ic_samples": feature.get("samples"),
        }
        injected.append(entry)
        if len(injected) >= max_injections:
            break
    if not injected:
        return []
    expr_payload.setdefault("expressions", []).extend(injected)
    expressions_path.write_text(json.dumps(expr_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return injected


def _ensure_theme_expressions(expressions_path: Path, feature_cfg: Dict[str, Any]) -> List[str]:
    """确保表达式列表包含关键主题特征（如 lstm/arb 派生列）。"""
    try:
        expr_payload = json.loads(expressions_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    expressions = expr_payload.setdefault("expressions", [])
    existing = {item.get("name") for item in expressions if item.get("name")}

    added: List[str] = []
    combos = feature_cfg.get("feature_combos") or []
    theme_keywords = ("lstm", "arb_")
    for combo in combos:
        name = combo.get("name")
        if (
            not name
            or name in existing
            or not any(keyword in name for keyword in theme_keywords)
        ):
            continue
        entry = {
            "name": name,
            "expression": name,
            "origin": "theme_hint",
            "description": f"Auto-added theme expression {name}",
        }
        expressions.append(entry)
        added.append(name)
        existing.add(name)

    if added:
        expr_payload["expressions"] = expressions
        expressions_path.write_text(json.dumps(expr_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def _write_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_spec_payload(spec_payload: Dict[str, Any]) -> Dict[str, Any]:
    constraints = spec_payload.get("expression_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    allowed_names = set(ALLOWED_FUNCS.keys())
    requested = constraints.get("allowed_functions")
    if isinstance(requested, list):
        filtered = [name for name in requested if name in allowed_names]
        if not filtered:
            filtered = sorted(allowed_names)
    else:
        filtered = sorted(allowed_names)
    constraints["allowed_functions"] = filtered
    disallowed = constraints.get("disallowed_patterns")
    if isinstance(disallowed, list):
        constraints["disallowed_patterns"] = [str(item) for item in disallowed if isinstance(item, str)]
    spec_payload["expression_constraints"] = constraints
    return spec_payload


def _prepare_rl_config(
    args: argparse.Namespace,
    config_data: Dict[str, Any],
    feature_cfg: Dict[str, Any],
    feature_path: Path,
    datadir_path: Path,
    freqai_identifier: str,
) -> Dict[str, Any]:
    exchange_cfg = config_data.get("exchange") or {}
    exchange_name = exchange_cfg.get("name") or feature_cfg.get("exchange")
    if not exchange_name:
        raise ValueError("配置缺少 exchange.name，无法执行 RL 训练。")
    pairs = exchange_cfg.get("pair_whitelist") or feature_cfg.get("pairs")
    if not pairs:
        raise ValueError("配置缺少 exchange.pair_whitelist，无法执行 RL 训练。")

    freqai_cfg = config_data.get("freqai", {})
    feature_params = freqai_cfg.get("feature_parameters", {})
    label_period = int(
        feature_cfg.get("label_period_candles")
        or feature_params.get("label_period_candles")
        or 12
    )

    model_dir = (Path(args.rl_output_dir).resolve() / freqai_identifier).resolve()
    if not datadir_path.exists():
        raise FileNotFoundError(f'数据目录不存在: {datadir_path}')
    data_config = {
        "feature_file": str(feature_path),
        "data_dir": str(datadir_path),
        "exchange": str(exchange_name),
        "timeframe": args.timeframe,
        "pairs": [str(item) for item in pairs],
        "label_period": label_period,
    }
    rl_config: Dict[str, Any] = {
        "data": data_config,
        "training": {
            "total_timesteps": int(args.rl_timesteps),
        },
        "output": {
            "model_dir": str(model_dir),
        },
        "policy": args.rl_policy,
        "algo": {
            "device": "auto",
        },
    }
    return rl_config


def _run_rl_training(
    args: argparse.Namespace,
    config_data: Dict[str, Any],
    feature_cfg: Dict[str, Any],
    feature_path: Path,
    datadir_path: Path,
    output_dir: Path,
    freqai_identifier: str,
) -> Optional[Dict[str, Any]]:
    if not args.rl_enabled:
        return None
    if RLTrainer is None:
        raise RuntimeError("stable-baselines3 未安装，无法执行 RL 训练。")

    rl_config = _prepare_rl_config(
        args,
        config_data,
        feature_cfg,
        feature_path,
        datadir_path,
        freqai_identifier,
    )
    trainer = RLTrainer(rl_config)
    result = trainer.train()
    summary = {
        "config": {
            "data": {
                "feature_file": rl_config["data"]["feature_file"],
                "data_dir": rl_config["data"]["data_dir"],
                "exchange": rl_config["data"]["exchange"],
                "timeframe": rl_config["data"]["timeframe"],
                "pairs": rl_config["data"]["pairs"],
                "label_period": rl_config["data"]["label_period"],
            },
            "training": rl_config["training"],
            "policy": rl_config["policy"],
            "algo": rl_config.get("algo"),
        },
        "result": {
            "model_path": str(result.model_path),
            "timesteps": int(result.timesteps),
        },
    }
    summary_path = output_dir / "rl_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pipeline] RL 训练完成 -> {result.model_path}")
    return {
        "summary_file": str(summary_path),
        **summary,
    }


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    feature_path = Path(args.feature_file).resolve()
    idea_path = output_dir / "idea_raw.json"
    spec_path = output_dir / "idea_spec.json"
    expressions_path = output_dir / "expressions.json"
    factor_path = output_dir / "factor_ic.json"
    corr_path = output_dir / "factor_corr.csv"
    iterations_path = output_dir / "iterations.json"

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    feature_cfg = json.loads(feature_path.read_text(encoding="utf-8"))
    freqai_identifier = config_data.get("freqai", {}).get("identifier", "freqai")
    if args.purge_models:
        model_dir = ROOT / "user_data" / "models" / freqai_identifier
        if model_dir.exists():
            shutil.rmtree(model_dir)

    llm_namespace_common = dict(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        llm_count=args.llm_count,
        llm_retries=args.llm_retries,
        llm_timeout=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
    )

    theme_key = (args.theme or "").strip().lower() or None
    theme_payload = THEME_CONTEXTS.get(theme_key or "", {}) if theme_key else {}
    idea_context = _merge_context(args.idea_context, theme_payload.get("idea"))
    refine_context = _merge_context(args.refine_context, theme_payload.get("refine"))

    # Step 1: idea agent
    idea_args = argparse.Namespace(
        idea=args.idea,
        idea_file=args.idea_file,
        context=idea_context,
        output=str(idea_path),
        **llm_namespace_common,
    )
    idea_payload = idea_mod.run(idea_args)
    idea_path.parent.mkdir(parents=True, exist_ok=True)
    idea_path.write_text(json.dumps(idea_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 2: refinement
    refine_args = argparse.Namespace(
        input=str(idea_path),
        context=refine_context,
        output=str(spec_path),
        **llm_namespace_common,
    )
    spec_payload = refine_mod.run(refine_args)
    spec_payload = _sanitize_spec_payload(spec_payload)
    spec_path.write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 3: factor generation via freqai_expression_agent
    python_cmd = shlex.split(args.python_exec, posix=os.name != "nt")
    backtest_runner_cmd = shlex.split(args.backtest_runner, posix=os.name != "nt") if args.backtest_runner else []
    freqtrade_executable = shutil.which("freqtrade")
    if freqtrade_executable:
        freqtrade_cli = backtest_runner_cmd + [freqtrade_executable]
    else:
        freqtrade_cli = backtest_runner_cmd + python_cmd + ["-m", "freqtrade"]
    backtest_runner_str = args.backtest_runner.strip()
    backtest_prefix = f"{backtest_runner_str} freqtrade backtesting" if backtest_runner_str else "freqtrade backtesting"

    datadir_value = Path(config_data.get("datadir", "resources/user_data/data"))
    if not datadir_value.is_absolute():
        candidates = [
            (config_path.parent / datadir_value).resolve(),
            (ROOT / datadir_value).resolve(),
            datadir_value.resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                datadir_path = candidate
                break
        else:
            datadir_path = (config_path.parent / datadir_value).resolve()
    else:
        datadir_path = datadir_value

    lstm_summary_file: Optional[Path] = None
    lstm_prediction_dir_path: Optional[Path] = None
    if args.lstm_enabled:
        exchange_cfg = config_data.get("exchange", {})
        whitelist = exchange_cfg.get("pair_whitelist") or []
        if not whitelist:
            raise ValueError("exchange.pair_whitelist is required for LSTM integration.")
        lstm_pairs = args.lstm_pairs or ",".join(whitelist)
        target_pair = args.lstm_target_pair or whitelist[0]
        exchange_name = exchange_cfg.get("name", "")
        data_dir_for_lstm = datadir_path
        if exchange_name:
            candidate = datadir_path / exchange_name
            if candidate.exists():
                data_dir_for_lstm = candidate

        lstm_prediction_dir_path = Path(args.lstm_prediction_dir).resolve()
        lstm_prediction_dir_path.mkdir(parents=True, exist_ok=True)
        lstm_summary_file = output_dir / f"lstm_summary_{target_pair.replace('/', '_')}.json"
        lstm_cmd = python_cmd + [
            "scripts/deep_seq_trainer.py",
            "--data-dir",
            str(data_dir_for_lstm),
            "--pairs",
            lstm_pairs,
            "--target-pair",
            target_pair,
            "--timeframe",
            args.timeframe,
            "--window",
            str(args.lstm_window),
            "--horizon",
            str(args.lstm_horizon),
            "--epochs",
            str(args.lstm_epochs),
            "--batch-size",
            str(args.lstm_batch_size),
            "--lr",
            str(args.lstm_lr),
            "--prediction-dir",
            str(lstm_prediction_dir_path),
            "--output",
            str(lstm_summary_file),
        ]
        print(f"[pipeline] training LSTM predictions for {target_pair} -> {lstm_prediction_dir_path}")
        _run_cmd(lstm_cmd)

    expr_cmd = python_cmd + [
        "freqtrade/scripts/freqai_expression_agent.py",
        "--config",
        str(config_path),
        "--feature-file",
        str(feature_path),
        "--output",
        str(expressions_path),
        "--timeframe",
        args.timeframe,
        "--llm-count",
        str(args.expression_llm_count),
        "--llm-loops",
        str(args.expression_llm_loops),
        "--top",
        str(args.expression_top),
        "--stability-min-samples",
        str(args.expression_stability_min_samples),
        "--spec-file",
        str(spec_path),
    ]
    env_override = {"FREQAI_EXPRESSIONS_FILE": str(expressions_path)}
    _run_cmd(expr_cmd, extra_env=env_override)
    theme_added = _ensure_theme_expressions(expressions_path, feature_cfg)
    if theme_added:
        print(f"[pipeline] ensured theme expressions: {theme_added}")

    raw_models = [m.strip() for m in args.train_models.split(",") if m.strip()]
    if not raw_models:
        raw_models = [args.freqaimodel]
    else:
        raw_models.append(args.freqaimodel)

    train_models: List[str] = []
    for model in raw_models:
        key = model.replace("_", "").lower()
        canonical = MODEL_ALIASES.get(key, model)
        if canonical not in train_models:
            train_models.append(canonical)
    train_timerange = args.train_timerange or args.timerange
    model_config_map: Dict[str, Dict[str, Any]] = {}
    trained_models: List[str] = []
    for model in train_models:
        if model in trained_models:
            continue
        model_identifier = f"{freqai_identifier}-{model.lower()}"
        model_config = copy.deepcopy(config_data)
        model_config.setdefault("freqai", {})
        model_config["freqai"]["identifier"] = model_identifier
        model_config["freqai"]["save_backtest_models"] = True
        model_dir = ROOT / "user_data" / "models" / model_identifier
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_config_path = output_dir / f"config_{model}.json"
        model_config_path.write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")
        model_config_map[model] = {
            "config_path": model_config_path,
            "identifier": model_identifier,
        }
        train_cmd = freqtrade_cli + [
            "backtesting",
            "--config",
            str(model_config_path),
            "--strategy",
            "ExpressionLongStrategy",
            "--strategy-path",
            "resources/user_data/strategies",
            "--timerange",
            train_timerange,
            "--freqaimodel",
            model,
        ]
        _run_cmd(train_cmd, extra_env=env_override)
        trained_models.append(model)

    # Step 4: factor evaluation
    compute_cmd = python_cmd + [
        "scripts/compute_factor_ic.py",
        "--config",
        str(config_path),
        "--feature-file",
        str(feature_path),
        "--expression-file",
        str(expressions_path),
        "--timeframe",
        args.timeframe,
        "--output",
        str(factor_path),
        "--corr-output",
        str(corr_path),
        "--quantiles",
        str(args.quantiles),
        "--update-expression-file",
    ]
    _run_cmd(compute_cmd, extra_env=env_override)

    iteration_summaries: Dict[str, Dict[str, Any]] = {}
    iterations_paths: Dict[str, Path] = {}
    for model in train_models:
        model_conf = model_config_map.get(model)
        model_config_path = model_conf["config_path"] if model_conf else config_path
        model_iterations_path = output_dir / f"iterations_{model}.json"
        iterations_paths[model] = model_iterations_path
        iterate_cmd = python_cmd + [
            "scripts/agent_iterate.py",
            "--config",
            str(model_config_path),
            "--feature-file",
            str(feature_path),
            "--expression-file",
            str(expressions_path),
            "--iterations",
            str(args.iterations),
            "--top-expressions",
            str(args.iteration_top_expressions),
            "--quantiles",
            str(args.quantiles),
            "--report-file",
            str(model_iterations_path),
            "--backtest-command",
            f"{backtest_prefix} --config {{config}} --strategy ExpressionLongStrategy --strategy-path resources/user_data/strategies --timerange {args.timerange} --export trades --export-filename {{export}} --freqaimodel {model}",
            "--backtest-export-prefix",
            str(output_dir / "backtest" / model),
        ]
        _run_cmd(iterate_cmd, extra_env=env_override)
        iteration_summaries[model] = _load_json(model_iterations_path)

    factor_summary = _load_json(factor_path)
    injected = _inject_high_ic_features(expressions_path, factor_summary)
    if injected:
        print(f"[pipeline] injected {len(injected)} high-IC expressions: {[item['name'] for item in injected]}")
    theme_added_post = _ensure_theme_expressions(expressions_path, feature_cfg)
    if theme_added_post:
        print(f"[pipeline] ensured theme expressions after IC: {theme_added_post}")

    backtest_dir = ROOT / "user_data" / "backtest_results"
    latest_archive = _read_last_result(backtest_dir)
    backtest_summary = (
        _summarize_backtest(latest_archive) if latest_archive else {"warning": "no backtest archive located"}
    )
    if latest_archive:
        meta_candidate = latest_archive.with_suffix("").with_suffix(".meta.json")
        backtest_summary["meta_file"] = str(meta_candidate) if meta_candidate.exists() else None

    rl_payload = _run_rl_training(
        args,
        config_data,
        feature_cfg,
        feature_path,
        datadir_path,
        output_dir,
        freqai_identifier,
    )

    for model, summary in iteration_summaries.items():
        iterations = summary.get("iterations", [])
        for entry in iterations:
            bt = entry.setdefault("backtest", {})
            bt.update(backtest_summary)
        summary_path = iterations_paths[model]
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        iteration_summaries[model] = summary

    summary_payload = {
        "idea": idea_payload,
        "spec": spec_payload,
        "expressions_file": str(expressions_path),
        "factor_report": str(factor_path),
        "iterations_reports": {model: str(path) for model, path in iterations_paths.items()},
        "factor_top_expressions": factor_summary.get("expressions", [])[: args.summary_top],
        "iterations": {model: summary.get("iterations", []) for model, summary in iteration_summaries.items()},
        "backtest_summary": backtest_summary,
        "trained_models": trained_models,
        "model_identifiers": {model: data["identifier"] for model, data in model_config_map.items()},
        "lstm_summary": str(lstm_summary_file) if lstm_summary_file else None,
        "lstm_prediction_dir": str(lstm_prediction_dir_path) if lstm_prediction_dir_path else None,
        "freqai_identifier": freqai_identifier,
        "theme": theme_key,
        "theme_contexts": theme_payload if theme_payload else None,
        "rl": rl_payload,
    }
    _write_summary(output_dir / "pipeline_summary.json", summary_payload)
    print(f"[pipeline] Completed idea refinement pipeline -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end idea → refinement → factor generation pipeline.")
    parser.add_argument("--idea", help="Inline idea text.")
    parser.add_argument("--idea-file", help="Path to text file describing the idea.")
    parser.add_argument("--idea-context", help="Supplementary notes passed to idea agent.")
    parser.add_argument("--refine-context", help="Supplementary notes passed to refiner agent.")
    parser.add_argument("--output-dir", default="user_data/reports/idea_pipeline")

    parser.add_argument("--config", required=True, help="Freqtrade config JSON path.")
    parser.add_argument(
        "--theme",
        choices=sorted(THEME_CONTEXTS.keys()),
        help="Optional thematic hint (e.g. 'arbitrage') to steer idea/refinement prompts.",
    )
    parser.add_argument("--feature-file", required=True, help="Feature configuration JSON.")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--timerange", default="20220101-20221231")
    parser.add_argument("--freqaimodel", default="LightGBMRegressor")
    parser.add_argument("--backtest-runner", default="", help="Optional command prefix for freqtrade backtesting (e.g. 'conda run -n freqtrade').")
    parser.add_argument("--train-models", default="", help="Comma-separated freqaimodel class names to pre-train before evaluation.")
    parser.add_argument("--train-timerange", default=None, help="Timerange to use when pre-training models (defaults to --timerange).")
    parser.add_argument("--purge-models", action="store_true", help="Delete existing models under user_data/models/<identifier> before training.")
    parser.add_argument("--lstm-enabled", action="store_true", help="Enable LSTM pre-training and prediction merging.")
    parser.add_argument("--lstm-pairs", default=None, help="Comma-separated pairs for LSTM training (defaults to exchange.pair_whitelist).")
    parser.add_argument("--lstm-target-pair", default=None, help="Target pair for LSTM predictions (defaults to first pair).")
    parser.add_argument("--lstm-window", type=int, default=64, help="LSTM sequence window length.")
    parser.add_argument("--lstm-horizon", type=int, default=6, help="LSTM prediction horizon.")
    parser.add_argument("--lstm-epochs", type=int, default=15, help="LSTM training epochs.")
    parser.add_argument("--lstm-batch-size", type=int, default=512, help="LSTM batch size.")
    parser.add_argument("--lstm-lr", type=float, default=5e-4, help="LSTM learning rate.")
    parser.add_argument("--lstm-prediction-dir", default="user_data/lstm_predictions", help="Directory to store LSTM predictions.")
    parser.add_argument("--rl-enabled", action="store_true", help="Enable RL training (requires stable-baselines3).")
    parser.add_argument("--rl-timesteps", type=int, default=20000, help="Total timesteps for PPO training.")
    parser.add_argument("--rl-output-dir", default="user_data/models/rl", help="Directory to store RL models.")
    parser.add_argument("--rl-policy", default="MlpPolicy", help="Stable-Baselines policy class name.")

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--iteration-top-expressions", type=int, default=6)
    parser.add_argument("--expression-top", type=int, default=20)
    parser.add_argument("--expression-llm-count", type=int, default=12)
    parser.add_argument("--expression-llm-loops", type=int, default=1)
    parser.add_argument("--expression-stability-min-samples", type=int, default=50)
    parser.add_argument("--summary-top", type=int, default=5)

    parser.add_argument("--python-exec", default=sys.executable)

    parser.add_argument("--llm-base-url", default=idea_mod.llm_utils.DEFAULT_BASE_URL)  # type: ignore[attr-defined]
    parser.add_argument("--llm-model", default=idea_mod.llm_utils.DEFAULT_MODEL)  # type: ignore[attr-defined]
    parser.add_argument("--llm-api-key", default=idea_mod.llm_utils.DEFAULT_API_KEY)  # type: ignore[attr-defined]
    parser.add_argument("--llm-count", type=int, default=1)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout", type=float, default=idea_mod.llm_utils.DEFAULT_TIMEOUT)  # type: ignore[attr-defined]
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=900)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
