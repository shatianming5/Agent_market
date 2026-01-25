from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market.backtest_results import write_latest_backtest_summary

logger = logging.getLogger(__name__)

TrainingPipeline = None
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_command(cmd: list[str], cwd: Optional[str] = None) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603,S607


def _resolve_path(path: str | Path, cwd: Optional[str] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if cwd:
        return (REPO_ROOT / cwd / p).resolve()
    return (REPO_ROOT / p).resolve()


def _resolve_cwd(cwd: Optional[str]) -> str:
    if not cwd:
        return str(REPO_ROOT)
    return str((REPO_ROOT / cwd).resolve())


def _extract_config_path_from_command(cmd: list[str]) -> Optional[str]:
    for flag in ("--config", "-c"):
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                return cmd[idx + 1]
    return None


def run_feature_generation(cfg: Dict[str, Any]) -> None:
    script = Path(cfg.get("script", "scripts/freqai_feature_agent.py"))
    args = list(map(str, cfg.get("args", [])))
    script_path = _resolve_path(script, cwd=cfg.get("cwd"))
    cmd = [sys.executable, str(script_path)] + args
    run_command(cmd, cwd=_resolve_cwd(cfg.get("cwd")))


def run_expression_generation(cfg: Dict[str, Any], feedback_path: Path) -> None:
    script = Path(cfg.get("script", "scripts/freqai_expression_agent.py"))
    args = list(map(str, cfg.get("args", [])))
    fb_path = Path(cfg.get("feedback_path", feedback_path))
    append_feedback = fb_path.exists() and "--feedback" not in args
    if append_feedback:
        args += ["--feedback", str(fb_path)]
        if "--feedback-top" not in args:
            args += ["--feedback-top", str(cfg.get("feedback_top", 10))]
        logger.info("Injecting feedback summary for expression generation: %s", fb_path)
    script_path = _resolve_path(script, cwd=cfg.get("cwd"))
    cmd = [sys.executable, str(script_path)] + args
    run_command(cmd, cwd=_resolve_cwd(cfg.get("cwd")))


def run_ml_training(cfg: Dict[str, Any]) -> None:
    pipeline_cls = TrainingPipeline
    if pipeline_cls is None:
        from agent_market.freqai.training.pipeline import TrainingPipeline as pipeline_cls  # noqa: WPS433

    configs = cfg.get("configs")
    single = cfg.get("config")
    if configs and single:
        raise ValueError("ml_training.config and ml_training.configs are mutually exclusive")
    if configs:
        if not isinstance(configs, list):
            raise ValueError("ml_training.configs must be a list")
        job_list = [item for item in configs if isinstance(item, dict)]
        if len(job_list) != len(configs):
            raise ValueError("ml_training.configs must contain JSON objects only")
    else:
        if not isinstance(single, dict):
            raise ValueError("ml_training.config must be provided as a JSON object")
        job_list = [single]

    total = len(job_list)
    for idx, job_cfg in enumerate(job_list, start=1):
        model_name = job_cfg.get("model", {}).get("name", "unknown")
        logger.info("Starting ML training job %s/%s with model=%s", idx, total, model_name)
        pipeline_cls(job_cfg).run()


def run_rl_training(cfg: Dict[str, Any]) -> None:
    config = cfg.get("config")
    if not isinstance(config, dict):
        raise ValueError("rl_training.config must be provided as a JSON object")
    logger.info("Starting RL training")

    model_dir = Path((config.get("output") or {}).get("model_dir") or "artifacts/models/rl_real")
    model_dir = _resolve_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "rl_training_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    script = REPO_ROOT / "scripts" / "train_rl.py"
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    run_command(cmd, cwd=str(REPO_ROOT))


def _maybe_generate_rl_signals_for_backtest(cfg: Dict[str, Any]) -> None:
    if cfg.get("generate_rl_signals") is False:
        logger.info("Skipping RL signal generation (generate_rl_signals=false)")
        return

    rl_summary = cfg.get("rl_summary", "artifacts/models/rl_real/training_summary.json")
    rl_summary_path = _resolve_path(str(rl_summary))
    if not rl_summary_path.exists():
        logger.info("RL summary not found (%s); skipping RL signal generation", rl_summary_path)
        return

    cmd_list: Optional[list[str]] = None
    backtest_cwd = cfg.get("cwd")
    if "command" in cfg:
        cmd_list = [str(part) for part in cfg["command"]]
        config_arg = _extract_config_path_from_command(cmd_list)
    else:
        config_arg = str(cfg.get("config")) if cfg.get("config") else None
    if not config_arg:
        logger.info("Backtest config path not found; skipping RL signal generation")
        return

    freqtrade_config_path = _resolve_path(config_arg, cwd=backtest_cwd)
    if not freqtrade_config_path.exists():
        logger.info("Freqtrade config not found (%s); skipping RL signal generation", freqtrade_config_path)
        return

    script = REPO_ROOT / "scripts" / "rl_generate_signals.py"
    gen_cmd = [sys.executable, str(script), "--config", str(freqtrade_config_path)]
    logger.info("Generating RL signals for backtest (%s)", freqtrade_config_path)
    run_command(gen_cmd, cwd=str(REPO_ROOT))


def run_backtest(cfg: Dict[str, Any], feedback_path: Path) -> None:
    _maybe_generate_rl_signals_for_backtest(cfg)

    if "command" in cfg:
        cmd = [str(part) for part in cfg["command"]]
    else:
        binary = cfg.get("binary", "freqtrade")
        cmd = [str(binary), "backtesting"]
        if cfg.get("config"):
            cmd += ["--config", str(cfg["config"])]
        if cfg.get("strategy"):
            cmd += ["--strategy", str(cfg["strategy"])]
        if cfg.get("strategy_path"):
            cmd += ["--strategy-path", str(cfg["strategy_path"])]
        if cfg.get("timerange"):
            cmd += ["--timerange", str(cfg["timerange"])]
        cmd += list(map(str, cfg.get("extra_args", [])))

    run_command(cmd, cwd=_resolve_cwd(cfg.get("cwd")))
    results_dir = Path(cfg.get("results_dir", "user_data/backtest_results"))
    out_path = Path(cfg.get("feedback_path", feedback_path))
    summary = write_latest_backtest_summary(results_dir, out_path)
    if summary is not None:
        logger.info("Backtest summary written to %s", out_path)


__all__ = [
    "run_backtest",
    "run_command",
    "run_expression_generation",
    "run_feature_generation",
    "run_ml_training",
    "run_rl_training",
]
