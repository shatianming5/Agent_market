from __future__ import annotations

import json
import logging
import subprocess
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_market.freqai.model import gradient_boosting  # noqa: F401
from agent_market.freqai.training.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


@dataclass
class AgentFlowConfig:
    download: Optional[Dict[str, Any]] = None
    feature: Optional[Dict[str, Any]] = None
    expression: Optional[Dict[str, Any]] = None
    ml_training: Optional[Dict[str, Any]] = None
    rl_training: Optional[Dict[str, Any]] = None
    backtest: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFlowConfig":
        known_keys = {"download", "feature", "expression", "ml_training", "rl_training", "backtest"}
        extra = set(data.keys()) - known_keys
        if extra:
            logger.warning("AgentFlowConfig received unknown keys: %s", ", ".join(sorted(extra)))
        return cls(
            download=data.get("download"),
            feature=data.get("feature"),
            expression=data.get("expression"),
            ml_training=data.get("ml_training"),
            rl_training=data.get("rl_training"),
            backtest=data.get("backtest"),
        )


def load_agent_flow_config(path: Path) -> AgentFlowConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse config JSON: {exc}") from exc
    return AgentFlowConfig.from_dict(payload)


class AgentFlow:
    STEP_ORDER = ["download", "feature", "expression", "ml", "rl", "backtest"]

    def __init__(self, config: AgentFlowConfig):
        self.config = config
        self.feedback_path = Path("user_data/llm_feedback/latest_backtest_summary.json")

    def run(self, steps: Optional[List[str]] = None) -> None:
        requested = None
        if steps:
            requested = [step.lower() for step in steps]
            unknown = [step for step in requested if step not in self.STEP_ORDER]
            if unknown:
                logger.warning("Ignoring unknown steps: %s", ", ".join(unknown))
            requested = [step for step in requested if step in self.STEP_ORDER]
        sequence = [
            ("download", self.config.download, self.run_download),
            ("feature", self.config.feature, self.run_feature_generation),
            ("expression", self.config.expression, self.run_expression_generation),
            ("ml", self.config.ml_training, self.run_ml_training),
            ("rl", self.config.rl_training, self.run_rl_training),
            ("backtest", self.config.backtest, self.run_backtest),
        ]
        for name, cfg, runner in sequence:
            if requested and name not in requested:
                continue
            if cfg:
                logger.info("Executing step: %s", name)
                runner(cfg)
            elif requested:
                logger.warning("Step '%s' requested but no configuration provided", name)

    def run_feature_generation(self, cfg: Dict[str, Any]) -> None:
        script = Path(cfg.get("script", "freqtrade/scripts/freqai_feature_agent.py"))
        args = list(map(str, cfg.get("args", [])))
        cmd = [sys.executable, str(script)] + args
        self._run_command(cmd, cwd=cfg.get("cwd"))

    def run_download(self, cfg: Dict[str, Any]) -> None:
        # Build a freqtrade download-data command
        cfg = dict(cfg)
        cmd: List[str]
        # Prefer console script, otherwise use module
        prefix = self._conda_prefix()
        if prefix:
            base = ['freqtrade', 'download-data']
        else:
            base = [sys.executable, '-m', 'freqtrade', 'download-data']
        cmd = (prefix or []) + base
        config_path = cfg.get('config')
        if not config_path:
            raise ValueError("download.config is required")
        cmd += ['--config', str(config_path)]

        timeframes = cfg.get('timeframes') or ([cfg['timeframe']] if cfg.get('timeframe') else [])
        for tf in timeframes:
            cmd += ['-t', str(tf)]

        pairs_file = cfg.get('pairs_file')
        if pairs_file:
            cmd += ['--pairs-file', str(pairs_file)]
        pairs = cfg.get('pairs') or []
        for pair in pairs:
            cmd += ['-p', str(pair)]

        if cfg.get('exchange'):
            cmd += ['--exchange', str(cfg['exchange'])]
        if cfg.get('timerange'):
            cmd += ['--timerange', str(cfg['timerange'])]
        if 'days' in cfg and cfg['days']:
            cmd += ['--days', str(int(cfg['days']))]
        if cfg.get('erase'):
            cmd += ['--erase']
        if cfg.get('new_pairs'):
            cmd += ['--new-pairs']
        if cfg.get('dl_trades'):
            cmd += ['--dl-trades']
        cmd += list(map(str, cfg.get('extra_args', [])))
        self._run_command(cmd, cwd=cfg.get("cwd"))

    def run_expression_generation(self, cfg: Dict[str, Any]) -> None:
        script = Path(cfg.get("script", "freqtrade/scripts/freqai_expression_agent.py"))
        args = list(map(str, cfg.get("args", [])))
        feedback_path = Path(cfg.get("feedback_path", self.feedback_path))
        append_feedback = feedback_path.exists() and "--feedback" not in args
        if append_feedback:
            args += ["--feedback", str(feedback_path)]
            if "--feedback-top" not in args:
                args += ["--feedback-top", str(cfg.get("feedback_top", 10))]
            logger.info("Injecting feedback summary for expression generation: %s", feedback_path)
        cmd = [sys.executable, str(script)] + args
        self._run_command(cmd, cwd=cfg.get("cwd"))

    def run_ml_training(self, cfg: Dict[str, Any]) -> None:
        configs = cfg.get("configs")
        single = cfg.get("config")
        if configs and single:
            raise ValueError("ml_training.config ? configs ??????")
        if configs:
            if not isinstance(configs, list):
                raise ValueError("ml_training.configs ???????")
            job_list = [item for item in configs if isinstance(item, dict)]
            if len(job_list) != len(configs):
                raise ValueError("ml_training.configs ????????")
        else:
            if not isinstance(single, dict):
                raise ValueError("ml_training.config must be provided")
            job_list = [single]
        total = len(job_list)
        for idx, job_cfg in enumerate(job_list, start=1):
            model_name = job_cfg.get('model', {}).get('name', 'unknown')
            logger.info(
                "Starting ML training job %s/%s with model=%s",
                idx,
                total,
                model_name,
            )
            TrainingPipeline(job_cfg).run()

    def run_rl_training(self, cfg: Dict[str, Any]) -> None:
        # Lazy import to avoid heavy deps (gymnasium) when RL is not used
        from agent_market.freqai.rl.trainer import RLTrainer  # type: ignore
        config = cfg.get("config")
        if not isinstance(config, dict):
            raise ValueError("rl_training.config must be provided")
        logger.info("Starting RL training")
        RLTrainer(config).train()

    def run_backtest(self, cfg: Dict[str, Any]) -> None:
        if "command" in cfg:
            cmd = [str(part) for part in cfg["command"]]
        else:
            binary = cfg.get("binary", "freqtrade")
            cmd = [binary, "backtesting"]
            if cfg.get("config"):
                cmd += ["--config", str(cfg["config"])]
            if cfg.get("strategy"):
                cmd += ["--strategy", str(cfg["strategy"])]
            if cfg.get("strategy_path"):
                cmd += ["--strategy-path", str(cfg["strategy_path"])]
            if cfg.get("timerange"):
                cmd += ["--timerange", str(cfg["timerange"])]
            cmd += list(map(str, cfg.get("extra_args", [])))
        self._run_command(cmd, cwd=cfg.get("cwd"))
        results_dir = Path(cfg.get("results_dir", "freqtrade/user_data/backtest_results"))
        feedback_path = Path(cfg.get("feedback_path", self.feedback_path))
        self._collect_backtest_feedback(results_dir, feedback_path)

    @staticmethod
    def _conda_prefix() -> Optional[List[str]]:
        try:
            out = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
            if out.returncode == 0 and 'freqtrade' in out.stdout:
                return ['conda', 'run', '--no-capture-output', '-n', 'freqtrade']
        except Exception:
            return None
        return None

    @staticmethod
    def _run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
        logger.info("Running command: %s", " ".join(cmd))
        # Prefix with conda run for bare 'freqtrade' binary
        prefix = None
        if cmd and os.path.basename(cmd[0]).lower() == 'freqtrade':
            prefix = AgentFlow._conda_prefix()
        full_cmd = (prefix or []) + cmd
        # 仅确保 src 可导入，避免覆盖 conda 内 freqtrade 包
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        py_paths = [str(root / 'src')]
        env['PYTHONPATH'] = os.pathsep.join(py_paths + [env.get('PYTHONPATH', '')])
        subprocess.run(full_cmd, cwd=cwd, check=True, env=env)

    def _collect_backtest_feedback(self, results_dir: Path, output_path: Path) -> None:
        try:
            latest_zip = self._find_latest_backtest_zip(results_dir)
        except FileNotFoundError as exc:
            logger.warning("Backtest summary skipped: %s", exc)
            return
        if latest_zip is None:
            logger.warning("No backtest archives found in %s", results_dir)
            return
        try:
            summary = self._build_backtest_summary(latest_zip)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to extract backtest summary from %s: %s", latest_zip, exc)
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Backtest summary written to %s", output_path)

    @staticmethod
    def _find_latest_backtest_zip(results_dir: Path) -> Optional[Path]:
        results_dir = results_dir.resolve()
        if not results_dir.exists():
            raise FileNotFoundError(f"Backtest results directory not found: {results_dir}")
        last_path = results_dir / '.last_result.json'
        if last_path.exists():
            try:
                latest_name = json.loads(last_path.read_text(encoding='utf-8')).get('latest_backtest')
                if latest_name:
                    candidate = results_dir / latest_name
                    if candidate.exists():
                        return candidate
            except json.JSONDecodeError:
                logger.warning("Malformed .last_result.json in %s", results_dir)
        zips = list(results_dir.glob('backtest-result-*.zip'))
        if not zips:
            return None
        return max(zips, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _build_backtest_summary(zip_path: Path) -> Dict[str, Any]:
        with zipfile.ZipFile(zip_path) as zf:
            json_members = [name for name in zf.namelist() if name.endswith('.json') and '_config' not in name]
            if not json_members:
                raise ValueError('No result JSON found in archive')
            data = json.loads(zf.read(json_members[0]))
        strategy_block = data.get('strategy', {})
        if not strategy_block:
            raise ValueError('Strategy block missing in backtest results')
        strategy_name, strategy_metrics = next(iter(strategy_block.items()))
        comparison = data.get('strategy_comparison', [])
        return {
            'source': zip_path.name,
            'strategy': strategy_name,
            'profit_total_pct': strategy_metrics.get('profit_total_pct'),
            'profit_total_abs': strategy_metrics.get('profit_total_abs'),
            'trades': strategy_metrics.get('trades'),
            'avg_profit_pct': strategy_metrics.get('profit_mean_pct'),
            'winrate': strategy_metrics.get('winrate'),
            'max_drawdown_abs': strategy_metrics.get('max_drawdown_abs'),
            'best_pair': strategy_metrics.get('best_pair', {}).get('key'),
            'worst_pair': strategy_metrics.get('worst_pair', {}).get('key'),
            'backtest_timerange': f"{strategy_metrics.get('backtest_start')} -> {strategy_metrics.get('backtest_end')}",
            'strategy_comparison': comparison,
        }


__all__ = ["AgentFlow", "AgentFlowConfig", "load_agent_flow_config"]

