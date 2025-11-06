from __future__ import annotations

import json
import logging
import subprocess
import os
import sys
import zipfile
import uuid
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from agent_market.freqai.model import gradient_boosting  # noqa: F401
from agent_market.freqai.training.pipeline import TrainingPipeline
from agent_market.runtime.concurrency import (
    CancellationToken,
    RateLimiter,
    RetryPolicy,
    TaskCancelledError,
    TaskResult,
    TaskScheduler,
    TaskSpec,
)
from agent_market.paths import (
    PROJECT_ROOT,
    APP_ROOT,
    SRC_ROOT,
    USER_DATA_DIR,
    RUNS_DIR,
    FREQTRADE_ROOT,
    FREQTRADE_SCRIPTS,
)

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
        self.feedback_path = (USER_DATA_DIR / "llm_feedback" / "latest_backtest_summary.json")
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._token: Optional[CancellationToken] = None
        self._runs_base = Path(os.environ.get("AGENT_MARKET_RUNS_DIR", RUNS_DIR))
        self._run_ctx: Optional[Dict[str, Any]] = None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        try:
            evt = dict(payload)
            if self._run_ctx:
                evt.setdefault("run_id", self._run_ctx.get("id"))
            text = json.dumps(evt, ensure_ascii=False)
            print(text, flush=True)
            if self._run_ctx:
                events_path = self._run_ctx.get("events_path")
                if events_path:
                    events_path.parent.mkdir(parents=True, exist_ok=True)
                    if not events_path.exists():
                        events_path.touch()
                    with events_path.open("a", encoding="utf-8") as fh:
                        fh.write(text + "\n")
        except Exception:
            logger.debug("Failed to emit event payload: %s", payload, exc_info=True)

    def _emit_progress(self, status: str, current: int, total: int, label: str) -> None:
        pct = int(min(100, (current / total * 100))) if total else (100 if status in {"completed", "finished"} else 0)
        payload = {
            "event": "progress",
            "status": status,
            "current": max(0, int(current)),
            "total": max(0, int(total)),
            "label": label,
            "percent": pct,
            "ts": self._now_iso(),
        }
        self._emit_event(payload)

    def _prepare_run(self) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        run_dir = self._runs_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = run_dir / "config.snapshot.json"
        try:
            cfg_dict = asdict(self.config)
        except TypeError:
            cfg_dict = {
                "download": self.config.download,
                "feature": self.config.feature,
                "expression": self.config.expression,
                "ml_training": self.config.ml_training,
                "rl_training": self.config.rl_training,
                "backtest": self.config.backtest,
            }
        snapshot_path.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        events_path = run_dir / "events.jsonl"
        events_path.touch(exist_ok=True)
        return {"id": run_id, "dir": run_dir, "config_snapshot": snapshot_path, "events_path": events_path}

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            root = PROJECT_ROOT
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=root,
                check=True,
            )
            commit = out.stdout.strip()
            return commit or None
        except Exception:
            return None

    def _write_manifest(
        self,
        run_ctx: Dict[str, Any],
        *,
        started_at: str,
        finished_at: str,
        status: str,
        steps: List[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        run_dir = Path(run_ctx["dir"])
        cfg_snapshot = run_ctx.get("config_snapshot")
        events_path = run_ctx.get("events_path")
        if cfg_snapshot:
            try:
                cfg_snapshot_rel = Path(cfg_snapshot).relative_to(run_dir)
            except ValueError:
                cfg_snapshot_rel = Path(cfg_snapshot)
        else:
            cfg_snapshot_rel = None
        if events_path:
            try:
                events_rel = Path(events_path).relative_to(run_dir)
            except ValueError:
                events_rel = Path(events_path)
        else:
            events_rel = None

        manifest = {
            "run_id": run_ctx.get("id"),
            "run_dir": str(run_dir),
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "error": error,
            "steps": steps,
            "config_snapshot": str(cfg_snapshot_rel) if cfg_snapshot_rel is not None else None,
            "events_log": str(events_rel) if events_rel is not None else None,
            "artifacts": {
                "feedback_summary": str(self.feedback_path) if self.feedback_path.exists() else None,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": self._git_commit(),
            },
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, steps: Optional[List[str]] = None) -> None:
        requested = None
        if steps:
            requested = [step.lower() for step in steps]
            unknown = [step for step in requested if step not in self.STEP_ORDER]
            if unknown:
                logger.warning("Ignoring unknown steps: %s", ", ".join(sorted(unknown)))
            requested = [step for step in requested if step in self.STEP_ORDER]

        sequence = [
            ("download", self.config.download, self.run_download),
            ("feature", self.config.feature, self.run_feature_generation),
            ("expression", self.config.expression, self.run_expression_generation),
            ("ml", self.config.ml_training, self.run_ml_training),
            ("rl", self.config.rl_training, self.run_rl_training),
            ("backtest", self.config.backtest, self.run_backtest),
        ]
        planned = [(n, c, r) for (n, c, r) in sequence if c and ((not requested) or (n in requested))]
        total = len(planned) if planned else 0
        current = 0
        self._token = CancellationToken()
        started_at = self._now_iso()
        steps_meta: List[Dict[str, Any]] = []
        status_label = "completed"
        error_detail: Optional[str] = None
        run_ctx = self._prepare_run()
        self._run_ctx = run_ctx
        try:
            if total:
                self._emit_progress("queued", 0, total, "start")
            for name, cfg, runner in planned:
                self._check_cancelled()
                current += 1
                step_meta: Dict[str, Any] = {
                    "name": name,
                    "status": "pending",
                    "started_at": self._now_iso(),
                }
                steps_meta.append(step_meta)
                self._emit_progress("running", current, total, name)
                try:
                    if requested and name not in requested:
                        step_meta["status"] = "skipped"
                        step_meta["finished_at"] = self._now_iso()
                        self._emit_progress("skipped", current, total, name)
                        continue
                    if cfg:
                        logger.info("Executing step: %s", name)
                        runner(cfg)
                        step_meta["status"] = "completed"
                    else:
                        step_meta["status"] = "skipped"
                        self._emit_progress("skipped", current, total, name)
                except TaskCancelledError as exc:
                    step_meta["status"] = "cancelled"
                    step_meta["error"] = str(exc)
                    status_label = "cancelled"
                    error_detail = str(exc)
                    self._emit_progress("cancelled", current, total, name)
                    raise
                except Exception as exc:
                    step_meta["status"] = "failed"
                    step_meta["error"] = str(exc)
                    status_label = "failed"
                    error_detail = str(exc)
                    self._emit_progress("failed", current, total, name)
                    raise
                finally:
                    step_meta.setdefault("finished_at", self._now_iso())
                if step_meta.get("status") == "completed":
                    self._emit_progress("completed", current, total, name)
            if total:
                self._emit_progress("completed", total, total, "complete")
                self._emit_progress("finished", total, total, "finished")
        finally:
            finished_at = self._now_iso()
            try:
                self._write_manifest(
                    run_ctx,
                    started_at=started_at,
                    finished_at=finished_at,
                    status=status_label,
                    steps=steps_meta,
                    error=error_detail,
                )
            finally:
                self._run_ctx = None
                self._token = None

    def cancel(self) -> None:
        if self._token:
            self._token.cancel()

    def _check_cancelled(self) -> None:
        if self._token:
            self._token.throw_if_cancelled()

    def _resolve_project_path(self, value: Any) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        text = str(path).replace("\\", "/")
        if text.startswith("freqtrade/") or text.startswith("scripts/"):
            path = Path("app") / path
        elif text.startswith("configs/") or text.startswith("conf/"):
            path = Path("config") / path
        elif any(text.startswith(prefix) for prefix in ("data/", "artifacts/", "backtests/")):
            path = Path("resources") / path
        return (PROJECT_ROOT / path).resolve()

    def _resolve_step_options(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        runtime_cfg = cfg.get("runtime") or {}
        concurrency = int(runtime_cfg.get("concurrency") or cfg.get("concurrency") or 1)
        if concurrency <= 0:
            concurrency = 1
        fail_fast = bool(runtime_cfg.get("fail_fast", cfg.get("fail_fast", True)))
        return {
            "concurrency": concurrency,
            "fail_fast": fail_fast,
            "retry": runtime_cfg.get("retry") or cfg.get("retry"),
            "rate_limit": runtime_cfg.get("rate_limit") or cfg.get("rate_limit"),
            "timeout": runtime_cfg.get("timeout") or cfg.get("timeout"),
        }

    @staticmethod
    def _build_retry_policy(cfg: Optional[Dict[str, Any]], default_attempts: int = 1) -> RetryPolicy:
        return RetryPolicy.from_config(cfg, default_attempts=default_attempts)

    def _ensure_rate_limiter(self, key: Optional[str], cfg: Optional[Dict[str, Any]]) -> Optional[str]:
        if not key or not cfg:
            return None
        interval = cfg.get("interval") or cfg.get("seconds")
        if interval is None and cfg.get("requests") and cfg.get("per_seconds"):
            interval = float(cfg["per_seconds"]) / float(cfg["requests"])
        if interval is None and cfg.get("requests") and cfg.get("per"):
            interval = float(cfg["per"]) / float(cfg["requests"])
        if interval is None:
            return None
        interval = float(interval)
        if interval <= 0:
            return None
        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimiter(interval)
        return key

    def _execute_tasks(
        self,
        tasks: List[TaskSpec],
        *,
        concurrency: int = 1,
        fail_fast: bool = True,
        default_retry: Optional[RetryPolicy] = None,
    ) -> List[TaskResult]:
        scheduler = TaskScheduler(
            max_workers=concurrency,
            rate_limiters=self._rate_limiters,
            default_retry=default_retry or RetryPolicy(attempts=1),
        )
        token = self._token or CancellationToken()
        results = scheduler.run(tasks, cancellation_token=token, fail_fast=fail_fast)
        for res in results:
            status = "succeeded" if res.success else "failed"
            logger.info(
                "Task %s %s in %.2fs after %s attempt(s)",
                res.name,
                status,
                res.duration,
                res.attempts,
            )
        failures = [res for res in results if not res.success]
        if failures:
            first = failures[0]
            if first.error:
                raise first.error
            raise RuntimeError(f"Task '{first.name}' failed without error context")
        return results

    @staticmethod
    def _run_ml_job(job_cfg: Dict[str, Any]) -> Any:
        return TrainingPipeline(job_cfg).run()

    @staticmethod
    def _run_rl_job(config: Dict[str, Any]) -> Any:
        from agent_market.freqai.rl.trainer import RLTrainer  # type: ignore

        return RLTrainer(config).train()

    def _invoke_command(self, cmd: List[str], *, cwd: Optional[str] = None, timeout: Optional[float] = None) -> None:
        self._check_cancelled()
        self._run_command(cmd, cwd=cwd, timeout=timeout)


    def _execute_command(self, label: str, cmd: List[str], cfg: Dict[str, Any], options: Dict[str, Any]) -> None:
        rate_cfg = options.get('rate_limit')
        rate_key = None
        if rate_cfg:
            rate_key = rate_cfg.get('key') or f'{label}:{cmd[0]}'
            rate_key = self._ensure_rate_limiter(rate_key, rate_cfg)
        retry_policy = self._build_retry_policy(options.get('retry'), default_attempts=1)
        task = TaskSpec(
            name=f'{label}:{cmd[0]}',
            func=self._invoke_command,
            args=(cmd,),
            kwargs={'cwd': cfg.get('cwd'), 'timeout': options.get('timeout')},
            retry=retry_policy,
            rate_limit=rate_key,
        )
        self._execute_tasks([task], concurrency=1, fail_fast=options.get('fail_fast', True), default_retry=retry_policy)

    def run_feature_generation(self, cfg: Dict[str, Any]) -> None:
        default_script = FREQTRADE_SCRIPTS / "freqai_feature_agent.py"
        raw_script = cfg.get("script", default_script)
        script = self._resolve_project_path(raw_script)
        args = list(map(str, cfg.get("args", [])))
        cmd = [sys.executable, str(script)] + args
        options = self._resolve_step_options(cfg)
        self._execute_command("feature", cmd, cfg, options)

    def run_download(self, cfg: Dict[str, Any]) -> None:
        cfg_local = dict(cfg)
        use_conda = cfg_local.get('use_conda', True)
        prefix = self._conda_prefix() if use_conda else None
        if prefix:
            base = ['freqtrade', 'download-data']
        else:
            base = [sys.executable, '-m', 'freqtrade', 'download-data']
        cmd = (prefix or []) + base
        config_path = cfg_local.get('config')
        if not config_path:
            raise ValueError('download.config is required')
        cmd += ['--config', str(self._resolve_project_path(config_path))]
        timeframes = cfg_local.get('timeframes') or ([cfg_local['timeframe']] if cfg_local.get('timeframe') else [])
        for tf in timeframes:
            cmd += ['-t', str(tf)]
        pairs_file = cfg_local.get('pairs_file')
        if pairs_file:
            cmd += ['--pairs-file', str(self._resolve_project_path(pairs_file))]
        pairs = cfg_local.get('pairs') or []
        for pair in pairs:
            cmd += ['-p', str(pair)]
        if cfg_local.get('exchange'):
            cmd += ['--exchange', str(cfg_local['exchange'])]
        if cfg_local.get('timerange'):
            cmd += ['--timerange', str(cfg_local['timerange'])]
        if 'days' in cfg_local and cfg_local['days']:
            cmd += ['--days', str(int(cfg_local['days']))]
        if cfg_local.get('erase'):
            cmd += ['--erase']
        if cfg_local.get('new_pairs'):
            cmd += ['--new-pairs']
        if cfg_local.get('dl_trades'):
            cmd += ['--dl-trades']
        if cfg_local.get('prepend'):
            cmd += ['--prepend']
        cmd += list(map(str, cfg_local.get('extra_args', [])))
        options = self._resolve_step_options(cfg_local)
        self._execute_command('download', cmd, cfg_local, options)

    def run_expression_generation(self, cfg: Dict[str, Any]) -> None:
        wrapper = (APP_ROOT / 'scripts' / 'expr_agent_wrapper.py')
        default_script = FREQTRADE_SCRIPTS / 'freqai_expression_agent.py'
        if cfg.get('script'):
            script = self._resolve_project_path(cfg['script'])
        else:
            script = wrapper if wrapper.exists() else default_script
            script = Path(script)
        args = list(map(str, cfg.get('args', [])))

        def _resolve_arg_value(flag: str, args_list: list[str]) -> None:
            if flag in args_list:
                idx = args_list.index(flag)
                if idx + 1 < len(args_list):
                    value = Path(args_list[idx + 1])
                    if not value.is_absolute():
                        value = self._resolve_project_path(value)
                    args_list[idx + 1] = str(value)

        _resolve_arg_value('--feature-file', args)
        _resolve_arg_value('--output', args)

        feedback_path = Path(cfg.get('feedback_path', self.feedback_path))
        if not feedback_path.is_absolute():
            feedback_path = self._resolve_project_path(feedback_path)
        append_feedback = feedback_path.exists() and '--feedback' not in args
        if append_feedback:
            args += ['--feedback', str(feedback_path)]
            if '--feedback-top' not in args:
                args += ['--feedback-top', str(cfg.get('feedback_top', 10))]
            logger.info('Injecting feedback summary for expression generation: %s', feedback_path)
        cmd = [sys.executable, str(script)] + args
        options = self._resolve_step_options(cfg)
        self._execute_command('expression', cmd, cfg, options)

    def run_ml_training(self, cfg: Dict[str, Any]) -> None:
        configs = cfg.get('configs')
        single = cfg.get('config')
        if configs and single:
            raise ValueError('ml_training.config and configs cannot be provided together')
        if configs:
            if not isinstance(configs, list):
                raise ValueError('ml_training.configs must be a list of dicts')
            job_list = [item for item in configs if isinstance(item, dict)]
            if len(job_list) != len(configs):
                raise ValueError('ml_training.configs must only contain objects')
        else:
            if not isinstance(single, dict):
                raise ValueError('ml_training.config must be provided')
            job_list = [single]
        if not job_list:
            logger.warning('No ML training jobs scheduled')
            return
        options = self._resolve_step_options(cfg)
        base_retry = self._build_retry_policy(options.get('retry'), default_attempts=1)
        rate_cfg = options.get('rate_limit')
        rate_key = None
        if rate_cfg:
            rate_key = rate_cfg.get('key') or 'ml-training'
            rate_key = self._ensure_rate_limiter(rate_key, rate_cfg)
        tasks: List[TaskSpec] = []
        total = len(job_list)
        for idx, job_cfg in enumerate(job_list, start=1):
            model_name = job_cfg.get('model', {}).get('name', 'unknown')
            logger.info('Queueing ML training job %s/%s with model=%s', idx, total, model_name)
            task_retry = self._build_retry_policy(job_cfg.get('retry'), default_attempts=base_retry.attempts)
            tasks.append(TaskSpec(
                name=f'ml_training:{model_name}:{idx}',
                func=self._run_ml_job,
                args=(job_cfg,),
                retry=task_retry,
                rate_limit=rate_key,
            ))
        self._execute_tasks(
            tasks,
            concurrency=options['concurrency'],
            fail_fast=options['fail_fast'],
            default_retry=base_retry,
        )

    def run_rl_training(self, cfg: Dict[str, Any]) -> None:
        config = cfg.get('config')
        if not isinstance(config, dict):
            raise ValueError('rl_training.config must be provided')
        logger.info('Queueing RL training job')
        options = self._resolve_step_options(cfg)
        base_retry = self._build_retry_policy(options.get('retry'), default_attempts=1)
        rate_cfg = options.get('rate_limit')
        rate_key = None
        if rate_cfg:
            rate_key = rate_cfg.get('key') or 'rl-training'
            rate_key = self._ensure_rate_limiter(rate_key, rate_cfg)
        task = TaskSpec(
            name='rl_training',
            func=self._run_rl_job,
            args=(config,),
            retry=base_retry,
            rate_limit=rate_key,
        )
        self._execute_tasks([task], concurrency=1, fail_fast=options['fail_fast'], default_retry=base_retry)

    def run_backtest(self, cfg: Dict[str, Any]) -> None:
        if 'command' in cfg:
            cmd = [str(part) for part in cfg['command']]
        else:
            binary = cfg.get('binary', 'freqtrade')
            cmd = [binary, 'backtesting']
            if cfg.get('config'):
                cmd += ['--config', str(self._resolve_project_path(cfg['config']))]
            if cfg.get('strategy'):
                cmd += ['--strategy', str(cfg['strategy'])]
            if cfg.get('strategy_path'):
                cmd += ['--strategy-path', str(self._resolve_project_path(cfg['strategy_path']))]
            if cfg.get('timerange'):
                cmd += ['--timerange', str(cfg['timerange'])]
            cmd += list(map(str, cfg.get('extra_args', [])))
        options = self._resolve_step_options(cfg)
        env_overrides = cfg.get('env') or {}
        old_env: Dict[str, Optional[str]] = {}
        try:
            for key, value in env_overrides.items():
                old_env[key] = os.environ.get(key)
                resolved = self._resolve_project_path(Path(value))
                os.environ[key] = str(resolved)
            self._execute_command('backtest', cmd, cfg, options)
        finally:
            for key, old in old_env.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
        default_results = USER_DATA_DIR / 'backtest_results'
        results_dir = Path(cfg.get('results_dir', default_results))
        if not results_dir.is_absolute():
            results_dir = self._resolve_project_path(results_dir)
        feedback_path = Path(cfg.get('feedback_path', self.feedback_path))
        if not feedback_path.is_absolute():
            feedback_path = self._resolve_project_path(feedback_path)
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
    def _run_command(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[float] = None) -> None:
        logger.info('Running command: %s', ' '.join(cmd))
        prefix = None
        if cmd and os.path.basename(cmd[0]).lower() == 'freqtrade':
            prefix = AgentFlow._conda_prefix()
        full_cmd = (prefix or []) + cmd
        env = os.environ.copy()
        py_paths = [str(SRC_ROOT)]
        env['PYTHONPATH'] = os.pathsep.join(py_paths + [env.get('PYTHONPATH', '')])
        subprocess.run(full_cmd, cwd=cwd, check=True, env=env, timeout=timeout)

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

