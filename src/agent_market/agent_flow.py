from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_market.utils import sha256_bytes
from agent_market.flow_ext import steps as flow_steps
from agent_market.flow_ext.step_spec import STEP_CONFIG_FIELDS, STEP_ORDER
from agent_market.flow_ext.step_dispatch import STEP_HANDLERS, StepContext
from agent_market.run_artifacts import RunArtifacts
from agent_market import paths

logger = logging.getLogger(__name__)
REPO_ROOT = paths.REPO_ROOT


def run_agent_flow_preflight(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    from agent_market.runtime_preflight import run_agent_flow_preflight as _run_agent_flow_preflight

    return _run_agent_flow_preflight(*args, **kwargs)


def _relpath(path: Path) -> str:
    return paths.relpath_for_meta(path if path.is_absolute() else (REPO_ROOT / path))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _extract_flag_value(args: Any, flag: str) -> Optional[str]:
    if not isinstance(args, list):
        return None
    value: Optional[str] = None
    for idx, item in enumerate(args):
        if str(item) == flag and idx + 1 < len(args):
            value = str(args[idx + 1])
    return value


def _config_snapshot_info(cfg: "AgentFlowConfig", cfg_path: Optional[Path]) -> Dict[str, Any]:
    if cfg_path is not None:
        try:
            p = cfg_path
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            payload = p.read_bytes()
            return {"path": _relpath(p), "sha256": sha256_bytes(payload), "source": "file"}
        except Exception as exc:
            logger.warning("Failed to hash config file (%s): %s", cfg_path, exc)

    try:
        snapshot = {
            "capture": cfg.capture,
            "lob_rebuild": cfg.lob_rebuild,
            "feature": cfg.feature,
            "micro_feature": cfg.micro_feature,
            "portfolio": cfg.portfolio,
            "expression": cfg.expression,
            "factor_compile": cfg.factor_compile,
            "factor_eval": cfg.factor_eval,
            "ml_training": cfg.ml_training,
            "rl_training": cfg.rl_training,
            "backtest": cfg.backtest,
            "tca": cfg.tca,
            "report": cfg.report,
            "strategy_miner": cfg.strategy_miner,
            "experiment": cfg.experiment,
        }
        payload = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {"path": None, "sha256": sha256_bytes(payload), "source": "object"}
    except Exception as exc:  # pragma: no cover
        return {"path": None, "sha256": None, "source": "error", "error": str(exc)}


@dataclass
class AgentFlowConfig:
    capture: Optional[Dict[str, Any]] = None
    lob_rebuild: Optional[Dict[str, Any]] = None
    feature: Optional[Dict[str, Any]] = None
    micro_feature: Optional[Dict[str, Any]] = None
    portfolio: Optional[Dict[str, Any]] = None
    expression: Optional[Dict[str, Any]] = None
    factor_compile: Optional[Dict[str, Any]] = None
    factor_eval: Optional[Dict[str, Any]] = None
    ml_training: Optional[Dict[str, Any]] = None
    rl_training: Optional[Dict[str, Any]] = None
    backtest: Optional[Dict[str, Any]] = None
    tca: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    strategy_miner: Optional[Dict[str, Any]] = None
    experiment: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFlowConfig":
        known_keys = {
            "capture",
            "lob_rebuild",
            "feature",
            "micro_feature",
            "portfolio",
            "expression",
            "factor_compile",
            "factor_eval",
            "ml_training",
            "rl_training",
            "backtest",
            "tca",
            "report",
            "strategy_miner",
            "experiment",
        }
        extra = set(data.keys()) - known_keys
        if extra:
            logger.warning(
                "AgentFlowConfig received unknown keys: %s", ", ".join(sorted(extra))
            )
        return cls(
            capture=data.get("capture"),
            lob_rebuild=data.get("lob_rebuild"),
            feature=data.get("feature"),
            micro_feature=data.get("micro_feature"),
            portfolio=data.get("portfolio"),
            expression=data.get("expression"),
            factor_compile=data.get("factor_compile"),
            factor_eval=data.get("factor_eval"),
            ml_training=data.get("ml_training"),
            rl_training=data.get("rl_training"),
            backtest=data.get("backtest"),
            tca=data.get("tca"),
            report=data.get("report"),
            strategy_miner=data.get("strategy_miner"),
            experiment=data.get("experiment"),
        )


def load_agent_flow_config(path: Path) -> AgentFlowConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse config JSON: {exc}") from exc
    return AgentFlowConfig.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Market end-to-end orchestrator")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Optional subset of steps to run (feature, expression, ml, rl, backtest)",
    )
    parser.add_argument(
        "--log-dir",
        default="user_data/agent_logs",
        help="Directory to store agent flow log files",
    )
    return parser


def _configure_cli_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_flow_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger().info("Agent Flow log file: %s", log_file)
    return log_file


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_cli_logging(paths.resolve_repo_path(args.log_dir))

    cfg_path = paths.resolve_repo_path(args.config)
    cfg = load_agent_flow_config(cfg_path)
    flow = AgentFlow(cfg, config_path=cfg_path)
    flow.run(args.steps)
    return 0


class AgentFlow:
    STEP_ORDER = list(STEP_ORDER)

    def __init__(
        self,
        config: AgentFlowConfig,
        config_path: Path | str | None = None,
        feedback_path: Path | str | None = None,
    ):
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None
        self.feedback_path = paths.resolve_repo_path(
            feedback_path if feedback_path is not None else paths.default_feedback_path()
        )
        self._last_preflight_report: Optional[Dict[str, Any]] = None

    def run(self, steps: Optional[List[str]] = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        meta_latest_path = paths.run_meta_latest_path()
        meta_run_path = paths.run_meta_path(run_id)
        run_dir = meta_run_path.parent

        requested: Optional[list[str]] = None
        if steps:
            requested = [str(step).lower() for step in steps]
            unknown = sorted({step for step in requested if step not in self.STEP_ORDER})
            if unknown:
                logger.warning("Ignoring unknown steps: %s", ", ".join(unknown))
            requested = [step for step in requested if step in self.STEP_ORDER]

        sequence: list[tuple[str, Optional[Dict[str, Any]]]] = [
            (name, getattr(self.config, field)) for name, field in STEP_CONFIG_FIELDS
        ]

        logger.info("[FLOW] RUN_ID %s", run_id)

        arts = RunArtifacts()
        ctx = StepContext(
            run_id=run_id, run_dir=run_dir,
            feedback_path=self.feedback_path, full_config=self.config,
            config_path=self.config_path,
        )
        self._last_preflight_report = run_agent_flow_preflight(
            self.config,
            run_dir=run_dir,
            requested_steps=requested,
            feedback_path=self.feedback_path,
        )
        status = "success"
        error_info: Optional[dict[str, Any]] = None
        steps_meta: list[dict[str, Any]] = []
        meta_write_error: Optional[BaseException] = None
        flow_exception: Optional[BaseException] = None

        try:
            for name, cfg in sequence:
                if requested and name not in requested:
                    continue
                if cfg is None:
                    if requested:
                        logger.warning("Step '%s' requested but no configuration provided", name)
                    continue

                handler = STEP_HANDLERS.get(name)
                if handler is None:  # pragma: no cover
                    raise ValueError(f"Unknown step: {name}")

                step_started = datetime.now(timezone.utc).isoformat()
                logger.info("[FLOW] STEP_START %s", name)
                logger.info("[FLOW] PHASE %s prepare", name)
                try:
                    logger.info("[FLOW] PHASE %s execute", name)
                    handler(cfg, arts, ctx)
                except Exception as exc:
                    logger.error("[FLOW] STEP_FAIL %s: %s", name, exc)
                    status = "failed"
                    error_info = {
                        "step": name,
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    steps_meta.append({
                        "name": name, "status": "failed",
                        "started_at": step_started,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    })
                    raise
                else:
                    logger.info("[FLOW] PHASE %s summarize", name)
                    logger.info("[FLOW] STEP_OK %s", name)
                    steps_meta.append({
                        "name": name, "status": "ok",
                        "started_at": step_started,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                    })
        except BaseException as exc:  # pragma: no cover
            flow_exception = exc
            if status == "success":
                status = "failed"
                error_info = {
                    "step": None,
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            raise
        finally:
            try:
                try:
                    from agent_market.strategy_factory import finalize_strategy_factory_artifacts  # noqa: WPS433

                    extra_artifacts = finalize_strategy_factory_artifacts(
                        run_id=run_id,
                        run_dir=run_dir,
                        experiment_cfg=dict(self.config.experiment or {}),
                        status=status,
                        started_at=started_at,
                        steps_meta=steps_meta,
                        error_info=error_info,
                        arts=arts,
                    )
                    arts.experiment_registry = extra_artifacts.get("experiment_registry")
                    arts.budget_plan_json = extra_artifacts.get("budget_plan_json")
                    arts.replay_manifest_json = extra_artifacts.get("replay_manifest_json")
                    arts.lineage_graph_json = extra_artifacts.get("lineage_graph_json")
                    arts.promotion_chain_json = extra_artifacts.get("promotion_chain_json")
                    arts.resource_dashboard_json = extra_artifacts.get("resource_dashboard_json")
                except Exception as exc:
                    logger.error("[FLOW] STRATEGY_FACTORY_FINALIZE_FAIL %s: %s", run_id, exc)
                meta = self._build_run_meta(
                    run_id, started_at, status, requested, steps_meta,
                    error_info, arts, meta_latest_path, meta_run_path,
                )
                _write_json_atomic(meta_latest_path, meta)
                _write_json_atomic(meta_run_path, meta)
                logger.info("[FLOW] META_OK %s", _relpath(meta_latest_path))
            except BaseException as exc:  # pragma: no cover
                meta_write_error = exc
                logger.error("[FLOW] META_FAIL %s: %s", run_id, exc)

            if flow_exception is None and status == "success" and meta_write_error is not None:
                raise RuntimeError(
                    f"Failed to write run metadata: {meta_write_error}"
                ) from meta_write_error
        return run_id

    # ------------------------------------------------------------------
    # Metadata assembly (extracted from run())
    # ------------------------------------------------------------------
    def _build_run_meta(
        self,
        run_id: str,
        started_at: str,
        status: str,
        requested: Optional[List[str]],
        steps_meta: List[Dict[str, Any]],
        error_info: Optional[Dict[str, Any]],
        arts: RunArtifacts,
        meta_latest_path: Path,
        meta_run_path: Path,
    ) -> Dict[str, Any]:
        cfg_info = _config_snapshot_info(self.config, self.config_path)

        # Preserve any paths captured by step handlers (they may point to run-local copies).
        if self.config.feature and not arts.feature_output:
            feature_out = _extract_flag_value(self.config.feature.get("args"), "--output")
            if feature_out:
                arts.feature_output = _relpath(paths.resolve_repo_path(feature_out))
        if self.config.expression and not arts.expression_output:
            expr_out = _extract_flag_value(self.config.expression.get("args"), "--output")
            if expr_out:
                arts.expression_output = _relpath(paths.resolve_repo_path(expr_out))

        model_dirs: list[str] = []
        if self.config.ml_training:
            jobs_list = self.config.ml_training.get("configs") or []
            if isinstance(jobs_list, list):
                for job in jobs_list:
                    if not isinstance(job, dict):
                        continue
                    output_cfg = job.get("output") or {}
                    model_cfg = (job.get("model") or {}).get("params") or {}
                    model_dir = output_cfg.get("model_dir") or model_cfg.get("model_dir")
                    if model_dir:
                        model_dirs.append(str(model_dir))

        results_dir = None
        if self.config.backtest:
            results_dir = str(
                self.config.backtest.get("results_dir")
                or str(paths.user_data_root() / "backtest_results")
            )
        # Avoid scanning global artifacts: record only the artifacts produced by this run.
        training_summaries: list[str] = [arts.training_summary_json] if arts.training_summary_json else []
        bt_zip = arts.backtest_zip_run or arts.backtest_zip
        bt_zips: list[str] = [bt_zip] if bt_zip else []
        feedback_summary = arts.feedback_summary_json or _relpath(self.feedback_path)

        return {
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "requested_steps": requested,
            "config": cfg_info,
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "freqtrade": flow_steps.get_freqtrade_version(),
            "preflight": {
                "ok": bool((self._last_preflight_report or {}).get("ok")),
                "warnings": int((self._last_preflight_report or {}).get("warnings") or 0),
                "errors": int((self._last_preflight_report or {}).get("errors") or 0),
                "applied_env": dict((self._last_preflight_report or {}).get("applied_env") or {}),
                "report": _relpath((meta_run_path.parent / "preflight.json").resolve()),
            },
            "artifacts": arts.to_dict(
                feedback_summary=feedback_summary,
                model_dirs=model_dirs,
                training_summaries=training_summaries,
                backtest_results_dir=results_dir,
                backtest_zips=bt_zips,
            ),
            "steps": steps_meta,
            "error": error_info,
            "paths": {
                "run_meta_latest": _relpath(meta_latest_path),
                "run_meta": _relpath(meta_run_path),
            },
        }


__all__ = [
    "AgentFlow",
    "AgentFlowConfig",
    "build_parser",
    "load_agent_flow_config",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
