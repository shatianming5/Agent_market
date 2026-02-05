from __future__ import annotations

import hashlib
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

from agent_market.flow_ext import steps as flow_steps

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relpath(path: Path) -> str:
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path.resolve())


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
            return {"path": _relpath(p), "sha256": _sha256_bytes(payload), "source": "file"}
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
        }
        payload = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {"path": None, "sha256": _sha256_bytes(payload), "source": "object"}
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
        )


def load_agent_flow_config(path: Path) -> AgentFlowConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse config JSON: {exc}") from exc
    return AgentFlowConfig.from_dict(payload)


class AgentFlow:
    STEP_ORDER = [
        "capture",
        "lob_rebuild",
        "feature",
        "micro_feature",
        "portfolio",
        "expression",
        "factor_compile",
        "factor_eval",
        "ml",
        "rl",
        "backtest",
        "tca",
        "report",
    ]

    def __init__(
        self,
        config: AgentFlowConfig,
        config_path: Path | str | None = None,
        feedback_path: Path | str = "user_data/llm_feedback/latest_backtest_summary.json",
    ):
        self.config = config
        self.config_path = Path(config_path) if config_path is not None else None
        self.feedback_path = Path(feedback_path)

    def run(self, steps: Optional[List[str]] = None) -> None:
        run_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        meta_latest_path = (REPO_ROOT / "artifacts" / "run_meta.json").resolve()
        meta_run_path = (REPO_ROOT / "artifacts" / "runs" / run_id / "run_meta.json").resolve()
        run_dir = meta_run_path.parent

        requested: Optional[list[str]] = None
        if steps:
            requested = [str(step).lower() for step in steps]
            unknown = sorted({step for step in requested if step not in self.STEP_ORDER})
            if unknown:
                logger.warning("Ignoring unknown steps: %s", ", ".join(unknown))
            requested = [step for step in requested if step in self.STEP_ORDER]

        sequence: list[tuple[str, Optional[Dict[str, Any]]]] = [
            ("capture", self.config.capture),
            ("lob_rebuild", self.config.lob_rebuild),
            ("feature", self.config.feature),
            ("micro_feature", self.config.micro_feature),
            ("portfolio", self.config.portfolio),
            ("expression", self.config.expression),
            ("factor_compile", self.config.factor_compile),
            ("factor_eval", self.config.factor_eval),
            ("ml", self.config.ml_training),
            ("rl", self.config.rl_training),
            ("backtest", self.config.backtest),
            ("tca", self.config.tca),
            ("report", self.config.report),
        ]

        logger.info("[FLOW] RUN_ID %s", run_id)

        status = "success"
        error_info: Optional[dict[str, Any]] = None
        steps_meta: list[dict[str, Any]] = []
        portfolio_weights: Optional[str] = None
        portfolio_report: Optional[str] = None
        portfolio_returns: Optional[str] = None
        micro_features_parquet: Optional[str] = None
        micro_features_manifest: Optional[str] = None
        capture_manifest: Optional[str] = None
        capture_match_path: Optional[str] = None
        capture_level2_path: Optional[str] = None
        lob_state_parquet: Optional[str] = None
        rebuild_report: Optional[str] = None
        factor_spec_json: Optional[str] = None
        factor_ast_json: Optional[str] = None
        factor_expression_txt: Optional[str] = None
        factor_expression_json: Optional[str] = None
        factor_eval_meta: Optional[str] = None
        factor_scores_json: Optional[str] = None
        factor_pareto_csv: Optional[str] = None
        tca_report: Optional[str] = None
        tca_html: Optional[str] = None
        compiled_expression_path: Optional[Path] = None
        capture_dir_path: Optional[Path] = None
        bundle_zip: Optional[str] = None
        bundle_manifest: Optional[str] = None
        meta_write_error: Optional[BaseException] = None
        flow_exception: Optional[BaseException] = None

        try:
            for name, cfg in sequence:
                if requested and name not in requested:
                    continue
                if not cfg:
                    if requested:
                        logger.warning(
                            "Step '%s' requested but no configuration provided", name
                        )
                    continue

                step_started = datetime.now(timezone.utc).isoformat()
                logger.info("[FLOW] STEP_START %s", name)
                logger.info("[FLOW] PHASE %s prepare", name)
                try:
                    logger.info("[FLOW] PHASE %s execute", name)
                    if name == "feature":
                        flow_steps.run_feature_generation(cfg)
                    elif name == "capture":
                        out = flow_steps.run_capture(
                            cfg,
                            out_dir=(run_dir / "capture").resolve(),
                        )
                        capture_dir_path = Path(out["capture_dir"]).resolve()
                        capture_manifest = _relpath(Path(out["manifest_json"]))
                        capture_match_path = _relpath(Path(out["match_path"]))
                        capture_level2_path = _relpath(Path(out["level2_path"]))
                    elif name == "lob_rebuild":
                        cap_dir_cfg = cfg.get("capture_dir") if isinstance(cfg, dict) else None
                        cap_dir = (
                            (REPO_ROOT / str(cap_dir_cfg)).resolve()
                            if cap_dir_cfg and not Path(str(cap_dir_cfg)).is_absolute()
                            else (Path(str(cap_dir_cfg)).resolve() if cap_dir_cfg else capture_dir_path)
                        )
                        if cap_dir is None:
                            raise ValueError("lob_rebuild requires capture_dir (or run capture step first)")
                        out = flow_steps.run_lob_rebuild(
                            cfg,
                            capture_dir=cap_dir,
                            out_dir=(run_dir / "lob_rebuild").resolve(),
                        )
                        lob_state_parquet = _relpath(Path(out["lob_state_parquet"]))
                        rebuild_report = _relpath(Path(out["rebuild_report_json"]))
                    elif name == "micro_feature":
                        mf_cfg = dict(cfg)
                        if not mf_cfg.get("config") and self.config.backtest:
                            mf_cfg["config"] = self.config.backtest.get("config")
                        out = flow_steps.run_micro_feature(
                            mf_cfg,
                            run_id=run_id,
                            out_dir=(run_dir / "micro_feature").resolve(),
                        )
                        micro_features_parquet = _relpath(Path(out["features_parquet"]))
                        micro_features_manifest = _relpath(Path(out["manifest_json"]))
                    elif name == "portfolio":
                        from agent_market.portfolio_opt import (  # noqa: WPS433
                            compute_returns,
                            load_prices_from_feather,
                            optimize_hrp,
                        )

                        ex = str(cfg.get("exchange") or "").strip()
                        pairs = cfg.get("pairs") or []
                        if isinstance(pairs, str):
                            pairs = [p for p in pairs.split(",") if p.strip()]
                        if not isinstance(pairs, list) or not pairs:
                            raise ValueError("portfolio.pairs must be a non-empty list")

                        timeframe = str(cfg.get("timeframe") or "1h").strip()
                        timerange = cfg.get("timerange") or (self.config.backtest or {}).get("timerange")
                        data_dir = cfg.get("data_dir") or "user_data/data"
                        returns_kind = str(cfg.get("returns") or "log").strip().lower()

                        prices = load_prices_from_feather(
                            REPO_ROOT,
                            exchange=ex,
                            pairs=[str(p) for p in pairs],
                            timeframe=timeframe,
                            timerange=str(timerange) if timerange else None,
                            data_dir=str(data_dir),
                        )
                        returns = compute_returns(prices, returns_kind)
                        result = optimize_hrp(returns)

                        out_dir = (run_dir / "portfolio").resolve()
                        out_dir.mkdir(parents=True, exist_ok=True)
                        weights_path = out_dir / "weights.json"
                        report_path = out_dir / "report.json"
                        returns_path = out_dir / "returns.parquet"

                        weights_payload = {
                            "method": "hrp",
                            "weights": result.get("weights") or {},
                        }
                        weights_path.write_text(
                            json.dumps(weights_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        try:
                            returns.to_parquet(returns_path, index=True)
                        except Exception:
                            returns_path = None  # type: ignore[assignment]

                        report = {
                            "method": "hrp",
                            "weights": result.get("weights") or {},
                            "stats": result.get("stats") or {},
                            "inputs": {
                                "exchange": ex,
                                "pairs": [str(p) for p in pairs],
                                "timeframe": timeframe,
                                "timerange": str(timerange) if timerange else None,
                                "returns_kind": returns_kind,
                                "data_dir": str(data_dir),
                            },
                        }
                        report_path.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                        portfolio_weights = _relpath(weights_path)
                        portfolio_report = _relpath(report_path)
                        if returns_path is not None:
                            portfolio_returns = _relpath(returns_path)
                    elif name == "expression":
                        flow_steps.run_expression_generation(cfg, self.feedback_path)
                    elif name == "factor_compile":
                        out = flow_steps.run_factor_compile(
                            cfg,
                            run_id=run_id,
                            out_dir=(run_dir / "factor_compile").resolve(),
                        )
                        factor_spec_json = _relpath(Path(out["factor_spec_json"]))
                        factor_ast_json = _relpath(Path(out["factor_ast_json"]))
                        factor_expression_txt = _relpath(Path(out["compiled_expression_txt"]))
                        factor_expression_json = _relpath(Path(out["compiled_expression_json"]))
                        compiled_expression_path = Path(out["compiled_expression_txt"]).resolve()
                    elif name == "factor_eval":
                        out = flow_steps.run_factor_eval(
                            cfg,
                            run_id=run_id,
                            out_dir=(run_dir / "factor_eval").resolve(),
                            compiled_expression_path=compiled_expression_path,
                        )
                        factor_eval_meta = _relpath(Path(out["factor_eval_meta"]))
                        factor_scores_json = _relpath(Path(out["factor_scores_json"]))
                        factor_pareto_csv = _relpath(Path(out["pareto_csv"]))
                    elif name == "ml":
                        flow_steps.run_ml_training(cfg)
                    elif name == "rl":
                        flow_steps.run_rl_training(cfg)
                    elif name == "backtest":
                        flow_steps.run_backtest(cfg, self.feedback_path)
                    elif name == "tca":
                        out = flow_steps.run_tca(
                            cfg,
                            run_id=run_id,
                            out_dir=(run_dir / "tca").resolve(),
                        )
                        tca_report = _relpath(Path(out["tca_report"]))
                        if out.get("tca_html"):
                            tca_html = _relpath(Path(out["tca_html"]))
                    elif name == "report":
                        artifacts = {
                            "capture_manifest": capture_manifest,
                            "capture_match_path": capture_match_path,
                            "capture_level2_path": capture_level2_path,
                            "lob_state_parquet": lob_state_parquet,
                            "rebuild_report": rebuild_report,
                            "micro_feature_parquet": micro_features_parquet,
                            "micro_feature_manifest": micro_features_manifest,
                            "factor_spec_json": factor_spec_json,
                            "factor_ast_json": factor_ast_json,
                            "factor_expression_txt": factor_expression_txt,
                            "factor_scores_json": factor_scores_json,
                            "factor_pareto_csv": factor_pareto_csv,
                            "tca_report": tca_report,
                            "tca_html": tca_html,
                            "feedback_summary": _relpath(self.feedback_path),
                            "config_path": _relpath(self.config_path) if self.config_path else None,
                        }
                        out = flow_steps.run_report_bundle(
                            cfg,
                            run_id=run_id,
                            out_dir=(run_dir / "bundle").resolve(),
                            artifacts=artifacts,
                        )
                        bundle_zip = _relpath(Path(out["bundle_zip"]))
                        bundle_manifest = _relpath(Path(out["bundle_manifest"]))
                    else:  # pragma: no cover
                        raise ValueError(f"Unknown step: {name}")
                except Exception as exc:
                    logger.error("[FLOW] STEP_FAIL %s: %s", name, exc)
                    status = "failed"
                    error_info = {
                        "step": name,
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    steps_meta.append(
                        {
                            "name": name,
                            "status": "failed",
                            "started_at": step_started,
                            "ended_at": datetime.now(timezone.utc).isoformat(),
                            "error": {"type": exc.__class__.__name__, "message": str(exc)},
                        }
                    )
                    raise
                else:
                    logger.info("[FLOW] PHASE %s summarize", name)
                    logger.info("[FLOW] STEP_OK %s", name)
                    steps_meta.append(
                        {
                            "name": name,
                            "status": "ok",
                            "started_at": step_started,
                            "ended_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
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
                cfg_info = _config_snapshot_info(self.config, self.config_path)

                feature_out = None
                if self.config.feature:
                    feature_out = _extract_flag_value(
                        self.config.feature.get("args"), "--output"
                    )
                expr_out = None
                if self.config.expression:
                    expr_out = _extract_flag_value(
                        self.config.expression.get("args"), "--output"
                    )

                model_dirs: list[str] = []
                if self.config.ml_training:
                    jobs = self.config.ml_training.get("configs") or []
                    if isinstance(jobs, list):
                        for job in jobs:
                            if not isinstance(job, dict):
                                continue
                            output_cfg = job.get("output") or {}
                            model_cfg = (job.get("model") or {}).get("params") or {}
                            model_dir = output_cfg.get("model_dir") or model_cfg.get(
                                "model_dir"
                            )
                            if model_dir:
                                model_dirs.append(str(model_dir))

                results_dir = None
                if self.config.backtest:
                    results_dir = str(
                        self.config.backtest.get("results_dir")
                        or "user_data/backtest_results"
                    )

                models_root = (REPO_ROOT / "artifacts" / "models").resolve()
                training_summaries = (
                    sorted(models_root.rglob("training_summary.json"))
                    if models_root.exists()
                    else []
                )
                bt_dir = (REPO_ROOT / (results_dir or "user_data/backtest_results")).resolve()
                bt_zips = (
                    sorted(bt_dir.glob("backtest-result-*.zip")) if bt_dir.exists() else []
                )

                ended_at = datetime.now(timezone.utc).isoformat()
                meta = {
                    "run_id": run_id,
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "requested_steps": requested,
                    "config": cfg_info,
                    "python": {
                        "version": sys.version,
                        "executable": sys.executable,
                        "platform": platform.platform(),
                    },
                    "freqtrade": flow_steps.get_freqtrade_version(),
                    "artifacts": {
                        "feature_output": feature_out,
                        "micro_feature_parquet": micro_features_parquet,
                        "micro_feature_manifest": micro_features_manifest,
                        "capture_manifest": capture_manifest,
                        "capture_match_path": capture_match_path,
                        "capture_level2_path": capture_level2_path,
                        "lob_state_parquet": lob_state_parquet,
                        "rebuild_report": rebuild_report,
                        "portfolio_weights": portfolio_weights,
                        "portfolio_report": portfolio_report,
                        "portfolio_returns": portfolio_returns,
                        "expression_output": expr_out,
                        "factor_spec_json": factor_spec_json,
                        "factor_ast_json": factor_ast_json,
                        "factor_expression_txt": factor_expression_txt,
                        "factor_expression_json": factor_expression_json,
                        "factor_eval_meta": factor_eval_meta,
                        "factor_scores_json": factor_scores_json,
                        "factor_pareto_csv": factor_pareto_csv,
                        "feedback_summary": _relpath(self.feedback_path),
                        "model_dirs": model_dirs,
                        "training_summaries": [_relpath(p) for p in training_summaries],
                        "backtest_results_dir": results_dir,
                        "backtest_zips": [_relpath(p) for p in bt_zips],
                        "tca_report": tca_report,
                        "tca_html": tca_html,
                        "bundle_zip": bundle_zip,
                        "bundle_manifest": bundle_manifest,
                    },
                    "steps": steps_meta,
                    "error": error_info,
                    "paths": {
                        "run_meta_latest": _relpath(meta_latest_path),
                        "run_meta": _relpath(meta_run_path),
                    },
                }
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


__all__ = ["AgentFlow", "AgentFlowConfig", "load_agent_flow_config"]
