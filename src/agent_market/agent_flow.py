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

from agent_market import flow_steps

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
            "feature": cfg.feature,
            "portfolio": cfg.portfolio,
            "expression": cfg.expression,
            "ml_training": cfg.ml_training,
            "rl_training": cfg.rl_training,
            "backtest": cfg.backtest,
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
    feature: Optional[Dict[str, Any]] = None
    portfolio: Optional[Dict[str, Any]] = None
    expression: Optional[Dict[str, Any]] = None
    ml_training: Optional[Dict[str, Any]] = None
    rl_training: Optional[Dict[str, Any]] = None
    backtest: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentFlowConfig":
        known_keys = {"feature", "portfolio", "expression", "ml_training", "rl_training", "backtest"}
        extra = set(data.keys()) - known_keys
        if extra:
            logger.warning(
                "AgentFlowConfig received unknown keys: %s", ", ".join(sorted(extra))
            )
        return cls(
            feature=data.get("feature"),
            portfolio=data.get("portfolio"),
            expression=data.get("expression"),
            ml_training=data.get("ml_training"),
            rl_training=data.get("rl_training"),
            backtest=data.get("backtest"),
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
    STEP_ORDER = ["feature", "portfolio", "expression", "ml", "rl", "backtest"]

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
            ("feature", self.config.feature),
            ("portfolio", self.config.portfolio),
            ("expression", self.config.expression),
            ("ml", self.config.ml_training),
            ("rl", self.config.rl_training),
            ("backtest", self.config.backtest),
        ]

        logger.info("[FLOW] RUN_ID %s", run_id)

        status = "success"
        error_info: Optional[dict[str, Any]] = None
        steps_meta: list[dict[str, Any]] = []
        portfolio_weights: Optional[str] = None
        portfolio_report: Optional[str] = None
        portfolio_returns: Optional[str] = None
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
                    elif name == "ml":
                        flow_steps.run_ml_training(cfg)
                    elif name == "rl":
                        flow_steps.run_rl_training(cfg)
                    elif name == "backtest":
                        flow_steps.run_backtest(cfg, self.feedback_path)
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
                        "portfolio_weights": portfolio_weights,
                        "portfolio_report": portfolio_report,
                        "portfolio_returns": portfolio_returns,
                        "expression_output": expr_out,
                        "feedback_summary": _relpath(self.feedback_path),
                        "model_dirs": model_dirs,
                        "training_summaries": [_relpath(p) for p in training_summaries],
                        "backtest_results_dir": results_dir,
                        "backtest_zips": [_relpath(p) for p in bt_zips],
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
