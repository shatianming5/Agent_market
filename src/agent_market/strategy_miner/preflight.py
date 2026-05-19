"""Startup preflight checks for strategy miner runs."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from agent_market import paths
from agent_market.runtime_preflight import (
    _check,
    _iso_now,
    check_freqtrade_config as _shared_check_freqtrade_config,
    check_freqtrade_cli as _check_freqtrade_cli,
    check_openai_compatible as _shared_check_openai_compatible,
    check_opencli as _check_opencli,
    check_opencode_ready as _check_opencode_ready,
    check_writable_dir as _check_writable_dir,
)

from .dtypes import MinerConfig, Phase

logger = logging.getLogger(__name__)


def _check_openai_compatible(config: MinerConfig, applied_env: dict[str, str]) -> dict[str, Any]:
    model = str(config.model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "").strip()
    base_url = str(
        config.base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    ).strip()
    api_key = str(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    return _shared_check_openai_compatible(
        model=model,
        base_url=base_url,
        api_key=api_key,
        applied_env=applied_env,
        env_key_name="LLM_API_KEY",
    )


def _check_provider(config: MinerConfig, applied_env: dict[str, str]) -> list[dict[str, Any]]:
    provider = str(config.provider or "auto").strip().lower() or "auto"
    if provider in ("openai", "openai_compatible"):
        return [_check_openai_compatible(config, applied_env)]

    model = str(config.model or os.environ.get("OPENCODE_MODEL") or "").strip()
    agent_url = str(config.base_url or os.environ.get("OPENCODE_URL") or "").strip()
    opencode_check = _check_opencode_ready(
        name="llm.opencode",
        model=model,
        agent_url=agent_url,
        require_model=True,
        unavailable_severity="error" if provider == "opencode" else "warning",
    )
    if provider == "opencode":
        return [opencode_check]

    checks: list[dict[str, Any]] = [opencode_check]
    openai_check = _check_openai_compatible(config, applied_env)
    if opencode_check["ok"] and not openai_check["ok"]:
        openai_check["severity"] = "warning"
    checks.append(openai_check)
    if not opencode_check["ok"] and not openai_check["ok"]:
        checks.append(
            _check(
                "llm.auto",
                ok=False,
                severity="error",
                detail="Neither opencode nor openai_compatible provider is ready",
            )
        )
    return checks


def _check_freqtrade_config(config: MinerConfig) -> list[dict[str, Any]]:
    return _shared_check_freqtrade_config(config.freqtrade_config)


def _log_check(item: dict[str, Any]) -> None:
    text = f"[preflight] {item.get('name')}: {item.get('detail')}"
    severity = str(item.get("severity") or "info").lower()
    if severity == "error":
        logger.error(text)
    elif severity == "warning":
        logger.warning(text)
    else:
        logger.info(text)


def run_startup_preflight(
    config: MinerConfig,
    *,
    miner_dir: Path,
    phase: Optional[Phase] = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Run startup checks before an active miner phase begins."""
    current_phase = phase.value if isinstance(phase, Phase) else str(phase or "")
    if current_phase == Phase.COMPLETE.value:
        miner_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": _iso_now(),
            "ok": True,
            "skipped": True,
            "phase": current_phase,
            "checks": [],
            "errors": 0,
            "warnings": 0,
            "applied_env": {},
        }
        (miner_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    applied_env: dict[str, str] = {}
    checks: list[dict[str, Any]] = []

    for path_name, path_value in (
        ("path.artifacts_root", paths.artifacts_root()),
        ("path.user_data_root", paths.user_data_root()),
        ("path.runs_root", paths.runs_root()),
        ("path.models_root", paths.models_root()),
        ("path.control_plane_root", paths.control_plane_root()),
        ("path.miner_dir", miner_dir),
        ("path.model_output_root", paths.resolve_repo_path(config.model_output_root)),
    ):
        checks.append(_check_writable_dir(path_name, path_value))

    if bool(getattr(config, "use_global_memory", True)):
        global_kb_path_raw = str(getattr(config, "global_strategy_knowledge_base_path", "") or "").strip()
        global_kb_path = (
            paths.resolve_repo_path(global_kb_path_raw)
            if global_kb_path_raw
            else paths.global_strategy_knowledge_base_path()
        )
        checks.append(_check_writable_dir("path.global_strategy_memory", global_kb_path.parent))

    checks.extend(_check_provider(config, applied_env))
    checks.append(_check_opencli())
    checks.append(_check_freqtrade_cli())
    checks.extend(_check_freqtrade_config(config))

    errors = sum(1 for item in checks if str(item.get("severity")).lower() == "error" and not item.get("ok"))
    warnings = sum(1 for item in checks if str(item.get("severity")).lower() == "warning")
    report = {
        "timestamp": _iso_now(),
        "ok": errors == 0,
        "skipped": False,
        "phase": current_phase,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "applied_env": applied_env,
    }

    miner_dir.mkdir(parents=True, exist_ok=True)
    (miner_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in checks:
        _log_check(item)

    if errors and raise_on_error:
        failed = [f"{item['name']}: {item['detail']}" for item in checks if str(item.get("severity")).lower() == "error" and not item.get("ok")]
        raise RuntimeError("Startup preflight failed: " + "; ".join(failed))
    return report


__all__ = ["run_startup_preflight"]
