"""Startup preflight checks for strategy miner runs."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Optional

from agent_market import paths
from agent_market.agents.executor import OpenAIChatExecutor

from . import research as research_mod
from ._helpers import _load_freqtrade_payload
from .dtypes import MinerConfig, Phase

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check(
    name: str,
    *,
    ok: bool,
    severity: str,
    detail: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "name": str(name),
        "ok": bool(ok),
        "severity": str(severity),
        "detail": str(detail),
    }
    if data:
        payload["data"] = dict(data)
    return payload


def _touch_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".agent_market_touch_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "writable"
    except Exception as exc:
        return False, str(exc)


def _check_writable_dir(name: str, path: Path) -> dict[str, Any]:
    ok, detail = _touch_directory(path)
    severity = "info" if ok else "error"
    suffix = "touch ok" if ok else f"touch failed: {detail}"
    return _check(name, ok=ok, severity=severity, detail=f"{path} — {suffix}", data={"path": str(path.resolve())})


def _normalize_base_url(raw: str) -> str:
    return OpenAIChatExecutor._normalize_base_url(raw or "")


def _request_json(url: str, *, api_key: str, timeout: float = 5.0) -> tuple[bool, Any, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
        return True, json.loads(body), ""
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return False, None, f"http_{exc.code}: {detail[:300]}"
    except urllib.error.URLError as exc:
        return False, None, f"url_error: {exc.reason}"
    except Exception as exc:
        return False, None, str(exc)


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

    if not model:
        return _check(
            "llm.openai_compatible",
            ok=False,
            severity="error",
            detail="Missing model for openai_compatible provider",
        )

    models_url = _normalize_base_url(base_url).rstrip("/") + "/models"
    api_key_source = "environment"
    if not api_key:
        ok, payload, err = _request_json(models_url, api_key="_")
        if ok:
            api_key = "_"
            os.environ.setdefault("LLM_API_KEY", api_key)
            applied_env["LLM_API_KEY"] = api_key
            api_key_source = "preflight_dummy"
            models = payload.get("data") if isinstance(payload, dict) else []
            model_ids = [str(item.get("id") or "") for item in models if isinstance(item, dict)]
            listed = model in model_ids
            detail = f"models probe ok via placeholder key; model={'listed' if listed else 'not_listed'}"
            severity = "info" if listed else "warning"
            return _check(
                "llm.openai_compatible",
                ok=True,
                severity=severity,
                detail=detail,
                data={"base_url": base_url, "model": model, "api_key_source": api_key_source},
            )
        return _check(
            "llm.openai_compatible",
            ok=False,
            severity="error",
            detail=f"Missing api_key and placeholder probe failed at {models_url}: {err}",
            data={"base_url": base_url, "model": model},
        )

    ok, payload, err = _request_json(models_url, api_key=api_key)
    if not ok:
        severity = "warning" if err.startswith("http_404") else "error"
        return _check(
            "llm.openai_compatible",
            ok=severity != "error",
            severity=severity,
            detail=f"models probe failed at {models_url}: {err}",
            data={"base_url": base_url, "model": model, "api_key_source": api_key_source},
        )

    models = payload.get("data") if isinstance(payload, dict) else []
    model_ids = [str(item.get("id") or "") for item in models if isinstance(item, dict)]
    listed = model in model_ids
    detail = "models probe ok"
    severity = "info"
    if model_ids and not listed:
        detail = f"models probe ok but configured model `{model}` not listed"
        severity = "warning"
    return _check(
        "llm.openai_compatible",
        ok=True,
        severity=severity,
        detail=detail,
        data={"base_url": base_url, "model": model, "api_key_source": api_key_source},
    )


def _check_provider(config: MinerConfig, applied_env: dict[str, str]) -> list[dict[str, Any]]:
    provider = str(config.provider or "auto").strip().lower() or "auto"
    if provider in ("openai", "openai_compatible"):
        return [_check_openai_compatible(config, applied_env)]
    if provider == "opencode":
        import shutil

        binary = shutil.which("opencode")
        model = str(config.model or os.environ.get("OPENCODE_MODEL") or "").strip()
        agent_url = str(config.base_url or os.environ.get("OPENCODE_URL") or "").strip()
        ok = bool(model and (agent_url or binary))
        severity = "info" if ok else "error"
        detail = "opencode ready" if ok else "Missing OpenCode model or CLI/agent URL"
        return [
            _check(
                "llm.opencode",
                ok=ok,
                severity=severity,
                detail=detail,
                data={"model": model, "agent_url": agent_url, "binary": binary or ""},
            )
        ]

    import shutil

    binary = shutil.which("opencode")
    model = str(config.model or os.environ.get("OPENCODE_MODEL") or "").strip()
    opencode_ok = bool(model and (str(config.base_url or os.environ.get("OPENCODE_URL") or "").strip() or binary))
    checks: list[dict[str, Any]] = [
        _check(
            "llm.opencode",
            ok=opencode_ok,
            severity="info" if opencode_ok else "warning",
            detail="opencode ready" if opencode_ok else "OpenCode unavailable; will require openai_compatible fallback",
            data={"model": model, "binary": binary or ""},
        )
    ]
    openai_check = _check_openai_compatible(config, applied_env)
    if opencode_ok and not openai_check["ok"]:
        openai_check["severity"] = "warning"
    checks.append(openai_check)
    if not opencode_ok and not openai_check["ok"]:
        checks.append(
            _check(
                "llm.auto",
                ok=False,
                severity="error",
                detail="Neither opencode nor openai_compatible provider is ready",
            )
        )
    return checks


def _check_opencli() -> dict[str, Any]:
    ok = bool(research_mod._opencli_available())
    detail = "opencli ready" if ok else "opencli unavailable; web research will be disabled"
    data = {
        "command": list(research_mod._OPENCLI_COMMAND or []),
        "path_prefix": str(research_mod._OPENCLI_PATH_PREFIX or ""),
    }
    return _check("system.opencli", ok=ok, severity="info" if ok else "warning", detail=detail, data=data)


def _check_freqtrade_cli() -> dict[str, Any]:
    wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
    cmd = [sys.executable, str(wrapper), "--version"]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(paths.REPO_ROOT),
        )
    except Exception as exc:
        return _check(
            "system.freqtrade",
            ok=False,
            severity="error",
            detail=f"freqtrade wrapper probe failed: {exc}",
            data={"command": cmd},
        )

    if proc.returncode != 0:
        blob = (proc.stderr or proc.stdout or "").strip()
        return _check(
            "system.freqtrade",
            ok=False,
            severity="error",
            detail=f"freqtrade unavailable: {blob[:300]}",
            data={"command": cmd, "returncode": proc.returncode},
        )

    detail = (proc.stdout or proc.stderr or "freqtrade ok").strip().splitlines()
    version = detail[0] if detail else "freqtrade ok"
    return _check(
        "system.freqtrade",
        ok=True,
        severity="info",
        detail=version,
        data={"command": cmd},
    )


def _check_freqtrade_config(config: MinerConfig) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    ft_path = paths.resolve_repo_path(config.freqtrade_config)
    if not ft_path.exists():
        return [
            _check(
                "config.freqtrade",
                ok=False,
                severity="error",
                detail=f"freqtrade_config not found: {ft_path}",
                data={"path": str(ft_path)},
            )
        ]

    payload = _load_freqtrade_payload(str(ft_path))
    pairs = ((payload.get("exchange") or {}).get("pair_whitelist")) or []
    checks.append(
        _check(
            "config.freqtrade",
            ok=bool(isinstance(pairs, list) and pairs),
            severity="info" if isinstance(pairs, list) and pairs else "error",
            detail=f"loaded {ft_path}" if isinstance(pairs, list) and pairs else "pair_whitelist is empty",
            data={"path": str(ft_path), "pair_count": len(pairs) if isinstance(pairs, list) else 0},
        )
    )

    datadir = paths.resolve_repo_path(str(payload.get("datadir") or "user_data/data"))
    checks.append(_check_writable_dir("config.freqtrade.datadir", datadir))
    return checks


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
