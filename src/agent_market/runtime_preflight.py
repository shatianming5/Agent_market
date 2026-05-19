"""Shared startup preflight checks for runtime entrypoints."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent_market import paths

logger = logging.getLogger(__name__)

_HOME_BIN_DIRS = (
    Path.home() / ".local" / "bin",
    Path.home() / "bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
)


def load_project_dotenv() -> Optional[Path]:
    path = paths.REPO_ROOT / ".env"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in raw.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.lower().startswith("export "):
            item = item[7:].strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1] == "'") or (value[0] == value[-1] == '"')):
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return path


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


def _log_check(item: dict[str, Any]) -> None:
    message = f"[preflight] {item.get('name')}: {item.get('detail')}"
    severity = str(item.get("severity") or "info").lower()
    if severity == "error":
        logger.error(message)
    elif severity == "warning":
        logger.warning(message)
    else:
        logger.info(message)


def _touch_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".agent_market_touch_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "writable"
    except Exception as exc:
        return False, str(exc)


def check_writable_dir(name: str, path: Path) -> dict[str, Any]:
    ok, detail = _touch_directory(path)
    severity = "info" if ok else "error"
    suffix = "touch ok" if ok else f"touch failed: {detail}"
    return _check(
        name,
        ok=ok,
        severity=severity,
        detail=f"{path} — {suffix}",
        data={"path": str(path.resolve())},
    )


def check_existing_file(name: str, path: Path, *, severity: str = "error") -> dict[str, Any]:
    ok = path.exists() and path.is_file()
    return _check(
        name,
        ok=ok,
        severity="info" if ok else severity,
        detail=f"found {path}" if ok else f"missing file: {path}",
        data={"path": str(path.resolve())},
    )


def check_existing_dir(name: str, path: Path, *, severity: str = "error") -> dict[str, Any]:
    ok = path.exists() and path.is_dir()
    return _check(
        name,
        ok=ok,
        severity="info" if ok else severity,
        detail=f"found {path}" if ok else f"missing directory: {path}",
        data={"path": str(path.resolve())},
    )


def check_nonempty_value(name: str, value: Any, *, detail: str) -> dict[str, Any]:
    ok = bool(str(value or "").strip())
    return _check(name, ok=ok, severity="info" if ok else "error", detail=detail)


def _normalize_base_url(raw: str) -> str:
    from agent_market.agents.executor import OpenAIChatExecutor

    return OpenAIChatExecutor._normalize_base_url(raw or "")


def normalize_opencode_model(model: str | None, *, default_provider: str = "openai", default_model: str = "gpt-5.2") -> str:
    raw = str(model or "").strip()
    if not raw:
        raw = default_model
    if "/" not in raw:
        return f"{default_provider}/{raw}"
    provider_id, model_id = raw.split("/", 1)
    provider_id = provider_id.strip() or default_provider
    model_id = model_id.strip() or default_model
    return f"{provider_id}/{model_id}"


def split_provider_model(model: str | None, *, default_provider: str = "openai", default_model: str = "gpt-5.2") -> tuple[str, str]:
    normalized = normalize_opencode_model(model, default_provider=default_provider, default_model=default_model)
    provider_id, model_id = normalized.split("/", 1)
    return provider_id, model_id


def resolve_openai_compatible_settings(
    *,
    model: str | None = None,
    default_provider: str = "openai",
    default_model: str = "gpt-5.2",
    default_base_url: str = "https://api.openai.com/v1",
) -> dict[str, str]:
    provider_id, model_id = split_provider_model(
        model,
        default_provider=default_provider,
        default_model=default_model,
    )
    base_url = str(
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or default_base_url
    ).strip()
    api_key = str(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    ).strip()
    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "opencode_model": f"{provider_id}/{model_id}",
        "base_url": base_url,
        "api_key": api_key,
    }


def _request_json(url: str, *, api_key: str, timeout: float = 5.0) -> tuple[bool, Any, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
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


def _resolve_executable(name_or_path: str, *, env_var: str | None = None) -> Optional[str]:
    if env_var:
        configured = str(os.environ.get(env_var) or "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists():
                return str(candidate.resolve())

    path_candidate = Path(name_or_path).expanduser()
    if path_candidate.is_absolute() or any(sep in name_or_path for sep in ("/", "\\")):
        return str(path_candidate.resolve()) if path_candidate.exists() else None

    found = shutil.which(name_or_path)
    if found:
        return found

    sibling = Path(sys.executable).with_name(name_or_path)
    if sibling.exists():
        return str(sibling.resolve())

    for candidate in (
        paths.REPO_ROOT / ".venv" / "bin" / name_or_path,
        paths.REPO_ROOT / ".venv" / "Scripts" / f"{name_or_path}.exe",
        paths.REPO_ROOT / "venv" / "bin" / name_or_path,
        paths.REPO_ROOT / "venv" / "Scripts" / f"{name_or_path}.exe",
        *_HOME_BIN_DIRS,
    ):
        probe = candidate if candidate.name == name_or_path else candidate / name_or_path
        if probe.exists():
            return str(probe.resolve())
    return None


def check_opencli() -> dict[str, Any]:
    from agent_market.strategy_miner import research as research_mod

    ok = bool(research_mod._opencli_available())
    detail = "opencli ready" if ok else "opencli unavailable; web research will be disabled"
    return _check(
        "system.opencli",
        ok=ok,
        severity="info" if ok else "warning",
        detail=detail,
        data={
            "command": list(research_mod._OPENCLI_COMMAND or []),
            "path_prefix": str(research_mod._OPENCLI_PATH_PREFIX or ""),
        },
    )


def check_freqtrade_cli() -> dict[str, Any]:
    wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
    command = [sys.executable, str(wrapper), "--version"]
    try:
        proc = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(paths.REPO_ROOT),
            check=False,
        )
    except Exception as exc:
        return _check(
            "system.freqtrade",
            ok=False,
            severity="error",
            detail=f"freqtrade wrapper probe failed: {exc}",
            data={"command": command},
        )

    if proc.returncode != 0:
        blob = (proc.stderr or proc.stdout or "").strip()
        return _check(
            "system.freqtrade",
            ok=False,
            severity="error",
            detail=f"freqtrade unavailable: {blob[:300]}",
            data={"command": command, "returncode": proc.returncode},
        )

    version = (proc.stdout or proc.stderr or "freqtrade ok").strip().splitlines()
    return _check(
        "system.freqtrade",
        ok=True,
        severity="info",
        detail=version[0] if version else "freqtrade ok",
        data={"command": command},
    )


def check_opencode_cli(*, model: Optional[str] = None) -> dict[str, Any]:
    binary = _resolve_executable("opencode")
    if not binary:
        return _check(
            "system.opencode",
            ok=False,
            severity="error",
            detail="opencode unavailable",
            data={"model": str(model or "")},
        )
    detail = f"opencode ready: {binary}"
    data: dict[str, Any] = {"binary": binary, "model": str(model or "")}
    try:
        proc = subprocess.run(  # noqa: S603
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = (proc.stdout or proc.stderr or "").strip().splitlines()
        if proc.returncode == 0 and version:
            detail = version[0]
            data["version"] = version[0]
    except Exception as exc:
        data["version_probe_error"] = str(exc)
    return _check("system.opencode", ok=True, severity="info", detail=detail, data=data)


def check_python_imports(
    name: str,
    modules: Sequence[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if not modules:
        return _check(name, ok=True, severity="info", detail="no import targets")
    script = "; ".join(f"import {module}" for module in modules)
    command = [sys.executable, "-c", script]
    try:
        proc = subprocess.run(  # noqa: S603
            command,
            cwd=str(cwd or paths.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env=os.environ.copy(),
        )
    except Exception as exc:
        return _check(
            name,
            ok=False,
            severity="error",
            detail=f"python import probe failed: {exc}",
            data={"modules": list(modules)},
        )

    if proc.returncode != 0:
        blob = (proc.stderr or proc.stdout or "").strip()
        return _check(
            name,
            ok=False,
            severity="error",
            detail=f"import failed: {blob[:300]}",
            data={"modules": list(modules), "returncode": proc.returncode},
        )
    return _check(name, ok=True, severity="info", detail="imports ready", data={"modules": list(modules)})


def check_openai_compatible(
    *,
    model: str,
    base_url: str,
    api_key: Optional[str],
    applied_env: dict[str, str],
    env_key_name: str = "LLM_API_KEY",
) -> dict[str, Any]:
    model_name = str(model or "").strip()
    base = str(base_url or "").strip()
    key = str(api_key or "").strip()

    if not model_name:
        return _check(
            "llm.openai_compatible",
            ok=False,
            severity="error",
            detail="Missing model for openai_compatible provider",
        )
    if not base:
        return _check(
            "llm.openai_compatible",
            ok=False,
            severity="error",
            detail="Missing base_url for openai_compatible provider",
            data={"model": model_name},
        )

    models_url = _normalize_base_url(base).rstrip("/") + "/models"
    api_key_source = "environment"
    if not key:
        ok, payload, err = _request_json(models_url, api_key="_")
        if ok:
            key = "_"
            os.environ.setdefault(env_key_name, key)
            applied_env[env_key_name] = key
            api_key_source = "preflight_dummy"
            model_ids = [
                str(item.get("id") or "")
                for item in (payload.get("data") or [])
                if isinstance(item, dict)
            ] if isinstance(payload, dict) else []
            listed = model_name in model_ids
            detail = f"models probe ok via placeholder key; model={'listed' if listed else 'not_listed'}"
            return _check(
                "llm.openai_compatible",
                ok=True,
                severity="info" if listed else "warning",
                detail=detail,
                data={"base_url": base, "model": model_name, "api_key_source": api_key_source},
            )
        return _check(
            "llm.openai_compatible",
            ok=False,
            severity="error",
            detail=f"Missing api_key and placeholder probe failed at {models_url}: {err}",
            data={"base_url": base, "model": model_name},
        )

    ok, payload, err = _request_json(models_url, api_key=key)
    if not ok:
        severity = "warning" if err.startswith("http_404") else "error"
        return _check(
            "llm.openai_compatible",
            ok=severity != "error",
            severity=severity,
            detail=f"models probe failed at {models_url}: {err}",
            data={"base_url": base, "model": model_name, "api_key_source": api_key_source},
        )

    model_ids = [
        str(item.get("id") or "")
        for item in (payload.get("data") or [])
        if isinstance(item, dict)
    ] if isinstance(payload, dict) else []
    listed = model_name in model_ids if model_ids else True
    detail = "models probe ok" if listed else f"models probe ok but configured model `{model_name}` not listed"
    return _check(
        "llm.openai_compatible",
        ok=True,
        severity="info" if listed else "warning",
        detail=detail,
        data={"base_url": base, "model": model_name, "api_key_source": api_key_source},
    )


def check_freqtrade_config(path_like: str | Path) -> list[dict[str, Any]]:
    config_path = paths.resolve_repo_path(path_like)
    if not config_path.exists():
        return [
            _check(
                "config.freqtrade",
                ok=False,
                severity="error",
                detail=f"freqtrade_config not found: {config_path}",
                data={"path": str(config_path)},
            )
        ]

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return [
            _check(
                "config.freqtrade",
                ok=False,
                severity="error",
                detail=f"freqtrade_config parse failed: {exc}",
                data={"path": str(config_path)},
            )
        ]
    pairs = ((payload.get("exchange") or {}).get("pair_whitelist")) or []
    checks = [
        _check(
            "config.freqtrade",
            ok=bool(isinstance(pairs, list) and pairs),
            severity="info" if isinstance(pairs, list) and pairs else "error",
            detail=f"loaded {config_path}" if isinstance(pairs, list) and pairs else "pair_whitelist is empty",
            data={"path": str(config_path), "pair_count": len(pairs) if isinstance(pairs, list) else 0},
        )
    ]
    datadir = paths.resolve_repo_path(str(payload.get("datadir") or "user_data/data"))
    checks.append(check_writable_dir("config.freqtrade.datadir", datadir))
    return checks


def _extract_flag_value(args: Any, flag: str) -> Optional[str]:
    if not isinstance(args, list):
        return None
    value: Optional[str] = None
    for idx, item in enumerate(args):
        if str(item) == flag and idx + 1 < len(args):
            value = str(args[idx + 1])
    return value


def _has_flag(args: Any, flag: str) -> bool:
    return isinstance(args, list) and flag in [str(item) for item in args]


def _resolve_cfg_path(path_like: str | Path, *, cwd: Optional[str] = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    if cwd:
        return (paths.REPO_ROOT / cwd / path).resolve()
    return paths.resolve_repo_path(path)


def _finalize_report(
    *,
    kind: str,
    report_path: Path,
    checks: Iterable[dict[str, Any]],
    applied_env: Optional[dict[str, str]] = None,
    extra: Optional[dict[str, Any]] = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    items = list(checks)
    warnings = sum(1 for item in items if str(item.get("severity")).lower() == "warning")
    errors = sum(1 for item in items if str(item.get("severity")).lower() == "error")
    report = {
        "schema_version": 1,
        "kind": kind,
        "started_at": _iso_now(),
        "ended_at": _iso_now(),
        "ok": errors == 0,
        "warnings": warnings,
        "errors": errors,
        "applied_env": dict(applied_env or {}),
        "checks": items,
    }
    if extra:
        report.update(extra)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in items:
        _log_check(item)

    if raise_on_error and errors:
        failed = [f"{item.get('name')}: {item.get('detail')}" for item in items if item.get("severity") == "error"]
        raise RuntimeError("Startup preflight failed: " + "; ".join(failed))
    return report


def _selected_flow_steps(config: Any, requested_steps: Optional[Sequence[str]]) -> list[str]:
    if requested_steps:
        return [str(step).lower() for step in requested_steps]
    mapping = {
        "capture": "capture",
        "lob_rebuild": "lob_rebuild",
        "feature": "feature",
        "micro_feature": "micro_feature",
        "portfolio": "portfolio",
        "expression": "expression",
        "factor_compile": "factor_compile",
        "factor_eval": "factor_eval",
        "ml": "ml_training",
        "rl": "rl_training",
        "backtest": "backtest",
        "tca": "tca",
        "strategy_miner": "strategy_miner",
        "report": "report",
    }
    selected: list[str] = []
    for step, attr in mapping.items():
        if getattr(config, attr, None) is not None:
            selected.append(step)
    return selected


def _maybe_check_expression_llm(cfg: dict[str, Any], applied_env: dict[str, str]) -> list[dict[str, Any]]:
    args = list(cfg.get("args") or [])
    if _has_flag(args, "--no-llm"):
        return []

    llm_flags_present = any(
        _has_flag(args, flag)
        for flag in (
            "--llm-enabled",
            "--llm-provider",
            "--llm-base-url",
            "--llm-api-key",
            "--llm-model",
            "--llm-agent-url",
        )
    )
    if not llm_flags_present:
        return []

    provider = str(
        _extract_flag_value(args, "--llm-provider")
        or os.environ.get("LLM_PROVIDER")
        or "openai_compatible"
    ).strip().lower()
    model = str(
        _extract_flag_value(args, "--llm-model")
        or os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or ""
    ).strip()
    base_url = str(
        _extract_flag_value(args, "--llm-base-url")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()
    api_key = str(
        _extract_flag_value(args, "--llm-api-key")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    if provider == "opencode":
        return [check_opencode_cli(model=model)]
    if provider == "auto":
        checks = [check_opencode_cli(model=model)]
        if base_url or model:
            checks.append(
                check_openai_compatible(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    applied_env=applied_env,
                )
            )
        return checks
    return [
        check_openai_compatible(
            model=model,
            base_url=base_url,
            api_key=api_key,
            applied_env=applied_env,
        )
    ]


def _latest_backtest_zip(results_dir: Path) -> Optional[Path]:
    from agent_market.backtest_results import find_latest_backtest_zip  # noqa: WPS433

    return find_latest_backtest_zip(results_dir)


def run_agent_flow_preflight(
    config: Any,
    *,
    run_dir: Path,
    requested_steps: Optional[Sequence[str]] = None,
    feedback_path: Optional[Path] = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    load_project_dotenv()
    selected_steps = _selected_flow_steps(config, requested_steps)
    applied_env: dict[str, str] = {}
    checks: list[dict[str, Any]] = [
        check_writable_dir("paths.artifacts_root", paths.artifacts_root()),
        check_writable_dir("paths.user_data_root", paths.user_data_root()),
        check_writable_dir("paths.runs_root", paths.runs_root()),
        check_writable_dir("paths.models_root", paths.models_root()),
        check_writable_dir("paths.control_plane_root", paths.control_plane_root()),
        check_writable_dir("paths.run_dir", run_dir),
        check_writable_dir("paths.feedback_dir", (feedback_path or paths.default_feedback_path()).parent),
        check_opencli(),
    ]

    requires_freqtrade = any(step in selected_steps for step in ("ml", "backtest", "tca", "strategy_miner"))
    if requires_freqtrade:
        checks.append(check_freqtrade_cli())

    if "expression" in selected_steps and getattr(config, "expression", None):
        checks.extend(_maybe_check_expression_llm(dict(config.expression or {}), applied_env))

    if "capture" in selected_steps and getattr(config, "capture", None) is not None:
        capture_cfg = dict(config.capture or {})
        exchange = str(capture_cfg.get("exchange") or "kucoin").strip().lower()
        channels = [part.strip().lower() for part in str(capture_cfg.get("channels") or "match,level2").replace(",", " ").split() if part.strip()]
        symbols = [part.strip() for part in str(capture_cfg.get("symbols") or "BTC-USDT").replace(",", " ").split() if part.strip()]
        fixture = capture_cfg.get("fixture")
        checks.append(
            _check(
                "capture.exchange",
                ok=exchange == "kucoin",
                severity="info" if exchange == "kucoin" else "error",
                detail=f"exchange={exchange}",
            )
        )
        allowed_channels = {"match", "level2"}
        unknown_channels = sorted(set(channels) - allowed_channels)
        checks.append(
            _check(
                "capture.channels",
                ok=bool(channels) and not unknown_channels,
                severity="info" if channels and not unknown_channels else "error",
                detail="channels ready" if channels and not unknown_channels else f"unsupported channels: {unknown_channels}",
                data={"channels": channels},
            )
        )
        if fixture:
            checks.append(check_existing_file("capture.fixture", _resolve_cfg_path(str(fixture), cwd=capture_cfg.get("cwd"))))
        else:
            checks.append(
                _check(
                    "capture.symbols",
                    ok=bool(symbols),
                    severity="info" if symbols else "error",
                    detail="symbols ready" if symbols else "No symbols provided",
                    data={"symbols": symbols},
                )
            )
        checks.append(check_writable_dir("capture.out_dir", (run_dir / "capture").resolve()))

    if "lob_rebuild" in selected_steps and getattr(config, "lob_rebuild", None) is not None:
        lob_cfg = dict(config.lob_rebuild or {})
        capture_dir_cfg = lob_cfg.get("capture_dir")
        capture_dir = _resolve_cfg_path(str(capture_dir_cfg), cwd=lob_cfg.get("cwd")) if capture_dir_cfg else (run_dir / "capture").resolve()
        snapshot = lob_cfg.get("snapshot")
        if "capture" not in selected_steps or capture_dir_cfg:
            checks.append(check_existing_dir("lob_rebuild.capture_dir", capture_dir))
            checks.append(check_existing_file("lob_rebuild.level2", capture_dir / "level2.ndjson.gz"))
        if snapshot:
            checks.append(check_existing_file("lob_rebuild.snapshot", _resolve_cfg_path(str(snapshot), cwd=lob_cfg.get("cwd"))))
        else:
            checks.append(
                _check(
                    "lob_rebuild.snapshot",
                    ok=False,
                    severity="error",
                    detail="lob_rebuild.snapshot must be provided",
                )
            )
        checks.append(check_nonempty_value("lob_rebuild.symbol", lob_cfg.get("symbol") or "BTC-USDT", detail="symbol ready"))
        checks.append(check_writable_dir("lob_rebuild.out_dir", (run_dir / "lob_rebuild").resolve()))

    if "backtest" in selected_steps and getattr(config, "backtest", None) is not None:
        backtest_cfg = dict(config.backtest or {})
        config_path = backtest_cfg.get("config")
        command = backtest_cfg.get("command") or []
        if not config_path:
            config_path = _extract_flag_value(command, "--config") or _extract_flag_value(command, "-c")
        if config_path:
            checks.extend(check_freqtrade_config(_resolve_cfg_path(str(config_path), cwd=backtest_cfg.get("cwd"))))
        else:
            checks.append(
                _check(
                    "backtest.config",
                    ok=False,
                    severity="error",
                    detail="backtest config path not found",
                )
            )

    if "tca" in selected_steps and getattr(config, "tca", None) is not None:
        tca_cfg = dict(config.tca or {})
        backtest_zip = tca_cfg.get("backtest_zip")
        results_dir = paths.resolve_repo_path(str(tca_cfg.get("results_dir") or "user_data/backtest_results"))
        if backtest_zip:
            checks.append(check_existing_file("tca.backtest_zip", _resolve_cfg_path(str(backtest_zip), cwd=tca_cfg.get("cwd"))))
        elif "backtest" not in selected_steps:
            latest = _latest_backtest_zip(results_dir)
            checks.append(
                _check(
                    "tca.backtest_zip",
                    ok=latest is not None,
                    severity="info" if latest is not None else "error",
                    detail=f"latest zip ready: {latest}" if latest is not None else f"No backtest-result-*.zip found in {results_dir}",
                    data={"results_dir": str(results_dir)},
                )
            )
        checks.append(check_writable_dir("tca.out_dir", (run_dir / "tca").resolve()))

    if "report" in selected_steps:
        checks.append(check_writable_dir("report.out_dir", (run_dir / "bundle").resolve()))

    return _finalize_report(
        kind="agent_flow",
        report_path=(run_dir / "preflight.json").resolve(),
        checks=checks,
        applied_env=applied_env,
        extra={"selected_steps": selected_steps},
        raise_on_error=raise_on_error,
    )


def run_capture_preflight(
    *,
    exchange: str,
    channels: Sequence[str],
    symbols: Sequence[str],
    fixture: Optional[Path],
    out_dir: Path,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    load_project_dotenv()
    allowed_channels = {"match", "level2"}
    unknown_channels = sorted(set(str(item).lower() for item in channels) - allowed_channels)
    checks = [
        check_writable_dir("paths.user_data_root", paths.user_data_root()),
        check_writable_dir("capture.out_dir", out_dir),
        check_opencli(),
        _check(
            "capture.exchange",
            ok=str(exchange).strip().lower() == "kucoin",
            severity="info" if str(exchange).strip().lower() == "kucoin" else "error",
            detail=f"exchange={exchange}",
        ),
        _check(
            "capture.channels",
            ok=bool(channels) and not unknown_channels,
            severity="info" if channels and not unknown_channels else "error",
            detail="channels ready" if channels and not unknown_channels else f"unsupported channels: {unknown_channels}",
            data={"channels": list(channels)},
        ),
    ]
    if fixture is not None:
        checks.append(check_existing_file("capture.fixture", fixture))
    else:
        checks.append(
            _check(
                "capture.symbols",
                ok=bool(list(symbols)),
                severity="info" if list(symbols) else "error",
                detail="symbols ready" if list(symbols) else "No symbols provided",
                data={"symbols": list(symbols)},
            )
        )
    return _finalize_report(
        kind="capture",
        report_path=(out_dir / "preflight.json").resolve(),
        checks=checks,
        raise_on_error=raise_on_error,
    )


def run_lob_rebuild_preflight(
    *,
    capture_dir: Path,
    snapshot: Path,
    symbol: str,
    out_dir: Path,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    load_project_dotenv()
    checks = [
        check_writable_dir("lob_rebuild.out_dir", out_dir),
        check_existing_dir("lob_rebuild.capture_dir", capture_dir),
        check_existing_file("lob_rebuild.level2", capture_dir / "level2.ndjson.gz"),
        check_existing_file("lob_rebuild.snapshot", snapshot),
        check_nonempty_value("lob_rebuild.symbol", symbol, detail="symbol ready"),
        check_opencli(),
    ]
    return _finalize_report(
        kind="lob_rebuild",
        report_path=(out_dir / "preflight.json").resolve(),
        checks=checks,
        raise_on_error=raise_on_error,
    )


def run_tca_preflight(
    *,
    run_id: str,
    backtest_zip: Optional[Path],
    results_dir: Path,
    out_path: Path,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    load_project_dotenv()
    checks = [
        check_writable_dir("tca.out_dir", out_path.parent),
        check_opencli(),
        check_nonempty_value("tca.run_id", run_id, detail="run_id ready"),
    ]
    if backtest_zip is not None:
        checks.append(check_existing_file("tca.backtest_zip", backtest_zip))
    else:
        latest = _latest_backtest_zip(results_dir)
        checks.append(
            _check(
                "tca.backtest_zip",
                ok=latest is not None,
                severity="info" if latest is not None else "error",
                detail=f"latest zip ready: {latest}" if latest is not None else f"No backtest-result-*.zip found in {results_dir}",
                data={"results_dir": str(results_dir.resolve())},
            )
        )
    return _finalize_report(
        kind="tca",
        report_path=(out_path.parent / "preflight.json").resolve(),
        checks=checks,
        raise_on_error=raise_on_error,
    )


def run_ws_production_preflight(
    *,
    workspace: Path,
    model: str,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    load_project_dotenv()
    workspace = workspace.resolve()
    repo_root = workspace.parent
    market_data_dir = repo_root / "user_data" / "data" / "gate"
    market_data_files = sorted(market_data_dir.glob("*-1h.feather")) if market_data_dir.exists() else []
    applied_env: dict[str, str] = {}
    llm_settings = resolve_openai_compatible_settings(model=model)
    checks = [
        check_existing_dir("ws.workspace", workspace),
        check_existing_file("ws.guide", workspace / "GUIDE.md"),
        check_existing_file("ws.meta", workspace / "meta.json"),
        check_existing_file("ws.sop", workspace / "sop.json"),
        check_existing_dir("ws.strategies", workspace / "strategies"),
        check_writable_dir("ws.results", workspace / "results"),
        check_writable_dir("ws.reports", workspace / "reports"),
        check_writable_dir("ws.validation", workspace / "validation"),
        check_writable_dir("ws.backtests", workspace / "backtests"),
        check_writable_dir("ws.paper", workspace / "paper"),
        check_writable_dir("ws.signals", workspace / "signals"),
        _check(
            "ws.market_data",
            ok=bool(market_data_files),
            severity="info" if market_data_files else "error",
            detail=(
                f"{market_data_dir} — {len(market_data_files)} x 1h feather files"
                if market_data_files
                else f"No Gate 1h feather files found under {market_data_dir}"
            ),
            data={"path": str(market_data_dir)},
        ),
        check_opencli(),
        check_freqtrade_cli(),
        check_opencode_cli(model=llm_settings["opencode_model"]),
        check_python_imports(
            "ws.python_imports",
            ["workspace.gate_pipeline", "workspace.continuous_runner"],
            cwd=paths.REPO_ROOT,
        ),
    ]
    if llm_settings["provider_id"] in {"openai", "custom"}:
        checks.append(
            check_openai_compatible(
                model=llm_settings["model_id"],
                base_url=llm_settings["base_url"],
                api_key=llm_settings["api_key"],
                applied_env=applied_env,
                env_key_name="OPENAI_API_KEY",
            )
        )
    return _finalize_report(
        kind="ws_production",
        report_path=(workspace / "results" / "preflight.json").resolve(),
        checks=checks,
        applied_env=applied_env,
        extra={"workspace": str(workspace), "model": llm_settings["opencode_model"]},
        raise_on_error=raise_on_error,
    )


__all__ = [
    "check_existing_dir",
    "check_existing_file",
    "check_freqtrade_cli",
    "check_freqtrade_config",
    "check_openai_compatible",
    "check_opencli",
    "check_opencode_cli",
    "check_python_imports",
    "check_writable_dir",
    "load_project_dotenv",
    "normalize_opencode_model",
    "resolve_openai_compatible_settings",
    "run_agent_flow_preflight",
    "run_capture_preflight",
    "run_lob_rebuild_preflight",
    "run_tca_preflight",
    "run_ws_production_preflight",
    "split_provider_model",
]
