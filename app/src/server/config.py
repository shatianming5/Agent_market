from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from agent_market.paths import USER_DATA_DIR


@dataclass
class Settings:
    api_title: str = "Agent Market Server"
    api_version: str = "0.1.1"
    host: str = "127.0.0.1"
    port: int = 8032
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    db_path: Path = USER_DATA_DIR / "app.db"
    job_max_seconds: int = 180
    force_fast: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    otel_traces_enabled: bool = False
    otel_endpoint: str | None = None
    otel_headers: Dict[str, str] = field(default_factory=dict)
    otel_service_name: str = "agent-market"
    secrets_master_key: str | None = None
    secret_admin_token: str | None = None
    secret_read_token: str | None = None
    secret_scopes: List[str] = field(default_factory=list)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _parse_origins(value: str | None) -> List[str]:
    if not value:
        return ["*"]
    v = value.strip()
    # try JSON array first
    if v.startswith("["):
        try:
            arr = json.loads(v)
            return [str(x) for x in arr if x]
        except Exception:
            pass
    # comma-separated fallback
    return [p.strip() for p in v.split(",") if p.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_headers(value: str | None) -> Dict[str, str]:
    if not value:
        return {}
    headers: Dict[str, str] = {}
    parts = [p.strip() for p in value.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        headers[k.strip()] = v.strip()
    return headers


def _parse_scopes(value: str | None) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def load_settings() -> Settings:
    host = _env("APP_HOST", "127.0.0.1") or "127.0.0.1"
    port = int(_env("APP_PORT", "8032") or 8032)
    db_env = _env("APP_DB_PATH")
    if db_env:
        db_path = Path(db_env)
        if not db_path.is_absolute():
            db_path = (USER_DATA_DIR / db_path).resolve()
    else:
        db_path = USER_DATA_DIR / "app.db"
    origins = _parse_origins(_env("APP_ALLOWED_ORIGINS"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    job_max = int(_env("APP_JOB_MAX_SECONDS", "180") or 180)
    force_fast = _env_bool("APP_FORCE_FAST", False)
    log_level = (_env("APP_LOG_LEVEL", "INFO") or "INFO").upper()
    log_json = _env_bool("APP_LOG_JSON", True)
    otel_enabled = _env_bool("APP_ENABLE_TRACING", False)
    otel_endpoint = _env("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_headers = _parse_headers(_env("OTEL_EXPORTER_OTLP_HEADERS"))
    otel_service_name = _env("OTEL_SERVICE_NAME", "agent-market") or "agent-market"
    secrets_master_key = _env("SECRETS_MASTER_KEY")
    secret_admin_token = _env("APP_SECRET_TOKEN")
    secret_read_token = _env("APP_SECRET_READ_TOKEN")
    secret_scopes = _parse_scopes(_env("APP_SECRET_SCOPES"))
    return Settings(
        host=host,
        port=port,
        allowed_origins=origins,
        db_path=db_path,
        job_max_seconds=job_max,
        force_fast=force_fast,
        log_level=log_level,
        log_json=log_json,
        otel_traces_enabled=otel_enabled and bool(otel_endpoint),
        otel_endpoint=otel_endpoint,
        otel_headers=otel_headers,
        otel_service_name=otel_service_name,
        secrets_master_key=secrets_master_key,
        secret_admin_token=secret_admin_token,
        secret_read_token=secret_read_token,
        secret_scopes=secret_scopes,
    )
