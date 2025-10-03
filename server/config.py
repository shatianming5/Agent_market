from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Settings:
    api_title: str = "Agent Market Server"
    api_version: str = "0.1.1"
    host: str = "127.0.0.1"
    port: int = 8032
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    db_path: Path = Path("user_data/app.db")
    job_max_seconds: int = 180
    force_fast: bool = False


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


def load_settings() -> Settings:
    host = _env("APP_HOST", "127.0.0.1") or "127.0.0.1"
    port = int(_env("APP_PORT", "8032") or 8032)
    db_path = Path(_env("APP_DB_PATH", "user_data/app.db") or "user_data/app.db")
    origins = _parse_origins(_env("APP_ALLOWED_ORIGINS"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    job_max = int(_env("APP_JOB_MAX_SECONDS", "180") or 180)
    force_fast = (_env("APP_FORCE_FAST", "0") or "0").strip() in ("1", "true", "yes")
    return Settings(host=host, port=port, allowed_origins=origins, db_path=db_path, job_max_seconds=job_max, force_fast=force_fast)
