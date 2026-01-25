from __future__ import annotations

import os
from pathlib import Path

from .job_manager import JobManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SETTINGS_PATH = ROOT / "user_data" / "server_settings.json"


def load_dotenv_into_environ(env_path: Path) -> None:
    """Lightweight .env loader without extra deps.

    Supports KEY=VALUE lines, ignores comments and empty lines.
    Best-effort: never raises.
    """
    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not os.environ.get(key):
                os.environ[key] = value
    except Exception:
        pass


# Load .env from project root into process env
load_dotenv_into_environ(ROOT / ".env")


jobs = JobManager()

