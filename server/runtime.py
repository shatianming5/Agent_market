from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .job_manager import JobManager, DEFAULT_KEEP_RECENT

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

logger = logging.getLogger(__name__)


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
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            if os.environ.get(key):
                continue
            os.environ[key] = value
    except Exception as _exc:
        logger.debug("Suppressed: %s", _exc)


# Load .env from project root into process env
load_dotenv_into_environ(ROOT / ".env")

# Make sure `agent_market` is importable for API routes.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from agent_market import paths as _am_paths  # type: ignore

    SETTINGS_PATH = (_am_paths.user_data_root() / "server_settings.json").resolve()
    _JOB_LOG_DIR = (_am_paths.user_data_root() / "job_logs").resolve()
    _JOB_REGISTRY_DIR = (_am_paths.user_data_root() / "job_registry").resolve()
except Exception:  # pragma: no cover
    SETTINGS_PATH = (ROOT / "user_data" / "server_settings.json").resolve()
    _JOB_LOG_DIR = (ROOT / "user_data" / "job_logs").resolve()
    _JOB_REGISTRY_DIR = (ROOT / "user_data" / "job_registry").resolve()


jobs = JobManager(
    log_dir=_JOB_LOG_DIR,
    registry_dir=_JOB_REGISTRY_DIR,
    keep_recent=int(os.environ.get("AGENT_MARKET_KEEP_RECENT_JOBS", str(DEFAULT_KEEP_RECENT)) or DEFAULT_KEEP_RECENT),
    max_running=int(os.environ.get("AGENT_MARKET_MAX_CONCURRENT_JOBS", "2") or "2"),
    max_queued=int(os.environ.get("AGENT_MARKET_MAX_QUEUED_JOBS", "50") or "50"),
)

# Optional background GC janitor for long-running single-machine deployments.
try:  # pragma: no cover
    from .janitor import start_janitor_thread

    _JANITOR_THREAD = start_janitor_thread()
except Exception:
    _JANITOR_THREAD = None
