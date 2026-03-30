from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_env_path(var: str, *, default: Path) -> Path:
    raw = os.environ.get(var)
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        return p
    return default.resolve()


def artifacts_root() -> Path:
    return _resolve_env_path("AGENT_MARKET_ARTIFACTS_ROOT", default=REPO_ROOT / "artifacts")


def user_data_root() -> Path:
    return _resolve_env_path("AGENT_MARKET_USER_DATA_ROOT", default=REPO_ROOT / "user_data")


def runs_root() -> Path:
    default = artifacts_root() / "runs"
    return _resolve_env_path("AGENT_MARKET_RUNS_ROOT", default=default)


def models_root() -> Path:
    default = artifacts_root() / "models"
    return _resolve_env_path("AGENT_MARKET_MODELS_ROOT", default=default)


def run_meta_latest_path() -> Path:
    default = artifacts_root() / "run_meta.json"
    return _resolve_env_path("AGENT_MARKET_RUN_META_LATEST_PATH", default=default)


def run_dir(run_id: str) -> Path:
    return (runs_root() / str(run_id)).resolve()


def run_meta_path(run_id: str) -> Path:
    return (run_dir(run_id) / "run_meta.json").resolve()


def default_feedback_path() -> Path:
    return (user_data_root() / "llm_feedback" / "latest_backtest_summary.json").resolve()


def resolve_repo_path(path: Union[str, Path]) -> Path:
    """Resolve a repo-relative path while respecting output-root overrides.

    - Absolute paths are preserved.
    - Paths under `artifacts/` are mapped under `artifacts_root()`.
    - Paths under `user_data/` are mapped under `user_data_root()`.
    - Other relative paths are resolved under the repo root.
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    # Normalize for matching (avoid Windows separators here; repo is macOS).
    parts = p.parts
    if parts and parts[0] == "artifacts":
        return (artifacts_root() / Path(*parts[1:])).resolve()
    if parts and parts[0] == "user_data":
        return (user_data_root() / Path(*parts[1:])).resolve()
    return (REPO_ROOT / p).resolve()


def relpath_under_repo(path: Path) -> str:
    """Best-effort render a stable, repo-relative path for metadata/docs."""
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(REPO_ROOT))
    except Exception:
        logger.debug("Cannot compute relative path for %s under repo root", path, exc_info=True)
        return str(path.resolve())


def resolve_under_repo(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    return resolve_repo_path(raw)


def _allowed_roots() -> list[Path]:
    """Return the set of directory roots that external paths may resolve into."""
    return [
        REPO_ROOT.resolve(),
        artifacts_root().resolve(),
        user_data_root().resolve(),
        runs_root().resolve(),
        models_root().resolve(),
    ]


def safe_resolve(path: Union[str, Path], *, allow_absolute: bool = False) -> Path:
    """Resolve *path* and verify it stays within allowed project directories.

    This is the **recommended function for all external/API inputs**.
    Unlike ``resolve_repo_path``, it rejects ``..`` traversal that escapes
    project boundaries and (by default) rejects absolute paths entirely.

    Raises ``ValueError`` if the resolved path is outside allowed roots.
    """
    raw = str(path).strip()
    if not raw:
        raise ValueError("Empty path")

    p = Path(raw)

    # Block ".." components before resolution to catch obvious traversal.
    if ".." in p.parts:
        raise ValueError(f"Path traversal ('..') not allowed: {raw}")

    if p.is_absolute() and not allow_absolute:
        raise ValueError(f"Absolute paths not allowed: {raw}")

    resolved = resolve_repo_path(raw)

    # Verify the resolved path is under at least one allowed root.
    for root in _allowed_roots():
        if resolved == root or root in resolved.parents:
            return resolved

    raise ValueError(
        f"Path resolves outside allowed directories: {raw} -> {resolved}"
    )


__all__ = [
    "REPO_ROOT",
    "artifacts_root",
    "user_data_root",
    "runs_root",
    "models_root",
    "run_meta_latest_path",
    "run_dir",
    "run_meta_path",
    "default_feedback_path",
    "resolve_repo_path",
    "resolve_under_repo",
    "relpath_under_repo",
    "safe_resolve",
]

