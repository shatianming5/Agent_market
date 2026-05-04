from __future__ import annotations

from pathlib import Path

from agent_market import paths as repo_paths


def wq_brain_root() -> Path:
    return repo_paths.artifacts_root() / "wq_brain"


def wq_brain_run_dir(run_id: str) -> Path:
    return wq_brain_root() / "runs" / str(run_id)


def alpha_pool_path(tag: str) -> Path:
    return wq_brain_root() / "pools" / f"{tag}.json"


def wq_brain_registry_path() -> Path:
    return wq_brain_root() / "run_registry.jsonl"


def hermes_home_dir(run_id: str) -> Path:
    return wq_brain_run_dir(run_id) / "hermes_home"


def kb_index_path() -> Path:
    return wq_brain_root() / "kb_index.json"
