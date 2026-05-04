"""Path resolution for wq_brain artifacts."""
from __future__ import annotations

from pathlib import Path

from agent_market.paths import artifacts_root


def wq_brain_root() -> Path:
    return artifacts_root() / "wq_brain"


def wq_brain_run_dir(run_id: str) -> Path:
    return wq_brain_root() / "runs" / run_id


def alpha_pool_path(tag: str) -> Path:
    return wq_brain_root() / "pools" / f"{tag}.json"


def tried_exprs_path(tag: str) -> Path:
    """Cross-loop ledger: every simulated expression appended as JSONL."""
    return wq_brain_root() / "tried_exprs" / f"{tag}.jsonl"


def run_registry_path() -> Path:
    return wq_brain_root() / "run_registry.jsonl"


def repo_root() -> Path:
    """Project root — used to resolve scripts/wq_tools.py and prompts/."""
    return Path(__file__).resolve().parents[3]
