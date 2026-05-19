from __future__ import annotations

"""
Artifact conventions for Flow steps (plan.md alignment).

This module is intentionally small: it defines stable helpers to build well-known
artifact paths for a given run_id. The rest of the system may still write into
existing locations (`artifacts/`, `user_data/`) while gradually adopting these helpers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ArtifactPaths:
    run_id: str
    run_dir: Path

    def step_dir(self, step_id: str) -> Path:
        return (self.run_dir / str(step_id)).resolve()

    def as_dict(self) -> Dict[str, str]:
        return {"run_id": str(self.run_id), "run_dir": str(self.run_dir)}


def run_dir(run_id: str, *, root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return (base / "artifacts" / "runs" / str(run_id)).resolve()


def artifact_paths(run_id: str, *, root: Optional[Path] = None) -> ArtifactPaths:
    return ArtifactPaths(run_id=str(run_id), run_dir=run_dir(run_id, root=root))


__all__ = ["ArtifactPaths", "artifact_paths", "run_dir"]
