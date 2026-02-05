from __future__ import annotations

"""
Validation helpers shared by API/UI (plan.md alignment).

The API currently performs run_meta artifact existence checks inside
`server/api/routes/flow.py`. This module provides reusable primitives so that
future refactors can consolidate checks without changing behavior.
"""

from pathlib import Path
from typing import Any, Dict, Optional


def resolve_under_root(root: Path, path: str | Path) -> Optional[Path]:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = (root / p).resolve()
        else:
            p = p.resolve()
        root_resolved = root.resolve()
        if p == root_resolved or root_resolved in p.parents:
            return p
    except Exception:
        return None
    return None


def check_path(root: Path, path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"path": path, "exists": False}
    resolved = resolve_under_root(root, path)
    if resolved is None:
        return {"path": path, "exists": False, "error": "path_outside_root"}
    try:
        rel = str(resolved.relative_to(root.resolve()))
    except Exception:
        rel = str(resolved)
    return {"path": rel, "exists": resolved.exists()}


__all__ = ["check_path", "resolve_under_root"]

