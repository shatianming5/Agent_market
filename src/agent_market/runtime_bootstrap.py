from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def discover_shared_user_data_root(
    *,
    repo_root: Path,
    explicit_root: str | Path | None = None,
) -> Optional[Path]:
    candidates: list[Path] = []

    configured = explicit_root or os.environ.get("AGENT_MARKET_SHARED_USER_DATA_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())

    repo_name = repo_root.name
    parent = repo_root.parent
    base_name = repo_name.split("_release_", 1)[0]

    if "_release_" in repo_name:
        candidates.append(parent / base_name / "user_data")
        candidates.append(parent / "Agent_market" / "user_data")

    local_user_data = repo_root / "user_data"
    candidates.append(local_user_data)

    if "_release_" not in repo_name:
        candidates.append(parent / "Agent_market" / "user_data")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if (candidate / "data").exists():
            return candidate.resolve()
    return None


def ensure_repo_user_data_data_link(
    *,
    repo_root: Path,
    shared_user_data_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    repo_user_data = repo_root / "user_data"
    repo_data_dir = repo_user_data / "data"
    repo_user_data.mkdir(parents=True, exist_ok=True)

    shared_root = discover_shared_user_data_root(
        repo_root=repo_root,
        explicit_root=shared_user_data_root,
    )

    if repo_data_dir.exists() and not repo_data_dir.is_symlink() and shared_root is None:
        return {
            "status": "present",
            "data_path": str(repo_data_dir.resolve()),
            "linked": False,
        }

    if shared_root is None:
        return {
            "status": "missing",
            "data_path": str(repo_data_dir),
            "linked": False,
        }

    shared_data_dir = (shared_root / "data").resolve()
    if not shared_data_dir.exists():
        return {
            "status": "missing",
            "data_path": str(repo_data_dir),
            "shared_data_path": str(shared_data_dir),
            "linked": False,
        }

    if repo_data_dir.exists() and not repo_data_dir.is_symlink():
        linked_entries: list[str] = []
        for child in sorted(shared_data_dir.iterdir()):
            target = repo_data_dir / child.name
            if target.exists():
                continue
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
            linked_entries.append(child.name)
        return {
            "status": "linked_children" if linked_entries else "present",
            "data_path": str(repo_data_dir.resolve()),
            "shared_data_path": str(shared_data_dir),
            "linked": bool(linked_entries),
            "linked_entries": linked_entries,
        }

    if repo_data_dir.is_symlink():
        current_target = repo_data_dir.resolve()
        if current_target == shared_data_dir:
            return {
                "status": "linked",
                "data_path": str(repo_data_dir.resolve()),
                "shared_data_path": str(shared_data_dir),
                "linked": True,
            }
        repo_data_dir.unlink()

    if repo_data_dir.exists():
        return {
            "status": "present",
            "data_path": str(repo_data_dir.resolve()),
            "shared_data_path": str(shared_data_dir),
            "linked": False,
        }

    repo_data_dir.symlink_to(shared_data_dir, target_is_directory=True)
    return {
        "status": "linked",
        "data_path": str(repo_data_dir.resolve()),
        "shared_data_path": str(shared_data_dir),
        "linked": True,
    }


__all__ = [
    "discover_shared_user_data_root",
    "ensure_repo_user_data_data_link",
]
