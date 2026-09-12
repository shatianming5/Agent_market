#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path, PureWindowsPath


DEFAULT_TARGETS = [
    ".pytest_cache",
    "artifacts",
    "user_data/agent_logs",
    "user_data/backtest_results",
    "user_data/logs",
    "user_data/models",
    "user_data/notebooks",
    "user_data/plot",
    "user_data/freqaimodels",
    "user_data/hyperopts",
    "user_data/hyperopt_results",
]


def rm_rf(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if path.is_dir():
        shutil.rmtree(path)


def validated_target(root: Path, target: str) -> Path:
    relative = Path(target)
    windows_path = PureWindowsPath(target)
    if (relative.is_absolute() or windows_path.drive or windows_path.root
            or ".." in relative.parts or ".." in windows_path.parts):
        raise ValueError(f"Unsafe cleanup target {target!r}: use a repo-relative descendant")
    path = root / relative
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Unsafe cleanup target {target!r}: must stay strictly inside {root}")
    # Keep the lexical path so a final symlink is unlinked, not followed.
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean temporary and generated directories")
    parser.add_argument("targets", nargs="*", help="Optional paths to clean (default uses built-in list)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    parser.add_argument("--keep-dirs", action="store_true", help="Recreate empty directories after cleaning")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    targets = args.targets or DEFAULT_TARGETS
    try:
        paths = [validated_target(root, rel) for rel in targets]
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    removed = []

    for rel, p in zip(targets, paths):
        if not p.exists() and not p.is_symlink():
            continue
        if args.dry_run:
            print(f"[dry] would remove: {p}")
            continue
        print(f"remove: {p}")
        try:
            rm_rf(p)
            removed.append(p)
            if args.keep_dirs and rel.endswith(("agent_logs","backtest_results","logs","artifacts")):
                p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            parser.exit(1, f"Cleanup failed for {p}: {exc}\n")

    print(f"done. removed {len(removed)} items")


if __name__ == "__main__":
    main()
