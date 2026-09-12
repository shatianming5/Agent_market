"""Cleanup — prevents disk bloat in long-running systems.

Usage:
    from workspace.cleanup import auto_cleanup
    cleaned = auto_cleanup()  # removes old results, keeps recent
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]


def auto_cleanup(
    *,
    results_dir: Optional[str | Path] = None,
    max_run_files: int = 100,
    max_age_days: int = 30,
    max_model_files: int = 5,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Clean up old result files, models, and logs.

    Keeps:
    - Most recent `max_run_files` run_*.json files
    - Models from last `max_model_files` unique strategies
    - All lifecycle/versions/calibration files (small, important)

    Returns summary of what was cleaned.
    """
    if results_dir is None:
        results_dir = ROOT / "workspace" / "results"
    results_dir = Path(results_dir)

    cleaned = {"files_removed": 0, "bytes_freed": 0, "details": []}

    if not results_dir.exists():
        return cleaned

    # 1. Clean old run_*.json (keep newest max_run_files)
    run_files = sorted(results_dir.glob("run_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(run_files) > max_run_files:
        for f in run_files[max_run_files:]:
            size = f.stat().st_size
            if not dry_run:
                f.unlink()
            cleaned["files_removed"] += 1
            cleaned["bytes_freed"] += size

    # 2. Clean old walkforward_*.json (keep newest 50)
    wf_files = sorted(results_dir.glob("walkforward_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(wf_files) > 50:
        for f in wf_files[50:]:
            size = f.stat().st_size
            if not dry_run:
                f.unlink()
            cleaned["files_removed"] += 1
            cleaned["bytes_freed"] += size

    # 3. Clean old cycle_*.json (keep newest 20)
    cycle_files = sorted(results_dir.glob("cycle_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if len(cycle_files) > 20:
        for f in cycle_files[20:]:
            size = f.stat().st_size
            if not dry_run:
                f.unlink()
            cleaned["files_removed"] += 1
            cleaned["bytes_freed"] += size

    # 4. Clean old model directories (keep newest max_model_files)
    model_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("model_")],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    if len(model_dirs) > max_model_files:
        for d in model_dirs[max_model_files:]:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if not dry_run:
                shutil.rmtree(d)
            cleaned["files_removed"] += 1
            cleaned["bytes_freed"] += size
            cleaned["details"].append(f"removed model dir: {d.name} ({size/1024:.0f} KB)")

    # 5. Clean old report files (keep newest 10)
    reports_dir = ROOT / "workspace" / "reports"
    if reports_dir.exists():
        for pattern in ["report_*.md", "daily_*.json"]:
            files = sorted(reports_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
            if len(files) > 10:
                for f in files[10:]:
                    size = f.stat().st_size
                    if not dry_run:
                        f.unlink()
                    cleaned["files_removed"] += 1
                    cleaned["bytes_freed"] += size

    cleaned["bytes_freed_mb"] = round(cleaned["bytes_freed"] / 1024 / 1024, 2)
    return cleaned


def disk_usage(results_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Report current disk usage."""
    if results_dir is None:
        results_dir = ROOT / "workspace" / "results"
    results_dir = Path(results_dir)

    if not results_dir.exists():
        return {"total_mb": 0}

    total = 0
    by_type = {}
    for f in results_dir.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total += size
            ext = f.suffix or "other"
            by_type[ext] = by_type.get(ext, 0) + size

    return {
        "total_mb": round(total / 1024 / 1024, 2),
        "total_files": len(list(results_dir.rglob("*"))),
        "by_type": {k: round(v / 1024 / 1024, 2) for k, v in sorted(by_type.items(), key=lambda x: -x[1])},
    }


__all__ = ["auto_cleanup", "disk_usage"]
