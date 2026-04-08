#!/usr/bin/env python3
"""Continuously monitor a ws_production loop and persist status snapshots."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    cleaned = [_strip_ansi(line).rstrip() for line in lines[-limit:]]
    return [line for line in cleaned if line]


def _file_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
    }
    if exists:
        stat = path.stat()
        payload.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return payload


def _recent_files(directories: list[Path], *, limit: int = 5) -> list[dict[str, Any]]:
    items: dict[str, Path] = {}
    for directory in directories:
        if not directory.exists():
            continue
        try:
            candidates = [item for item in directory.iterdir() if item.is_file()]
        except OSError:
            continue
        for item in candidates:
            items[str(item.resolve())] = item
    ranked = sorted(items.values(), key=lambda item: item.stat().st_mtime, reverse=True)
    output: list[dict[str, Any]] = []
    for item in ranked[:limit]:
        output.append(_file_info(item))
    return output


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tmux_session_info(session: str) -> dict[str, Any]:
    proc = subprocess.run(  # noqa: S603
        ["tmux", "list-panes", "-t", session, "-F", "pid=#{pane_pid}\tcmd=#{pane_current_command}\tpath=#{pane_current_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "name": session,
            "exists": False,
            "panes": [],
            "error": (proc.stderr or proc.stdout or "").strip(),
        }

    panes: list[dict[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        fields: dict[str, str] = {}
        for chunk in line.split("\t"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            fields[key] = value
        panes.append(fields)

    return {"name": session, "exists": True, "panes": panes}


def gather_status(workspace: Path, *, session: str, tail_lines: int = 20) -> dict[str, Any]:
    canonical_workspace = workspace.parent / "workspace"
    results_dir = workspace / "results"
    reports_dirs = [workspace / "reports"]
    validation_dirs = [workspace / "validation", results_dir / "factor_validation"]
    backtests_dirs = [workspace / "backtests"]
    paper_dirs = [workspace / "paper"]
    signals_dirs = [workspace / "signals"]
    if canonical_workspace != workspace:
        reports_dirs.append(canonical_workspace / "reports")
        validation_dirs.append(canonical_workspace / "validation")
        backtests_dirs.append(canonical_workspace / "backtests")
        paper_dirs.append(canonical_workspace / "paper")
        signals_dirs.append(canonical_workspace / "signals")

    loop_logs = sorted(results_dir.glob("ws_production_loop_*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
    latest_loop_log = loop_logs[0] if loop_logs else None
    agent_loop_log = results_dir / "agent_loop.log"
    preflight_json = results_dir / "preflight.json"
    last_cycle_json = results_dir / "last_cycle.json"
    factor_cycle_json = results_dir / "factor_cycle_latest.json"

    status = {
        "timestamp": _utc_now(),
        "workspace": str(workspace),
        "canonical_workspace": str(canonical_workspace) if canonical_workspace.exists() else None,
        "session": _tmux_session_info(session),
        "logs": {
            "latest_loop_log": _file_info(latest_loop_log) if latest_loop_log is not None else None,
            "agent_loop_log": _file_info(agent_loop_log),
            "latest_loop_tail": _tail_lines(latest_loop_log, tail_lines) if latest_loop_log is not None else [],
            "agent_loop_tail": _tail_lines(agent_loop_log, tail_lines),
        },
        "preflight": _read_json(preflight_json),
        "last_cycle": _read_json(last_cycle_json),
        "factor_cycle": _read_json(factor_cycle_json),
        "artifacts": {
            "results_recent": _recent_files([results_dir]),
            "reports_recent": _recent_files(reports_dirs),
            "validation_recent": _recent_files(validation_dirs),
            "backtests_recent": _recent_files(backtests_dirs),
            "paper_recent": _recent_files(paper_dirs),
            "signals_recent": _recent_files(signals_dirs),
        },
    }
    return status


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a ws_production tmux loop and persist status snapshots.")
    parser.add_argument("--workspace", default="ws_production", help="Path to ws_production workspace")
    parser.add_argument("--session", default="ws_production_loop", help="tmux session name to monitor")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--tail-lines", type=int, default=20, help="How many log lines to keep in the snapshot")
    parser.add_argument("--output", default=None, help="Status JSON path (default: workspace/results/monitor_status.json)")
    parser.add_argument("--history", default=None, help="History JSONL path (default: workspace/results/monitor_history.jsonl)")
    parser.add_argument("--once", action="store_true", help="Collect one snapshot then exit")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    results_dir = workspace / "results"
    output_path = Path(args.output).resolve() if args.output else (results_dir / "monitor_status.json")
    history_path = Path(args.history).resolve() if args.history else (results_dir / "monitor_history.jsonl")

    while True:
        status = gather_status(workspace, session=str(args.session), tail_lines=int(args.tail_lines))
        _write_json(output_path, status)
        _append_jsonl(history_path, status)

        latest_loop = (status.get("logs") or {}).get("latest_loop_log") or {}
        latest_name = Path(str(latest_loop.get("path") or "")).name if latest_loop.get("path") else "-"
        session_ok = bool((status.get("session") or {}).get("exists"))
        print(
            f"[{status['timestamp']}] session={'up' if session_ok else 'down'} "
            f"log={latest_name} reports={len(((status.get('artifacts') or {}).get('reports_recent') or []))} "
            f"validation={len(((status.get('artifacts') or {}).get('validation_recent') or []))}",
            flush=True,
        )

        if args.once:
            return 0
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
