from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_ws_production_monitor_writes_status_snapshot(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "ws_production_monitor.py"

    workspace = tmp_path / "ws_production"
    for name in ("results", "reports", "validation", "backtests", "paper", "signals"):
        (workspace / name).mkdir(parents=True, exist_ok=True)

    (workspace / "results" / "preflight.json").write_text('{"ok": true}\n', encoding="utf-8")
    (workspace / "results" / "last_cycle.json").write_text('{"summary": {"total_strategies": 2}}\n', encoding="utf-8")
    (workspace / "results" / "agent_loop.log").write_text("line1\nline2\n", encoding="utf-8")
    (workspace / "results" / "ws_production_loop_20260408-010620.log").write_text(
        "\x1b[0mhello\nworld\n",
        encoding="utf-8",
    )
    (workspace / "reports" / "daily_report.json").write_text("{}\n", encoding="utf-8")

    output = workspace / "results" / "monitor_status.json"
    history = workspace / "results" / "monitor_history.jsonl"
    subprocess.run(  # noqa: S603,S607
        [
            sys.executable,
            str(script),
            "--workspace",
            str(workspace),
            "--session",
            "missing_session",
            "--once",
            "--output",
            str(output),
            "--history",
            str(history),
        ],
        cwd=str(root),
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workspace"] == str(workspace.resolve())
    assert payload["session"]["exists"] is False
    assert payload["preflight"]["ok"] is True
    assert payload["last_cycle"]["summary"]["total_strategies"] == 2
    assert payload["logs"]["latest_loop_tail"][-1] == "world"
    assert len(history.read_text(encoding="utf-8").splitlines()) == 1
