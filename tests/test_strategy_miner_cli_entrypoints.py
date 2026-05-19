from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_strategy_miner_script_help_uses_package_cli() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "strategy_miner.py"), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Strategy-level mining via LLM Agent" in proc.stdout
    assert "--allow-defaults" in proc.stdout


def test_strategy_miner_module_help_uses_same_cli() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "agent_market.strategy_miner", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Strategy-level mining via LLM Agent" in proc.stdout
    assert "--allow-defaults" in proc.stdout
