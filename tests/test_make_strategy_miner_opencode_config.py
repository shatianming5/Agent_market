from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_make_strategy_miner_opencode_config_renders_runtime_file(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    out_path = tmp_path / "out.json"
    base_path.write_text(
        json.dumps(
            {
                "budget": {
                    "provider": "openai_compatible",
                    "model": "gpt-5.2",
                    "base_url": "http://127.0.0.1:4141/v1",
                },
                "evaluation": {"hyperopt_jobs": 12},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "make_strategy_miner_opencode_config.py"),
            "--base",
            str(base_path),
            "--output",
            str(out_path),
            "--model",
            "custom/gpt-5.2",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    rendered = json.loads(out_path.read_text(encoding="utf-8"))
    original = json.loads(base_path.read_text(encoding="utf-8"))

    assert proc.stdout.strip() == str(out_path)
    assert rendered["budget"]["provider"] == "opencode"
    assert rendered["budget"]["model"] == "custom/gpt-5.2"
    assert "base_url" not in rendered["budget"]
    assert original["budget"]["provider"] == "openai_compatible"
