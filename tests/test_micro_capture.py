from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_micro_capture_fixture_mode_writes_files(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "micro_capture.py"
    fixture = root / "tests" / "fixtures" / "kucoin_ws_sample.jsonl"

    out_dir = tmp_path / "capture"
    cmd = [
        sys.executable,
        str(script),
        "--exchange",
        "kucoin",
        "--channels",
        "match,level2",
        "--fixture",
        str(fixture),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(root), check=True)  # noqa: S603,S607

    assert (out_dir / "match.ndjson.gz").exists()
    assert (out_dir / "level2.ndjson.gz").exists()
    manifest = out_dir / "manifest.json"
    preflight = out_dir / "preflight.json"
    assert manifest.exists()
    assert preflight.exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["exchange"] == "kucoin"
    files = {item["channel"]: item for item in payload.get("files") or []}
    assert files["match"]["count"] > 0
    assert files["level2"]["count"] > 0

    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    assert preflight_payload["kind"] == "capture"
    assert preflight_payload["ok"] is True
