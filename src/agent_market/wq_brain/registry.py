"""Run registry: append-only JSONL log of completed wq_brain runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .paths import wq_brain_registry_path


def append_run_meta(run_id: str, summary: dict[str, Any]) -> Path:
    path = wq_brain_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **summary,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def load_registry() -> list[dict[str, Any]]:
    path = wq_brain_registry_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
