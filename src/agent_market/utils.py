"""Shared utility functions for agent_market package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def as_float(value: Any) -> Optional[float]:
    """Safely convert *value* to float, returning ``None`` on failure."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError, Exception):
        return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write *payload* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()
