"""Shared utilities for scripts/ — eliminates duplicated helpers."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Path bootstrapping — ensures src/ and project root are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0].parent  # repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_market import paths  # noqa: E402


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(path: str | Path) -> Path:
    """Resolve *path* relative to repo root via ``paths.resolve_repo_path``."""
    return paths.resolve_repo_path(path)


# ---------------------------------------------------------------------------
# JSON / YAML I/O
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    raw = str(value).replace(",", " ").split()
    return [x.strip() for x in raw if x.strip()]


# ---------------------------------------------------------------------------
# FreqAI config helpers
# ---------------------------------------------------------------------------

def resolve_timeframe(cfg: Dict[str, Any], cli_timeframe: Optional[str]) -> str:
    if cli_timeframe:
        return cli_timeframe
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    tfs = fp.get("include_timeframes") or []
    if isinstance(tfs, list) and tfs and isinstance(tfs[0], str) and tfs[0]:
        return tfs[0]
    tf = cfg.get("timeframe")
    return str(tf) if tf else "1h"


def resolve_label_period(cfg: Dict[str, Any], override: Any = None) -> int:
    if override is not None:
        return int(override)
    freqai = cfg.get("freqai") or {}
    fp = freqai.get("feature_parameters") or {}
    return int(fp.get("label_period_candles") or 12)
