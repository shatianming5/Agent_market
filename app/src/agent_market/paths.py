from __future__ import annotations

import os
from pathlib import Path


_FILE = Path(__file__).resolve()
SRC_ROOT = _FILE.parents[1]
APP_ROOT = _FILE.parents[2]
PROJECT_ROOT = _FILE.parents[3]

RESOURCES_ROOT = Path(os.environ.get("AGENT_MARKET_RESOURCES_DIR", PROJECT_ROOT / "resources")).resolve()
CONFIG_ROOT = Path(os.environ.get("AGENT_MARKET_CONFIG_DIR", PROJECT_ROOT / "config")).resolve()
FREQTRADE_ROOT = Path(os.environ.get("AGENT_MARKET_FREQTRADE_DIR", APP_ROOT / "freqtrade")).resolve()
FREQTRADE_SCRIPTS = FREQTRADE_ROOT / "freqtrade" / "scripts"

DATA_DIR = RESOURCES_ROOT / "data"
USER_DATA_DIR = RESOURCES_ROOT / "user_data"
ARTIFACTS_DIR = RESOURCES_ROOT / "artifacts"
BACKTESTS_DIR = RESOURCES_ROOT / "backtests"
RUNS_DIR = USER_DATA_DIR / "runs"


__all__ = [
    "SRC_ROOT",
    "APP_ROOT",
    "PROJECT_ROOT",
    "RESOURCES_ROOT",
    "CONFIG_ROOT",
    "FREQTRADE_ROOT",
    "FREQTRADE_SCRIPTS",
    "DATA_DIR",
    "USER_DATA_DIR",
    "ARTIFACTS_DIR",
    "BACKTESTS_DIR",
    "RUNS_DIR",
]
