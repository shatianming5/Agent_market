"""Canonical paths for factor_lab subsystem."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]

USER_DATA = ROOT / "user_data"
KUCOIN_DIR = USER_DATA / "data" / "kucoin"
OKX_FUTURES_DIR = USER_DATA / "data" / "okx" / "futures"
BYBIT_FUTURES_DIR = USER_DATA / "data" / "bybit" / "futures"
BINANCE_FUTURES_DIR = USER_DATA / "data" / "binance" / "futures"
FUNDING_DIR = USER_DATA / "data" / "funding"

FEATURE_FILE = USER_DATA / "freqai_features_real.json"
EXPRESSIONS_FILE = USER_DATA / "freqai_expressions.json"
EXPRESSIONS_SCORED = USER_DATA / "freqai_expressions_scored_all.json"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
LAB_STATE = ARTIFACTS / "factor_lab"

# Default pair universe
DEFAULT_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
]

# Default training/OOS split (label_period=12h, strict temporal separation)
DEFAULT_TRAIN = ("2023-05-15", "2025-11-01")  # 2.5 years
DEFAULT_OOS = ("2025-11-01", "2026-04-12")     # 5.5 months — legacy two-section
# Three-section split: miners/GA select on VAL only; REAL_TEST is held for a
# single final check and must not be peeked at during parameter tuning.
DEFAULT_TRAIN3 = ("2023-05-15", "2025-07-01")    # fit region (~2.1 years)
DEFAULT_VAL3   = ("2025-07-01", "2025-12-01")    # miner's holdout (~5 months)
DEFAULT_REAL_TEST3 = ("2025-12-01", "2026-04-12")# final hold-out (~4.4 months)
# Rolling sub-windows inside VAL3 for multi-period fitness (start → end pairs)
DEFAULT_VAL_WINDOWS = (
    ("2025-07-01", "2025-08-15"),
    ("2025-08-15", "2025-10-01"),
    ("2025-10-01", "2025-11-15"),
    ("2025-11-15", "2025-12-01"),
)
DEFAULT_LABEL_PERIOD = 12
DEFAULT_CLASS_THRESHOLD = 0.005

# Economic constants used by the composite-fitness penalty.
DEFAULT_TAKER_FEE = 0.0008        # 0.08% KuCoin taker
DEFAULT_FUNDING_DAILY = 0.0001    # ~0.01% avg funding/day (for futures)
DEFAULT_SLIPPAGE = 0.0003         # 3bps estimate


def kucoin_feather(pair: str, timeframe: str = "1h") -> Path:
    """Return the canonical feather path for a pair at a given timeframe."""
    return KUCOIN_DIR / f"{pair.replace('/', '_')}-{timeframe}.feather"


def feather_for_pair(pair: str, timeframe: str = "1h", data_dir: Path | str | None = None) -> Path:
    """Return a spot OHLCV feather path under a configurable data directory."""
    root = Path(data_dir) if data_dir is not None else KUCOIN_DIR
    return root / f"{pair.replace('/', '_')}-{timeframe}.feather"


def pair_from_feather(path: Path, timeframe: str = "1h") -> str:
    suffix = f"-{timeframe}.feather"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"not a {timeframe} feather: {path}")
    return name[: -len(suffix)].replace("_", "/")


def discover_pairs(data_dir: Path | str | None = None, timeframe: str = "1h") -> list[str]:
    """Discover pair names from `*-<timeframe>.feather` files in one directory."""
    root = Path(data_dir) if data_dir is not None else KUCOIN_DIR
    if not root.exists():
        return []
    return [pair_from_feather(path, timeframe) for path in sorted(root.glob(f"*-{timeframe}.feather"))]


def resolve_pairs(
    pairs: Sequence[str] | str | None = None,
    *,
    data_dir: Path | str | None = None,
    timeframe: str = "1h",
) -> list[str]:
    """Resolve explicit/default/auto pair universes for FactorLab commands."""
    if isinstance(pairs, str):
        raw = pairs.strip()
        if not raw or raw.lower() == "default":
            return list(DEFAULT_PAIRS)
        if raw.lower() == "auto":
            discovered = discover_pairs(data_dir=data_dir, timeframe=timeframe)
            return discovered or list(DEFAULT_PAIRS)
        return [part.strip() for part in raw.split(",") if part.strip()]
    if pairs is None:
        return list(DEFAULT_PAIRS)
    return list(pairs)


# Timeframe → label-period (bars). `label_period` stays at 12 bars = 12h horizon
# on 1h, 3 bars on 4h, 720 bars on 1m. Keep horizon consistent across frequencies
# by scaling the bar count.
TIMEFRAME_LABEL_BARS = {"1m": 720, "5m": 144, "15m": 48, "1h": 12, "4h": 3, "1d": 1}
