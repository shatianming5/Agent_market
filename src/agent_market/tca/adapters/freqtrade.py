from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _pick_backtest_json_member(zf: zipfile.ZipFile) -> str:
    members = [name for name in zf.namelist() if name.endswith(".json")]
    # Exclude config json and any other aux files.
    members = [name for name in members if "_config" not in name]
    if not members:
        raise ValueError("No backtest JSON found in archive")
    # Typically only one member.
    return members[0]


def load_freqtrade_backtest_trades(zip_path: Path) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Load strategy trades from a freqtrade backtest result zip.

    Returns: (strategy_name, trades, meta)
    """
    zp = Path(zip_path).resolve()
    if not zp.exists():
        raise FileNotFoundError(f"Backtest archive not found: {zp}")

    with zipfile.ZipFile(zp) as zf:
        member = _pick_backtest_json_member(zf)
        data = json.loads(zf.read(member))

    strategy_block = data.get("strategy") or {}
    if not isinstance(strategy_block, dict) or not strategy_block:
        raise ValueError("Strategy block missing in backtest JSON")
    strategy_name, strategy_metrics = next(iter(strategy_block.items()))
    if not isinstance(strategy_metrics, dict):
        raise ValueError("Invalid strategy metrics in backtest JSON")

    trades = strategy_metrics.get("trades") or []
    if not isinstance(trades, list):
        trades = []
    # Keep only JSON objects (freqtrade uses dicts per trade).
    trades = [t for t in trades if isinstance(t, dict)]

    meta: Dict[str, Any] = {
        "zip_member": member,
        "has_trades": bool(trades),
        "strategy_metrics_keys": sorted(strategy_metrics.keys())[:200],
    }
    # Common helpful fields (best-effort).
    for key in ("timeframe", "timeframe_detail", "backtest_start", "backtest_end", "max_open_trades"):
        if key in strategy_metrics:
            meta[key] = strategy_metrics.get(key)

    return str(strategy_name), trades, meta


__all__ = ["load_freqtrade_backtest_trades"]

