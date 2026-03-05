from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from agent_market.utils import as_float

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _trades_count(strategy_metrics: Dict[str, Any]) -> Optional[int]:
    total = _as_int(strategy_metrics.get("total_trades"))
    if total is not None:
        return total
    trades = strategy_metrics.get("trades")
    if isinstance(trades, list):
        return len(trades)
    return _as_int(trades)


def find_latest_backtest_zip(results_dir: Path) -> Optional[Path]:
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Backtest results directory not found: {results_dir}")
    last_path = results_dir / ".last_result.json"
    if last_path.exists():
        try:
            latest_name = json.loads(last_path.read_text(encoding="utf-8")).get(
                "latest_backtest"
            )
            if latest_name:
                candidate = results_dir / str(latest_name)
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            logger.warning("Malformed .last_result.json in %s", results_dir)
    zips = list(results_dir.glob("backtest-result-*.zip"))
    if not zips:
        return None
    return max(zips, key=lambda p: p.stat().st_mtime)


def build_backtest_summary(zip_path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        json_members = [name for name in zf.namelist() if name.endswith(".json")]
        json_members = [name for name in json_members if "_config" not in name]
        json_members = [name for name in json_members if not name.startswith("trades-")]
        if not json_members:
            raise ValueError("No result JSON found in archive")
        data = json.loads(zf.read(json_members[0]))
    strategy_block = data.get("strategy", {})
    if not strategy_block:
        raise ValueError("Strategy block missing in backtest results")
    strategy_name, strategy_metrics = next(iter(strategy_block.items()))
    comparison = data.get("strategy_comparison", []) or []
    comparison_row = next(
        (row for row in comparison if row.get("key") == strategy_name),
        comparison[0] if comparison else None,
    )
    profit_total_pct = strategy_metrics.get("profit_total_pct")
    if profit_total_pct is None and comparison_row is not None:
        profit_total_pct = comparison_row.get("profit_total_pct")
    if profit_total_pct is None:
        profit_total_pct = as_float(strategy_metrics.get("profit_total"))
        profit_total_pct = profit_total_pct * 100.0 if profit_total_pct is not None else None

    avg_profit_pct = strategy_metrics.get("profit_mean_pct")
    if avg_profit_pct is None and comparison_row is not None:
        avg_profit_pct = comparison_row.get("profit_mean_pct")
    if avg_profit_pct is None:
        avg_profit_pct = as_float(strategy_metrics.get("profit_mean"))
        avg_profit_pct = avg_profit_pct * 100.0 if avg_profit_pct is not None else None

    trades_count = _trades_count(strategy_metrics)
    if trades_count is None and comparison_row is not None:
        trades_count = _as_int(comparison_row.get("trades"))

    winrate = strategy_metrics.get("winrate")
    if winrate is None and comparison_row is not None:
        winrate = comparison_row.get("winrate")
    max_drawdown_abs = strategy_metrics.get("max_drawdown_abs")
    if max_drawdown_abs is None and comparison_row is not None:
        max_drawdown_abs = comparison_row.get("max_drawdown_abs")
    return {
        "source": zip_path.name,
        "strategy": strategy_name,
        "profit_total_pct": as_float(profit_total_pct),
        "profit_total_abs": as_float(strategy_metrics.get("profit_total_abs")),
        "trades": trades_count,
        "avg_profit_pct": as_float(avg_profit_pct),
        "winrate": as_float(winrate),
        "max_drawdown_abs": as_float(max_drawdown_abs),
        "best_pair": strategy_metrics.get("best_pair", {}).get("key"),
        "worst_pair": strategy_metrics.get("worst_pair", {}).get("key"),
        "backtest_timerange": f"{strategy_metrics.get('backtest_start')} -> {strategy_metrics.get('backtest_end')}",
        "strategy_comparison": comparison,
    }


def read_backtest_trades(zip_path: Path) -> Optional[Any]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            trade_members = [
                name
                for name in zf.namelist()
                if name.endswith(".json") and name.startswith("trades-")
            ]
            if not trade_members:
                return None
            return json.loads(zf.read(trade_members[0]))
    except Exception:
        logger.warning("Failed to read backtest trades from %s", zip_path, exc_info=True)
        return None


def write_latest_backtest_summary(results_dir: Path, output_path: Path) -> Optional[Dict[str, Any]]:
    try:
        latest_zip = find_latest_backtest_zip(results_dir)
    except FileNotFoundError as exc:
        logger.warning("Backtest summary skipped: %s", exc)
        return None
    if latest_zip is None:
        logger.warning("No backtest archives found in %s", results_dir)
        return None
    try:
        summary = build_backtest_summary(latest_zip)
    except Exception as exc:
        logger.warning("Failed to extract backtest summary from %s: %s", latest_zip, exc)
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "build_backtest_summary",
    "find_latest_backtest_zip",
    "read_backtest_trades",
    "write_latest_backtest_summary",
]
