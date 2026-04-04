from __future__ import annotations

import json
import logging
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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


def _normalize_pct(value: Any) -> Optional[float]:
    raw = as_float(value)
    if raw is None or not math.isfinite(raw):
        return None
    raw = abs(raw)
    return raw * 100.0 if raw <= 1.0 else raw


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _daily_profit_rows(value: Any) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        day = None
        profit = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            day = str(item[0])
            profit = as_float(item[1])
        elif isinstance(item, dict):
            day = str(item.get("date") or item.get("day") or "")
            profit = as_float(
                item.get("profit_abs", item.get("profit_total_abs", item.get("profit")))
            )
        if day and profit is not None and math.isfinite(profit):
            rows.append((day, float(profit)))
    return rows


def _trade_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _stdev(values: Iterable[float]) -> float:
    seq = [float(v) for v in values if math.isfinite(float(v))]
    if len(seq) < 2:
        return 0.0
    try:
        return float(statistics.stdev(seq))
    except statistics.StatisticsError:
        return 0.0


def _derive_metrics(
    *,
    strategy_metrics: Dict[str, Any],
    comparison_row: Optional[Dict[str, Any]],
    profit_total_pct: Optional[float],
    trades_count: Optional[int],
    winrate: Optional[float],
) -> Dict[str, Any]:
    native_sharpe = as_float(strategy_metrics.get("sharpe"))
    native_sortino = as_float(strategy_metrics.get("sortino"))
    native_calmar = as_float(strategy_metrics.get("calmar"))
    native_profit_factor = as_float(strategy_metrics.get("profit_factor"))
    native_expectancy = as_float(strategy_metrics.get("expectancy"))
    native_expectancy_ratio = as_float(strategy_metrics.get("expectancy_ratio"))
    native_sqn = as_float(strategy_metrics.get("sqn"))
    native_cagr = as_float(strategy_metrics.get("cagr"))
    native_drawdown_raw = as_float(strategy_metrics.get("max_drawdown_account"))
    native_drawdown_pct = _normalize_pct(
        strategy_metrics.get("max_drawdown_account", strategy_metrics.get("max_relative_drawdown"))
    )

    starting_balance = as_float(strategy_metrics.get("starting_balance"))
    final_balance = as_float(strategy_metrics.get("final_balance"))
    profit_total_abs = as_float(strategy_metrics.get("profit_total_abs"))
    if final_balance is None and starting_balance is not None and profit_total_abs is not None:
        final_balance = starting_balance + profit_total_abs

    daily_rows = _daily_profit_rows(strategy_metrics.get("daily_profit"))
    trades_rows = _trade_rows(strategy_metrics.get("trades"))
    period_days = _as_int(strategy_metrics.get("backtest_days"))
    if period_days is None and daily_rows:
        period_days = len(daily_rows)

    positive_days = 0
    negative_days = 0
    flat_days = 0
    daily_returns: list[float] = []
    equity_curve: list[float] = []
    equity = starting_balance if starting_balance is not None else 1.0
    equity_curve.append(equity)
    for _day, profit_abs in daily_rows:
        base_equity = equity if abs(equity) > 1e-12 else 1.0
        daily_returns.append(float(profit_abs) / float(base_equity))
        equity += float(profit_abs)
        equity_curve.append(equity)
        if profit_abs > 1e-9:
            positive_days += 1
        elif profit_abs < -1e-9:
            negative_days += 1
        else:
            flat_days += 1

    if starting_balance is None or final_balance is None:
        if equity_curve:
            starting_balance = equity_curve[0]
            final_balance = equity_curve[-1]

    daily_mean = float(sum(daily_returns) / len(daily_returns)) if daily_returns else 0.0
    daily_std = _stdev(daily_returns)
    downside_returns = [r for r in daily_returns if r < 0]
    downside_std = _stdev(downside_returns)
    daily_sharpe = _safe_ratio(daily_mean, daily_std)
    daily_sortino = _safe_ratio(daily_mean, downside_std if downside_std > 0 else daily_std)

    peak = equity_curve[0] if equity_curve else (starting_balance or 1.0)
    equity_drawdown_pct = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            equity_drawdown_pct = max(equity_drawdown_pct, (peak - value) / peak * 100.0)
    max_drawdown_pct = max(
        value
        for value in (
            equity_drawdown_pct,
            native_drawdown_pct or 0.0,
        )
    )

    positive_days_ratio = _safe_ratio(float(positive_days), float(len(daily_rows))) if daily_rows else 0.0
    active_days = positive_days + negative_days
    active_days_ratio = _safe_ratio(float(active_days), float(len(daily_rows))) if daily_rows else 0.0

    trade_profit_ratios = [
        float(ratio)
        for ratio in (as_float(trade.get("profit_ratio")) for trade in trades_rows)
        if ratio is not None and math.isfinite(float(ratio))
    ]
    trade_profit_mean_pct = float(sum(trade_profit_ratios) / len(trade_profit_ratios) * 100.0) if trade_profit_ratios else 0.0
    trade_profit_median_pct = float(statistics.median(trade_profit_ratios) * 100.0) if trade_profit_ratios else 0.0
    trade_profit_std_pct = _stdev(trade_profit_ratios) * 100.0 if trade_profit_ratios else 0.0

    realized_winrate = winrate if winrate is not None else None
    if realized_winrate is None and comparison_row is not None:
        realized_winrate = as_float(comparison_row.get("winrate"))
    if realized_winrate is None and trade_profit_ratios:
        realized_winrate = _safe_ratio(
            float(sum(1 for ratio in trade_profit_ratios if ratio > 0)),
            float(len(trade_profit_ratios)),
        )

    fee_total_abs = 0.0
    for trade in trades_rows:
        stake_amount = as_float(trade.get("stake_amount")) or 0.0
        profit_abs = as_float(trade.get("profit_abs")) or 0.0
        fee_open = as_float(trade.get("fee_open")) or 0.0
        fee_close = as_float(trade.get("fee_close")) or 0.0
        gross_exit_notional = max(0.0, stake_amount + profit_abs)
        fee_total_abs += stake_amount * fee_open
        fee_total_abs += gross_exit_notional * fee_close
    fee_drag_pct = 0.0
    if starting_balance is not None and starting_balance > 0:
        fee_drag_pct = fee_total_abs / starting_balance * 100.0

    profit_total_pct_safe = float(profit_total_pct or 0.0)
    return_over_drawdown = _safe_ratio(profit_total_pct_safe, max(1.0, max_drawdown_pct))
    realistic_calmar = return_over_drawdown

    metric_flags: list[str] = []
    if native_sharpe is not None and abs(native_sharpe) > max(6.0, abs(daily_sharpe) * 5.0):
        metric_flags.append("native_sharpe_inflated")
    if native_sortino is not None and abs(native_sortino) > max(8.0, abs(daily_sortino) * 5.0):
        metric_flags.append("native_sortino_inflated")
    if native_calmar is not None and abs(native_calmar) > max(10.0, abs(realistic_calmar) * 5.0):
        metric_flags.append("native_calmar_inflated")
    if native_drawdown_pct is not None and equity_drawdown_pct > 0:
        dd_ratio = max(native_drawdown_pct, equity_drawdown_pct) / max(
            min(native_drawdown_pct, equity_drawdown_pct), 1e-9
        )
        if dd_ratio > 4.0:
            metric_flags.append("drawdown_mismatch")

    return {
        "starting_balance": starting_balance,
        "final_balance": final_balance,
        "observation_days": int(len(daily_rows)),
        "active_days": int(active_days),
        "positive_days": int(positive_days),
        "negative_days": int(negative_days),
        "flat_days": int(flat_days),
        "positive_days_ratio": round(float(positive_days_ratio), 6),
        "active_days_ratio": round(float(active_days_ratio), 6),
        "daily_return_mean": round(float(daily_mean), 8),
        "daily_return_std": round(float(daily_std), 8),
        "daily_downside_std": round(float(downside_std), 8),
        "realistic_sharpe": round(float(daily_sharpe), 6),
        "realistic_sortino": round(float(daily_sortino), 6),
        "realistic_calmar": round(float(realistic_calmar), 6),
        "return_over_drawdown": round(float(return_over_drawdown), 6),
        "max_drawdown_pct": round(float(max_drawdown_pct), 6),
        "equity_drawdown_pct": round(float(equity_drawdown_pct), 6),
        "native_max_drawdown_pct": round(float(native_drawdown_pct), 6)
        if native_drawdown_pct is not None
        else None,
        "native_max_drawdown_account_raw": native_drawdown_raw,
        "native_sharpe": native_sharpe,
        "native_sortino": native_sortino,
        "native_calmar": native_calmar,
        "native_profit_factor": native_profit_factor,
        "native_expectancy": native_expectancy,
        "native_expectancy_ratio": native_expectancy_ratio,
        "native_sqn": native_sqn,
        "native_cagr": native_cagr,
        "trade_profit_mean_pct": round(float(trade_profit_mean_pct), 6),
        "trade_profit_median_pct": round(float(trade_profit_median_pct), 6),
        "trade_profit_std_pct": round(float(trade_profit_std_pct), 6),
        "fee_total_abs": round(float(fee_total_abs), 6),
        "fee_drag_pct": round(float(fee_drag_pct), 6),
        "metrics_trusted": not bool(metric_flags),
        "metric_flags": metric_flags,
        "profit_positive": profit_total_pct_safe > 0.0,
        "trade_count": int(trades_count or 0),
        "realized_winrate": realized_winrate,
        "period_days": int(period_days or len(daily_rows) or 0),
    }


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

    trades_per_day = None
    if trades_count is not None:
        duration_days = as_float(strategy_metrics.get("backtest_days"))
        if duration_days and duration_days > 0:
            trades_per_day = round(trades_count / duration_days, 4)

    derived = _derive_metrics(
        strategy_metrics=strategy_metrics,
        comparison_row=comparison_row,
        profit_total_pct=as_float(profit_total_pct),
        trades_count=trades_count,
        winrate=as_float(winrate),
    )

    return {
        "source": zip_path.name,
        "strategy": strategy_name,
        "profit_total_pct": as_float(profit_total_pct),
        "profit_total_abs": as_float(strategy_metrics.get("profit_total_abs")),
        "trades": trades_count,
        "avg_profit_pct": as_float(avg_profit_pct),
        "winrate": as_float(winrate),
        "max_drawdown_abs": as_float(max_drawdown_abs),
        "max_drawdown_account": derived.get("max_drawdown_pct"),
        "max_drawdown_pct": derived.get("max_drawdown_pct"),
        "sharpe": as_float(strategy_metrics.get("sharpe")),
        "sortino": as_float(strategy_metrics.get("sortino")),
        "calmar": as_float(strategy_metrics.get("calmar")),
        "profit_factor": as_float(strategy_metrics.get("profit_factor")),
        "expectancy": as_float(strategy_metrics.get("expectancy")),
        "expectancy_ratio": as_float(strategy_metrics.get("expectancy_ratio")),
        "sqn": as_float(strategy_metrics.get("sqn")),
        "cagr": as_float(strategy_metrics.get("cagr")),
        "realistic_sharpe": derived.get("realistic_sharpe"),
        "realistic_sortino": derived.get("realistic_sortino"),
        "realistic_calmar": derived.get("realistic_calmar"),
        "return_over_drawdown": derived.get("return_over_drawdown"),
        "positive_days_ratio": derived.get("positive_days_ratio"),
        "active_days_ratio": derived.get("active_days_ratio"),
        "observation_days": derived.get("observation_days"),
        "active_days": derived.get("active_days"),
        "positive_days": derived.get("positive_days"),
        "negative_days": derived.get("negative_days"),
        "flat_days": derived.get("flat_days"),
        "daily_return_mean": derived.get("daily_return_mean"),
        "daily_return_std": derived.get("daily_return_std"),
        "daily_downside_std": derived.get("daily_downside_std"),
        "native_sharpe": derived.get("native_sharpe"),
        "native_sortino": derived.get("native_sortino"),
        "native_calmar": derived.get("native_calmar"),
        "native_max_drawdown_pct": derived.get("native_max_drawdown_pct"),
        "native_max_drawdown_account_raw": derived.get("native_max_drawdown_account_raw"),
        "metrics_trusted": derived.get("metrics_trusted"),
        "metric_flags": derived.get("metric_flags"),
        "trade_profit_mean_pct": derived.get("trade_profit_mean_pct"),
        "trade_profit_median_pct": derived.get("trade_profit_median_pct"),
        "trade_profit_std_pct": derived.get("trade_profit_std_pct"),
        "fee_total_abs": derived.get("fee_total_abs"),
        "fee_drag_pct": derived.get("fee_drag_pct"),
        "trades_per_day": trades_per_day,
        "max_consecutive_wins": _as_int(strategy_metrics.get("max_consecutive_wins")),
        "max_consecutive_losses": _as_int(strategy_metrics.get("max_consecutive_losses")),
        "holding_avg_s": as_float(strategy_metrics.get("holding_avg_s")),
        "draw_days": _as_int(strategy_metrics.get("draw_days")),
        "winning_days": _as_int(strategy_metrics.get("winning_days")),
        "losing_days": _as_int(strategy_metrics.get("losing_days")),
        "profit_median": as_float(strategy_metrics.get("profit_median")),
        "market_change": as_float(strategy_metrics.get("market_change")),
        "best_pair": strategy_metrics.get("best_pair", {}).get("key"),
        "worst_pair": strategy_metrics.get("worst_pair", {}).get("key"),
        "backtest_timerange": f"{strategy_metrics.get('backtest_start')} -> {strategy_metrics.get('backtest_end')}",
        "strategy_comparison": comparison,
        "results_per_pair": strategy_metrics.get("results_per_pair", []),
        "daily_profit": strategy_metrics.get("daily_profit", []),
        "starting_balance": derived.get("starting_balance"),
        "final_balance": derived.get("final_balance"),
        "period_days": derived.get("period_days"),
    }


def read_backtest_trades(zip_path: Path) -> Optional[Any]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            trade_members = [
                name
                for name in zf.namelist()
                if name.endswith(".json") and name.startswith("trades-")
            ]
            if trade_members:
                return json.loads(zf.read(trade_members[0]))
            json_members = [
                name
                for name in zf.namelist()
                if name.endswith(".json")
                and "_config" not in name
                and not name.startswith("trades-")
            ]
            if not json_members:
                return None
            data = json.loads(zf.read(json_members[0]))
            strategy_block = data.get("strategy", {})
            if not strategy_block:
                return None
            _name, strategy_metrics = next(iter(strategy_block.items()))
            return strategy_metrics.get("trades")
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
