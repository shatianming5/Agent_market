"""Paper trading replay for freqtrade-backed strategies."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from workspace.backtest_api import run_backtest

ROOT = Path(__file__).resolve().parents[1]


def _timeframe_delta(timeframe: str) -> timedelta:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    mapping = {
        "m": timedelta(minutes=value),
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
    }
    if unit not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[unit]


def _closed_candle_start(now: datetime, timeframe: str) -> pd.Timestamp:
    delta = _timeframe_delta(timeframe)
    seconds = int(delta.total_seconds())
    epoch = int(now.timestamp())
    floored = epoch - (epoch % seconds)
    return pd.Timestamp(floored, unit="s", tz="UTC")


class FreqtradePaperCycle:
    """Replays freqtrade backtest trades into day-level paper returns."""

    def __init__(
        self,
        *,
        exchange: str = "gate",
        timeframe: str = "1h",
        initial_equity: float = 1000.0,
        paper_dir: Optional[str | Path] = None,
    ):
        self.exchange = exchange
        self.timeframe = timeframe
        self.initial_equity = float(initial_equity)
        self.paper_dir = Path(paper_dir or ROOT / "workspace" / "paper")
        self.paper_dir.mkdir(parents=True, exist_ok=True)

    def run_strategy(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        strategy_path = self._resolve_path(config.get("strategy_path"))
        if strategy_path is None or not strategy_path.exists():
            return {"ok": False, "strategy": name, "error": "missing strategy_path"}

        state_path = self.paper_dir / f"{name}.json"
        state = self._load_state(state_path) or {}
        latest_closed = _closed_candle_start(datetime.now(timezone.utc), self.timeframe) - _timeframe_delta(self.timeframe)
        start_stamp = self._paper_start(config, state, latest_closed)
        start_day = start_stamp.date()
        end_day = latest_closed.date()
        if end_day < start_day:
            end_day = start_day
        timerange = f"{start_day.strftime('%Y%m%d')}-{end_day.strftime('%Y%m%d')}"

        bt = run_backtest(
            strategy_path=strategy_path,
            strategy_name=config.get("strategy_name"),
            config_path=self._resolve_path(config.get("config_path")) or None,
            pairs=list(config.get("pairs") or []),
            exchange_name=str(config.get("exchange") or self.exchange),
            timeframe=str(config.get("timeframe") or self.timeframe),
            timerange=timerange,
            extra_args=list(config.get("extra_args") or []),
        )
        if not bt.get("ok"):
            return {
                "ok": False,
                "strategy": name,
                "error": str(bt.get("error") or "backtest failed"),
                "backtest": bt,
            }

        details = self._load_backtest_details(bt.get("backtest_zip"), strategy_name=str(bt.get("strategy") or ""))
        starting_balance = float(details.get("starting_balance") or self.initial_equity)
        daily_equity = self._daily_equity(
            start_day=start_day,
            end_day=end_day,
            starting_balance=starting_balance,
            trades=list(details.get("trades") or []),
        )
        previous_days = set((state.get("daily_equity") or {}).keys())
        current_equity = float(next(reversed(daily_equity.values()), starting_balance))
        payload = {
            "strategy": name,
            "strategy_path": str(strategy_path),
            "strategy_name": str(bt.get("strategy") or config.get("strategy_name") or ""),
            "exchange": str(config.get("exchange") or self.exchange),
            "timeframe": str(config.get("timeframe") or self.timeframe),
            "started_at": start_stamp.isoformat(),
            "last_processed_at": latest_closed.isoformat(),
            "daily_equity": daily_equity,
            "current_equity": current_equity,
            "backtest_run_id": bt.get("run_id"),
            "backtest_zip": bt.get("backtest_zip"),
            "trades": int(bt.get("trades") or len(details.get("trades") or [])),
            "params": dict(config.get("params") or {}),
        }
        self._save_state(state_path, payload)
        metrics = self._paper_metrics(daily_equity=daily_equity, current_equity=current_equity, initial_equity=starting_balance)
        return {
            "ok": True,
            "strategy": name,
            "initialized": not bool(state),
            "backfilled": bool(not state and start_stamp < latest_closed),
            "new_days": max(0, len(set(daily_equity.keys()) - previous_days)),
            "orders_added": int(bt.get("trades") or 0),
            "daily_equity": daily_equity,
            "current_equity": current_equity,
            "last_processed_at": latest_closed.isoformat(),
            "state_path": str(state_path),
            "strategy_path": str(strategy_path),
            "timerange": timerange,
            "backtest_run_id": bt.get("run_id"),
            "backtest_zip": bt.get("backtest_zip"),
            "trades": int(bt.get("trades") or 0),
            **metrics,
        }

    def _paper_start(self, config: Dict[str, Any], state: Dict[str, Any], latest_closed: pd.Timestamp) -> pd.Timestamp:
        for raw in (
            state.get("started_at"),
            config.get("paper_started_at"),
            state.get("last_processed_at"),
        ):
            stamp = self._coerce_timestamp(raw)
            if stamp is not None:
                return stamp
        return latest_closed

    def _resolve_path(self, raw: Any) -> Optional[Path]:
        if raw in (None, ""):
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve()

    def _load_state(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_state(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _coerce_timestamp(self, raw: Any) -> Optional[pd.Timestamp]:
        if raw in (None, ""):
            return None
        stamp = pd.Timestamp(raw)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp

    def _load_backtest_details(self, backtest_zip: Any, *, strategy_name: str) -> Dict[str, Any]:
        if backtest_zip in (None, ""):
            return {"trades": [], "starting_balance": self.initial_equity}
        zip_path = Path(str(backtest_zip))
        if not zip_path.is_absolute():
            zip_path = ROOT / zip_path
        with zipfile.ZipFile(zip_path) as zf:
            name = next(
                item for item in zf.namelist()
                if item.endswith(".json") and "_config" not in item
            )
            payload = json.loads(zf.read(name))
        strategies = payload.get("strategy") or {}
        block = strategies.get(strategy_name) if strategy_name else None
        if block is None and strategies:
            block = next(iter(strategies.values()))
        block = block or {}
        return {
            "trades": list(block.get("trades") or []),
            "starting_balance": float(block.get("starting_balance") or self.initial_equity),
            "final_balance": float(block.get("final_balance") or block.get("starting_balance") or self.initial_equity),
        }

    def _daily_equity(
        self,
        *,
        start_day,
        end_day,
        starting_balance: float,
        trades: list[dict[str, Any]],
    ) -> Dict[str, float]:
        realized_by_day: Dict[str, float] = {}
        for trade in trades:
            close_raw = trade.get("close_date") or trade.get("close_timestamp")
            if close_raw in (None, ""):
                continue
            if isinstance(close_raw, (int, float)):
                close_stamp = pd.Timestamp(int(close_raw), unit="ms", tz="UTC")
            else:
                close_stamp = pd.Timestamp(close_raw)
                if close_stamp.tzinfo is None:
                    close_stamp = close_stamp.tz_localize("UTC")
            close_day = close_stamp.date().isoformat()
            realized_by_day[close_day] = realized_by_day.get(close_day, 0.0) + float(trade.get("profit_abs") or 0.0)

        equity = float(starting_balance)
        cursor = start_day
        daily_equity: Dict[str, float] = {}
        while cursor <= end_day:
            key = cursor.isoformat()
            equity += float(realized_by_day.get(key, 0.0))
            daily_equity[key] = round(equity, 8)
            cursor += timedelta(days=1)
        return daily_equity

    def _paper_metrics(
        self,
        *,
        daily_equity: Dict[str, float],
        current_equity: float,
        initial_equity: float,
    ) -> Dict[str, Any]:
        ordered = sorted((str(day), float(value)) for day, value in daily_equity.items())
        cumulative_return_pct = ((current_equity / initial_equity) - 1.0) * 100.0 if initial_equity > 0 else 0.0
        latest_day_return_pct = None
        if ordered:
            previous = initial_equity
            for _, equity in ordered:
                latest_day_return_pct = 0.0 if previous <= 0 else ((equity / previous) - 1.0) * 100.0
                previous = equity
        return {
            "paper_days": len(ordered),
            "cumulative_return_pct": round(float(cumulative_return_pct), 8),
            "latest_day_return_pct": round(float(latest_day_return_pct), 8) if latest_day_return_pct is not None else None,
        }


__all__ = ["FreqtradePaperCycle"]
