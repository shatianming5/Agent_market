"""Forward-only paper trading on fresh market data."""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from workspace.paper_trader import PaperTrader
from workspace.pairs_engine import PairsEngine

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


class PairsPaperCycle:
    """Executes forward-only paper trading for spot pairs strategies."""

    def __init__(
        self,
        *,
        exchange: str = "gate",
        timeframe: str = "1h",
        fee_bps: float = 10.0,
        initial_equity: float = 1000.0,
        position_pct: float = 0.5,
        min_refresh_interval_sec: int = 300,
        paper_dir: Optional[str | Path] = None,
    ):
        self.exchange = exchange
        self.timeframe = timeframe
        self.fee_bps = fee_bps
        self.initial_equity = initial_equity
        self.position_pct = position_pct
        self.min_refresh_interval_sec = min_refresh_interval_sec
        self.paper_dir = Path(paper_dir or ROOT / "workspace" / "paper")
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self._exchange_client = None

    def run_pairs_strategy(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        pair_a = str(config.get("pair_a") or "")
        pair_b = str(config.get("pair_b") or "")
        if not pair_a or not pair_b:
            return {"ok": False, "error": "missing pair config", "strategy": name}

        params = self._resolve_params(config)
        sync_result = self.sync_market_data(pair_a, pair_b)
        merged = self._load_merged_frame(pair_a, pair_b)
        if merged.empty:
            return {"ok": False, "error": "no market data", "strategy": name, "sync": sync_result}

        state_path = self.paper_dir / f"{name}.json"
        state = self._load_state(state_path)
        latest_closed = merged["date"].iloc[-1]
        bootstrapped = False
        if state is None:
            trader = PaperTrader(
                initial_equity=self.initial_equity,
                fee_bps=self.fee_bps,
                pairs=[pair_a, pair_b],
            )
            trader.update_prices({
                pair_a: float(merged["price_a"].iloc[-1]),
                pair_b: float(merged["price_b"].iloc[-1]),
            })
            payload = {
                "strategy": name,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "pair_a": pair_a,
                "pair_b": pair_b,
                "params": params,
                "initialized_at": datetime.now(timezone.utc).isoformat(),
                "last_processed_at": latest_closed.isoformat(),
                "daily_equity": {},
                "trader": trader.to_dict(),
            }
            bootstrap_from = self._bootstrap_start(config, merged)
            if bootstrap_from is None:
                metrics = self._paper_metrics(
                    daily_equity=payload.get("daily_equity") or {},
                    current_equity=float(trader.total_equity()),
                )
                self._save_state(state_path, payload)
                return {
                    "ok": True,
                    "strategy": name,
                    "initialized": True,
                    "new_bars": 0,
                    "daily_equity": {},
                    "current_equity": trader.total_equity(),
                    "last_processed_at": latest_closed.isoformat(),
                    "state_path": str(state_path),
                    "sync": sync_result,
                    "params": params,
                    **metrics,
                }
            state = payload
            last_processed_at = bootstrap_from - _timeframe_delta(self.timeframe)
            bootstrapped = True
        else:
            trader = PaperTrader.from_dict(state.get("trader") or {})
            last_processed_at = self._coerce_timestamp(state.get("last_processed_at"))
            if last_processed_at is None:
                fallback = self._bootstrap_start(config, merged)
                last_processed_at = (fallback or latest_closed) - _timeframe_delta(self.timeframe)
        if last_processed_at.tzinfo is None:
            last_processed_at = last_processed_at.tz_localize("UTC")

        signals = self._signals_for_frame(merged, params)
        if "signal" not in signals.columns:
            signals = signals.assign(signal="hold")
        merged_with_signals = merged.merge(signals[["date", "signal"]], on="date", how="left")
        merged_with_signals["signal"] = merged_with_signals["signal"].fillna("hold")
        new_rows = merged_with_signals.loc[merged_with_signals["date"] > last_processed_at].reset_index(drop=True)
        orders_before = len(trader.orders)
        for row in new_rows.itertuples(index=False):
            trader.update_prices({
                pair_a: float(row.price_a),
                pair_b: float(row.price_b),
            })
            self._apply_signal(
                trader=trader,
                pair_a=pair_a,
                pair_b=pair_b,
                signal=str(row.signal),
                price_a=float(row.price_a),
                price_b=float(row.price_b),
            )
            trader.mark_to_market(timestamp=row.date.isoformat())
            state.setdefault("daily_equity", {})[row.date.date().isoformat()] = float(trader.total_equity())
            state["last_processed_at"] = row.date.isoformat()

        state["params"] = params
        state["trader"] = trader.to_dict()
        self._save_state(state_path, state)
        metrics = self._paper_metrics(
            daily_equity=state.get("daily_equity") or {},
            current_equity=float(trader.total_equity()),
        )
        return {
            "ok": True,
            "strategy": name,
            "initialized": False,
            "bootstrapped": bootstrapped,
            "new_bars": int(len(new_rows)),
            "orders_added": int(len(trader.orders) - orders_before),
            "daily_equity": dict(state.get("daily_equity") or {}),
            "current_equity": float(trader.total_equity()),
            "last_processed_at": str(state.get("last_processed_at") or latest_closed.isoformat()),
            "state_path": str(state_path),
            "sync": sync_result,
            "params": params,
            **metrics,
        }

    def _resolve_params(self, config: Dict[str, Any]) -> Dict[str, float]:
        params = config.get("params") or {}
        return {
            "lookback": int(params.get("lookback", config.get("lookback", 30))),
            "entry_z": float(params.get("entry_z", config.get("entry_z", 2.0))),
            "exit_z": float(params.get("exit_z", config.get("exit_z", 0.5))),
        }

    def _state_path(self, strategy_name: str) -> Path:
        return self.paper_dir / f"{strategy_name}.json"

    def _load_state(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _coerce_timestamp(self, raw: Any) -> Optional[pd.Timestamp]:
        if raw in (None, ""):
            return None
        stamp = pd.Timestamp(raw)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp

    def _save_state(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _data_path(self, pair: str) -> Path:
        slug = pair.replace("/", "_")
        return ROOT / "user_data" / "data" / self.exchange / f"{slug}-{self.timeframe}.feather"

    def sync_market_data(self, pair_a: str, pair_b: str) -> Dict[str, Any]:
        return {
            pair_a: self._sync_pair_data(pair_a),
            pair_b: self._sync_pair_data(pair_b),
        }

    def _sync_pair_data(self, pair: str) -> Dict[str, Any]:
        path = self._data_path(pair)
        existing = self._read_single_pair(path)
        now = datetime.now(timezone.utc)
        latest_expected = _closed_candle_start(now, self.timeframe) - _timeframe_delta(self.timeframe)
        latest_existing = existing["date"].iloc[-1] if not existing.empty else None
        stale_bars = self._stale_bars(latest_existing, latest_expected)
        if (
            path.exists()
            and (time.time() - path.stat().st_mtime) < self.min_refresh_interval_sec
            and stale_bars <= 0
        ):
            return {
                "ok": True,
                "pair": pair,
                "rows": int(len(existing)),
                "latest_candle": latest_existing.isoformat() if latest_existing is not None else None,
                "skipped": "fresh_cache",
            }

        since_ms = None
        if not existing.empty:
            overlap = _timeframe_delta(self.timeframe) * 4
            since = existing["date"].iloc[-1].to_pydatetime() - overlap
            since_ms = int(since.timestamp() * 1000)
        max_batches = self._required_batches(stale_bars)
        try:
            fetched = self._fetch_pair_data(pair=pair, since_ms=since_ms, max_batches=max_batches)
        except Exception as exc:
            return {"ok": False, "pair": pair, "error": str(exc)[:200]}

        if fetched.empty and not existing.empty:
            return {
                "ok": True,
                "pair": pair,
                "rows": int(len(existing)),
                "latest_candle": latest_existing.isoformat() if latest_existing is not None else None,
                "stale_bars": stale_bars,
                "skipped": "no_remote_update",
            }

        merged = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
        if merged.empty:
            return {"ok": False, "pair": pair, "error": "no data"}
        merged["date"] = pd.to_datetime(merged["date"], utc=True)
        cutoff = _closed_candle_start(now, self.timeframe)
        merged = merged.loc[merged["date"] < cutoff]
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_feather(path)
        latest_merged = merged["date"].iloc[-1] if not merged.empty else None
        return {
            "ok": True,
            "pair": pair,
            "rows": int(len(merged)),
            "latest_candle": latest_merged.isoformat() if latest_merged is not None else None,
            "stale_bars_before": stale_bars,
            "fetched_rows": int(len(fetched)),
            "max_batches": max_batches,
        }

    def _read_single_pair(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        frame = pd.read_feather(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        return frame

    def _fetch_pair_data(self, *, pair: str, since_ms: Optional[int], max_batches: int) -> pd.DataFrame:
        import ccxt
        from workspace.download_data import download_pair

        if self._exchange_client is None:
            exchange_cls = getattr(ccxt, self.exchange)
            self._exchange_client = exchange_cls({"enableRateLimit": True})
            self._exchange_client.load_markets()
        if pair not in self._exchange_client.markets:
            raise ValueError(f"pair not supported: {pair}")

        if since_ms is None:
            delta = _timeframe_delta(self.timeframe)
            since = datetime.now(timezone.utc) - (delta * 500)
            since_ms = int(since.timestamp() * 1000)
        return download_pair(
            self._exchange_client,
            pair,
            self.timeframe,
            since_ms=since_ms,
            limit=1000,
            max_batches=max_batches,
        )

    def _load_merged_frame(self, pair_a: str, pair_b: str) -> pd.DataFrame:
        return PairsEngine(pair_a, pair_b, exchange=self.exchange, timeframe=self.timeframe).load_data().copy()

    def _signals_for_frame(self, frame: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
        pe = PairsEngine("A/USDT", "B/USDT", exchange=self.exchange, timeframe=self.timeframe)
        pe._df = frame.reset_index(drop=True).copy()
        return pe.generate_signals(
            lookback=int(params["lookback"]),
            entry_z=float(params["entry_z"]),
            exit_z=float(params["exit_z"]),
        )

    def _signal_for_history(self, history: pd.DataFrame, params: Dict[str, float]) -> str:
        signals = self._signals_for_frame(history, params)
        return str(signals.iloc[-1]["signal"])

    def _bootstrap_start(self, config: Dict[str, Any], merged: pd.DataFrame) -> Optional[pd.Timestamp]:
        raw = config.get("paper_started_at")
        if not raw:
            return None
        stamp = self._coerce_timestamp(raw)
        if stamp is None:
            return None
        eligible = merged.loc[merged["date"] >= stamp, "date"]
        if eligible.empty:
            return None
        return eligible.iloc[0]

    def _apply_signal(
        self,
        *,
        trader: PaperTrader,
        pair_a: str,
        pair_b: str,
        signal: str,
        price_a: float,
        price_b: float,
    ) -> None:
        held_pair = self._held_pair(trader, pair_a, pair_b)
        if signal == "enter_long_spread":
            if held_pair == pair_b:
                self._close_pair(trader, pair_b, price_b)
                held_pair = None
            if held_pair is None:
                self._open_pair(trader, pair_a, price_a)
        elif signal == "enter_short_spread":
            if held_pair == pair_a:
                self._close_pair(trader, pair_a, price_a)
                held_pair = None
            if held_pair is None:
                self._open_pair(trader, pair_b, price_b)
        elif signal == "exit" and held_pair is not None:
            if held_pair == pair_a:
                self._close_pair(trader, pair_a, price_a)
            elif held_pair == pair_b:
                self._close_pair(trader, pair_b, price_b)

    def _held_pair(self, trader: PaperTrader, pair_a: str, pair_b: str) -> Optional[str]:
        for pair in (pair_a, pair_b):
            pos = trader.positions.get(pair)
            if pos and pos.qty > 0:
                return pair
        return None

    def _open_pair(self, trader: PaperTrader, pair: str, price: float) -> None:
        size_usd = min(trader.cash, trader.total_equity() * self.position_pct)
        if size_usd <= 0 or price <= 0:
            return
        trader.submit_order(pair, "buy", size_usd=size_usd, price=price)

    def _close_pair(self, trader: PaperTrader, pair: str, price: float) -> None:
        pos = trader.positions.get(pair)
        if not pos or pos.qty <= 0 or price <= 0:
            return
        trader.submit_order(pair, "sell", size_usd=pos.qty * price, price=price)

    def _stale_bars(self, latest_existing: Optional[pd.Timestamp], latest_expected: pd.Timestamp) -> int:
        if latest_existing is None:
            return 10_000
        if latest_existing.tzinfo is None:
            latest_existing = latest_existing.tz_localize("UTC")
        if latest_existing >= latest_expected:
            return 0
        delta = _timeframe_delta(self.timeframe)
        gap = latest_expected.to_pydatetime() - latest_existing.to_pydatetime()
        return max(0, int(gap.total_seconds() // delta.total_seconds()))

    def _required_batches(self, stale_bars: int) -> int:
        if stale_bars <= 0:
            return 3
        return max(3, min(48, int(math.ceil((stale_bars + 96) / 1000.0))))

    def _paper_metrics(self, *, daily_equity: Dict[str, float], current_equity: float) -> Dict[str, Any]:
        ordered = sorted((str(day), float(value)) for day, value in daily_equity.items())
        cumulative_return_pct = ((current_equity / self.initial_equity) - 1.0) * 100.0 if self.initial_equity > 0 else 0.0
        latest_day_return_pct = None
        if ordered:
            previous = self.initial_equity
            for _, equity in ordered:
                latest_day_return_pct = 0.0 if previous <= 0 else ((equity / previous) - 1.0) * 100.0
                previous = equity
        return {
            "paper_days": len(ordered),
            "cumulative_return_pct": round(float(cumulative_return_pct), 6),
            "latest_day_return_pct": round(float(latest_day_return_pct), 6) if latest_day_return_pct is not None else None,
        }


__all__ = ["PairsPaperCycle"]
