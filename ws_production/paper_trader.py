"""Paper Trader — simulated live trading with real-time data.

Connects to exchange WebSocket for real prices, executes trades on paper,
tracks P&L, compares with backtest expectations.

Usage:
    from workspace.paper_trader import PaperTrader
    pt = PaperTrader(exchange="gate", pairs=["BTC/USDT"], initial_equity=1000)
    pt.submit_order("BTC/USDT", "buy", size_usd=100)
    pt.update_prices({"BTC/USDT": 84500.0})
    print(pt.status())
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PaperOrder:
    """A simulated order."""
    order_id: str
    pair: str
    side: str           # "buy" or "sell"
    size_usd: float
    price: float        # fill price
    timestamp: str
    fee_bps: float = 10.0

    @property
    def fee_usd(self) -> float:
        return self.size_usd * self.fee_bps / 10000.0

    @property
    def qty(self) -> float:
        return self.size_usd / self.price if self.price > 0 else 0.0


@dataclass
class PaperPosition:
    """Current position in a pair."""
    pair: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0

    def update_mark_price(self, price: float) -> None:
        if self.qty > 0 and price > 0:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.qty
        else:
            self.unrealized_pnl = 0.0


class PaperTrader:
    """Simulated trading engine for paper trading validation."""

    def __init__(
        self,
        *,
        initial_equity: float = 1000.0,
        fee_bps: float = 10.0,
        pairs: Optional[List[str]] = None,
    ):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.fee_bps = fee_bps
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: List[PaperOrder] = []
        self.prices: Dict[str, float] = {}
        self.equity_history: List[Dict[str, Any]] = []
        self._order_counter = 0

        if pairs:
            for pair in pairs:
                self.positions[pair] = PaperPosition(pair=pair)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current market prices."""
        self.prices.update(prices)
        for pair, pos in self.positions.items():
            if pair in prices:
                pos.update_mark_price(prices[pair])

    def mark_to_market(self, timestamp: Optional[str] = None) -> None:
        """Record an equity snapshot even when no order is executed."""
        self._snapshot_equity(timestamp=timestamp)

    def submit_order(
        self,
        pair: str,
        side: str,
        size_usd: float,
        price: Optional[float] = None,
    ) -> PaperOrder:
        """Submit a paper order. Uses current price if not specified."""
        if price is None:
            price = self.prices.get(pair, 0.0)
        if price <= 0:
            raise ValueError(f"No price available for {pair}")

        self._order_counter += 1
        order = PaperOrder(
            order_id=f"paper_{self._order_counter:06d}",
            pair=pair,
            side=side,
            size_usd=size_usd,
            price=price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fee_bps=self.fee_bps,
        )

        # Execute immediately (paper trading = instant fill)
        self._execute_order(order)
        self.orders.append(order)
        self._snapshot_equity()
        return order

    def _execute_order(self, order: PaperOrder) -> None:
        """Execute a paper order against positions."""
        if order.pair not in self.positions:
            self.positions[order.pair] = PaperPosition(pair=order.pair)
        pos = self.positions[order.pair]
        fee = order.fee_usd

        if order.side == "buy":
            # Add to position
            new_qty = order.qty
            total_cost = order.size_usd + fee
            if total_cost > self.cash:
                return  # insufficient funds
            old_value = pos.qty * pos.avg_entry_price
            pos.qty += new_qty
            if pos.qty > 0:
                pos.avg_entry_price = (old_value + order.size_usd) / pos.qty
            pos.total_fees += fee
            self.cash -= total_cost

        elif order.side == "sell":
            # Reduce position
            sell_qty = min(order.qty, pos.qty)
            if sell_qty <= 0:
                return
            pnl = (order.price - pos.avg_entry_price) * sell_qty
            pos.realized_pnl += pnl
            pos.qty -= sell_qty
            pos.total_fees += fee
            self.cash += sell_qty * order.price - fee
            if pos.qty <= 0:
                pos.qty = 0
                pos.avg_entry_price = 0
                pos.unrealized_pnl = 0

    def _snapshot_equity(self, timestamp: Optional[str] = None) -> None:
        """Record equity snapshot."""
        equity = self.total_equity()
        self.equity_history.append({
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "cash": self.cash,
            "positions_value": equity - self.cash,
            "n_orders": len(self.orders),
        })

    def total_equity(self) -> float:
        """Current total equity (cash + positions)."""
        positions_value = sum(
            pos.qty * self.prices.get(pos.pair, pos.avg_entry_price)
            for pos in self.positions.values()
            if pos.qty > 0
        )
        return self.cash + positions_value

    def total_pnl(self) -> float:
        """Total P&L since inception."""
        return self.total_equity() - self.initial_equity

    def total_fees(self) -> float:
        """Total fees paid."""
        return sum(pos.total_fees for pos in self.positions.values())

    def status(self) -> Dict[str, Any]:
        """Complete paper trading status."""
        equity = self.total_equity()
        pnl = self.total_pnl()
        peak = max((s["equity"] for s in self.equity_history), default=self.initial_equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0

        return {
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / self.initial_equity * 100, 2),
            "total_fees": round(self.total_fees(), 2),
            "total_orders": len(self.orders),
            "max_drawdown_pct": round(dd, 2),
            "positions": {
                pair: {
                    "qty": round(pos.qty, 8),
                    "avg_entry": round(pos.avg_entry_price, 2),
                    "unrealized_pnl": round(pos.unrealized_pnl, 2),
                    "realized_pnl": round(pos.realized_pnl, 2),
                }
                for pair, pos in self.positions.items()
                if pos.qty > 0 or pos.realized_pnl != 0
            },
        }

    def save_log(self, path: Optional[str | Path] = None) -> Path:
        """Save trading log to JSON."""
        if path is None:
            path = ROOT / "workspace" / "results" / f"paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "status": self.status(),
            "orders": [
                {
                    "id": o.order_id, "pair": o.pair, "side": o.side,
                    "size_usd": o.size_usd, "price": o.price,
                    "fee": o.fee_usd, "ts": o.timestamp,
                }
                for o in self.orders
            ],
            "equity_history": self.equity_history,
        }
        path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trader state for persistence."""
        return {
            "initial_equity": self.initial_equity,
            "cash": self.cash,
            "fee_bps": self.fee_bps,
            "prices": self.prices,
            "order_counter": self._order_counter,
            "positions": {
                pair: {
                    "pair": pos.pair,
                    "qty": pos.qty,
                    "avg_entry_price": pos.avg_entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_fees": pos.total_fees,
                }
                for pair, pos in self.positions.items()
            },
            "orders": [
                {
                    "order_id": order.order_id,
                    "pair": order.pair,
                    "side": order.side,
                    "size_usd": order.size_usd,
                    "price": order.price,
                    "timestamp": order.timestamp,
                    "fee_bps": order.fee_bps,
                }
                for order in self.orders
            ],
            "equity_history": self.equity_history,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PaperTrader":
        """Restore trader state from a persisted payload."""
        trader = cls(
            initial_equity=float(payload.get("initial_equity", 1000.0)),
            fee_bps=float(payload.get("fee_bps", 10.0)),
            pairs=[],
        )
        trader.cash = float(payload.get("cash", trader.initial_equity))
        trader.prices = {str(k): float(v) for k, v in (payload.get("prices") or {}).items()}
        trader._order_counter = int(payload.get("order_counter", 0))
        trader.positions = {
            str(pair): PaperPosition(
                pair=str(pair),
                qty=float((raw or {}).get("qty", 0.0)),
                avg_entry_price=float((raw or {}).get("avg_entry_price", 0.0)),
                unrealized_pnl=float((raw or {}).get("unrealized_pnl", 0.0)),
                realized_pnl=float((raw or {}).get("realized_pnl", 0.0)),
                total_fees=float((raw or {}).get("total_fees", 0.0)),
            )
            for pair, raw in (payload.get("positions") or {}).items()
        }
        trader.orders = [
            PaperOrder(
                order_id=str(raw.get("order_id", "")),
                pair=str(raw.get("pair", "")),
                side=str(raw.get("side", "")),
                size_usd=float(raw.get("size_usd", 0.0)),
                price=float(raw.get("price", 0.0)),
                timestamp=str(raw.get("timestamp", "")),
                fee_bps=float(raw.get("fee_bps", trader.fee_bps)),
            )
            for raw in (payload.get("orders") or [])
        ]
        trader.equity_history = list(payload.get("equity_history") or [])
        return trader


__all__ = ["PaperTrader", "PaperOrder", "PaperPosition"]
