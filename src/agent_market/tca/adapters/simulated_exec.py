from __future__ import annotations

"""
Simulated execution adapter (plan.md alignment).

This module provides a minimal, deterministic simulated execution path that can
be used to generate synthetic `orders`/`fills` for TCA studies when real exchange
order/fill logs are unavailable.

The current implementation focuses on a simple market-order model so that it is
usable in offline tests without external data dependencies.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple


def _utc_iso(ts: Optional[datetime] = None) -> str:
    return (ts or datetime.now(timezone.utc)).isoformat()


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class SimOrder:
    order_id: str
    symbol: str
    side: Side
    qty: float
    submit_ts: str
    type: Literal["MKT", "LMT"] = "MKT"
    limit_px: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.type,
            "qty": float(self.qty),
            "limit_px": float(self.limit_px) if self.limit_px is not None else None,
            "submit_ts": self.submit_ts,
            "cancel_ts": None,
        }


@dataclass(frozen=True, slots=True)
class SimFill:
    order_id: str
    fill_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    ts: str
    fee: float = 0.0
    liquidity: Optional[Literal["TAKER", "MAKER"]] = "TAKER"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "fill_id": self.fill_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": float(self.qty),
            "price": float(self.price),
            "ts": self.ts,
            "fee": float(self.fee),
            "liquidity": self.liquidity,
        }


def simulate_market_orders(
    *,
    symbol: str,
    sides: Sequence[Side],
    qtys: Sequence[float],
    prices: Sequence[float],
    timestamps: Optional[Sequence[str]] = None,
    fee_bps: float = 0.0,
) -> Tuple[List[SimOrder], List[SimFill]]:
    """
    Deterministic market-order simulation: each (side, qty, price, ts) becomes an order+fill.
    """

    if len(sides) != len(qtys) or len(qtys) != len(prices):
        raise ValueError("sides/qtys/prices must have the same length")
    if timestamps is not None and len(timestamps) != len(prices):
        raise ValueError("timestamps must have the same length as prices")

    orders: List[SimOrder] = []
    fills: List[SimFill] = []
    for i, (side, qty, px) in enumerate(zip(sides, qtys, prices)):
        ts = str(timestamps[i]) if timestamps is not None else _utc_iso()
        order_id = f"sim_{i:06d}"
        orders.append(SimOrder(order_id=order_id, symbol=str(symbol), side=side, qty=float(qty), submit_ts=ts))

        fee = float(abs(float(qty) * float(px)) * float(fee_bps) / 10000.0)
        fills.append(
            SimFill(
                order_id=order_id,
                fill_id=f"{order_id}_fill0",
                symbol=str(symbol),
                side=side,
                qty=float(qty),
                price=float(px),
                ts=ts,
                fee=fee,
                liquidity="TAKER",
            )
        )

    return orders, fills


__all__ = [
    "Side",
    "SimFill",
    "SimOrder",
    "simulate_market_orders",
]

