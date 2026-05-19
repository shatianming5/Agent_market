"""Risk Manager — position sizing, drawdown circuit breaker, exposure limits.

Usage:
    from workspace.risk_manager import RiskManager
    rm = RiskManager(max_drawdown_pct=5.0, max_position_pct=25.0)
    decision = rm.check_trade(signal=0.8, current_equity=980, peak_equity=1000)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TradeDecision:
    """Risk manager's decision on a proposed trade."""
    allowed: bool
    position_size_pct: float    # % of equity to allocate (0 = blocked)
    reason: str
    drawdown_pct: float         # current drawdown from peak
    kelly_fraction: float       # Kelly criterion position size


class RiskManager:
    """Portfolio-level risk management."""

    def __init__(
        self,
        *,
        initial_equity: float = 1000.0,
        max_drawdown_pct: float = 5.0,
        max_position_pct: float = 25.0,
        max_total_exposure_pct: float = 80.0,
        max_correlated_exposure_pct: float = 40.0,
        consecutive_loss_limit: int = 5,
        kelly_fraction_cap: float = 0.25,
    ):
        self.initial_equity = initial_equity
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_correlated_exposure_pct = max_correlated_exposure_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.kelly_fraction_cap = kelly_fraction_cap

        # State
        self.peak_equity = initial_equity
        self.current_equity = initial_equity
        self.consecutive_losses = 0
        self.total_exposure_pct = 0.0
        self.circuit_breaker_active = False

    def update_equity(self, equity: float) -> None:
        """Update equity after a trade result."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.consecutive_losses = 0

    def record_trade_result(self, profit: float) -> None:
        """Record a trade result to update risk state."""
        self.current_equity += profit
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak as percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity * 100.0

    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion position fraction.

        Kelly % = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
        Capped at kelly_fraction_cap for safety (fractional Kelly).
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        R = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / R
        kelly = max(0.0, kelly)
        return min(kelly, self.kelly_fraction_cap)

    def check_trade(
        self,
        *,
        signal_strength: float = 1.0,
        win_rate: float = 0.5,
        avg_win_pct: float = 1.0,
        avg_loss_pct: float = 1.0,
        pair_correlation: float = 0.0,
    ) -> TradeDecision:
        """Check if a new trade should be allowed and at what size.

        Returns TradeDecision with allowed, position_size_pct, and reason.
        """
        dd = self.current_drawdown_pct()

        # Circuit breaker: max drawdown exceeded
        if dd >= self.max_drawdown_pct:
            self.circuit_breaker_active = True
            return TradeDecision(
                allowed=False,
                position_size_pct=0.0,
                reason=f"CIRCUIT BREAKER: drawdown {dd:.1f}% >= limit {self.max_drawdown_pct}%",
                drawdown_pct=dd,
                kelly_fraction=0.0,
            )

        # Reset circuit breaker if equity recovers
        if self.circuit_breaker_active and dd < self.max_drawdown_pct * 0.5:
            self.circuit_breaker_active = False

        if self.circuit_breaker_active:
            return TradeDecision(
                allowed=False,
                position_size_pct=0.0,
                reason=f"CIRCUIT BREAKER active, waiting for recovery (DD={dd:.1f}%)",
                drawdown_pct=dd,
                kelly_fraction=0.0,
            )

        # Consecutive loss limit
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return TradeDecision(
                allowed=False,
                position_size_pct=0.0,
                reason=f"Consecutive losses: {self.consecutive_losses} >= {self.consecutive_loss_limit}",
                drawdown_pct=dd,
                kelly_fraction=0.0,
            )

        # Total exposure limit
        if self.total_exposure_pct >= self.max_total_exposure_pct:
            return TradeDecision(
                allowed=False,
                position_size_pct=0.0,
                reason=f"Total exposure {self.total_exposure_pct:.0f}% >= {self.max_total_exposure_pct}%",
                drawdown_pct=dd,
                kelly_fraction=0.0,
            )

        # Kelly sizing
        kelly = self.kelly_criterion(win_rate, avg_win_pct, avg_loss_pct)

        # Scale by signal strength and drawdown proximity
        dd_scale = max(0.2, 1.0 - dd / self.max_drawdown_pct)
        position_pct = kelly * 100.0 * abs(signal_strength) * dd_scale
        position_pct = min(position_pct, self.max_position_pct)

        # Reduce for correlated assets
        if pair_correlation > 0.7:
            position_pct *= 0.5

        if position_pct < 1.0:
            return TradeDecision(
                allowed=False,
                position_size_pct=0.0,
                reason=f"Position too small: {position_pct:.1f}% (Kelly={kelly:.3f})",
                drawdown_pct=dd,
                kelly_fraction=kelly,
            )

        return TradeDecision(
            allowed=True,
            position_size_pct=round(position_pct, 2),
            reason=f"OK: Kelly={kelly:.3f}, DD_scale={dd_scale:.2f}, size={position_pct:.1f}%",
            drawdown_pct=dd,
            kelly_fraction=kelly,
        )

    def status(self) -> Dict[str, Any]:
        """Current risk manager state."""
        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": round(self.current_drawdown_pct(), 2),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker": self.circuit_breaker_active,
            "total_exposure_pct": self.total_exposure_pct,
        }


__all__ = ["RiskManager", "TradeDecision"]
