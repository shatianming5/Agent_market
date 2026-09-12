"""Strategy IR (Intermediate Representation) — D3.

Structured representation that separates strategy concerns into modules:
- Alpha (signals)
- Regime (market filter)
- Exit (ROI, stoploss, trailing)
- Execution (position management)
- Risk (portfolio-level controls)

This enables:
- Static validation of strategy intent
- Module-level reuse across candidates
- Safer compilation than raw LLM code
- Schema-based comparison and search
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AlphaModule:
    """Entry/exit signal definitions."""
    indicators: List[Dict[str, Any]] = field(default_factory=list)
    # Each indicator: {"name": "ema", "params": {"period": 20}, "column": "ema_20"}
    entry_conditions: List[str] = field(default_factory=list)
    # Each condition: a string expression, e.g., "close > ema_20"
    exit_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicators": self.indicators,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AlphaModule:
        return cls(
            indicators=d.get("indicators", []),
            entry_conditions=d.get("entry_conditions", []),
            exit_conditions=d.get("exit_conditions", []),
        )


@dataclass
class RegimeModule:
    """Market regime filter — when to trade."""
    filters: List[Dict[str, Any]] = field(default_factory=list)
    # Each filter: {"type": "adx_threshold", "params": {"min_adx": 25}}

    def to_dict(self) -> Dict[str, Any]:
        return {"filters": self.filters}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RegimeModule:
        return cls(filters=d.get("filters", []))


@dataclass
class ExitModule:
    """Exit parameters — ROI, stoploss, trailing."""
    minimal_roi: Dict[str, float] = field(default_factory=lambda: {"0": 0.1})
    stoploss: float = -0.10
    trailing_stop: bool = False
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.02
    trailing_only_offset_is_reached: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimal_roi": self.minimal_roi,
            "stoploss": self.stoploss,
            "trailing_stop": self.trailing_stop,
            "trailing_stop_positive": self.trailing_stop_positive,
            "trailing_stop_positive_offset": self.trailing_stop_positive_offset,
            "trailing_only_offset_is_reached": self.trailing_only_offset_is_reached,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExitModule:
        return cls(
            minimal_roi=d.get("minimal_roi", {"0": 0.1}),
            stoploss=float(d.get("stoploss", -0.10)),
            trailing_stop=bool(d.get("trailing_stop", False)),
            trailing_stop_positive=float(d.get("trailing_stop_positive", 0.01)),
            trailing_stop_positive_offset=float(d.get("trailing_stop_positive_offset", 0.02)),
            trailing_only_offset_is_reached=bool(d.get("trailing_only_offset_is_reached", True)),
        )


@dataclass
class ExecutionModule:
    """Position management — DCA, grid, etc."""
    mode: str = "signal"  # signal | dca | grid
    max_dca_orders: int = 0
    dca_scale_factor: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_dca_orders": self.max_dca_orders,
            "dca_scale_factor": self.dca_scale_factor,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionModule:
        return cls(
            mode=d.get("mode", "signal"),
            max_dca_orders=int(d.get("max_dca_orders", 0)),
            dca_scale_factor=float(d.get("dca_scale_factor", 1.5)),
        )


@dataclass
class RiskModule:
    """Portfolio-level risk controls."""
    max_open_trades: int = 5
    stake_currency: str = "USDT"
    stake_amount: str = "unlimited"
    max_drawdown_pct: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_open_trades": self.max_open_trades,
            "stake_currency": self.stake_currency,
            "stake_amount": self.stake_amount,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RiskModule:
        return cls(
            max_open_trades=int(d.get("max_open_trades", 5)),
            stake_currency=d.get("stake_currency", "USDT"),
            stake_amount=str(d.get("stake_amount", "unlimited")),
            max_drawdown_pct=float(d.get("max_drawdown_pct", 50.0)),
        )


@dataclass
class StrategySpec:
    """Complete strategy specification in IR form.

    schema_version tracks format evolution for forward/backward compat.
    """
    schema_version: str = "1.0"
    name: str = ""
    timeframe: str = "5m"
    description: str = ""

    alpha: AlphaModule = field(default_factory=AlphaModule)
    regime: RegimeModule = field(default_factory=RegimeModule)
    exit: ExitModule = field(default_factory=ExitModule)
    execution: ExecutionModule = field(default_factory=ExecutionModule)
    risk: RiskModule = field(default_factory=RiskModule)

    # Hyperopt search space hints
    search_space: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: List[str] = field(default_factory=list)
    family: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "timeframe": self.timeframe,
            "description": self.description,
            "alpha": self.alpha.to_dict(),
            "regime": self.regime.to_dict(),
            "exit": self.exit.to_dict(),
            "execution": self.execution.to_dict(),
            "risk": self.risk.to_dict(),
            "search_space": self.search_space,
            "tags": self.tags,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StrategySpec:
        return cls(
            schema_version=d.get("schema_version", "1.0"),
            name=d.get("name", ""),
            timeframe=d.get("timeframe", "5m"),
            description=d.get("description", ""),
            alpha=AlphaModule.from_dict(d.get("alpha", {})),
            regime=RegimeModule.from_dict(d.get("regime", {})),
            exit=ExitModule.from_dict(d.get("exit", {})),
            execution=ExecutionModule.from_dict(d.get("execution", {})),
            risk=RiskModule.from_dict(d.get("risk", {})),
            search_space=d.get("search_space", {}),
            tags=d.get("tags", []),
            family=d.get("family", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> StrategySpec:
        return cls.from_dict(json.loads(text))

    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors: List[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.alpha.indicators:
            errors.append("alpha.indicators is empty")
        if not self.alpha.entry_conditions:
            errors.append("alpha.entry_conditions is empty")
        if self.exit.stoploss >= 0:
            errors.append("exit.stoploss must be negative")
        return errors
