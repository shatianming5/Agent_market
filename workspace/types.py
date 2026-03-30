"""Typed data models for workspace — replaces loose Dict[str, Any] returns.

Used by: gate_pipeline, backtest_api, evaluator, pairs_engine, tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BacktestResult:
    """Standardized backtest result."""
    ok: bool = False
    error: str = ""
    strategy: str = ""
    trades: int = 0
    profit_pct: float = 0.0
    profit_abs: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_duration: str = ""
    cagr: float = 0.0
    sqn: float = 0.0
    expectancy: float = 0.0
    # Metadata
    strategy_path: str = ""
    strategy_sha256: str = ""
    config_path: str = ""
    timerange: str = ""
    timestamp: str = ""
    backtest_zip: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (backwards compatible with existing code)."""
        return {k: v for k, v in self.__dict__.items() if v or k in ("ok", "trades")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestResult":
        """Create from dict (backwards compatible)."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class EvaluationResult:
    """Multi-objective evaluation result."""
    total_score: float = 0.0
    grade: str = "F"
    details: Dict[str, Any] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class GatePipelineResult:
    """Result of running a strategy through the gate pipeline."""
    strategy: str = ""
    strategy_type: str = ""
    timestamp: str = ""
    final_gate_passed: Optional[str] = None
    recommendation: str = ""
    gates: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.final_gate_passed == "gate_3"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class ExperimentRecord:
    """A single experiment record."""
    id: int = 0
    timestamp: str = ""
    strategy_name: str = ""
    strategy_path: str = ""
    model_name: Optional[str] = None
    backtest_ok: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Result of adaptive parameter optimization."""
    pair_a: str = ""
    pair_b: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    train_sharpe: float = 0.0
    valid_sharpe: float = 0.0
    overfit_ratio: float = 0.0
    optimized_at: str = ""


@dataclass
class HealthCheckResult:
    """Result of a performance health check."""
    timestamp: str = ""
    checked: int = 0
    healthy: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)


__all__ = [
    "BacktestResult",
    "CalibrationResult",
    "EvaluationResult",
    "ExperimentRecord",
    "GatePipelineResult",
    "HealthCheckResult",
]
