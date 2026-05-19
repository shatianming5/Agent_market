"""Typed data models for Factor Hub."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ============================================================
# Factor
# ============================================================

FACTOR_STATUS = ("active", "shadow", "deprecated", "retired", "candidate")


@dataclass
class Factor:
    id: Optional[int] = None
    name: str = ""
    expression: str = ""
    expression_hash: str = ""
    category: str = "misc"
    origin: str = "unknown"
    description: str = ""
    features_used: List[str] = field(default_factory=list)
    complexity: int = 0
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: str = ""
    source_lib: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Evaluation
# ============================================================

EVAL_TYPES = ("ic", "walk_forward", "sub_period", "random_baseline",
              "lgb_gain", "custom")


@dataclass
class Evaluation:
    id: Optional[int] = None
    factor_id: int = 0
    eval_type: str = "ic"
    period_start: str = ""
    period_end: str = ""
    metric_name: str = "ic"
    metric_value: float = 0.0
    n_samples: int = 0
    sign_agree: int = 0
    eval_at: Optional[str] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Deployment (a named set of factors that's "live")
# ============================================================

@dataclass
class Deployment:
    id: Optional[int] = None
    name: str = "production"
    factor_ids: List[int] = field(default_factory=list)
    deployed_at: Optional[str] = None
    deployed_by: str = ""
    is_active: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Event (audit log / pub-sub payload)
# ============================================================

EVENT_TYPES = (
    "factor.created", "factor.updated", "factor.evaluated",
    "factor.status_changed", "deployment.switched",
    "mining.started", "mining.loop_completed", "mining.finished",
    "backtest.started", "backtest.finished",
)


@dataclass
class Event:
    id: Optional[int] = None
    event_type: str = ""
    factor_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
