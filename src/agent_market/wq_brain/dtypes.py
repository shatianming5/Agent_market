"""Pure data types — no I/O, no logic beyond serialization."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class AlphaSettings:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 6
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    language: str = "FASTEXPR"
    instrument_type: str = "EQUITY"
    pasteurization: str = "ON"
    unit_handling: str = "VERIFY"
    nan_handling: str = "ON"

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "instrumentType": self.instrument_type,
            "region": self.region,
            "universe": self.universe,
            "delay": self.delay,
            "decay": self.decay,
            "neutralization": self.neutralization,
            "truncation": self.truncation,
            "pasteurization": self.pasteurization,
            "unitHandling": self.unit_handling,
            "nanHandling": self.nan_handling,
            "language": self.language,
            "visualization": False,
        }


@dataclass
class SimulationResult:
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    returns: Optional[float] = None
    turnover: Optional[float] = None
    drawdown: Optional[float] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    alpha_id: Optional[str] = None
    status: str = "UNKNOWN"
    error: Optional[str] = None
    checks: list = field(default_factory=list)

    def passes_quality(self, *, sharpe_min: float = 1.25, fitness_min: float = 1.0) -> bool:
        return (
            self.sharpe is not None
            and self.sharpe >= sharpe_min
            and self.fitness is not None
            and self.fitness >= fitness_min
        )

    def failed_check_names(self) -> list[str]:
        return [c.get("name", "") for c in self.checks if c.get("result") == "FAIL"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaCandidate:
    expr: str
    settings: AlphaSettings
    sim_result: Optional[SimulationResult] = None
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    submitted: bool = False
    source: str = "agent"  # "agent" | "scan"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "candidate_id": self.candidate_id,
            "expr": self.expr,
            "settings": asdict(self.settings),
            "submitted": self.submitted,
            "source": self.source,
        }
        if self.sim_result is not None:
            d["sim_result"] = self.sim_result.to_dict()
        return d


@dataclass
class AlphaPoolEntry:
    alpha_id: str
    expr: str
    settings_dict: dict[str, Any]
    sharpe: float
    fitness: float
    returns: float
    turnover: float
    submitted_at: float = field(default_factory=time.time)
    tag: str = ""
    source: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlphaPoolEntry":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
