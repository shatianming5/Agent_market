from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    PREPARE = "PREPARE"
    ALPHA_GEN = "ALPHA_GEN"
    SIMULATE = "SIMULATE"
    EVALUATE = "EVALUATE"
    SUBMIT = "SUBMIT"
    ANALYSIS = "ANALYSIS"
    COMPLETE = "COMPLETE"


@dataclass
class AlphaSettings:
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
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
    checks: list = field(default_factory=list)  # WQ quality check results

    def failed_check_names(self) -> list[str]:
        return [c["name"] for c in self.checks if c.get("result") == "FAIL"]

    def check_quality(self, *, sharpe_min: float = 1.25, fitness_min: float = 1.0) -> bool:
        return (
            self.sharpe is not None and self.sharpe >= sharpe_min
            and self.fitness is not None and self.fitness >= fitness_min
        )

    @property
    def passes_quality(self) -> bool:
        return self.check_quality()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaCandidate:
    expr: str
    settings: AlphaSettings
    sim_result: Optional[SimulationResult] = None
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    submitted_to_pool: bool = False
    correlation_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "candidate_id": self.candidate_id,
            "expr": self.expr,
            "settings": asdict(self.settings),
            "submitted_to_pool": self.submitted_to_pool,
            "correlation_ok": self.correlation_ok,
        }
        if self.sim_result:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlphaPoolEntry:
        return cls(**d)


@dataclass
class WQBrainConfig:
    tag: str
    run_id: str = field(default_factory=lambda: "")
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "SUBINDUSTRY"
    truncation: float = 0.08
    max_iterations: int = 50
    batch_size: int = 5
    max_concurrent: int = 3
    auto_submit: bool = False
    model: str = ""
    hermes_provider: str = ""
    hermes_yolo: bool = True
    hermes_toolsets: str = "terminal,file"
    hermes_reasoning_effort: str = ""
    max_turns: int = 30
    stale_timeout: float = 600.0
    quality_sharpe_min: float = 1.25
    quality_fitness_min: float = 1.0
    corr_max: float = 0.7
    dry_run: bool = False

    def default_settings(self) -> AlphaSettings:
        return AlphaSettings(
            region=self.region,
            universe=self.universe,
            delay=self.delay,
            decay=self.decay,
            neutralization=self.neutralization,
            truncation=self.truncation,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WQBrainConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WQBrainState:
    config: WQBrainConfig
    iteration: int = 0
    phase: Phase = Phase.PREPARE
    total_generated: int = 0
    total_simulated: int = 0
    total_passed: int = 0
    total_submitted: int = 0
    score_history: list[dict[str, Any]] = field(default_factory=list)
    # Rolling window of all simulated expressions with actual results (last 30)
    tried_exprs: list[dict[str, Any]] = field(default_factory=list)

    _TRIED_EXPRS_MAXLEN: int = field(default=80, init=False, repr=False, compare=False)

    def add_tried(self, expr: str, sharpe: Optional[float], fitness: Optional[float], turnover: Optional[float]) -> None:
        self.tried_exprs.append({"expr": expr, "sh": sharpe, "fi": fitness, "to": turnover})
        if len(self.tried_exprs) > self._TRIED_EXPRS_MAXLEN:
            self.tried_exprs = self.tried_exprs[-self._TRIED_EXPRS_MAXLEN:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "iteration": self.iteration,
            "phase": self.phase.value,
            "total_generated": self.total_generated,
            "total_simulated": self.total_simulated,
            "total_passed": self.total_passed,
            "total_submitted": self.total_submitted,
            "score_history": self.score_history,
            "tried_exprs": self.tried_exprs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WQBrainState:
        cfg = WQBrainConfig.from_dict(d["config"])
        return cls(
            config=cfg,
            iteration=d.get("iteration", 0),
            phase=Phase(d.get("phase", Phase.PREPARE.value)),
            total_generated=d.get("total_generated", 0),
            total_simulated=d.get("total_simulated", 0),
            total_passed=d.get("total_passed", 0),
            total_submitted=d.get("total_submitted", 0),
            score_history=d.get("score_history", []),
            tried_exprs=d.get("tried_exprs", []),
        )
