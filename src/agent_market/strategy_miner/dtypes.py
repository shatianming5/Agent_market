"""Data types for the strategy miner."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Phase(Enum):
    STRATEGY_GEN = "strategy_gen"
    BACKTEST = "backtest"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    EVOLVE = "evolve"
    COMPLETE = "complete"


@dataclass
class MinerConfig:
    model: str = ""
    base_url: Optional[str] = None
    max_turns: int = 30
    max_iterations: int = 10
    max_retries: int = 2
    freqtrade_config: str = "user_data/config_freqai.json"
    timerange: str = "20250101-20260101"
    backtest_timeout: int = 300
    stale_timeout: float = 180.0
    evolve_enabled: bool = True
    evolve_every_n: int = 2
    mutation_intensity: float = 0.3
    crossover_prob: float = 0.3
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe": 0.30,
        "profit_pct": 0.20,
        "max_drawdown": 0.15,
        "winrate": 0.10,
        "trade_count": 0.05,
        "stability": 0.10,
    })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MinerConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class StrategyCandidate:
    name: str
    code: str
    strategy_path: Path
    iteration: int = 0
    validation_passed: bool = False
    backtest_summary: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    diagnosis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "strategy_path": str(self.strategy_path),
            "iteration": self.iteration,
            "validation_passed": self.validation_passed,
            "backtest_summary": self.backtest_summary,
            "reward": self.reward,
            "diagnosis": self.diagnosis,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StrategyCandidate:
        d = dict(d)
        d["strategy_path"] = Path(d["strategy_path"])
        return cls(**{k: v for k, v in d.items() if k in {
            "name", "code", "strategy_path", "iteration",
            "validation_passed", "backtest_summary", "reward", "diagnosis",
        }})


@dataclass
class MinerState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.STRATEGY_GEN
    iteration: int = 0
    candidates: List[StrategyCandidate] = field(default_factory=list)
    best_reward: float = float("-inf")
    best_candidate: Optional[StrategyCandidate] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_reward": self.best_reward,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MinerState:
        state = cls(
            run_id=d["run_id"],
            phase=Phase(d["phase"]),
            iteration=d.get("iteration", 0),
            best_reward=d.get("best_reward", float("-inf")),
            history=d.get("history", []),
        )
        state.candidates = [
            StrategyCandidate.from_dict(c) for c in d.get("candidates", [])
        ]
        bc = d.get("best_candidate")
        if bc is not None:
            state.best_candidate = StrategyCandidate.from_dict(bc)
        return state
