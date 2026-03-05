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
    # Agent provider
    provider: str = "auto"  # auto|opencode|openai_compatible|template

    # Agent budget / retries
    model: str = ""  # primary model name (opencode)
    base_url: Optional[str] = None  # opencode server base url (optional)
    max_turns: int = 30
    max_iterations: int = 10
    max_retries: int = 2
    stale_timeout: float = 180.0

    # Backtest
    freqtrade_config: str = "user_data/config_freqai.json"
    timerange: str = "20250101-20260101"
    backtest_timeout: int = 300

    # Self-repair / retries
    repair_attempts: int = 1

    # Tool policy (OpenCode tool loop)
    tool_allowlist: List[str] = field(default_factory=lambda: ["file", "bash"])
    bash_allow: bool = True
    bash_timeout: int = 60
    bash_allowlist: List[str] = field(default_factory=lambda: [
        # Common safe inspection helpers
        "ls ", "find ", "cat ", "head ", "tail ", "sed ", "grep ",
        # Python invocations
        "python ", "python3 ",
        # Freqtrade (module or wrapper)
        "freqtrade ",
    ])

    # Evolution
    evolve_enabled: bool = True
    evolve_every_n: int = 2
    mutation_intensity: float = 0.3
    crossover_prob: float = 0.3

    # Evaluation
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sharpe": 0.30,
            "profit_pct": 0.20,
            "max_drawdown": 0.15,
            "winrate": 0.10,
            "trade_count": 0.05,
            "stability": 0.10,
        }
    )

    # Risk constraints / gating (optional)
    min_trades: int = 10
    max_abs_drawdown: float = 50.0
    min_winrate: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MinerConfig:
        """Parse config from a dict.

        Backward compatible:
        - Accepts flat keys (legacy)
        - Accepts nested sections: budget/tools/evaluation/risk_constraints
        """
        d2: Dict[str, Any] = dict(d or {})

        # Flatten nested sections (new style)
        budget = d2.get("budget")
        if isinstance(budget, dict):
            for k in (
                "provider",
                "model",
                "base_url",
                "max_turns",
                "max_iterations",
                "max_retries",
                "stale_timeout",
                "backtest_timeout",
                "repair_attempts",
            ):
                if k in budget and k not in d2:
                    d2[k] = budget[k]

        tools = d2.get("tools")
        if isinstance(tools, dict):
            for k in (
                "tool_allowlist",
                "bash_allow",
                "bash_timeout",
                "bash_allowlist",
            ):
                if k in tools and k not in d2:
                    d2[k] = tools[k]

        backtest = d2.get("backtest")
        if isinstance(backtest, dict):
            for k in (
                "freqtrade_config",
                "timerange",
                "backtest_timeout",
            ):
                if k in backtest and k not in d2:
                    d2[k] = backtest[k]

        evolution = d2.get("evolution")
        if isinstance(evolution, dict):
            for k in (
                "evolve_enabled",
                "evolve_every_n",
                "mutation_intensity",
                "crossover_prob",
            ):
                if k in evolution and k not in d2:
                    d2[k] = evolution[k]

        evaluation = d2.get("evaluation")
        if isinstance(evaluation, dict):
            if "reward_weights" in evaluation and "reward_weights" not in d2:
                d2["reward_weights"] = evaluation.get("reward_weights")
            for k in ("min_trades", "max_abs_drawdown", "min_winrate"):
                if k in evaluation and k not in d2:
                    d2[k] = evaluation[k]

        risk = d2.get("risk_constraints")
        if isinstance(risk, dict):
            for k in ("min_trades", "max_abs_drawdown", "min_winrate"):
                if k in risk and k not in d2:
                    d2[k] = risk[k]

        known = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in d2.items() if k in known}
        return cls(**payload)


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
        return cls(
            **{
                k: v
                for k, v in d.items()
                if k
                in {
                    "name",
                    "code",
                    "strategy_path",
                    "iteration",
                    "validation_passed",
                    "backtest_summary",
                    "reward",
                    "diagnosis",
                }
            }
        )


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
        state.candidates = [StrategyCandidate.from_dict(c) for c in d.get("candidates", [])]
        bc = d.get("best_candidate")
        if bc is not None:
            state.best_candidate = StrategyCandidate.from_dict(bc)
        return state
