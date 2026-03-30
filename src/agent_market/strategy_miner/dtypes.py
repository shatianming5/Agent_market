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
    COMPLETE = "complete"


@dataclass
class MinerConfig:
    # Agent provider
    provider: str = "auto"  # auto|opencode|openai_compatible

    # Multi-agent (planner/coder/reviewer) generation pipeline
    multiagent_enabled: bool = True
    multiagent_refine_rounds: int = 1  # reviewer->coder refinement loops per candidate

    # Multi-candidate generation
    candidates_per_iteration: int = 1

    # Concurrency controls (0=auto)
    max_parallel_candidates: int = 0
    max_parallel_roles: int = 1

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
    repair_attempts: int = 3

    # Tool policy (OpenCode tool loop)
    tool_allowlist: List[str] = field(default_factory=lambda: ["file", "bash"])
    bash_allow: bool = True
    bash_timeout: int = 60
    bash_allowlist: List[str] = field(
        default_factory=lambda: [
            # Common safe inspection helpers
            "ls ",
            "find ",
            "cat ",
            "head ",
            "tail ",
            "sed ",
            "grep ",
            # Python invocations
            "python ",
            "python3 ",
            # Freqtrade (module or wrapper)
            "freqtrade ",
        ]
    )

    # Risk constraints / gating (optional)
    min_trades: int = 10
    max_abs_drawdown: float = 50.0
    max_drawdown_pct: float = 0.0  # max_drawdown_account percentage (0 = disabled)
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
                "multiagent_enabled",
                "multiagent_refine_rounds",
                "candidates_per_iteration",
                "max_parallel_candidates",
                "max_parallel_roles",
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

        generation = d2.get("generation")
        if isinstance(generation, dict):
            for k in (
                "candidates_per_iteration",
                "max_parallel_candidates",
                "max_parallel_roles",
                "multiagent_enabled",
                "multiagent_refine_rounds",
            ):
                if k in generation and k not in d2:
                    d2[k] = generation[k]

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

        evaluation = d2.get("evaluation")
        if isinstance(evaluation, dict):
            for k in ("min_trades", "max_abs_drawdown", "max_drawdown_pct", "min_winrate"):
                if k in evaluation and k not in d2:
                    d2[k] = evaluation[k]

        risk = d2.get("risk_constraints")
        if isinstance(risk, dict):
            for k in ("min_trades", "max_abs_drawdown", "max_drawdown_pct", "min_winrate"):
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
    candidate_slot: int = 0
    source_provider: str = ""
    source_model: Optional[str] = None
    agent_traces: Dict[str, str] = field(default_factory=dict)
    validation_passed: bool = False
    backtest_summary: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    diagnosis: str = ""

    # Provenance (for audit + enforcing no-template mode)
    generation_provider: str = ""
    generation_model: Optional[str] = None

    # Multi-agent traces (truncated for checkpoint size)
    planner_notes: str = ""
    reviewer_notes: str = ""
    backtester_notes: str = ""

    # Failure categorization (validation/backtest)
    failure_category: str = ""

    # Risk constraint gating (computed during evaluation)
    constraints_ok: bool = True
    constraint_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "strategy_path": str(self.strategy_path),
            "iteration": self.iteration,
            "candidate_slot": int(getattr(self, "candidate_slot", 0) or 0),
            "source_provider": self.source_provider,
            "source_model": self.source_model,
            "agent_traces": dict(self.agent_traces or {}),
            "validation_passed": self.validation_passed,
            "backtest_summary": self.backtest_summary,
            "reward": self.reward,
            "diagnosis": self.diagnosis,
            "generation_provider": self.generation_provider,
            "generation_model": self.generation_model,
            "planner_notes": self.planner_notes,
            "reviewer_notes": self.reviewer_notes,
            "backtester_notes": self.backtester_notes,
            "failure_category": self.failure_category,
            "constraints_ok": self.constraints_ok,
            "constraint_violations": list(self.constraint_violations or []),
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
                    "candidate_slot",
                    "source_provider",
                    "source_model",
                    "agent_traces",
                    "validation_passed",
                    "backtest_summary",
                    "reward",
                    "diagnosis",
                    "generation_provider",
                    "generation_model",
                    "planner_notes",
                    "reviewer_notes",
                    "backtester_notes",
                    "failure_category",
                    "constraints_ok",
                    "constraint_violations",
                }
            }
        )


@dataclass
class MinerState:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: Phase = Phase.STRATEGY_GEN
    iteration: int = 0
    candidates: List[StrategyCandidate] = field(default_factory=list)
    best_score: float = float("-inf")
    best_candidate: Optional[StrategyCandidate] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Multi-candidate scheduling within an iteration
    pending_candidate_idxs: List[int] = field(default_factory=list)
    active_candidate_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "candidates": [c.to_dict() for c in self.candidates],
            "best_score": self.best_score,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "history": self.history,
            "pending_candidate_idxs": list(self.pending_candidate_idxs or []),
            "active_candidate_idx": self.active_candidate_idx,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MinerState:
        state = cls(
            run_id=d["run_id"],
            phase=Phase(d["phase"]),
            iteration=d.get("iteration", 0),
            best_score=d.get("best_score", d.get("best_reward", float("-inf"))),
            history=d.get("history", []),
        )
        state.candidates = [StrategyCandidate.from_dict(c) for c in d.get("candidates", [])]
        bc = d.get("best_candidate")
        if bc is not None:
            state.best_candidate = StrategyCandidate.from_dict(bc)

        state.pending_candidate_idxs = list(d.get("pending_candidate_idxs") or [])
        state.active_candidate_idx = d.get("active_candidate_idx")
        return state
