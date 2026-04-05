"""Data types for the strategy miner."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class BacktestSummary(TypedDict, total=False):
    """Typed structure for backtest result summaries."""
    source: str
    strategy: str
    profit_total_pct: Optional[float]
    profit_total_abs: Optional[float]
    trades: Optional[int]
    avg_profit_pct: Optional[float]
    winrate: Optional[float]
    max_drawdown_abs: Optional[float]
    max_drawdown_pct: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]
    calmar: Optional[float]
    profit_factor: Optional[float]
    realistic_sharpe: Optional[float]
    realistic_sortino: Optional[float]
    realistic_calmar: Optional[float]
    return_over_drawdown: Optional[float]
    positive_days_ratio: Optional[float]
    observation_days: Optional[int]
    metric_flags: List[str]
    metrics_trusted: bool
    fee_drag_pct: Optional[float]
    trades_per_day: Optional[float]


class HistoryRow(TypedDict, total=False):
    """Typed structure for iteration history entries."""
    iteration: int
    name: str
    candidate_type: str
    model_family: str
    sharpe: float
    native_sharpe: float
    sortino: float
    native_sortino: float
    calmar: float
    native_calmar: float
    profit_factor: float
    profit_pct: float
    trades: int
    winrate: float
    max_drawdown_pct: float
    positive_days_ratio: float
    return_over_drawdown: float
    metric_flags: List[str]
    expectancy: float
    sqn: float
    cagr: float
    training_summary: Optional[Dict[str, Any]]
    constraints_ok: bool
    constraint_violations: List[str]
    diagnosis: str
    analysis_structured: Dict[str, Any]


class Phase(Enum):
    STRATEGY_GEN = "strategy_gen"
    TRAIN_MODEL = "train_model"
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
    max_strategy_timeframe: str = ""
    allowed_informative_timeframes: List[str] = field(default_factory=list)
    candidate_types: List[str] = field(default_factory=lambda: ["rule"])
    model_feature_file: str = "user_data/freqai_features_real.json"
    model_expressions_file: str = "user_data/freqai_expressions_selected.json"
    model_training_pairs: List[str] = field(default_factory=list)
    model_output_root: str = "artifacts/models/strategy_miner"
    training_validation_ratio: float = 0.2
    training_rolling_splits: int = 3
    training_scaler: str = "robust"
    rl_total_timesteps: int = 5000
    enable_dl: bool = False
    enable_rl: bool = False
    train_timerange: str = ""
    enable_quick_funnel: bool = True
    quick_backtest_pairs: List[str] = field(default_factory=list)
    quick_backtest_timerange: str = ""
    quick_backtest_timeout: int = 120
    quick_min_trades: int = 8
    quick_min_profit_factor: float = 0.9
    quick_min_profit_pct: float = -1.0
    quick_max_drawdown_pct: float = 35.0

    # Hyperopt integration
    hyperopt_enabled: bool = False
    hyperopt_epochs: int = 80
    hyperopt_spaces: List[str] = field(default_factory=lambda: ["buy", "sell", "roi", "stoploss"])
    hyperopt_loss: str = "SharpeHyperOptLoss"
    hyperopt_jobs: int = 2
    hyperopt_min_trades: int = 10

    # Position management (DCA / grid / martingale support)
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = 0  # 0=disabled, 3-5 for DCA/grid
    strategy_archetypes: List[str] = field(default_factory=lambda: ["signal"])
    # Valid archetypes: signal (default single-entry), dca, grid, martingale

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

    # Sealed holdout (final validation, touched only once at run completion)
    selection_timerange: str = ""   # used for iteration scoring (replaces timerange if set)
    holdout_timerange: str = ""     # sealed final validation window

    # Walk-forward OOS validation (optional — default off for backward compat)
    walkforward_enabled: bool = False
    walkforward_folds: int = 3
    walkforward_train_ratio: float = 0.6

    # Risk constraints / gating (optional)
    min_trades: int = 10
    max_abs_drawdown: float = 50.0
    max_drawdown_pct: float = 0.0  # max_drawdown_account percentage (0 = disabled)
    min_winrate: float = 0.0
    min_profit_factor: float = 0.0
    min_profit_pct: float = 0.0
    min_positive_days_ratio: float = 0.0
    min_return_over_drawdown: float = 0.0
    min_pair_profit_pct: float = -0.5
    target_trades: int = 20
    min_acceptable_trades: int = 10
    roi_target_min_pct: float = 5.0
    roi_target_max_pct: float = 10.0
    stoploss_min_pct: float = 5.0
    stoploss_max_pct: float = 10.0
    preferred_patterns: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)

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
                "selection_timerange",
                "holdout_timerange",
                "backtest_timeout",
                "max_strategy_timeframe",
                "allowed_informative_timeframes",
                "candidate_types",
                "model_feature_file",
                "model_expressions_file",
                "model_training_pairs",
                "model_output_root",
                "training_validation_ratio",
                "training_rolling_splits",
                "training_scaler",
                "rl_total_timesteps",
                "train_timerange",
                "enable_quick_funnel",
                "quick_backtest_pairs",
                "quick_backtest_timerange",
                "quick_backtest_timeout",
                "position_adjustment_enable",
                "max_entry_position_adjustment",
                "strategy_archetypes",
            ):
                if k in backtest and k not in d2:
                    d2[k] = backtest[k]

        evaluation = d2.get("evaluation")
        if isinstance(evaluation, dict):
            for k in (
                "min_trades",
                "max_abs_drawdown",
                "max_drawdown_pct",
                "min_winrate",
                "min_profit_factor",
                "min_profit_pct",
                "min_positive_days_ratio",
                "min_return_over_drawdown",
                "min_pair_profit_pct",
                "target_trades",
                "min_acceptable_trades",
                "selection_timerange",
                "holdout_timerange",
                "walkforward_enabled",
                "walkforward_folds",
                "walkforward_train_ratio",
                "quick_min_trades",
                "quick_min_profit_factor",
                "quick_min_profit_pct",
                "quick_max_drawdown_pct",
                "hyperopt_enabled",
                "hyperopt_epochs",
                "hyperopt_spaces",
                "hyperopt_loss",
                "hyperopt_jobs",
                "hyperopt_min_trades",
            ):
                if k in evaluation and k not in d2:
                    d2[k] = evaluation[k]

        risk = d2.get("risk_constraints")
        if isinstance(risk, dict):
            for k in (
                "min_trades",
                "max_abs_drawdown",
                "max_drawdown_pct",
                "min_winrate",
                "min_profit_factor",
                "min_profit_pct",
                "min_positive_days_ratio",
                "min_return_over_drawdown",
                "min_pair_profit_pct",
                "target_trades",
                "min_acceptable_trades",
            ):
                if k in risk and k not in d2:
                    d2[k] = risk[k]

        strategy_profile = d2.get("strategy_profile")
        if isinstance(strategy_profile, dict):
            for k in (
                "max_strategy_timeframe",
                "allowed_informative_timeframes",
                "roi_target_min_pct",
                "roi_target_max_pct",
                "stoploss_min_pct",
                "stoploss_max_pct",
                "preferred_patterns",
                "avoid_patterns",
            ):
                if k in strategy_profile and k not in d2:
                    d2[k] = strategy_profile[k]

        model_mining = d2.get("model_mining")
        if isinstance(model_mining, dict):
            for k in (
                "candidate_types",
                "model_feature_file",
                "model_expressions_file",
                "model_training_pairs",
                "model_output_root",
                "training_validation_ratio",
                "training_rolling_splits",
                "training_scaler",
                "rl_total_timesteps",
                "enable_dl",
                "enable_rl",
                "train_timerange",
            ):
                if k in model_mining and k not in d2:
                    d2[k] = model_mining[k]

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
    backtest_summary: Optional[BacktestSummary] = None  # type: ignore[assignment]
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
    candidate_type: str = "rule"
    model_family: str = ""
    candidate_payload: Dict[str, Any] = field(default_factory=dict)
    training_config: Optional[Dict[str, Any]] = None
    training_summary: Optional[Dict[str, Any]] = None
    quick_backtest_summary: Optional[Dict[str, Any]] = None
    candidate_family: str = ""
    funnel_state: Dict[str, Any] = field(default_factory=dict)

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
            "candidate_type": self.candidate_type,
            "model_family": self.model_family,
            "candidate_payload": self.candidate_payload,
            "training_config": self.training_config,
            "training_summary": self.training_summary,
            "quick_backtest_summary": self.quick_backtest_summary,
            "candidate_family": self.candidate_family,
            "funnel_state": dict(self.funnel_state or {}),
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
                    "candidate_type",
                    "model_family",
                    "candidate_payload",
                    "training_config",
                    "training_summary",
                    "quick_backtest_summary",
                    "candidate_family",
                    "funnel_state",
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
    history: List[HistoryRow] = field(default_factory=list)  # type: ignore[assignment]

    # Multi-candidate scheduling within an iteration
    pending_candidate_idxs: List[int] = field(default_factory=list)
    active_candidate_idx: Optional[int] = None

    # Retry counter for iterations that produce no results (persisted in checkpoint)
    gen_retries: int = 0

    # Bandit scheduler state (persisted across checkpoints)
    bandit_state: Dict[str, Any] = field(default_factory=dict)

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
            "gen_retries": self.gen_retries,
            "bandit_state": self.bandit_state,
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
        state.gen_retries = int(d.get("gen_retries", 0) or 0)
        state.bandit_state = d.get("bandit_state", {})
        return state
