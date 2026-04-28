"""Goal Contract — machine-readable objective spec (D1).

The GoalContract is the **single source of truth** for:
- What the harness is optimizing for (objective weights)
- Hard constraint gates (must-pass to survive)
- Evaluation protocol (timeranges, walk-forward, fee model)
- Search space bounds (candidate types, families, iterations)
- Experiment metadata (hypothesis, expected improvement, budget, stop/promotion)
- Lineage (parent run/experiment that spawned this one)

All runtime scoring, constraint checking, promotion gating, and budget decisions
MUST consume GoalContract, not scattered MinerConfig fields.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GoalContract:
    """Machine-readable goal specification for strategy mining.

    This is the control plane for the entire harness — generation,
    scheduling, evaluation, and promotion all read from here.
    """

    schema_version: str = "2.0"

    # --- Experiment identity & hypothesis -----------------------------------
    experiment_id: str = ""
    hypothesis: str = ""  # what we're testing this run
    expected_improvement: str = ""  # which axis should improve (sharpe/drawdown/etc)
    parent_run_id: str = ""  # lineage: which run spawned this experiment
    parent_experiment_id: str = ""  # lineage: which experiment is the parent

    # --- Budget & stop conditions -------------------------------------------
    budget: Dict[str, Any] = field(default_factory=lambda: {
        "max_iterations": 10,
        "max_candidates": 50,
        "max_wall_seconds": 7200,  # 2 hours
        "max_backtest_seconds": 3600,
        "max_tokens": 0,  # 0 = unlimited
        "max_cost_usd": 0.0,  # 0 = unlimited
    })

    stop_conditions: List[str] = field(default_factory=lambda: [
        "budget_exhausted",  # any budget limit hit
        "target_achieved",  # best_score >= target_sharpe
        "stagnation",  # no improvement for N iterations
    ])

    # --- Promotion conditions -----------------------------------------------
    promotion_conditions: Dict[str, Any] = field(default_factory=lambda: {
        "min_sharpe": 0.5,
        "holdout_delta_max_pct": 50.0,
        "walkforward_min_folds_passed": 2,
        "constraints_all_passed": True,
        "min_observation_days": 30,
    })

    # --- Objective weights (higher = more important) -----------------------
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe": 0.40,
        "sortino": 0.20,
        "profit_factor": 0.15,
        "calmar": 0.10,
        "profit_pct": 0.10,
        "winrate": 0.05,
        "return_over_drawdown": 0.0,
    })

    # --- Hard constraints (must-pass gates) --------------------------------
    constraints: Dict[str, Any] = field(default_factory=lambda: {
        "min_trades": 10,
        "max_abs_drawdown": 50.0,
        "max_drawdown_pct": 0.0,
        "min_profit_factor": 0.0,
        "min_winrate": 0.0,
        "min_profit_pct": 0.0,
        "min_positive_days_ratio": 0.0,
        "min_return_over_drawdown": 0.0,
        "min_pair_profit_pct": -0.5,
        "holdout_delta_max_pct": 50.0,
    })

    # --- Evaluation protocol -----------------------------------------------
    evaluation_protocol: Dict[str, Any] = field(default_factory=lambda: {
        "train_timerange": "",
        "selection_timerange": "",
        "holdout_timerange": "",
        "walkforward_folds": 0,
        "walkforward_train_ratio": 0.6,
        "fee_model": "realistic",
        "benchmark_suite": "",  # path to frozen benchmark pack
    })

    # --- Search space bounds -----------------------------------------------
    search_space: Dict[str, Any] = field(default_factory=lambda: {
        "candidate_types": ["rule"],
        "families": [],
        "max_iterations": 10,
        "candidates_per_iteration": 1,
    })

    # --- Metadata ----------------------------------------------------------
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "expected_improvement": self.expected_improvement,
            "parent_run_id": self.parent_run_id,
            "parent_experiment_id": self.parent_experiment_id,
            "budget": dict(self.budget),
            "stop_conditions": list(self.stop_conditions),
            "promotion_conditions": dict(self.promotion_conditions),
            "objective_weights": dict(self.objective_weights),
            "constraints": dict(self.constraints),
            "evaluation_protocol": dict(self.evaluation_protocol),
            "search_space": dict(self.search_space),
            "description": self.description,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GoalContract:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> GoalContract:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: Path) -> GoalContract:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def sha256(self) -> str:
        """Deterministic hash for immutable snapshots."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # -----------------------------------------------------------------------
    # Runtime queries (single source of truth)
    # -----------------------------------------------------------------------

    def get_constraint(self, key: str, default: float = 0.0) -> float:
        """Get a constraint value. This is the canonical accessor."""
        return float(self.constraints.get(key, default) or default)

    def max_iterations(self) -> int:
        return int(self.budget.get("max_iterations",
                   self.search_space.get("max_iterations", 10)))

    def is_budget_exhausted(self, state_economics: Dict[str, Any],
                            iteration: int, n_candidates: int) -> bool:
        """Check if any budget limit is hit."""
        b = self.budget
        if iteration >= int(b.get("max_iterations", 10)):
            return True
        if int(b.get("max_candidates", 0)) > 0 and n_candidates >= int(b["max_candidates"]):
            return True
        wall = float(state_economics.get("total_wall_seconds", 0))
        if float(b.get("max_wall_seconds", 0)) > 0 and wall >= float(b["max_wall_seconds"]):
            return True
        bt = float(state_economics.get("total_backtest_seconds", 0))
        if float(b.get("max_backtest_seconds", 0)) > 0 and bt >= float(b["max_backtest_seconds"]):
            return True
        tokens = int(state_economics.get("total_tokens", 0))
        if int(b.get("max_tokens", 0)) > 0 and tokens >= int(b["max_tokens"]):
            return True
        # D13: Cost budget check
        cost = float(state_economics.get("total_cost_usd", 0))
        if float(b.get("max_cost_usd", 0)) > 0 and cost >= float(b["max_cost_usd"]):
            return True
        return False

    def is_promotable(self, sharpe: float, holdout_delta_pct: Optional[float],
                      wf_folds_passed: int, constraints_ok: bool,
                      observation_days: int) -> bool:
        """Check if a candidate meets promotion conditions."""
        pc = self.promotion_conditions
        if sharpe < float(pc.get("min_sharpe", 0)):
            return False
        if not constraints_ok and bool(pc.get("constraints_all_passed", True)):
            return False
        if holdout_delta_pct is not None:
            if abs(holdout_delta_pct) > float(pc.get("holdout_delta_max_pct", 50)):
                return False
        if wf_folds_passed < int(pc.get("walkforward_min_folds_passed", 0)):
            return False
        if observation_days < int(pc.get("min_observation_days", 0)):
            return False
        return True

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors: List[str] = []
        # Weights must sum to ~1.0
        wt_sum = sum(self.objective_weights.values())
        if abs(wt_sum - 1.0) > 0.05:
            errors.append(f"objective_weights sum={wt_sum:.3f}, expected ~1.0")
        # Constraints sanity
        if self.constraints.get("min_trades", 0) < 0:
            errors.append("min_trades must be >= 0")
        max_dd = self.constraints.get("max_abs_drawdown", 50)
        if max_dd is not None and float(max_dd) < 0:
            errors.append("max_abs_drawdown must be >= 0")
        # Budget sanity
        if int(self.budget.get("max_iterations", 0)) <= 0:
            errors.append("budget.max_iterations must be > 0")
        return errors

    # -----------------------------------------------------------------------
    # Factory: build from MinerConfig (backward compatible)
    # -----------------------------------------------------------------------

    @classmethod
    def from_miner_config(cls, config: Any) -> GoalContract:
        """Extract a GoalContract from legacy MinerConfig fields."""
        gc = cls()
        gc.constraints = {
            "min_trades": int(getattr(config, "min_trades", 10)),
            "max_abs_drawdown": float(getattr(config, "max_abs_drawdown", 50.0)),
            "max_drawdown_pct": float(getattr(config, "max_drawdown_pct", 0.0)),
            "min_profit_factor": float(getattr(config, "min_profit_factor", 0.0)),
            "min_winrate": float(getattr(config, "min_winrate", 0.0)),
            "min_profit_pct": float(getattr(config, "min_profit_pct", 0.0)),
            "min_positive_days_ratio": float(getattr(config, "min_positive_days_ratio", 0.0)),
            "min_return_over_drawdown": float(getattr(config, "min_return_over_drawdown", 0.0)),
            "min_pair_profit_pct": float(getattr(config, "min_pair_profit_pct", -0.5)),
            "holdout_delta_max_pct": 50.0,
        }
        gc.evaluation_protocol = {
            "train_timerange": str(getattr(config, "train_timerange", "") or ""),
            "selection_timerange": str(getattr(config, "selection_timerange", "") or ""),
            "holdout_timerange": str(getattr(config, "holdout_timerange", "") or ""),
            "walkforward_folds": int(getattr(config, "walkforward_folds", 0)),
            "walkforward_train_ratio": float(getattr(config, "walkforward_train_ratio", 0.6)),
            "fee_model": "realistic",
            "benchmark_suite": str(getattr(config, "benchmark_suite", "") or ""),
        }
        gc.search_space = {
            "candidate_types": list(getattr(config, "candidate_types", ["rule"])),
            "families": list(getattr(config, "search_families", []) or []),
            "max_iterations": int(getattr(config, "max_iterations", 10)),
            "candidates_per_iteration": int(getattr(config, "candidates_per_iteration", 1)),
        }
        objective_weights = dict(getattr(config, "objective_weights", {}) or {})
        if objective_weights:
            gc.objective_weights = objective_weights
        gc.budget = {
            "max_iterations": int(getattr(config, "max_iterations", 10)),
            "max_candidates": 0,
            "max_wall_seconds": 0,
            "max_backtest_seconds": 0,
            "max_tokens": 0,
            "max_cost_usd": 0.0,
        }
        return gc
