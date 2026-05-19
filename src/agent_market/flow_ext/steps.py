from __future__ import annotations

"""
Flow extension shim layer (plan.md alignment).

The repository historically kept flow execution logic in `agent_market.flow_steps`.
`plan.md` proposes a dedicated `flow_ext/` domain for step definitions and artifact
conventions. To avoid breaking existing imports while moving towards that structure,
this module re-exports the canonical step runner functions.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from agent_market import flow_steps as _legacy
from agent_market import paths
from agent_market.flow_ext.step_spec import STEP_ORDER

# Legacy step runners (canonical behavior lives in agent_market.flow_steps).
run_command = _legacy.run_command
run_feature_generation = _legacy.run_feature_generation
run_expression_generation = _legacy.run_expression_generation
get_freqtrade_version = _legacy.get_freqtrade_version
run_ml_training = _legacy.run_ml_training
run_rl_training = _legacy.run_rl_training
run_backtest = _legacy.run_backtest
run_micro_feature = _legacy.run_micro_feature
run_tca = _legacy.run_tca
run_factor_compile = _legacy.run_factor_compile
run_factor_eval = _legacy.run_factor_eval
run_capture = _legacy.run_capture
run_lob_rebuild = _legacy.run_lob_rebuild
run_report_bundle = _legacy.run_report_bundle
run_strategy_miner_step = _legacy.run_strategy_miner_step


def run_dir_for_run_id(run_id: str) -> Path:
    return (paths.runs_root() / str(run_id)).resolve()


def step_out_dir(run_id: str, step_id: str) -> Path:
    return (run_dir_for_run_id(run_id) / str(step_id)).resolve()


def merge_artifacts(existing: Dict[str, Optional[str]], updates: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    out = dict(existing or {})
    for key, value in (updates or {}).items():
        if value is None or value == "":
            continue
        out[str(key)] = str(value)
    return out


__all__ = [
    "STEP_ORDER",
    "merge_artifacts",
    "run_backtest",
    "run_capture",
    "run_command",
    "run_dir_for_run_id",
    "run_expression_generation",
    "run_factor_compile",
    "run_factor_eval",
    "run_feature_generation",
    "get_freqtrade_version",
    "run_lob_rebuild",
    "run_micro_feature",
    "run_ml_training",
    "run_report_bundle",
    "run_rl_training",
    "run_tca",
    "run_strategy_miner_step",
    "step_out_dir",
]
