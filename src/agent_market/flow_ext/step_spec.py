"""Canonical Flow step order and AgentFlowConfig field mapping."""
from __future__ import annotations


STEP_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("capture", "capture"),
    ("lob_rebuild", "lob_rebuild"),
    ("feature", "feature"),
    ("micro_feature", "micro_feature"),
    ("portfolio", "portfolio"),
    ("expression", "expression"),
    ("factor_compile", "factor_compile"),
    ("factor_eval", "factor_eval"),
    ("ml", "ml_training"),
    ("rl", "rl_training"),
    ("backtest", "backtest"),
    ("tca", "tca"),
    ("strategy_miner", "strategy_miner"),
    ("report", "report"),
)

STEP_ORDER: list[str] = [name for name, _field in STEP_CONFIG_FIELDS]
CONFIG_FIELD_BY_STEP: dict[str, str] = dict(STEP_CONFIG_FIELDS)


__all__ = ["CONFIG_FIELD_BY_STEP", "STEP_CONFIG_FIELDS", "STEP_ORDER"]
