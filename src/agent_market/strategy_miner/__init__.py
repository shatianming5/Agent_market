"""Strategy-level mining system using LLM Agent autonomous loop."""
from __future__ import annotations

from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .knowledge_base import KnowledgeBase


def __getattr__(name: str):
    if name == "run_strategy_miner":
        from .runner import run_strategy_miner

        return run_strategy_miner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KnowledgeBase",
    "MinerConfig",
    "MinerState",
    "Phase",
    "StrategyCandidate",
    "run_strategy_miner",
]
