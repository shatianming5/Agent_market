"""Phase handlers for the strategy mining loop.

Split into submodules for maintainability:
  _helpers.py    — shared utilities (timeframe, config, candidate management)
  _scoring.py    — metric computation (PSR, effective score, robustness)
  _rendering.py  — ML/RL strategy code rendering + model payload normalization
  _generation.py — phase_strategy_gen (LLM-based strategy generation)
  _backtest.py   — phase_train_model + phase_backtest + repair logic
  _evaluation.py — phase_evaluation + phase_analysis
"""
from __future__ import annotations

from ._backtest import phase_backtest, phase_train_model
from ._evaluation import phase_analysis, phase_evaluation
from ._generation import phase_strategy_gen

__all__ = [
    "phase_strategy_gen",
    "phase_train_model",
    "phase_backtest",
    "phase_evaluation",
    "phase_analysis",
]
