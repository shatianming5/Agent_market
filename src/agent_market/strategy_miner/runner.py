"""Main runner loop for strategy mining with checkpoint support."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from agent_market import paths

from .agent_adapter import StrategyAgent
from .dtypes import MinerConfig, MinerState, Phase
from .knowledge_base import KnowledgeBase
from .phases import (
    phase_analysis,
    phase_backtest,
    phase_evaluation,
    phase_evolve,
    phase_strategy_gen,
)

logger = logging.getLogger(__name__)


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.json"


def _save_checkpoint(state: MinerState, run_dir: Path) -> None:
    """Atomic checkpoint write."""
    cp_path = _checkpoint_path(run_dir)
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    tmp = cp_path.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.rename(cp_path)
    logger.debug("Checkpoint saved: %s", cp_path)


def _load_checkpoint(cp_path: Path) -> MinerState:
    data = json.loads(cp_path.read_text(encoding="utf-8"))
    return MinerState.from_dict(data)


def _update_knowledge_base(kb: KnowledgeBase, state: MinerState) -> None:
    """Update knowledge base after evaluation/analysis phase."""
    if not state.candidates:
        return
    candidate = state.candidates[-1]
    if candidate.reward is not None and candidate.backtest_summary is not None:
        kb.add_elite(
            name=candidate.name,
            code=candidate.code,
            reward=candidate.reward,
            backtest_summary=candidate.backtest_summary,
            iteration=state.iteration,
        )
    elif candidate.diagnosis:
        failure_type = "validation" if not candidate.validation_passed else "backtest"
        kb.add_failure(
            name=candidate.name,
            iteration=state.iteration,
            failure_type=failure_type,
            detail=candidate.diagnosis,
        )


def _should_evolve(config: MinerConfig, state: MinerState) -> bool:
    """Decide whether to attempt evolution this iteration."""
    if not config.evolve_enabled:
        return False
    if state.best_candidate is None:
        return False
    if state.iteration < 1:
        return False
    return state.iteration % config.evolve_every_n == 0


def run_strategy_miner(
    config: MinerConfig,
    resume: Optional[Path] = None,
) -> MinerState:
    """Run the strategy mining loop.

    State machine flow:
        STRATEGY_GEN → BACKTEST → EVALUATION → ANALYSIS → EVOLVE → STRATEGY_GEN
                                                             ↓ (if evolve succeeds)
                                                          BACKTEST (evolved candidate)

    Args:
        config: Mining configuration.
        resume: Path to checkpoint.json for resuming.

    Returns:
        Final MinerState.
    """
    if resume and resume.exists():
        state = _load_checkpoint(resume)
        logger.info("Resumed from checkpoint: run_id=%s iteration=%d phase=%s",
                     state.run_id, state.iteration, state.phase.value)
    else:
        state = MinerState()
        logger.info("Starting new mining run: run_id=%s", state.run_id)

    run_dir = paths.artifacts_root() / "strategy_miner" / state.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    kb = KnowledgeBase(run_dir / "knowledge_base.json")

    # Create agent workspace (sandbox parent)
    workspace = run_dir / f"iter_{state.iteration}" / "sandbox"
    workspace.mkdir(parents=True, exist_ok=True)

    agent: Optional[StrategyAgent] = None
    try:
        agent = StrategyAgent(
            workspace=workspace,
            model=config.model,
            base_url=config.base_url,
            max_turns=config.max_turns,
            stale_timeout=config.stale_timeout,
            max_retries=config.max_retries,
        )

        while state.phase != Phase.COMPLETE:
            logger.info(
                "=== Iteration %d | Phase %s ===",
                state.iteration, state.phase.value,
            )

            if state.phase == Phase.STRATEGY_GEN:
                workspace = run_dir / f"iter_{state.iteration}" / "sandbox"
                workspace.mkdir(parents=True, exist_ok=True)
                phase_strategy_gen(state, config, run_dir, agent, kb=kb)

            elif state.phase == Phase.BACKTEST:
                phase_backtest(state, config, run_dir)

            elif state.phase == Phase.EVALUATION:
                phase_evaluation(state, config, run_dir=run_dir)
                _update_knowledge_base(kb, state)

            elif state.phase == Phase.ANALYSIS:
                phase_analysis(state, config, run_dir, agent)
                _update_knowledge_base(kb, state)
                # After analysis, decide: evolve or next iteration
                if state.phase == Phase.STRATEGY_GEN and _should_evolve(config, state):
                    state.phase = Phase.EVOLVE

            elif state.phase == Phase.EVOLVE:
                elite_codes = kb.top_elite_codes(3)
                evolved = phase_evolve(
                    state, config, run_dir, elite_codes=elite_codes,
                )
                if evolved is not None:
                    # Evolved candidate ready → go to backtest
                    state.phase = Phase.BACKTEST
                    logger.info("Evolve succeeded, proceeding to backtest evolved candidate")
                else:
                    # Evolve failed → proceed to normal strategy gen
                    state.phase = Phase.STRATEGY_GEN
                    logger.info("Evolve produced nothing, falling back to strategy gen")

            _save_checkpoint(state, run_dir)

        logger.info(
            "Mining complete: run_id=%s iterations=%d best_reward=%.4f",
            state.run_id, state.iteration, state.best_reward,
        )
        if state.best_candidate:
            logger.info(
                "Best strategy: %s (reward=%.4f)",
                state.best_candidate.name, state.best_candidate.reward or 0,
            )

    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass

    _save_checkpoint(state, run_dir)
    return state
