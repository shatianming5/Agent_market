"""Main runner loop for strategy mining with checkpoint support."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from agent_market import paths

from .agent_factory import build_strategy_agent
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


def miner_run_dir(run_id: str) -> Path:
    """Standardized miner output directory.

    Layout: ``runs/<run_id>/strategy_miner`` (runs root is configurable).
    """

    return paths.run_dir(str(run_id)) / "strategy_miner"


def _checkpoint_path(miner_dir: Path) -> Path:
    return miner_dir / "checkpoint.json"


def _save_checkpoint(state: MinerState, miner_dir: Path) -> None:
    """Atomic checkpoint write."""

    cp_path = _checkpoint_path(miner_dir)
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
    """Update knowledge base (best-effort) based on the most recent candidate."""

    if not state.candidates:
        return

    # Prefer active candidate when available, else fallback to last.
    cand = None
    idx = getattr(state, "active_candidate_idx", None)
    if idx is not None and isinstance(idx, int) and 0 <= idx < len(state.candidates):
        cand = state.candidates[idx]
    else:
        cand = state.candidates[-1]

    if cand.reward is not None and cand.backtest_summary is not None:
        kb.add_elite(
            name=cand.name,
            code=cand.code,
            reward=cand.reward,
            backtest_summary=cand.backtest_summary,
            iteration=state.iteration,
        )
    elif cand.diagnosis:
        failure_type = "validation" if not cand.validation_passed else "backtest"
        kb.add_failure(
            name=cand.name,
            iteration=state.iteration,
            failure_type=failure_type,
            detail=cand.diagnosis,
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
    *,
    run_id: Optional[str] = None,
    resume: Optional[Path] = None,
) -> MinerState:
    """Run the strategy mining loop.

    State machine flow:
        STRATEGY_GEN → BACKTEST → EVALUATION → ANALYSIS → EVOLVE → STRATEGY_GEN
                                                             ↓ (if evolve succeeds)
                                                          BACKTEST (evolved candidate)

    Args:
        config: Mining configuration.
        run_id: Optional run id (used when starting a new run).
        resume: Path to checkpoint.json for resuming.

    Returns:
        Final MinerState.
    """

    if resume and resume.exists():
        state = _load_checkpoint(resume)
        logger.info(
            "Resumed from checkpoint: run_id=%s iteration=%d phase=%s",
            state.run_id,
            state.iteration,
            state.phase.value,
        )
    else:
        state = MinerState(run_id=str(run_id)) if run_id else MinerState()
        logger.info("Starting new mining run: run_id=%s", state.run_id)

    miner_dir = miner_run_dir(state.run_id)
    miner_dir.mkdir(parents=True, exist_ok=True)

    # Persist a stable proposal artifact for API/audit.
    from .artifacts import write_proposal

    write_proposal(miner_dir, run_id=state.run_id, config=config, overwrite=False)

    kb = KnowledgeBase(miner_dir / "knowledge_base.json")

    try:
        while state.phase != Phase.COMPLETE:
            logger.info(
                "=== Iteration %d | Phase %s ===",
                state.iteration,
                state.phase.value,
            )

            if state.phase == Phase.STRATEGY_GEN:
                phase_strategy_gen(state, config, miner_dir, kb=kb)

            elif state.phase == Phase.BACKTEST:
                phase_backtest(state, config, miner_dir, kb=kb)

            elif state.phase == Phase.EVALUATION:
                phase_evaluation(state, config, run_dir=miner_dir, kb=kb)

            elif state.phase == Phase.ANALYSIS:
                # Optional LLM diagnosis for best candidate in the iteration.
                agent = None
                try:
                    # Use the last generated sandbox as workspace if possible.
                    if state.candidates:
                        sandbox = state.candidates[-1].strategy_path.parent.parent.parent
                        agent = build_strategy_agent(config, sandbox)
                except Exception:
                    agent = None

                try:
                    phase_analysis(state, config, miner_dir, agent)
                finally:
                    if agent is not None:
                        try:
                            agent.close()
                        except Exception:
                            pass

                # After analysis, decide: evolve or next iteration
                if state.phase == Phase.STRATEGY_GEN and _should_evolve(config, state):
                    state.phase = Phase.EVOLVE

            elif state.phase == Phase.EVOLVE:
                elite_codes = kb.top_elite_codes(3)
                evolved = phase_evolve(
                    state,
                    config,
                    miner_dir,
                    elite_codes=elite_codes,
                )
                if evolved is not None:
                    # Evolved candidate ready → go to backtest
                    state.phase = Phase.BACKTEST
                    logger.info("Evolve succeeded, proceeding to backtest evolved candidate")
                else:
                    # Evolve failed → proceed to normal strategy gen
                    state.phase = Phase.STRATEGY_GEN
                    logger.info("Evolve produced nothing, falling back to strategy gen")

            _save_checkpoint(state, miner_dir)

        logger.info(
            "Mining complete: run_id=%s iterations=%d best_reward=%.4f",
            state.run_id,
            state.iteration,
            state.best_reward,
        )
        if state.best_candidate:
            logger.info(
                "Best strategy: %s (reward=%.4f)",
                state.best_candidate.name,
                state.best_candidate.reward or 0,
            )

    finally:
        _save_checkpoint(state, miner_dir)

    return state
