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

    agent: Optional[StrategyAgent] = None
    agent_workspace: Optional[Path] = None

    def _build_tool_policy(workspace: Path):
        from runner_fsm.opencode.tool_executor import ToolPolicy  # noqa: WPS433

        allowed = (
            frozenset(
                str(x).strip().lower()
                for x in (config.tool_allowlist or [])
                if str(x).strip()
            )
            if (config.tool_allowlist is not None)
            else None
        )
        return ToolPolicy(
            repo=workspace.resolve(),
            unattended="strict",
            allowed_tool_kinds=allowed,
            bash_allow=bool(config.bash_allow),
            bash_allowlist=tuple(
                str(x) for x in (config.bash_allowlist or []) if str(x).strip()
            ),
            bash_timeout_seconds=int(config.bash_timeout or 60),
        )

    def _ensure_agent() -> StrategyAgent:
        nonlocal agent
        nonlocal agent_workspace

        workspace = miner_dir / f"iter_{state.iteration}" / "sandbox"
        workspace.mkdir(parents=True, exist_ok=True)

        if agent is not None and agent_workspace is not None:
            try:
                if agent_workspace.resolve() == workspace.resolve():
                    return agent
            except Exception:
                pass

        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass

        tool_policy = _build_tool_policy(workspace)
        agent = StrategyAgent(
            workspace=workspace,
            model=config.model,
            base_url=config.base_url,
            max_turns=config.max_turns,
            stale_timeout=config.stale_timeout,
            max_retries=config.max_retries,
            provider=config.provider,
            tool_policy=tool_policy,
        )
        agent_workspace = workspace
        return agent

    try:
        while state.phase != Phase.COMPLETE:
            logger.info(
                "=== Iteration %d | Phase %s ===",
                state.iteration,
                state.phase.value,
            )

            if state.phase == Phase.STRATEGY_GEN:
                agent_i = _ensure_agent()
                phase_strategy_gen(state, config, miner_dir, agent_i, kb=kb)

            elif state.phase == Phase.BACKTEST:
                agent_i = _ensure_agent()
                phase_backtest(state, config, miner_dir, agent=agent_i)

            elif state.phase == Phase.EVALUATION:
                phase_evaluation(state, config, run_dir=miner_dir)
                _update_knowledge_base(kb, state)

            elif state.phase == Phase.ANALYSIS:
                agent_i = _ensure_agent()
                phase_analysis(state, config, miner_dir, agent_i)
                _update_knowledge_base(kb, state)
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
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass

    _save_checkpoint(state, miner_dir)
    return state
