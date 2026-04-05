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
    phase_strategy_gen,
    phase_train_model,
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
    """Atomic checkpoint write with fsync for durability."""
    import os as _os

    cp_path = _checkpoint_path(miner_dir)
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    tmp = cp_path.with_suffix(".tmp")
    fd = _os.open(str(tmp), _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC)
    try:
        _os.write(fd, data.encode("utf-8"))
        _os.fsync(fd)
    finally:
        _os.close(fd)
    tmp.rename(cp_path)
    # Fsync parent directory to ensure rename is durable
    try:
        dir_fd = _os.open(str(cp_path.parent), _os.O_RDONLY)
        try:
            _os.fsync(dir_fd)
        finally:
            _os.close(dir_fd)
    except OSError:
        pass  # Best-effort on platforms that don't support dir fsync
    logger.debug("Checkpoint saved: %s", cp_path)


def _load_checkpoint(cp_path: Path) -> MinerState:
    tmp_path = cp_path.with_suffix(".tmp")

    # Try loading the main checkpoint first
    if cp_path.exists():
        try:
            data = json.loads(cp_path.read_text(encoding="utf-8"))
            # Clean up orphan .tmp if main checkpoint loads fine
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return MinerState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Main checkpoint corrupt (%s), attempting .tmp recovery", exc)

    # Recover from .tmp if main checkpoint is missing/corrupt
    if tmp_path.exists():
        try:
            data = json.loads(tmp_path.read_text(encoding="utf-8"))
            logger.info("Recovered checkpoint from orphan .tmp file: %s", tmp_path)
            # Promote .tmp to main checkpoint
            tmp_path.rename(cp_path)
            # Fsync parent directory to ensure promotion is durable
            try:
                import os as _os
                dir_fd = _os.open(str(cp_path.parent), _os.O_RDONLY)
                try:
                    _os.fsync(dir_fd)
                finally:
                    _os.close(dir_fd)
            except OSError:
                pass
            return MinerState.from_dict(data)
        except Exception as exc:
            logger.warning("Failed to recover from .tmp: %s", exc)

    # Fallback: re-raise if we can't load anything
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
        if getattr(cand, "constraints_ok", True):
            kb.add_elite(
                name=cand.name,
                code=cand.code,
                reward=cand.reward,  # stores Sharpe
                backtest_summary=cand.backtest_summary,
                iteration=state.iteration,
            )
        else:
            kb.add_failure(
                name=cand.name,
                iteration=state.iteration,
                failure_type="constraint_violation",
                detail=f"Constraint violations: {', '.join(cand.constraint_violations or [])}",
            )
    elif cand.diagnosis:
        failure_type = "validation" if not cand.validation_passed else "backtest"
        kb.add_failure(
            name=cand.name,
            iteration=state.iteration,
            failure_type=failure_type,
            detail=cand.diagnosis,
        )


def run_strategy_miner(
    config: MinerConfig,
    *,
    run_id: Optional[str] = None,
    resume: Optional[Path] = None,
) -> MinerState:
    """Run the strategy mining loop.

    State machine flow:
        STRATEGY_GEN → BACKTEST → EVALUATION → ANALYSIS → STRATEGY_GEN

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

    # --- Deep research: run once for new runs, load from file on resume ---
    _deep_research_result = None
    dr_path = miner_dir / "deep_research.json"
    if resume and dr_path.exists():
        try:
            import json as _json
            _deep_research_result = _json.loads(dr_path.read_text(encoding="utf-8"))
            logger.info("Loaded deep research from %s", dr_path)
        except Exception as e:
            logger.warning("Failed to load deep_research.json: %s", e)
    elif not resume and state.iteration == 0:
        try:
            from .deep_research import run_deep_research
            _deep_research_result = run_deep_research()
            import json as _json
            dr_path.write_text(_json.dumps(_deep_research_result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            logger.info("Deep research saved to %s", dr_path)
        except Exception as e:
            logger.warning("Deep research failed (non-fatal): %s", e)

    # Adaptive exploration tracking (P2-1)
    _stagnation_count = 0
    _last_best_score = state.best_score

    try:
        while state.phase != Phase.COMPLETE:
            logger.info(
                "=== Iteration %d | Phase %s ===",
                state.iteration,
                state.phase.value,
            )

            if state.phase == Phase.STRATEGY_GEN:
                # Track stagnation
                if state.best_score <= _last_best_score:
                    _stagnation_count += 1
                else:
                    _stagnation_count = 0
                    _last_best_score = state.best_score

                # Inject deep research into strategy generation
                _extra_kw = {}

                if _stagnation_count >= 5:
                    logger.info("Stagnation detected (%d rounds). Forcing full exploration.", _stagnation_count)
                    # Force all candidates into explore mode by clearing best code
                    _extra_kw["strategy_blueprints"] = (
                        "\n## EXPLORATION MODE (stagnation detected)\n"
                        "Best score has NOT improved for 5+ rounds. You MUST try a fundamentally "
                        "different approach. Do NOT reuse EMA/RSI/Keltner patterns from elite archive.\n"
                    )
                if _deep_research_result and not _deep_research_result.get("skipped"):
                    try:
                        from .deep_research import format_blueprints_for_prompt
                        from .research import format_paper_insights
                        bp_str = format_blueprints_for_prompt(
                            _deep_research_result.get("blueprints", []),
                            _deep_research_result.get("regime"),
                        )
                        paper_str = format_paper_insights(
                            _deep_research_result.get("literature", {}).get("papers_by_topic", {})
                        )
                        _extra_kw["research_insights"] = paper_str
                        _extra_kw["strategy_blueprints"] = bp_str
                    except Exception as e:
                        logger.debug("Research formatting failed: %s", e)
                phase_strategy_gen(state, config, miner_dir, kb=kb, **_extra_kw)

            elif state.phase == Phase.BACKTEST:
                phase_backtest(state, config, miner_dir, kb=kb)

            elif state.phase == Phase.TRAIN_MODEL:
                phase_train_model(state, config, miner_dir, kb=kb)

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

            _save_checkpoint(state, miner_dir)

        # Sealed holdout: run exactly once on the final champion
        if (
            state.best_candidate is not None
            and "holdout" not in (state.best_candidate.funnel_state or {})
        ):
            try:
                from ._holdout import run_sealed_holdout
                holdout_result = run_sealed_holdout(state.best_candidate, config, miner_dir)
                if holdout_result is not None:
                    state.best_candidate.funnel_state["holdout"] = holdout_result
                    _save_checkpoint(state, miner_dir)
            except Exception as exc:
                logger.warning("Sealed holdout failed: %s", exc)

        best_summary = state.best_candidate.backtest_summary if state.best_candidate else {}
        logger.info(
            "Mining complete: run_id=%s iterations=%d best_sharpe=%.4f best_profit=%.2f%%",
            state.run_id,
            state.iteration,
            state.best_score,
            float((best_summary or {}).get("profit_total_pct") or 0),
        )
        if state.best_candidate:
            logger.info(
                "Best strategy: %s (sharpe=%.4f)",
                state.best_candidate.name,
                state.best_candidate.reward or 0,
            )

    finally:
        _save_checkpoint(state, miner_dir)

    return state
