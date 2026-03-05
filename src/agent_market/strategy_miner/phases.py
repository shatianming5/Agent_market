"""Phase handlers for the strategy mining loop."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from agent_market import paths
from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip

from .agent_adapter import StrategyAgent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .evolution import evolve_strategy
from .grading import compute_enhanced_reward, compute_factor_score, compute_reward
from .prompts import build_analysis_prompt, build_strategy_gen_prompt
from .sandbox import find_strategy_files, prepare_sandbox, validate_strategy_code

logger = logging.getLogger(__name__)


def phase_strategy_gen(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: StrategyAgent,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Generate a strategy via the LLM agent."""
    sandbox = prepare_sandbox(config, run_dir, state.iteration)

    best_code = None
    if state.best_candidate is not None:
        best_code = state.best_candidate.code

    # Inject knowledge base context
    elite_summaries = None
    failure_summary = None
    if kb is not None:
        elite_summaries = kb.elites[:3] if kb.elites else None
        failure_summary = kb.failure_summary(5) if kb.failures else None

    prompt = build_strategy_gen_prompt(
        iteration=state.iteration,
        sandbox_path=str(sandbox),
        freqtrade_config=config.freqtrade_config,
        timerange=config.timerange,
        history=state.history,
        best_reward=state.best_reward,
        best_strategy_code=best_code,
        elite_summaries=elite_summaries,
        failure_summary=failure_summary,
    )

    logger.info("Phase STRATEGY_GEN: iteration=%d, sending prompt to agent", state.iteration)
    agent.generate_strategy(prompt)

    # Find generated strategy files
    strategy_files = find_strategy_files(sandbox)
    if not strategy_files:
        logger.warning("Agent did not produce any strategy files")
        state.phase = Phase.ANALYSIS
        return

    # Take the first (or newest) strategy file
    strat_file = max(strategy_files, key=lambda p: p.stat().st_mtime)
    code = strat_file.read_text(encoding="utf-8")
    name = strat_file.stem

    candidate = StrategyCandidate(
        name=name,
        code=code,
        strategy_path=strat_file,
        iteration=state.iteration,
    )
    state.candidates.append(candidate)
    state.phase = Phase.BACKTEST
    logger.info("Strategy generated: %s (%d bytes)", name, len(code))


def phase_backtest(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
) -> None:
    """Validate and backtest the latest candidate strategy."""
    if not state.candidates:
        logger.warning("No candidates to backtest")
        state.phase = Phase.ANALYSIS
        return

    candidate = state.candidates[-1]

    # Static validation
    passed, msg = validate_strategy_code(candidate.code)
    candidate.validation_passed = passed
    if not passed:
        logger.warning("Strategy validation failed: %s", msg)
        candidate.diagnosis = f"Validation failed: {msg}"
        state.phase = Phase.ANALYSIS
        return

    logger.info("Phase BACKTEST: running freqtrade backtesting for %s", candidate.name)

    sandbox = candidate.strategy_path.parent.parent.parent  # sandbox root
    strategies_dir = sandbox / "user_data" / "strategies"

    # Build backtest command
    ft_config = paths.resolve_repo_path(config.freqtrade_config)
    results_dir = sandbox / "user_data" / "backtest_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "freqtrade",
        "backtesting",
        "--config", str(ft_config),
        "--strategy", candidate.name,
        "--strategy-path", str(strategies_dir),
        "--timerange", config.timerange,
        "--userdir", str(sandbox / "user_data"),
    ]

    # Try wrapper script first
    wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
    if wrapper.exists():
        cmd = [
            sys.executable, str(wrapper),
            "backtesting",
            "--config", str(ft_config),
            "--strategy", candidate.name,
            "--strategy-path", str(strategies_dir),
            "--timerange", config.timerange,
            "--userdir", str(sandbox / "user_data"),
        ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(paths.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=config.backtest_timeout,
            check=False,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-500:]
            logger.warning("Backtest failed (rc=%d): %s", proc.returncode, stderr_tail)
            candidate.diagnosis = f"Backtest failed (rc={proc.returncode}): {stderr_tail}"
            state.phase = Phase.ANALYSIS
            return
    except subprocess.TimeoutExpired:
        logger.warning("Backtest timed out after %ds", config.backtest_timeout)
        candidate.diagnosis = f"Backtest timed out after {config.backtest_timeout}s"
        state.phase = Phase.ANALYSIS
        return

    # Parse results
    try:
        zip_path = find_latest_backtest_zip(results_dir)
        if zip_path is None:
            candidate.diagnosis = "No backtest result zip found"
            state.phase = Phase.ANALYSIS
            return
        summary = build_backtest_summary(zip_path)
        candidate.backtest_summary = summary
        state.phase = Phase.EVALUATION
        logger.info("Backtest completed: profit=%.2f%% trades=%s",
                     summary.get("profit_total_pct", 0), summary.get("trades"))
    except Exception as e:
        logger.warning("Failed to parse backtest results: %s", e)
        candidate.diagnosis = f"Backtest result parsing failed: {e}"
        state.phase = Phase.ANALYSIS


def phase_evaluation(
    state: MinerState,
    config: MinerConfig,
    run_dir: Optional[Path] = None,
) -> None:
    """Score the latest backtest results and update best candidate."""
    if not state.candidates:
        state.phase = Phase.ANALYSIS
        return

    candidate = state.candidates[-1]
    if candidate.backtest_summary is None:
        state.phase = Phase.ANALYSIS
        return

    # Try factor-level scoring if features are available
    factor_scores = None
    if run_dir is not None:
        iter_dir = run_dir / f"iter_{state.iteration}"
        features_candidates = list(iter_dir.rglob("features.parquet")) if iter_dir.exists() else []
        if features_candidates:
            factor_scores = compute_factor_score(
                features_parquet=features_candidates[0],
                expression=candidate.name,
                out_dir=iter_dir / "factor_scores",
            )

    if factor_scores is not None:
        reward, components = compute_enhanced_reward(
            candidate.backtest_summary,
            config.reward_weights,
            factor_scores=factor_scores,
        )
        logger.info("Enhanced scoring with factor quality: %s", factor_scores)
    else:
        reward, components = compute_reward(
            candidate.backtest_summary,
            config.reward_weights,
        )

    candidate.reward = reward

    logger.info(
        "Phase EVALUATION: reward=%.4f (best=%.4f) components=%s",
        reward, state.best_reward, components,
    )

    if reward > state.best_reward:
        state.best_reward = reward
        state.best_candidate = candidate
        logger.info("New best candidate: %s with reward=%.4f", candidate.name, reward)

    # Store in history
    state.history.append({
        "iteration": state.iteration,
        "name": candidate.name,
        "reward": reward,
        "components": components,
        "profit_pct": candidate.backtest_summary.get("profit_total_pct"),
        "trades": candidate.backtest_summary.get("trades"),
        "winrate": candidate.backtest_summary.get("winrate"),
        "max_drawdown": candidate.backtest_summary.get("max_drawdown_abs"),
        "factor_scores": factor_scores,
        "diagnosis": "",
    })

    state.phase = Phase.ANALYSIS


def phase_analysis(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: Optional[StrategyAgent] = None,
) -> None:
    """Analyze results and produce diagnosis for next iteration."""
    if not state.candidates:
        state.phase = Phase.COMPLETE
        return

    candidate = state.candidates[-1]

    # If we have backtest results and an agent, do LLM analysis
    if (
        candidate.backtest_summary is not None
        and candidate.reward is not None
        and agent is not None
    ):
        last_history = state.history[-1] if state.history else {}
        components = last_history.get("components", {})

        prompt = build_analysis_prompt(
            strategy_code=candidate.code,
            backtest_summary=candidate.backtest_summary,
            reward=candidate.reward,
            reward_components=components,
        )

        try:
            diagnosis = agent.run(prompt)
            candidate.diagnosis = diagnosis.strip()[:1000]
            if state.history:
                state.history[-1]["diagnosis"] = candidate.diagnosis
        except Exception as e:
            logger.warning("Analysis agent failed: %s", e)
            candidate.diagnosis = f"Analysis failed: {e}"
    elif not candidate.diagnosis:
        candidate.diagnosis = "No backtest results to analyze"

    logger.info("Phase ANALYSIS complete: %s", candidate.diagnosis[:200])

    # Decide next phase
    if state.iteration + 1 >= config.max_iterations:
        state.phase = Phase.COMPLETE
    else:
        state.iteration += 1
        state.phase = Phase.STRATEGY_GEN


def phase_evolve(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    elite_codes: Optional[List[str]] = None,
) -> Optional[StrategyCandidate]:
    """Evolve the best candidate through mutation/crossover.

    Returns a new evolved candidate or None.
    This phase is optional and can be inserted between ANALYSIS and STRATEGY_GEN.
    """
    if state.best_candidate is None:
        logger.info("No best candidate to evolve, skipping")
        return None

    code = state.best_candidate.code
    evolved_code, ops = evolve_strategy(
        code,
        elite_codes=elite_codes,
        mutation_intensity=config.mutation_intensity,
        indicator_swaps=1,
        crossover_prob=config.crossover_prob if elite_codes else 0.0,
    )

    if "no_change" in ops and len(ops) == 1:
        logger.info("Evolution produced no changes")
        return None

    logger.info("Evolution applied: %s", ", ".join(ops))

    # Validate evolved code
    passed, msg = validate_strategy_code(evolved_code)
    if not passed:
        logger.warning("Evolved strategy failed validation: %s", msg)
        return None

    # Write evolved strategy to sandbox
    sandbox = prepare_sandbox(config, run_dir, state.iteration)
    strategies_dir = sandbox / "user_data" / "strategies"
    evolved_name = f"{state.best_candidate.name}_evolved_{state.iteration}"
    evolved_path = strategies_dir / f"{evolved_name}.py"
    evolved_path.write_text(evolved_code, encoding="utf-8")

    candidate = StrategyCandidate(
        name=evolved_name,
        code=evolved_code,
        strategy_path=evolved_path,
        iteration=state.iteration,
        validation_passed=True,
    )
    state.candidates.append(candidate)
    logger.info("Evolved candidate: %s (%d bytes, ops=%s)", evolved_name, len(evolved_code), ops)
    return candidate
