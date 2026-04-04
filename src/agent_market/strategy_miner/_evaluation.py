"""Evaluation and analysis phase handlers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from .agent_adapter import StrategyAgent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .prompts import build_analysis_prompt
from ._helpers import (
    _parse_json_object,
    _pick_active_candidate,
    _advance_after_candidate,
    _walkforward_timeranges,
)
from ._scoring import (
    _compute_effective_score,
    _training_score_adjustment,
    _training_robustness_violations,
    _check_per_pair_robustness,
    _safe_metric,
    _safe_ratio_metric,
)

logger = logging.getLogger(__name__)


def _run_walkforward_backtests(
    candidate: StrategyCandidate,
    config: MinerConfig,
) -> Optional[Dict[str, Any]]:
    """Run walk-forward OOS backtests across multiple time folds.

    Returns a summary dict with per-fold scores and aggregate stats,
    or None if walk-forward is disabled/fails.
    """
    if not getattr(config, "walkforward_enabled", False):
        return None

    import subprocess
    import sys
    from agent_market import paths
    from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip

    folds = _walkforward_timeranges(
        config.timerange,
        folds=int(getattr(config, "walkforward_folds", 3) or 3),
        train_ratio=float(getattr(config, "walkforward_train_ratio", 0.6) or 0.6),
    )
    if not folds:
        logger.debug("Walk-forward: no valid folds for timerange=%s", config.timerange)
        return None

    sandbox = candidate.strategy_path.parent.parent.parent
    strategies_dir = sandbox / "user_data" / "strategies"
    ft_config = paths.resolve_repo_path(config.freqtrade_config)

    fold_results: list[Dict[str, Any]] = []
    for i, (_train_range, test_range) in enumerate(folds):
        results_dir = sandbox / "user_data" / "backtest_results" / f"wf_fold_{i}"
        results_dir.mkdir(parents=True, exist_ok=True)

        wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
        if wrapper.exists():
            cmd = [
                sys.executable, str(wrapper), "backtesting",
                "--config", str(ft_config),
                "--strategy", candidate.name,
                "--strategy-path", str(strategies_dir),
                "--timerange", test_range,
                "--userdir", str(sandbox / "user_data"),
            ]
        else:
            cmd = [
                sys.executable, "-m", "freqtrade", "backtesting",
                "--config", str(ft_config),
                "--strategy", candidate.name,
                "--strategy-path", str(strategies_dir),
                "--timerange", test_range,
                "--userdir", str(sandbox / "user_data"),
            ]

        try:
            from ._sandbox_exec import run_sandboxed
            proc = run_sandboxed(
                cmd,
                cwd=paths.REPO_ROOT,
                timeout=config.backtest_timeout,
                cpu_seconds=config.backtest_timeout + 60,
                mem_mb=4096,
            )
            if proc.returncode != 0:
                logger.debug("Walk-forward fold %d failed (rc=%d)", i, proc.returncode)
                continue

            # Parse fold results from default results dir
            default_results = sandbox / "user_data" / "backtest_results"
            zip_path = find_latest_backtest_zip(default_results)
            if zip_path is None:
                continue

            summary = build_backtest_summary(zip_path)
            fold_results.append({
                "fold": i,
                "test_range": test_range,
                "sharpe": summary.get("realistic_sharpe", summary.get("sharpe", 0)),
                "profit_pct": summary.get("profit_total_pct", 0),
                "trades": summary.get("trades", 0),
                "winrate": summary.get("winrate", 0),
                "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
            })
        except subprocess.TimeoutExpired:
            logger.debug("Walk-forward fold %d timed out", i)
        except Exception as exc:
            logger.debug("Walk-forward fold %d error: %s", i, exc)

    if not fold_results:
        return None

    import statistics
    sharpes = [f["sharpe"] or 0.0 for f in fold_results]
    profits = [f["profit_pct"] or 0.0 for f in fold_results]
    return {
        "folds": fold_results,
        "folds_completed": len(fold_results),
        "folds_total": len(folds),
        "sharpe_mean": float(statistics.mean(sharpes)),
        "sharpe_std": float(statistics.stdev(sharpes)) if len(sharpes) >= 2 else 0.0,
        "profit_mean": float(statistics.mean(profits)),
        "profit_std": float(statistics.stdev(profits)) if len(profits) >= 2 else 0.0,
    }


def _extract_indicator_names(code: str) -> List[str]:
    """Extract indicator names from strategy code for research queries."""
    import re

    indicators = set()
    code_lower = code.lower()
    for m in re.finditer(r'\bta\.(\w+)\s*\(', code):
        indicators.add(m.group(1).upper())
    for kw in ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "cci",
               "stoch", "keltner", "donchian", "bollinger", "vwap", "obv"]:
        if kw in code_lower:
            indicators.add(kw.upper())
    return sorted(indicators)[:10]


def phase_evaluation(
    state: MinerState,
    config: MinerConfig,
    run_dir: Optional[Path] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Score backtest results using robust daily metrics instead of native freqtrade ratios."""

    candidate = _pick_active_candidate(state)
    if candidate is None:
        state.phase = Phase.ANALYSIS
        return

    if candidate.backtest_summary is None:
        _advance_after_candidate(state)
        return

    summary = candidate.backtest_summary or {}

    sharpe = _safe_ratio_metric(summary, "realistic_sharpe", "sharpe")
    sortino = _safe_ratio_metric(summary, "realistic_sortino", "sortino")
    calmar = _safe_ratio_metric(summary, "realistic_calmar", "calmar")
    native_sharpe = _safe_metric(summary, "native_sharpe", _safe_metric(summary, "sharpe"))
    native_sortino = _safe_metric(summary, "native_sortino", _safe_metric(summary, "sortino"))
    native_calmar = _safe_metric(summary, "native_calmar", _safe_metric(summary, "calmar"))
    profit_factor = _safe_metric(summary, "profit_factor")
    profit_pct = _safe_metric(summary, "profit_total_pct")
    expectancy = _safe_metric(summary, "expectancy")
    sqn = _safe_metric(summary, "sqn")
    cagr = _safe_metric(summary, "cagr")
    max_dd_pct = _safe_ratio_metric(summary, "max_drawdown_pct", "max_drawdown_account")
    return_over_drawdown = _safe_metric(summary, "return_over_drawdown")
    positive_days_ratio = _safe_metric(summary, "positive_days_ratio")
    observation_days = max(
        0,
        int(_safe_metric(summary, "observation_days")),
        int(_safe_metric(summary, "period_days")),
    )
    metric_flags = list(summary.get("metric_flags") or [])

    try:
        trades = int(summary.get("trades") or 0)
    except Exception:
        trades = 0

    try:
        winrate = float(summary.get("winrate") or 0.0)
        if winrate > 1.0:
            winrate = winrate / 100.0
    except Exception:
        winrate = 0.0

    observations = max(3, observation_days or trades or 0)
    total_candidates_so_far = max(1, len(state.candidates))
    effective_sharpe = _compute_effective_score(
        config=config,
        sharpe=sharpe,
        observations=observations,
        trades=trades,
        winrate=winrate,
        profit_factor=profit_factor,
        profit_pct=profit_pct,
        max_drawdown_pct=max_dd_pct,
        positive_days_ratio=positive_days_ratio,
        return_over_drawdown=return_over_drawdown,
        metric_flags=metric_flags,
        total_candidates=total_candidates_so_far,
    )
    effective_sharpe += _training_score_adjustment(candidate)

    # Walk-forward OOS validation (optional)
    wf_result = _run_walkforward_backtests(candidate, config)
    if wf_result is not None:
        summary["walkforward"] = wf_result
        # Penalize unstable walk-forward performance
        wf_std = wf_result.get("sharpe_std", 0.0)
        if wf_std > 1.0:
            effective_sharpe -= 0.3 * min(1.0, wf_std - 1.0)
            logger.info(
                "Walk-forward instability penalty for %s: std=%.4f",
                candidate.name, wf_std,
            )
        # Blend walk-forward mean sharpe with single-period score
        wf_mean = wf_result.get("sharpe_mean", 0.0)
        if wf_result.get("folds_completed", 0) >= 2:
            # 70% original + 30% walk-forward mean
            effective_sharpe = 0.7 * effective_sharpe + 0.3 * wf_mean

    candidate.reward = effective_sharpe

    # Risk constraint gating
    violations: list[str] = []
    pair_robust, pair_violations = _check_per_pair_robustness(
        summary,
        min_pair_profit_pct=float(getattr(config, "min_pair_profit_pct", -0.5) or -0.5),
    )
    _ = pair_robust
    if pair_violations:
        violations.extend(pair_violations)
    min_trades = int(getattr(config, "min_trades", 0) or 0)
    if min_trades and trades < min_trades:
        violations.append(f"min_trades:{trades}<{min_trades}")

    min_winrate = float(getattr(config, "min_winrate", 0.0) or 0.0)
    if min_winrate and winrate < min_winrate:
        violations.append(f"min_winrate:{winrate:.4f}<{min_winrate}")

    min_profit_factor = float(getattr(config, "min_profit_factor", 0.0) or 0.0)
    if min_profit_factor and profit_factor < min_profit_factor:
        violations.append(f"min_profit_factor:{profit_factor:.4f}<{min_profit_factor}")

    min_profit_pct = float(getattr(config, "min_profit_pct", 0.0) or 0.0)
    if min_profit_pct and profit_pct < min_profit_pct:
        violations.append(f"min_profit_pct:{profit_pct:.2f}<{min_profit_pct:.2f}")

    min_positive_days_ratio = float(getattr(config, "min_positive_days_ratio", 0.0) or 0.0)
    if min_positive_days_ratio and positive_days_ratio < min_positive_days_ratio:
        violations.append(
            f"min_positive_days_ratio:{positive_days_ratio:.4f}<{min_positive_days_ratio:.4f}"
        )

    min_return_over_drawdown = float(getattr(config, "min_return_over_drawdown", 0.0) or 0.0)
    if min_return_over_drawdown and return_over_drawdown < min_return_over_drawdown:
        violations.append(
            f"min_return_over_drawdown:{return_over_drawdown:.4f}<{min_return_over_drawdown:.4f}"
        )

    try:
        max_dd = abs(float(summary.get("max_drawdown_abs") or 0.0))
    except Exception:
        max_dd = 0.0
    max_abs_dd = float(getattr(config, "max_abs_drawdown", 0.0) or 0.0)
    if max_abs_dd and max_dd > max_abs_dd:
        violations.append(f"max_abs_drawdown:{max_dd:.4f}>{max_abs_dd}")

    max_dd_pct_limit = float(getattr(config, "max_drawdown_pct", 0.0) or 0.0)
    if max_dd_pct_limit and max_dd_pct > max_dd_pct_limit:
        violations.append(f"max_drawdown_pct:{max_dd_pct:.2f}>{max_dd_pct_limit:.2f}")

    violations.extend(_training_robustness_violations(candidate))

    candidate.constraint_violations = violations
    candidate.constraints_ok = not violations
    if violations:
        logger.info("Risk constraints violated for %s: %s", candidate.name, violations)

    logger.info(
        "Phase EVALUATION: %s sharpe=%.4f native_sharpe=%.4f sortino=%.4f "
        "native_sortino=%.4f calmar=%.4f native_calmar=%.4f "
        "profit_factor=%.4f profit=%.2f%% trades=%d winrate=%.4f "
        "pos_days=%.2f max_dd=%.2f%% rod=%.4f sqn=%.4f flags=%s (best_score=%.4f)",
        candidate.name, sharpe, native_sharpe, sortino,
        native_sortino, calmar, native_calmar,
        profit_factor, profit_pct, trades, winrate,
        positive_days_ratio, max_dd_pct, return_over_drawdown, sqn,
        ",".join(metric_flags) if metric_flags else "-",
        state.best_score,
    )

    if candidate.constraints_ok and effective_sharpe > state.best_score:
        state.best_score = effective_sharpe
        state.best_candidate = candidate
        logger.info(
            "New best candidate: %s with effective_sharpe=%.4f (score_sharpe=%.4f, native_sharpe=%.4f, trades=%d)",
            candidate.name, effective_sharpe, sharpe, native_sharpe, trades,
        )

    state.history.append(
        {
            "iteration": state.iteration,
            "name": candidate.name,
            "candidate_type": getattr(candidate, "candidate_type", "rule"),
            "model_family": getattr(candidate, "model_family", ""),
            "sharpe": sharpe,
            "native_sharpe": native_sharpe,
            "sortino": sortino,
            "native_sortino": native_sortino,
            "calmar": calmar,
            "native_calmar": native_calmar,
            "profit_factor": profit_factor,
            "profit_pct": profit_pct,
            "trades": trades,
            "winrate": winrate,
            "max_drawdown_pct": max_dd_pct,
            "positive_days_ratio": positive_days_ratio,
            "return_over_drawdown": return_over_drawdown,
            "metric_flags": metric_flags,
            "expectancy": expectancy,
            "sqn": sqn,
            "cagr": cagr,
            "training_summary": getattr(candidate, "training_summary", None),
            "constraints_ok": bool(candidate.constraints_ok),
            "constraint_violations": list(candidate.constraint_violations or []),
            "diagnosis": "",
        }
    )

    if kb is not None and candidate.reward is not None and candidate.constraints_ok:
        kb.add_elite(
            name=candidate.name,
            code=candidate.code,
            reward=effective_sharpe,
            backtest_summary=candidate.backtest_summary,
            iteration=state.iteration,
        )

    if run_dir is not None:
        try:
            from .artifacts import write_leaderboard

            write_leaderboard(run_dir, state, config=config)
        except Exception:
            logger.debug("Leaderboard write failed", exc_info=True)

    _advance_after_candidate(state)


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

    # Multi-candidate: pick best from this iteration if possible.
    iter_candidates = [c for c in state.candidates if c.iteration == state.iteration]
    candidate = None
    if iter_candidates:
        evaluated = [c for c in iter_candidates if c.backtest_summary is not None and c.reward is not None]
        if evaluated:
            eligible = [c for c in evaluated if bool(getattr(c, "constraints_ok", True))]
            pool = eligible or evaluated
            candidate = max(pool, key=lambda c: float(c.reward or -1e9))
        else:
            candidate = iter_candidates[-1]
    else:
        candidate = state.candidates[-1]

    # If we have backtest results and an agent, do LLM analysis
    if candidate.backtest_summary is not None and candidate.reward is not None and agent is not None:
        last_history = state.history[-1] if state.history else {}

        _analysis_research_str = ""
        try:
            from .research import research_for_analysis, format_analysis_context
            _indicators = _extract_indicator_names(candidate.code)
            _analysis_ctx = research_for_analysis(_indicators)
            _analysis_research_str = format_analysis_context(_analysis_ctx)
        except Exception as e:
            logger.debug("Analysis research skipped: %s", e)

        prompt = build_analysis_prompt(
            strategy_code=candidate.code,
            backtest_summary=candidate.backtest_summary,
            metrics=last_history,
            analysis_research_context=_analysis_research_str,
        )

        # Find the matching history row for this candidate (not just [-1])
        def _find_history_row() -> Optional[Dict[str, Any]]:
            for row in reversed(state.history):
                if row.get("iteration") == candidate.iteration and row.get("name") == candidate.name:
                    return row
            return state.history[-1] if state.history else None

        try:
            raw_diagnosis = agent.run(prompt)
            parsed = _parse_json_object(raw_diagnosis)
            history_row = _find_history_row()
            if parsed and "summary" in parsed:
                candidate.diagnosis = str(parsed.get("summary", ""))[:1000]
                if history_row is not None:
                    history_row["diagnosis"] = candidate.diagnosis
                    history_row["analysis_structured"] = {
                        "strengths": parsed.get("strengths", []),
                        "weaknesses": parsed.get("weaknesses", []),
                        "suggestions": parsed.get("suggestions", []),
                        "verdict": parsed.get("verdict", ""),
                    }
            else:
                candidate.diagnosis = raw_diagnosis.strip()[:1000]
                if history_row is not None:
                    history_row["diagnosis"] = candidate.diagnosis
        except Exception as e:
            logger.warning("Analysis agent failed: %s", e)
            candidate.diagnosis = f"Analysis failed: {e}"
    elif not candidate.diagnosis:
        candidate.diagnosis = "No backtest results to analyze"

    logger.info("Phase ANALYSIS complete: %s", (candidate.diagnosis or "")[:200])

    iter_candidates = [c for c in state.candidates if c.iteration == state.iteration]
    has_results = any(c.backtest_summary is not None for c in iter_candidates)

    if not has_results and iter_candidates:
        if state.gen_retries < 2:
            state.gen_retries += 1
            logger.info(
                "No candidates produced results in iteration %d — retrying (%d/2) without incrementing iteration",
                state.iteration, state.gen_retries,
            )
            state.phase = Phase.STRATEGY_GEN
            return
        else:
            logger.warning(
                "No candidates produced results after %d retries — moving on",
                state.gen_retries,
            )
            state.gen_retries = 0

    state.gen_retries = 0

    if state.iteration + 1 >= config.max_iterations:
        state.phase = Phase.COMPLETE
    else:
        state.iteration += 1
        state.phase = Phase.STRATEGY_GEN
