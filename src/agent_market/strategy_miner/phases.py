"""Phase handlers for the strategy mining loop."""
from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from agent_market import paths
from agent_market.backtest_results import build_backtest_summary, find_latest_backtest_zip

from .agent_adapter import StrategyAgent
from .agent_factory import build_strategy_agent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .evolution import evolve_strategy
from .grading import compute_enhanced_reward, compute_factor_score, compute_reward
from .prompts import build_analysis_prompt, build_repair_prompt, build_strategy_gen_prompt
from .sandbox import (
    auto_fix_strategy_code,
    auto_fix_strategy_file,
    find_strategy_files,
    infer_strategy_class_name,
    prepare_sandbox,
    validate_strategy_code,
)

logger = logging.getLogger(__name__)


def _pick_active_candidate(state: MinerState) -> Optional[StrategyCandidate]:
    if not state.candidates:
        return None

    idx = state.active_candidate_idx
    if idx is None and state.pending_candidate_idxs:
        idx = state.pending_candidate_idxs[0]
        state.active_candidate_idx = idx

    if idx is None:
        return state.candidates[-1]

    if 0 <= idx < len(state.candidates):
        return state.candidates[idx]

    # Out-of-range (corrupt checkpoint) fallback
    state.active_candidate_idx = None
    return state.candidates[-1]


def _mark_candidate_done(state: MinerState) -> None:
    """Pop active candidate from queue and update next phase."""
    idx = state.active_candidate_idx
    if idx is not None:
        try:
            state.pending_candidate_idxs = [i for i in state.pending_candidate_idxs if i != idx]
        except Exception:
            state.pending_candidate_idxs = []

    state.active_candidate_idx = state.pending_candidate_idxs[0] if state.pending_candidate_idxs else None


def _advance_after_candidate(state: MinerState) -> None:
    _mark_candidate_done(state)
    state.phase = Phase.BACKTEST if state.active_candidate_idx is not None else Phase.ANALYSIS


def _rewrite_strategy_class_name(code: str, old: str, new: str) -> str:
    """Best-effort rename `class <old>(...)` to `class <new>(...)` (first match only)."""
    pat = re.compile(rf"(\bclass\s+){re.escape(old)}(\s*\()")

    def _repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{new}{m.group(2)}"

    out, n = pat.subn(_repl, code, count=1)
    return out if n else code


def _normalize_candidate_artifact(
    *,
    sandbox: Path,
    strategy_path: Path,
    iteration: int,
    candidate_idx: int,
    names_seen: set[str],
    names_seen_lock: threading.Lock,
) -> tuple[str, Path, str, list[str]]:
    """Return (strategy_name, strategy_path, code, fixes)."""

    code_raw = strategy_path.read_text(encoding="utf-8", errors="replace")
    code, fixes = auto_fix_strategy_code(code_raw)
    if fixes and code != code_raw:
        strategy_path.write_text(code, encoding="utf-8")

    inferred = infer_strategy_class_name(code)
    if not inferred:
        inferred = strategy_path.stem

    name = inferred

    # Ensure stable uniqueness within iteration to avoid artifact overwrites.
    with names_seen_lock:
        if name in names_seen:
            new_name = f"{name}_cand{candidate_idx}"
            code2 = _rewrite_strategy_class_name(code, old=name, new=new_name)
            if code2 != code:
                code = code2
                fixes.append("rename_duplicate_class")
                name = new_name
                strategy_path.write_text(code, encoding="utf-8")
        names_seen.add(name)

    # Align filename with class name when possible.
    desired_path = strategy_path.with_name(f"{name}.py")
    if desired_path != strategy_path:
        try:
            if not desired_path.exists():
                strategy_path.rename(desired_path)
                fixes.append("rename_file_to_match_class")
                strategy_path = desired_path
        except Exception:
            logger.debug("Failed to rename strategy file %s -> %s", strategy_path, desired_path, exc_info=True)

    return name, strategy_path, code, fixes


def _repair_candidate(
    *,
    agent: StrategyAgent,
    config: MinerConfig,
    run_dir: Path,
    sandbox: Path,
    candidate: StrategyCandidate,
    failure: str,
    attempt: int,
    max_attempts: int,
) -> bool:
    try:
        rel = candidate.strategy_path
        try:
            rel = candidate.strategy_path.resolve().relative_to(sandbox.resolve())
        except Exception:
            rel = Path("user_data") / "strategies" / candidate.strategy_path.name

        prompt = build_repair_prompt(
            sandbox_path=str(sandbox),
            strategy_rel_path=str(rel),
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            failure=failure,
            attempt=attempt,
            max_attempts=max_attempts,
            tool_allowlist=list(config.tool_allowlist or []),
            bash_allow=bool(config.bash_allow),
            bash_timeout=int(config.bash_timeout or 60),
            bash_allowlist=list(config.bash_allowlist or []),
        )

        repaired_path = agent.generate_strategy(prompt, filename_hint=candidate.strategy_path.name)
        if repaired_path is None or not repaired_path.exists():
            return False

        candidate.strategy_path = repaired_path
        candidate.code = repaired_path.read_text(encoding="utf-8", errors="replace")

        # Auto-fix any lingering tool tags / fences after repair.
        did_fix, _ = auto_fix_strategy_file(candidate.strategy_path)
        if did_fix:
            candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")

        candidate.name = infer_strategy_class_name(candidate.code) or repaired_path.stem
        candidate.backtest_summary = None
        candidate.reward = None

        try:
            from .artifacts import write_candidate_snapshot

            write_candidate_snapshot(run_dir, candidate)
        except Exception:
            logger.debug("Candidate snapshot write failed", exc_info=True)

        return True
    except Exception:
        logger.debug("Repair attempt failed", exc_info=True)
        return False


def phase_strategy_gen(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: Optional[StrategyAgent] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Generate strategies via the LLM agent (multi-candidate, parallel)."""

    # Reset candidate queue for this iteration.
    state.pending_candidate_idxs = []
    state.active_candidate_idx = None

    n = max(1, int(getattr(config, "candidates_per_iteration", 1) or 1))

    best_code = state.best_candidate.code if state.best_candidate is not None else None

    # Inject knowledge base context
    elite_summaries = None
    failure_summary = None
    if kb is not None:
        elite_summaries = kb.elites[:3] if kb.elites else None
        failure_summary = kb.failure_summary(5) if kb.failures else None

    names_seen: set[str] = set()
    names_seen_lock = threading.Lock()

    def _gen_one(candidate_idx: int) -> Optional[StrategyCandidate]:
        variant = None if n == 1 else f"cand_{candidate_idx:02d}"
        sandbox = prepare_sandbox(config, run_dir, state.iteration, variant=variant)

        prompt = build_strategy_gen_prompt(
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
            sandbox_path=str(sandbox),
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            history=state.history,
            best_reward=state.best_reward,
            best_strategy_code=best_code,
            elite_summaries=elite_summaries,
            failure_summary=failure_summary,
        )

        # Prefer a provided agent only in single-candidate legacy mode.
        local_agent = agent if (agent is not None and n == 1) else build_strategy_agent(config, sandbox)
        try:
            filename_hint = f"MinedStrategy_{state.iteration}_{candidate_idx}.py"
            out_path = local_agent.generate_strategy(prompt, filename_hint=filename_hint)
            if out_path is None or not Path(out_path).exists():
                # Tool-capable providers may still have written files even if return value is None.
                candidates = find_strategy_files(sandbox)
                if candidates:
                    out_path = max(candidates, key=lambda p: p.stat().st_mtime)
                else:
                    logger.warning("Candidate %d produced no strategy file", candidate_idx)
                    return None

            name, norm_path, code, fixes = _normalize_candidate_artifact(
                sandbox=sandbox,
                strategy_path=Path(out_path),
                iteration=state.iteration,
                candidate_idx=candidate_idx,
                names_seen=names_seen,
                names_seen_lock=names_seen_lock,
            )
            if fixes:
                logger.info("Candidate %d normalized: %s", candidate_idx, ",".join(fixes))

            cand = StrategyCandidate(
                name=name,
                code=code,
                strategy_path=norm_path,
                iteration=state.iteration,
            )

            try:
                from .artifacts import write_candidate_snapshot

                write_candidate_snapshot(run_dir, cand)
            except Exception:
                logger.debug("Candidate snapshot write failed", exc_info=True)

            return cand
        finally:
            if local_agent is not agent:
                try:
                    local_agent.close()
                except Exception:
                    pass

    logger.info(
        "Phase STRATEGY_GEN: iteration=%d generating %d candidates (parallel)",
        state.iteration,
        n,
    )

    results: list[tuple[int, StrategyCandidate]] = []
    # Run in parallel to reduce wall-clock; each candidate has isolated sandbox.
    with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
        fut_map = {pool.submit(_gen_one, i): i for i in range(n)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                cand = fut.result()
            except Exception as exc:
                logger.warning("Candidate %d generation crashed: %s", i, exc)
                cand = None
            if cand is not None:
                results.append((i, cand))

    results.sort(key=lambda x: x[0])

    if not results:
        logger.warning("Agent did not produce any strategy files")
        state.phase = Phase.ANALYSIS
        return

    start_idx = len(state.candidates)
    state.candidates.extend([c for _, c in results])

    new_idxs = list(range(start_idx, len(state.candidates)))
    state.pending_candidate_idxs = new_idxs
    state.active_candidate_idx = new_idxs[0]

    state.phase = Phase.BACKTEST
    logger.info("Generated %d candidates for iteration %d", len(new_idxs), state.iteration)


def _classify_backtest_failure(stderr: str, stdout: str, *, rc: int | None = None) -> str:
    blob = (stderr or "") + "\n" + (stdout or "")
    blob_l = blob.lower()

    if "no module named 'freqtrade'" in blob_l or "module not found" in blob_l and "freqtrade" in blob_l:
        return (
            "Backtest failed: dependency_missing(freqtrade). "
            "Install with: pip install -r requirements-full.txt"
        )

    if "no module named 'ccxt.static_dependencies" in blob_l:
        return (
            "Backtest failed: dependency_missing(ccxt_static_dependencies). "
            "Pin ccxt==4.5.4 (known-good) or reinstall ccxt."
        )

    if "unrecognized arguments" in blob_l or "invalid choice" in blob_l:
        return "Backtest failed: parameter_error. Check freqtrade args/config/timerange."

    if "strategy" in blob_l and ("not found" in blob_l or "could not" in blob_l) and "strategy" in blob_l:
        return "Backtest failed: strategy_load_error. Strategy name/path may be wrong."

    if "filenotfounderror" in blob_l and "config" in blob_l:
        return "Backtest failed: config_path_error. freqtrade_config path is invalid."

    tail = (stderr or "")[-500:] or (stdout or "")[-500:]
    rc_s = "" if rc is None else f"rc={rc} "
    return f"Backtest failed ({rc_s}tail={tail})"


def _compute_overfit_penalty(
    code: str,
    summary: dict[str, Any],
    *,
    min_trades: int,
) -> tuple[float, list[str]]:
    """Heuristic penalty for low-trade overfit and threshold hacking."""

    reasons: list[str] = []
    penalty = 0.0

    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    profit_pct = _as_float((summary or {}).get("profit_total_pct") or 0.0)
    trades = int(_as_float((summary or {}).get("trades") or 0.0))
    winrate = _as_float((summary or {}).get("winrate") or 0.0)
    if winrate > 1.0:
        winrate = winrate / 100.0

    # 1) Low-trade penalty (soft, even when min_trades is small for recovery).
    trade_floor = max(10, int(min_trades or 0))
    if trades and trades < trade_floor:
        penalty += 0.10
        reasons.append(f"low_trades:{trades}<{trade_floor}")

    # 2) Suspiciously high winrate/profit with few trades (overfit).
    if trades and trades < max(20, trade_floor) and winrate >= 0.80:
        penalty += 0.15
        reasons.append(f"high_winrate_low_trades:{winrate:.3f}@{trades}")

    if trades and trades < max(30, trade_floor) and abs(profit_pct) >= 50.0:
        penalty += 0.15
        reasons.append(f"extreme_profit_low_trades:{profit_pct:.1f}@{trades}")

    # 3) Threshold/constant hacking: many numeric constants used in comparisons.
    try:
        tree = ast.parse(code or "")
        thresholds: list[float] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                parts = [node.left, *list(node.comparators or [])]
                for part in parts:
                    if isinstance(part, ast.Constant) and isinstance(part.value, (int, float)):
                        thresholds.append(float(part.value))
        if len(thresholds) > 12:
            extra = len(thresholds) - 12
            p = min(0.30, extra * 0.02)
            penalty += p
            reasons.append(f"too_many_thresholds:{len(thresholds)}")
    except SyntaxError:
        pass

    penalty = min(0.60, penalty)
    return penalty, reasons


def phase_backtest(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: Optional[StrategyAgent] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Validate and backtest the active candidate.

    When ``config.repair_attempts`` > 0, failures will trigger:
    1) local auto-fix (syntax/tool-tag cleanup)
    2) optional agent-guided repair loop

    Multi-candidate: failures advance to the next candidate without stopping the iteration.
    """

    candidate = _pick_active_candidate(state)
    if candidate is None:
        logger.warning("No candidates to backtest")
        state.phase = Phase.ANALYSIS
        return

    sandbox = candidate.strategy_path.parent.parent.parent  # sandbox root

    max_repairs = max(0, int(getattr(config, "repair_attempts", 0) or 0))

    for attempt_idx in range(max_repairs + 1):
        # Always refresh code from disk if possible (repairs may have edited it).
        try:
            if candidate.strategy_path.exists():
                candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

        # Static validation
        passed, msg = validate_strategy_code(candidate.code)
        candidate.validation_passed = passed
        if not passed:
            failure = f"Validation failed: {msg}"
            candidate.diagnosis = failure

            # Local auto-fix first for syntax/tool-tag failures.
            if "syntax error" in msg.lower() or "<write" in candidate.code.lower():
                did, fixes = auto_fix_strategy_file(candidate.strategy_path)
                if did:
                    try:
                        candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        pass
                    passed2, msg2 = validate_strategy_code(candidate.code)
                    candidate.validation_passed = passed2
                    if passed2:
                        logger.info("Auto-fix succeeded for %s: %s", candidate.name, ",".join(fixes))
                    else:
                        failure = f"Validation failed after auto-fix({','.join(fixes)}): {msg2}"
                        candidate.diagnosis = failure

            if candidate.validation_passed:
                # Proceed to backtest without consuming an LLM repair attempt.
                pass
            else:
                if agent is not None and attempt_idx < max_repairs:
                    ok = _repair_candidate(
                        agent=agent,
                        config=config,
                        run_dir=run_dir,
                        sandbox=sandbox,
                        candidate=candidate,
                        failure=failure,
                        attempt=attempt_idx + 1,
                        max_attempts=max_repairs,
                    )
                    if ok:
                        continue

                if kb is not None:
                    kb.add_failure(
                        name=candidate.name,
                        iteration=state.iteration,
                        failure_type="validation",
                        detail=candidate.diagnosis,
                    )

                _advance_after_candidate(state)
                return

        logger.info(
            "Phase BACKTEST: running freqtrade backtesting for %s (attempt %d/%d)",
            candidate.name,
            attempt_idx,
            max_repairs,
        )

        strategies_dir = sandbox / "user_data" / "strategies"

        # Build backtest command
        ft_config = paths.resolve_repo_path(config.freqtrade_config)
        results_dir = sandbox / "user_data" / "backtest_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "freqtrade",
            "backtesting",
            "--config",
            str(ft_config),
            "--strategy",
            candidate.name,
            "--strategy-path",
            str(strategies_dir),
            "--timerange",
            config.timerange,
            "--userdir",
            str(sandbox / "user_data"),
        ]

        # Try wrapper script first
        wrapper = paths.REPO_ROOT / "scripts" / "freqtrade_cli.py"
        if wrapper.exists():
            cmd = [
                sys.executable,
                str(wrapper),
                "backtesting",
                "--config",
                str(ft_config),
                "--strategy",
                candidate.name,
                "--strategy-path",
                str(strategies_dir),
                "--timerange",
                config.timerange,
                "--userdir",
                str(sandbox / "user_data"),
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
        except subprocess.TimeoutExpired:
            candidate.diagnosis = f"Backtest timed out after {config.backtest_timeout}s"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    local_agent = build_strategy_agent(config, sandbox)
                try:
                    ok = _repair_candidate(
                        agent=local_agent,
                        config=config,
                        run_dir=run_dir,
                        sandbox=sandbox,
                        candidate=candidate,
                        failure=candidate.diagnosis,
                        attempt=attempt_idx + 1,
                        max_attempts=max_repairs,
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

        if proc.returncode != 0:
            candidate.diagnosis = _classify_backtest_failure(proc.stderr or "", proc.stdout or "", rc=proc.returncode)
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    local_agent = build_strategy_agent(config, sandbox)
                try:
                    ok = _repair_candidate(
                        agent=local_agent,
                        config=config,
                        run_dir=run_dir,
                        sandbox=sandbox,
                        candidate=candidate,
                        failure=candidate.diagnosis,
                        attempt=attempt_idx + 1,
                        max_attempts=max_repairs,
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

        # Parse results
        try:
            zip_path = find_latest_backtest_zip(results_dir)
            if zip_path is None:
                candidate.diagnosis = "No backtest result zip found"
                logger.warning("%s", candidate.diagnosis)

                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        local_agent = build_strategy_agent(config, sandbox)
                    try:
                        ok = _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=candidate.diagnosis,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                    finally:
                        if local_agent is not agent:
                            try:
                                local_agent.close()
                            except Exception:
                                pass
                    if ok:
                        continue

                if kb is not None:
                    kb.add_failure(
                        name=candidate.name,
                        iteration=state.iteration,
                        failure_type="backtest",
                        detail=candidate.diagnosis,
                    )

                _advance_after_candidate(state)
                return

            summary = build_backtest_summary(zip_path)
            candidate.backtest_summary = summary
            state.phase = Phase.EVALUATION

            try:
                from .artifacts import write_backtest_summary

                write_backtest_summary(run_dir, candidate, zip_path=zip_path)
            except Exception:
                logger.debug("Backtest summary artifact write failed", exc_info=True)

            logger.info(
                "Backtest completed: profit=%.2f%% trades=%s",
                summary.get("profit_total_pct", 0) or 0,
                summary.get("trades"),
            )
            return
        except Exception as e:
            candidate.diagnosis = f"Backtest result parsing failed: {e}"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    local_agent = build_strategy_agent(config, sandbox)
                try:
                    ok = _repair_candidate(
                        agent=local_agent,
                        config=config,
                        run_dir=run_dir,
                        sandbox=sandbox,
                        candidate=candidate,
                        failure=candidate.diagnosis,
                        attempt=attempt_idx + 1,
                        max_attempts=max_repairs,
                    )
                finally:
                    if local_agent is not agent:
                        try:
                            local_agent.close()
                        except Exception:
                            pass
                if ok:
                    continue

            if kb is not None:
                kb.add_failure(
                    name=candidate.name,
                    iteration=state.iteration,
                    failure_type="backtest",
                    detail=candidate.diagnosis,
                )

            _advance_after_candidate(state)
            return

    if kb is not None:
        kb.add_failure(
            name=candidate.name,
            iteration=state.iteration,
            failure_type="backtest",
            detail=candidate.diagnosis or "Backtest failed",
        )

    _advance_after_candidate(state)


def phase_evaluation(
    state: MinerState,
    config: MinerConfig,
    run_dir: Optional[Path] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Score backtest results for the active candidate and update best candidate."""

    candidate = _pick_active_candidate(state)
    if candidate is None:
        state.phase = Phase.ANALYSIS
        return

    if candidate.backtest_summary is None:
        _advance_after_candidate(state)
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

    penalty, penalty_reasons = _compute_overfit_penalty(
        candidate.code,
        candidate.backtest_summary or {},
        min_trades=int(getattr(config, "min_trades", 0) or 0),
    )
    if penalty:
        reward = max(-1.0, float(reward) - float(penalty))
        components["overfit_penalty"] = -float(penalty)
        logger.info("Applied overfit penalty %.3f for %s: %s", penalty, candidate.name, ";".join(penalty_reasons))

    candidate.reward = reward

    # Risk constraint gating (used by leaderboard/best selection)
    violations: list[str] = []
    summary = candidate.backtest_summary or {}
    try:
        trades = int(summary.get("trades") or 0)
    except Exception:
        trades = 0
    min_trades = int(getattr(config, "min_trades", 0) or 0)
    if min_trades and trades < min_trades:
        violations.append(f"min_trades:{trades}<{min_trades}")

    try:
        winrate = float(summary.get("winrate") or 0.0)
        if winrate > 1.0:
            winrate = winrate / 100.0
    except Exception:
        winrate = 0.0
    min_winrate = float(getattr(config, "min_winrate", 0.0) or 0.0)
    if min_winrate and winrate < min_winrate:
        violations.append(f"min_winrate:{winrate:.4f}<{min_winrate}")

    try:
        max_dd = abs(float(summary.get("max_drawdown_abs") or 0.0))
    except Exception:
        max_dd = 0.0
    max_abs_dd = float(getattr(config, "max_abs_drawdown", 0.0) or 0.0)
    if max_abs_dd and max_dd > max_abs_dd:
        violations.append(f"max_abs_drawdown:{max_dd:.4f}>{max_abs_dd}")

    candidate.constraint_violations = violations
    candidate.constraints_ok = not violations
    if violations:
        logger.info("Risk constraints violated for %s: %s", candidate.name, violations)

    logger.info(
        "Phase EVALUATION: reward=%.4f (best=%.4f) components=%s",
        reward,
        state.best_reward,
        components,
    )

    if candidate.constraints_ok and reward > state.best_reward:
        state.best_reward = reward
        state.best_candidate = candidate
        logger.info("New best candidate: %s with reward=%.4f", candidate.name, reward)

    # Store in history
    state.history.append(
        {
            "iteration": state.iteration,
            "name": candidate.name,
            "reward": reward,
            "constraints_ok": bool(candidate.constraints_ok),
            "constraint_violations": list(candidate.constraint_violations or []),
            "components": components,
            "overfit_penalty": float(penalty or 0.0),
            "overfit_reasons": list(penalty_reasons or []),
            "profit_pct": candidate.backtest_summary.get("profit_total_pct"),
            "trades": candidate.backtest_summary.get("trades"),
            "winrate": candidate.backtest_summary.get("winrate"),
            "max_drawdown": candidate.backtest_summary.get("max_drawdown_abs"),
            "factor_scores": factor_scores,
            "diagnosis": "",
        }
    )

    if kb is not None and candidate.reward is not None:
        kb.add_elite(
            name=candidate.name,
            code=candidate.code,
            reward=candidate.reward,
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
        # Prefer successful evaluated candidates
        evaluated = [c for c in iter_candidates if c.backtest_summary is not None and c.reward is not None]
        if evaluated:
            candidate = max(evaluated, key=lambda c: float(c.reward or -1e9))
        else:
            candidate = iter_candidates[-1]
    else:
        candidate = state.candidates[-1]

    # If we have backtest results and an agent, do LLM analysis
    if candidate.backtest_summary is not None and candidate.reward is not None and agent is not None:
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

    logger.info("Phase ANALYSIS complete: %s", (candidate.diagnosis or "")[:200])

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

    evolved_name = f"{state.best_candidate.name}_evolved_{state.iteration}"
    base_name = infer_strategy_class_name(evolved_code) or state.best_candidate.name
    if base_name != evolved_name:
        evolved_code = _rewrite_strategy_class_name(evolved_code, old=base_name, new=evolved_name)

    # Validate evolved code
    passed, msg = validate_strategy_code(evolved_code)
    if not passed:
        logger.warning("Evolved strategy failed validation: %s", msg)
        return None

    # Write evolved strategy to sandbox (legacy path, no variant)
    sandbox = prepare_sandbox(config, run_dir, state.iteration)
    strategies_dir = sandbox / "user_data" / "strategies"
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

    try:
        from .artifacts import write_candidate_snapshot

        write_candidate_snapshot(run_dir, candidate)
    except Exception:
        logger.debug("Candidate snapshot write failed", exc_info=True)

    logger.info(
        "Evolved candidate: %s (%d bytes, ops=%s)",
        evolved_name,
        len(evolved_code),
        ops,
    )
    return candidate
