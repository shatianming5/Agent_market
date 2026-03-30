"""Phase handlers for the strategy mining loop."""
from __future__ import annotations

import hashlib
import json
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
from .grading import compute_factor_score
from .prompts import (
    build_analysis_prompt,
    build_backtester_prompt,
    build_planner_prompt,
    build_repair_prompt,
    build_reviewer_prompt,
    build_strategy_gen_prompt,
)
from .sandbox import (
    auto_fix_strategy_code,
    auto_fix_strategy_file,
    ensure_freqtrade_strategy_compliance_file,
    find_strategy_files,
    infer_strategy_class_name,
    prepare_sandbox,
    validate_strategy_code,
)

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _freqtrade_config_defaults(freqtrade_config_path: str) -> tuple[str, bool]:
    """Return (timeframe, enforce_can_short_false) from a freqtrade config file."""
    try:
        ft_path = paths.resolve_repo_path(freqtrade_config_path)
        payload = json.loads(ft_path.read_text(encoding="utf-8-sig"))
        timeframe = str(payload.get("timeframe") or "1h").strip() or "1h"
        trading_mode = str(payload.get("trading_mode") or "spot").strip().lower() or "spot"
        enforce_can_short_false = trading_mode == "spot"
        return timeframe, enforce_can_short_false
    except Exception:
        return "1h", True



def _build_market_profile(freqtrade_config_path: str) -> Optional[str]:
    """Extract market profile info from freqtrade config for prompt injection."""
    try:
        ft_path = paths.resolve_repo_path(freqtrade_config_path)
        payload = json.loads(ft_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    lines = []
    pairs = payload.get("exchange", {}).get("pair_whitelist", [])
    if pairs:
        lines.append(f"- Trading pairs: {', '.join(pairs[:10])}" + (f" (+{len(pairs)-10} more)" if len(pairs) > 10 else ""))
    stake = payload.get("stake_currency")
    if stake:
        lines.append(f"- Stake currency: {stake}")
    mode = payload.get("trading_mode", "spot")
    lines.append(f"- Trading mode: {mode}")
    wallet = payload.get("dry_run_wallet")
    if wallet:
        lines.append(f"- Dry-run wallet: {wallet}")
    timeframe = payload.get("timeframe")
    if timeframe:
        lines.append(f"- Timeframe: {timeframe}")
    return "\n".join(lines) if lines else None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()

    # Fast path: pure JSON.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Best-effort extraction from mixed output.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _classify_validation_failure(msg: str) -> str:
    s = str(msg or "").strip().lower()
    if not s:
        return "validation.unknown"
    if "syntax error" in s:
        return "validation.syntax"
    if "forbidden import" in s:
        return "validation.forbidden_import"
    if "forbidden call" in s:
        return "validation.forbidden_call"
    if "look-ahead" in s or "look ahead" in s:
        return "validation.lookahead"
    if "must inherit" in s and "istrategy" in s:
        return "validation.inheritance"
    if "missing required methods" in s:
        return "validation.missing_methods"
    return "validation.other"


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
        original_path = candidate.strategy_path
        before_hash = None
        try:
            if original_path.exists():
                before_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
        except Exception:
            before_hash = None

        rel = original_path
        try:
            rel = original_path.resolve().relative_to(sandbox.resolve())
        except Exception:
            rel = Path("user_data") / "strategies" / original_path.name

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
            provider=config.provider,
        )

        repair_provider = ""
        repair_model = None
        attempts: list[dict[str, Any]] = []

        def _on_result(r: Any) -> None:
            nonlocal repair_provider, repair_model
            repair_provider = str(getattr(r, "provider", "") or "")
            repair_model = getattr(r, "model", None)
            attempts.append(
                {
                    "provider": repair_provider,
                    "model": repair_model,
                    "assistant_text": (getattr(r, "assistant_text", "") or "")[:4000],
                    "tool_trace": getattr(r, "tool_trace", None),
                }
            )

        try:
            repaired_path = agent.generate_strategy(
                prompt,
                filename_hint=original_path.name,
                on_result=_on_result,
            )
        except TypeError:
            repaired_path = agent.generate_strategy(prompt, filename_hint=original_path.name)

        if repaired_path is None or not repaired_path.exists():
            return False

        after_hash = None
        try:
            after_hash = hashlib.sha256(repaired_path.read_bytes()).hexdigest()
        except Exception:
            after_hash = None

        same_file = False
        try:
            same_file = repaired_path.resolve() == original_path.resolve()
        except Exception:
            same_file = str(repaired_path) == str(original_path)

        if same_file and before_hash is not None and after_hash is not None and before_hash == after_hash:
            logger.info(
                "Repair produced no changes for %s (attempt %d/%d)",
                original_path.name,
                int(attempt),
                int(max_attempts),
            )
            return False

        # Update candidate to point at repaired artifact.
        candidate.strategy_path = repaired_path
        if repair_provider:
            candidate.source_provider = repair_provider
            # Preserve original generation provider if present.
            if not getattr(candidate, "generation_provider", ""):
                candidate.generation_provider = repair_provider
        if repair_model is not None:
            candidate.source_model = repair_model
            if getattr(candidate, "generation_model", None) is None:
                candidate.generation_model = repair_model

        candidate.code = repaired_path.read_text(encoding="utf-8", errors="replace")

        # Auto-fix any lingering tool tags / fences after repair.
        did_fix, auto_fixes = auto_fix_strategy_file(candidate.strategy_path)
        if did_fix:
            candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
        else:
            auto_fixes = []

        # Ensure freqtrade sanity settings regardless of LLM output.
        compliance_fixes: list[str] = []
        try:
            tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
            did_comp, compliance_fixes = ensure_freqtrade_strategy_compliance_file(
                candidate.strategy_path,
                timeframe=tf,
                enforce_can_short_false=enforce_short,
            )
            if did_comp:
                candidate.code = candidate.strategy_path.read_text(encoding="utf-8", errors="replace")
                logger.info(
                    "Compliance auto-fix applied (repair) for %s: %s",
                    candidate.strategy_path.name,
                    ",".join(compliance_fixes),
                )
        except Exception:
            compliance_fixes = []

        candidate.name = infer_strategy_class_name(candidate.code) or repaired_path.stem
        candidate.backtest_summary = None
        candidate.reward = None
        candidate.failure_category = ""
        candidate.diagnosis = ""

        # Record repair trace (best-effort).
        try:
            from .artifacts import write_agent_trace

            slot = int(getattr(candidate, "candidate_slot", 0) or 0)
            failure_cat = str(getattr(candidate, "failure_category", "") or "unknown")
            role = f"repair_{int(attempt):02d}.{failure_cat}"
            p = write_agent_trace(
                run_dir,
                iteration=int(candidate.iteration),
                candidate_idx=slot,
                role=role,
                payload={
                    "failure_category": failure_cat,
                    "failure": failure,
                    "attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "provider": repair_provider,
                    "model": repair_model,
                    "repaired_path": str(repaired_path),
                    "attempts": attempts,
                    "same_file": bool(same_file),
                    "before_hash": before_hash,
                    "after_hash": after_hash,
                    "auto_fixes": list(auto_fixes or []),
                    "compliance_fixes": list(compliance_fixes or []),
                },
            )
            candidate.agent_traces = dict(getattr(candidate, "agent_traces", None) or {})
            candidate.agent_traces[role] = str(p)
        except Exception:
            logger.debug("Agent trace write failed (repair)", exc_info=True)

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
        cand_label = f"cand_{candidate_idx:02d}"

        if not bool(getattr(config, "multiagent_enabled", False)):
            variant = None if n == 1 else cand_label
            sandbox = prepare_sandbox(
                config,
                run_dir,
                state.iteration,
                variant=variant,
            )

            prompt = build_strategy_gen_prompt(
                iteration=state.iteration,
                candidate_idx=candidate_idx,
                candidates_per_iteration=n,
                sandbox_path=str(sandbox),
                freqtrade_config=config.freqtrade_config,
                timerange=config.timerange,
                history=state.history,
                best_score=state.best_score,
                best_strategy_code=best_code,
                elite_summaries=elite_summaries,
                failure_summary=failure_summary,
                provider=config.provider,
                market_profile=_build_market_profile(config.freqtrade_config),
            )

            gen_provider = ""
            gen_model = None

            # Prefer a provided agent only in single-candidate legacy mode.
            local_agent = agent if (agent is not None and n == 1) else build_strategy_agent(config, sandbox)

            def _on_result(r: Any) -> None:
                nonlocal gen_provider, gen_model
                gen_provider = str(getattr(r, "provider", "") or "")
                gen_model = getattr(r, "model", None)

            try:
                filename_hint = f"MinedStrategy_{state.iteration}_{candidate_idx}.py"
                try:
                    out_path = local_agent.generate_strategy(
                        prompt,
                        filename_hint=filename_hint,
                        on_result=_on_result,
                    )
                except TypeError:
                    out_path = local_agent.generate_strategy(prompt, filename_hint=filename_hint)

                if out_path is None or not Path(out_path).exists():
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

                # Ensure freqtrade sanity settings (order_types/time_in_force/can_short).
                try:
                    tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
                    did_comp, comp_fixes = ensure_freqtrade_strategy_compliance_file(
                        norm_path,
                        timeframe=tf,
                        enforce_can_short_false=enforce_short,
                    )
                    if did_comp:
                        code = norm_path.read_text(encoding="utf-8", errors="replace")
                        logger.info("Compliance auto-fix applied (gen) for candidate %d: %s", candidate_idx, ",".join(comp_fixes))
                except Exception:
                    logger.debug("Compliance auto-fix failed (gen) for candidate %d", candidate_idx, exc_info=True)

                cand = StrategyCandidate(
                    name=name,
                    code=code,
                    strategy_path=norm_path,
                    iteration=state.iteration,
                )
                if gen_provider:
                    cand.source_provider = gen_provider
                    cand.generation_provider = gen_provider
                if gen_model is not None:
                    cand.source_model = gen_model
                    cand.generation_model = gen_model

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

        # Role-specific sandboxes prevent tool conflicts and enable parallel roles.
        sandbox_coder = prepare_sandbox(
            config,
            run_dir,
            state.iteration,
            variant=f"{cand_label}/coder",
        )
        sandbox_planner = prepare_sandbox(
            config,
            run_dir,
            state.iteration,
            variant=f"{cand_label}/planner",
        )

        coder_prompt = build_strategy_gen_prompt(
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
            sandbox_path=str(sandbox_coder),
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            history=state.history,
            best_score=state.best_score,
            best_strategy_code=best_code,
            elite_summaries=elite_summaries,
            failure_summary=failure_summary,
            provider=config.provider,
            market_profile=_build_market_profile(config.freqtrade_config),
        )

        planner_prompt = build_planner_prompt(
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            history=state.history,
            elite_summaries=elite_summaries,
            failure_summary=failure_summary,
        )

        trace_paths: dict[str, str] = {}
        planner_notes = ""
        reviewer_notes = ""
        backtester_notes = ""

        generation_provider = ""
        generation_model = None

        def _write_trace(role: str, payload: dict[str, Any]) -> None:
            try:
                from .artifacts import write_agent_trace

                p = write_agent_trace(
                    run_dir,
                    iteration=state.iteration,
                    candidate_idx=candidate_idx,
                    role=role,
                    payload=payload,
                )
                trace_paths[role] = str(p)
            except Exception:
                logger.debug("Agent trace write failed (%s)", role, exc_info=True)

        def _planner_job() -> None:
            nonlocal planner_notes
            local_agent = build_strategy_agent(config, sandbox_planner)
            try:
                res = local_agent.run_result(planner_prompt)
                planner_notes = (res.assistant_text or "").strip()[:4000]
                _write_trace(
                    "planner",
                    {
                        "provider": res.provider,
                        "model": res.model,
                        "assistant_text": res.assistant_text,
                        "tool_trace": res.tool_trace,
                    },
                )
            finally:
                try:
                    local_agent.close()
                except Exception:
                    pass

        def _coder_job(prompt: str) -> Path | None:
            nonlocal generation_provider, generation_model

            # Prefer a provided agent only in single-candidate legacy mode.
            local_agent = agent if (agent is not None and n == 1) else build_strategy_agent(config, sandbox_coder)
            attempts: list[dict[str, Any]] = []
            last_provider = ""
            last_model = None

            def _on_result(r: Any) -> None:
                nonlocal last_provider, last_model
                last_provider = str(getattr(r, "provider", "") or "")
                last_model = getattr(r, "model", None)
                attempts.append(
                    {
                        "provider": last_provider,
                        "model": last_model,
                        "assistant_text": (getattr(r, "assistant_text", "") or "")[:4000],
                        "tool_trace": getattr(r, "tool_trace", None),
                    }
                )

            try:
                filename_hint = f"MinedStrategy_{state.iteration}_{candidate_idx}.py"
                try:
                    out_path = local_agent.generate_strategy(
                        prompt,
                        filename_hint=filename_hint,
                        on_result=_on_result,
                    )
                except TypeError:
                    out_path = local_agent.generate_strategy(
                        prompt,
                        filename_hint=filename_hint,
                    )

                generation_provider = last_provider
                generation_model = last_model

                _write_trace(
                    "coder",
                    {
                        "provider": last_provider,
                        "model": last_model,
                        "out_path": str(out_path) if out_path is not None else None,
                        "attempts": attempts,
                    },
                )

                return Path(out_path) if out_path is not None else None
            finally:
                if local_agent is not agent:
                    try:
                        local_agent.close()
                    except Exception:
                        pass

        coder_path: Path | None = None

        # Role parallelism: 1=sequential, 2=parallel (clamped).
        roles_workers = int(getattr(config, "max_parallel_roles", 2) or 2)
        roles_workers = max(1, min(2, roles_workers))

        # --- Run planner + coder (parallel when enabled) ---
        if roles_workers <= 1:
            try:
                _planner_job()
            except Exception as exc:
                logger.debug("Candidate %d planner failed: %s", candidate_idx, exc)

            coder_prompt2 = coder_prompt
            if planner_notes:
                coder_prompt2 = (
                    coder_prompt2
                    + "\n\n# Planner notes (follow these guidelines)\n"
                    + planner_notes
                    + "\n"
                )

            try:
                coder_path = _coder_job(coder_prompt2)
            except Exception as exc:
                logger.warning("Candidate %d coder crashed: %s", candidate_idx, exc)
                coder_path = None
        else:
            with ThreadPoolExecutor(max_workers=2) as roles_pool0:
                fut_p = roles_pool0.submit(_planner_job)
                fut_c = roles_pool0.submit(_coder_job, coder_prompt)
                try:
                    _ = fut_p.result()
                except Exception as exc:
                    logger.debug("Candidate %d planner failed: %s", candidate_idx, exc)
                try:
                    coder_path = fut_c.result()
                except Exception as exc:
                    logger.warning("Candidate %d coder crashed: %s", candidate_idx, exc)
                    coder_path = None


        out_path = coder_path
        if out_path is None or not out_path.exists():
            # Tool-capable providers may still have written files even if return value is None.
            candidates = find_strategy_files(sandbox_coder)
            if candidates:
                out_path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                logger.warning("Candidate %d produced no strategy file", candidate_idx)
                return None

        name, norm_path, code, fixes = _normalize_candidate_artifact(
            sandbox=sandbox_coder,
            strategy_path=Path(out_path),
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            names_seen=names_seen,
            names_seen_lock=names_seen_lock,
        )
        if fixes:
            logger.info("Candidate %d normalized: %s", candidate_idx, ",".join(fixes))

        # Ensure freqtrade sanity settings before reviewer/backtester prompts.
        try:
            tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
            did_comp, comp_fixes = ensure_freqtrade_strategy_compliance_file(
                norm_path,
                timeframe=tf,
                enforce_can_short_false=enforce_short,
            )
            if did_comp:
                code = norm_path.read_text(encoding="utf-8", errors="replace")
                logger.info("Compliance auto-fix applied (gen) for candidate %d: %s", candidate_idx, ",".join(comp_fixes))
        except Exception:
            logger.debug("Compliance auto-fix failed (gen) for candidate %d", candidate_idx, exc_info=True)

        def _try_parse_json(text_blob: str) -> dict[str, Any] | None:
            if not isinstance(text_blob, str) or not text_blob.strip():
                return None

            try:
                obj = json.loads(text_blob)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

            m = re.search(r"```json\s*(\{.*?\})\s*```", text_blob, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    pass

            i = text_blob.find("{")
            j = text_blob.rfind("}")
            if 0 <= i < j:
                try:
                    obj = json.loads(text_blob[i : j + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None

        # --- Reviewer + Backtester (parallel) ---
        sandbox_reviewer = prepare_sandbox(
            config,
            run_dir,
            state.iteration,
            variant=f"{cand_label}/reviewer",
        )
        sandbox_backtester = prepare_sandbox(
            config,
            run_dir,
            state.iteration,
            variant=f"{cand_label}/backtester",
        )

        reviewer_prompt = build_reviewer_prompt(
            strategy_code=code,
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
        )
        backtester_prompt = build_backtester_prompt(
            strategy_code=code,
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
        )

        reviewer_fixed_code: str | None = None

        def _reviewer_job() -> None:
            nonlocal reviewer_notes, reviewer_fixed_code
            local_agent = build_strategy_agent(config, sandbox_reviewer)
            try:
                res = local_agent.run_result(reviewer_prompt)
                reviewer_notes = (res.assistant_text or "").strip()[:4000]
                parsed = _try_parse_json(res.assistant_text or "") or {}
                fixed = parsed.get("fixed_code")
                if isinstance(fixed, str) and fixed.strip():
                    reviewer_fixed_code = fixed
                _write_trace(
                    "reviewer",
                    {
                        "provider": res.provider,
                        "model": res.model,
                        "assistant_text": res.assistant_text,
                        "tool_trace": res.tool_trace,
                        "parsed": parsed,
                    },
                )
            finally:
                try:
                    local_agent.close()
                except Exception:
                    pass

        def _backtester_job() -> None:
            nonlocal backtester_notes
            local_agent = build_strategy_agent(config, sandbox_backtester)
            try:
                res = local_agent.run_result(backtester_prompt)
                backtester_notes = (res.assistant_text or "").strip()[:4000]
                parsed = _try_parse_json(res.assistant_text or "")
                _write_trace(
                    "backtester",
                    {
                        "provider": res.provider,
                        "model": res.model,
                        "assistant_text": res.assistant_text,
                        "tool_trace": res.tool_trace,
                        "parsed": parsed,
                    },
                )
            finally:
                try:
                    local_agent.close()
                except Exception:
                    pass

        if roles_workers <= 1:
            try:
                _reviewer_job()
            except Exception as exc:
                logger.debug("Candidate %d reviewer failed: %s", candidate_idx, exc)
            try:
                _backtester_job()
            except Exception as exc:
                logger.debug("Candidate %d backtester failed: %s", candidate_idx, exc)
        else:
            with ThreadPoolExecutor(max_workers=2) as roles_pool2:
                fut_r = roles_pool2.submit(_reviewer_job)
                fut_b = roles_pool2.submit(_backtester_job)
                try:
                    _ = fut_r.result()
                except Exception as exc:
                    logger.debug("Candidate %d reviewer failed: %s", candidate_idx, exc)
                try:
                    _ = fut_b.result()
                except Exception as exc:
                    logger.debug("Candidate %d backtester failed: %s", candidate_idx, exc)

        # Apply reviewer fixed code if provided.
        if reviewer_fixed_code is not None:
            try:
                norm_path.write_text(reviewer_fixed_code, encoding="utf-8")
                did_fix, _ = auto_fix_strategy_file(norm_path)
                if did_fix:
                    reviewer_fixed_code = norm_path.read_text(encoding="utf-8", errors="replace")

                name2, norm_path2, code2, fixes2 = _normalize_candidate_artifact(
                    sandbox=sandbox_coder,
                    strategy_path=norm_path,
                    iteration=state.iteration,
                    candidate_idx=candidate_idx,
                    names_seen=names_seen,
                    names_seen_lock=names_seen_lock,
                )
                name, norm_path, code = name2, norm_path2, code2
                if fixes2:
                    logger.info("Candidate %d re-normalized after review: %s", candidate_idx, ",".join(fixes2))
            except Exception:
                logger.debug("Applying reviewer fixed_code failed", exc_info=True)

        # Ensure freqtrade sanity settings after reviewer edits.
        try:
            tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
            did_comp, comp_fixes = ensure_freqtrade_strategy_compliance_file(
                norm_path,
                timeframe=tf,
                enforce_can_short_false=enforce_short,
            )
            if did_comp:
                code = norm_path.read_text(encoding="utf-8", errors="replace")
                logger.info("Compliance auto-fix applied (post-review) for candidate %d: %s", candidate_idx, ",".join(comp_fixes))
        except Exception:
            logger.debug("Compliance auto-fix failed (post-review) for candidate %d", candidate_idx, exc_info=True)

        cand = StrategyCandidate(
            name=name,
            code=code,
            strategy_path=norm_path,
            iteration=state.iteration,
        )

        cand.candidate_slot = int(candidate_idx)

        cand.generation_provider = generation_provider
        cand.generation_model = generation_model
        cand.source_provider = generation_provider
        cand.source_model = generation_model
        cand.planner_notes = planner_notes
        cand.reviewer_notes = reviewer_notes
        cand.backtester_notes = backtester_notes
        cand.agent_traces = dict(trace_paths)

        try:
            from .artifacts import write_candidate_snapshot

            write_candidate_snapshot(run_dir, cand)
        except Exception:
            logger.debug("Candidate snapshot write failed", exc_info=True)

        return cand

    logger.info(
        "Phase STRATEGY_GEN: iteration=%d generating %d candidates (parallel)",
        state.iteration,
        n,
    )

    results: list[tuple[int, StrategyCandidate]] = []
    max_workers = min(8, n)
    max_parallel = int(getattr(config, "max_parallel_candidates", 0) or 0)
    if max_parallel > 0:
        max_workers = max(1, min(max_workers, max_parallel))

    # Run in parallel to reduce wall-clock; each candidate has isolated sandbox.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
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


def _classify_backtest_failure(stderr: str, stdout: str, *, rc: int | None = None) -> tuple[str, str]:
    blob = (stderr or "") + "\n" + (stdout or "")
    blob_l = blob.lower()

    if "no module named 'freqtrade'" in blob_l or ("module not found" in blob_l and "freqtrade" in blob_l):
        return (
            "backtest.dependency_missing.freqtrade",
            "Backtest failed: dependency_missing(freqtrade). Install with: pip install -r requirements-full.txt",
        )

    if "no module named 'ccxt.static_dependencies" in blob_l:
        return (
            "backtest.dependency_missing.ccxt_static_dependencies",
            "Backtest failed: dependency_missing(ccxt_static_dependencies). Pin ccxt==4.5.4 (known-good) or reinstall ccxt.",
        )

    if "no module named 'talib'" in blob_l or ("importerror" in blob_l and "talib" in blob_l) or "talib" in blob_l:
        return (
            "backtest.dependency_missing.talib",
            "Backtest failed: dependency_missing(talib). TA-Lib is NOT installed. "
            "Replace `import talib.abstract as ta` with `import pandas_ta as ta` and update ALL API calls: "
            "e.g. ta.EMA(dataframe, timeperiod=N) → ta.ema(dataframe['close'], length=N), "
            "ta.BBANDS(dataframe, ...) → bbands_df = ta.bbands(dataframe['close'], length=20, std=2); "
            "upper=bbands_df['BBU_20_2.0'], middle=bbands_df['BBM_20_2.0'], lower=bbands_df['BBL_20_2.0'].",
        )

    if "no module named 'pandas_ta'" in blob_l:
        return (
            "backtest.dependency_missing.pandas_ta",
            "Backtest failed: dependency_missing(pandas_ta). Install via requirements.txt or switch to manual indicators.",
        )

    if "no data" in blob_l and "found" in blob_l:
        return (
            "backtest.data_missing",
            "Backtest failed: data_missing. Ensure OHLCV exists for pairs/timeframe and timerange. If you only have 1h data, set strategy timeframe to 1h.",
        )

    if "order-types mapping is incomplete" in blob_l or "order_types mapping is incomplete" in blob_l:
        return (
            "backtest.strategy_config_incomplete",
            "Backtest failed: strategy_config_incomplete. Define complete order_types and order_time_in_force dicts in the strategy.",
        )

    if "unrecognized arguments" in blob_l or "invalid choice" in blob_l:
        return (
            "backtest.parameter_error",
            "Backtest failed: parameter_error. Check freqtrade args/config/timerange.",
        )

    if "strategy" in blob_l and ("not found" in blob_l or "could not" in blob_l) and "strategy" in blob_l:
        return (
            "backtest.strategy_load_error",
            "Backtest failed: strategy_load_error. Strategy class name/path may be wrong.",
        )

    if "filenotfounderror" in blob_l and "config" in blob_l:
        return (
            "backtest.config_path_error",
            "Backtest failed: config_path_error. freqtrade_config path is invalid.",
        )

    tail = (stderr or "")[-2000:] or (stdout or "")[-2000:]
    rc_s = "" if rc is None else f"rc={rc} "
    return "backtest.unknown", f"Backtest failed ({rc_s}tail={tail})"


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

    raw_repairs = int(getattr(config, "repair_attempts", 0) or 0)
    if raw_repairs <= 0:
        max_repairs = 0
    else:
        # Configurable repair rounds: clamp to [1, 8] when enabled.
        max_repairs = max(1, min(8, raw_repairs))

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
            category = _classify_validation_failure(msg)
            candidate.failure_category = category
            failure = f"[{category}] Validation failed: {msg}"
            candidate.diagnosis = failure

            # Local auto-fix first for syntax/tool-tag/forbidden-import failures.
            msg_lower = msg.lower()
            if "syntax error" in msg_lower or "<write" in candidate.code.lower() or "forbidden import" in msg_lower:
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
                        candidate.diagnosis = ""
                        candidate.failure_category = ""
                    else:
                        category2 = _classify_validation_failure(msg2)
                        candidate.failure_category = category2
                        failure = f"[{category2}] Validation failed after auto-fix({','.join(fixes)}): {msg2}"
                        candidate.diagnosis = failure

            if candidate.validation_passed:
                # Proceed to backtest without consuming an LLM repair attempt.
                pass
            else:
                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        try:
                            local_agent = build_strategy_agent(config, sandbox)
                        except Exception as exc:
                            logger.warning("Repair skipped (agent unavailable): %s", exc)
                            local_agent = None

                    try:
                        failure_for_repair = failure
                        if getattr(candidate, "backtester_notes", ""):
                            failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                                str(candidate.backtester_notes),
                                limit=1500,
                            )

                        ok = (
                            _repair_candidate(
                                agent=local_agent,
                                config=config,
                                run_dir=run_dir,
                                sandbox=sandbox,
                                candidate=candidate,
                                failure=failure_for_repair,
                                attempt=attempt_idx + 1,
                                max_attempts=max_repairs,
                            )
                            if local_agent is not None
                            else False
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
                        failure_type="validation",
                        detail=candidate.diagnosis,
                    )

                _advance_after_candidate(state)
                return

        # Preflight: ensure freqtrade sanity-required settings.
        try:
            tf, enforce_short = _freqtrade_config_defaults(config.freqtrade_config)
            did_comp, comp_fixes = ensure_freqtrade_strategy_compliance_file(
                candidate.strategy_path,
                timeframe=tf,
                enforce_can_short_false=enforce_short,
            )
            if did_comp:
                try:
                    candidate.code = candidate.strategy_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except Exception:
                    pass
                logger.info(
                    "Compliance auto-fix applied for %s: %s",
                    candidate.name,
                    ",".join(comp_fixes),
                )
        except Exception:
            logger.debug("Compliance auto-fix failed", exc_info=True)


        # Preflight: verify strategy is loadable via freqtrade list-strategies
        strategies_dir = sandbox / "user_data" / "strategies"
        try:
            ls_cmd = [
                sys.executable, "-m", "freqtrade", "list-strategies",
                "--strategy-path", str(strategies_dir),
            ]
            ls_proc = subprocess.run(
                ls_cmd, capture_output=True, text=True, timeout=30, check=False,
            )
            if ls_proc.returncode == 0 and (ls_proc.stdout or "").strip() and candidate.name not in ls_proc.stdout:
                logger.warning(
                    "Preflight: %s not found by freqtrade list-strategies — skipping to repair",
                    candidate.name,
                )
                candidate.failure_category = "backtest.preflight_not_found"
                candidate.diagnosis = (
                    f"[backtest.preflight_not_found] Strategy {candidate.name} not loadable by freqtrade. "
                    f"stderr: {(ls_proc.stderr or '')[:500]}"
                )
                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        try:
                            local_agent = build_strategy_agent(config, sandbox)
                        except Exception:
                            local_agent = None
                    if local_agent is not None:
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
                        if ok:
                            continue
                _advance_after_candidate(state)
                return
        except Exception:
            pass  # preflight is best-effort; continue to backtest

        logger.info(
            "Phase BACKTEST: running freqtrade backtesting for %s (attempt %d/%d)",
            candidate.name,
            attempt_idx,
            max_repairs,
        )

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
            candidate.failure_category = "backtest.timeout"
            candidate.diagnosis = f"[backtest.timeout] Backtest timed out after {config.backtest_timeout}s"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
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
            category, diag = _classify_backtest_failure(proc.stderr or "", proc.stdout or "", rc=proc.returncode)
            candidate.failure_category = category
            candidate.diagnosis = f"[{category}] {diag}"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
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
                candidate.failure_category = "backtest.result_missing_zip"
                candidate.diagnosis = "[backtest.result_missing_zip] No backtest result zip found"
                logger.warning("%s", candidate.diagnosis)

                if attempt_idx < max_repairs:
                    local_agent = agent
                    if local_agent is None:
                        try:
                            local_agent = build_strategy_agent(config, sandbox)
                        except Exception as exc:
                            logger.warning("Repair skipped (agent unavailable): %s", exc)
                            local_agent = None
                    try:
                        failure_for_repair = candidate.diagnosis
                        if getattr(candidate, "backtester_notes", ""):
                            failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                                str(candidate.backtester_notes),
                                limit=1500,
                            )
                        ok = (
                            _repair_candidate(
                                agent=local_agent,
                                config=config,
                                run_dir=run_dir,
                                sandbox=sandbox,
                                candidate=candidate,
                                failure=failure_for_repair,
                                attempt=attempt_idx + 1,
                                max_attempts=max_repairs,
                            )
                            if local_agent is not None
                            else False
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
            candidate.failure_category = ""
            candidate.diagnosis = ""
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
            candidate.failure_category = "backtest.result_parse_error"
            candidate.diagnosis = f"[backtest.result_parse_error] Backtest result parsing failed: {e}"
            logger.warning("%s", candidate.diagnosis)

            if attempt_idx < max_repairs:
                local_agent = agent
                if local_agent is None:
                    try:
                        local_agent = build_strategy_agent(config, sandbox)
                    except Exception as exc:
                        logger.warning("Repair skipped (agent unavailable): %s", exc)
                        local_agent = None
                try:
                    failure_for_repair = candidate.diagnosis
                    if getattr(candidate, "backtester_notes", ""):
                        failure_for_repair += "\n\nBacktester preflight:\n" + _truncate_text(
                            str(candidate.backtester_notes),
                            limit=1500,
                        )
                    ok = (
                        _repair_candidate(
                            agent=local_agent,
                            config=config,
                            run_dir=run_dir,
                            sandbox=sandbox,
                            candidate=candidate,
                            failure=failure_for_repair,
                            attempt=attempt_idx + 1,
                            max_attempts=max_repairs,
                        )
                        if local_agent is not None
                        else False
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


def _safe_metric(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Extract a float metric from backtest summary, defaulting on None/error."""
    try:
        v = summary.get(key)
        if v is None:
            return default
        f = float(v)
        import math
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def phase_evaluation(
    state: MinerState,
    config: MinerConfig,
    run_dir: Optional[Path] = None,
    kb: Optional["KnowledgeBase"] = None,
) -> None:
    """Score backtest results using professional quant metrics (Sharpe as primary)."""

    candidate = _pick_active_candidate(state)
    if candidate is None:
        state.phase = Phase.ANALYSIS
        return

    if candidate.backtest_summary is None:
        _advance_after_candidate(state)
        return

    summary = candidate.backtest_summary or {}

    # Extract professional metrics directly from backtest summary
    sharpe = _safe_metric(summary, "sharpe")
    sortino = _safe_metric(summary, "sortino")
    calmar = _safe_metric(summary, "calmar")
    profit_factor = _safe_metric(summary, "profit_factor")
    profit_pct = _safe_metric(summary, "profit_total_pct")
    expectancy = _safe_metric(summary, "expectancy")
    sqn = _safe_metric(summary, "sqn")
    cagr = _safe_metric(summary, "cagr")
    max_dd_pct = _safe_metric(summary, "max_drawdown_account")

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

    # Primary score = Sharpe with trade-count penalty for statistical significance
    effective_sharpe = sharpe * min(1.0, trades / 30) if trades > 0 else 0.0
    candidate.reward = effective_sharpe

    # Risk constraint gating
    violations: list[str] = []
    min_trades = int(getattr(config, "min_trades", 0) or 0)
    if min_trades and trades < min_trades:
        violations.append(f"min_trades:{trades}<{min_trades}")

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

    max_dd_pct_limit = float(getattr(config, "max_drawdown_pct", 0.0) or 0.0)
    if max_dd_pct_limit and max_dd_pct > max_dd_pct_limit:
        violations.append(f"max_drawdown_pct:{max_dd_pct:.2f}>{max_dd_pct_limit:.2f}")

    candidate.constraint_violations = violations
    candidate.constraints_ok = not violations
    if violations:
        logger.info("Risk constraints violated for %s: %s", candidate.name, violations)

    logger.info(
        "Phase EVALUATION: %s sharpe=%.4f sortino=%.4f calmar=%.4f "
        "profit_factor=%.4f profit=%.2f%% trades=%d winrate=%.4f "
        "max_dd=%.2f%% sqn=%.4f (best_score=%.4f)",
        candidate.name, sharpe, sortino, calmar,
        profit_factor, profit_pct, trades, winrate,
        max_dd_pct, sqn, state.best_score,
    )

    if candidate.constraints_ok and effective_sharpe > state.best_score:
        state.best_score = effective_sharpe
        state.best_candidate = candidate
        logger.info(
            "New best candidate: %s with effective_sharpe=%.4f (raw=%.4f, trades=%d)",
            candidate.name, effective_sharpe, sharpe, trades,
        )

    # Store in history with professional metrics
    state.history.append(
        {
            "iteration": state.iteration,
            "name": candidate.name,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "profit_factor": profit_factor,
            "profit_pct": profit_pct,
            "trades": trades,
            "winrate": winrate,
            "max_drawdown_pct": max_dd_pct,
            "expectancy": expectancy,
            "sqn": sqn,
            "cagr": cagr,
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

        prompt = build_analysis_prompt(
            strategy_code=candidate.code,
            backtest_summary=candidate.backtest_summary,
            metrics=last_history,
        )

        try:
            raw_diagnosis = agent.run(prompt)
            # Try to parse structured JSON response
            parsed = _parse_json_object(raw_diagnosis)
            if parsed and "summary" in parsed:
                candidate.diagnosis = str(parsed.get("summary", ""))[:1000]
                # Store structured data in history
                if state.history:
                    state.history[-1]["diagnosis"] = candidate.diagnosis
                    state.history[-1]["analysis_structured"] = {
                        "strengths": parsed.get("strengths", []),
                        "weaknesses": parsed.get("weaknesses", []),
                        "suggestions": parsed.get("suggestions", []),
                        "verdict": parsed.get("verdict", ""),
                    }
            else:
                candidate.diagnosis = raw_diagnosis.strip()[:1000]
                if state.history:
                    state.history[-1]["diagnosis"] = candidate.diagnosis
        except Exception as e:
            logger.warning("Analysis agent failed: %s", e)
            candidate.diagnosis = f"Analysis failed: {e}"
    elif not candidate.diagnosis:
        candidate.diagnosis = "No backtest results to analyze"

    logger.info("Phase ANALYSIS complete: %s", (candidate.diagnosis or "")[:200])

    # Check if this iteration produced any candidates with backtest results.
    # If not (pure infra failure), don't waste the iteration counter.
    iter_candidates = [c for c in state.candidates if c.iteration == state.iteration]
    has_results = any(c.backtest_summary is not None for c in iter_candidates)

    if not has_results and iter_candidates:
        gen_retries = getattr(state, "_gen_retries", 0)
        if gen_retries < 2:
            state._gen_retries = gen_retries + 1
            logger.info(
                "No candidates produced results in iteration %d — retrying (%d/2) without incrementing iteration",
                state.iteration, gen_retries + 1,
            )
            state.phase = Phase.STRATEGY_GEN
            return
        else:
            logger.warning(
                "No candidates produced results after %d retries — moving on",
                gen_retries,
            )
            state._gen_retries = 0

    # Reset retry counter on successful iteration
    if hasattr(state, "_gen_retries"):
        state._gen_retries = 0

    # Decide next phase
    if state.iteration + 1 >= config.max_iterations:
        state.phase = Phase.COMPLETE
    else:
        state.iteration += 1
        state.phase = Phase.STRATEGY_GEN


