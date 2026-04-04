"""Strategy generation phase – extracted from phases.py."""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase

from agent_market import paths

from .agent_adapter import StrategyAgent
from .agent_factory import build_strategy_agent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from .prompts import (
    build_backtester_prompt,
    build_model_candidate_prompt,
    build_model_planner_prompt,
    build_planner_prompt,
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
from ._helpers import (
    _freqtrade_config_defaults,
    _freqtrade_market_context,
    _prompt_objective_profile,
    _sanitize_candidate_name,
    _candidate_type_for_slot,
    _allowed_model_families,
    _normalize_candidate_type,
    _json_block_or_none,
    _parse_json_object,
    _phase_for_candidate,
    _pick_active_candidate,
    _rewrite_strategy_class_name,
)
from ._rendering import _normalize_model_candidate_payload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDICATOR_SETS = [
    {"role": "exploit", "instructions": "Improve upon the best strategy. You MAY reuse its indicators."},
    {"role": "explore_alt", "instructions": (
        "Create a strategy using DIFFERENT indicators than the current best. "
        "You MUST NOT use EMA crossovers or Keltner Channel as primary signals. "
        "Try: MACD, Stochastic RSI, CCI, Williams %R, OBV, VWAP, or Ichimoku."
    )},
    {"role": "explore_novel", "instructions": (
        "Create a fundamentally different strategy type. "
        "Try one of: volume-profile breakout, multi-timeframe confirmation, "
        "volatility regime switching (ADX-based), or pure momentum (ROC + acceleration). "
        "Do NOT use EMA or RSI as primary entry signals."
    )},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_indicator_names(code: str) -> List[str]:
    """Extract indicator names from strategy code for research queries."""
    indicators = set()
    code_lower = code.lower()
    # Match ta.xxx(...) calls
    for m in re.finditer(r'\bta\.(\w+)\s*\(', code):
        indicators.add(m.group(1).upper())
    # Match common indicator keywords
    for kw in ["ema", "sma", "rsi", "macd", "bbands", "atr", "adx", "cci",
               "stoch", "keltner", "donchian", "bollinger", "vwap", "obv"]:
        if kw in code_lower:
            indicators.add(kw.upper())
    return sorted(indicators)[:10]


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


# ---------------------------------------------------------------------------
# Main phase handler
# ---------------------------------------------------------------------------


def phase_strategy_gen(
    state: MinerState,
    config: MinerConfig,
    run_dir: Path,
    agent: Optional[StrategyAgent] = None,
    kb: Optional["KnowledgeBase"] = None,
    research_insights: Optional[str] = None,
    strategy_blueprints: Optional[str] = None,
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

    # --- Web research (only first 5 iterations to save tokens) ---
    _market_context_str = ""
    if state.iteration < 5:
        try:
            from .research import research_market_context, format_market_context
            _market_ctx = research_market_context()
            _market_context_str = format_market_context(_market_ctx)
            if _market_context_str:
                logger.info("Market research injected (%d chars)", len(_market_context_str))
        except Exception as e:
            logger.debug("Market research skipped: %s", e)

    # --- Previous analysis suggestions (P1-1: feedback loop) ---
    _prev_suggestions: Optional[List[str]] = None
    if state.history:
        last = state.history[-1]
        structured = last.get("analysis_structured") or {}
        suggs = structured.get("suggestions") or []
        if suggs:
            _prev_suggestions = suggs[:5]

    names_seen: set[str] = set()
    names_seen_lock = threading.Lock()
    objective_profile = _prompt_objective_profile(config)

    def _reserve_candidate_name(name_hint: str, candidate_idx: int, fallback: str) -> str:
        name = _sanitize_candidate_name(name_hint, fallback=fallback)
        with names_seen_lock:
            if name in names_seen:
                name = f"{name}_cand{candidate_idx}"
            names_seen.add(name)
        return name

    def _gen_model_candidate(candidate_idx: int, candidate_type: str) -> Optional[StrategyCandidate]:
        cand_label = f"cand_{candidate_idx:02d}"
        sandbox = prepare_sandbox(
            config,
            run_dir,
            state.iteration,
            variant=f"{cand_label}/model",
        )

        allowed_families = _allowed_model_families(candidate_type)
        if not allowed_families:
            logger.warning("Candidate %d requested unsupported candidate_type=%s", candidate_idx, candidate_type)
            return None

        _, pairs, _, _ = _freqtrade_market_context(config.freqtrade_config)
        pairs = list(getattr(config, "model_training_pairs", None) or []) or pairs

        planner_notes = ""
        trace_paths: dict[str, str] = {}
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

        if bool(getattr(config, "multiagent_enabled", False)):
            planner_prompt = build_model_planner_prompt(
                iteration=state.iteration,
                candidate_idx=candidate_idx,
                candidates_per_iteration=n,
                candidate_type=candidate_type,
                allowed_model_families=allowed_families,
                freqtrade_config=config.freqtrade_config,
                timerange=config.timerange,
                feature_file=str(getattr(config, "model_feature_file", "user_data/freqai_features_real.json")),
                expressions_file=getattr(config, "model_expressions_file", None),
                pairs=pairs,
                history=state.history,
                target_trades=int(objective_profile.get("target_trades") or 20),
                min_acceptable_trades=int(objective_profile.get("min_acceptable_trades") or 10),
            )
            planner_agent = build_strategy_agent(config, sandbox)
            try:
                planner_res = planner_agent.run_result(planner_prompt)
                planner_notes = (planner_res.assistant_text or "").strip()[:3000]
                _write_trace(
                    "planner",
                    {
                        "provider": planner_res.provider,
                        "model": planner_res.model,
                        "assistant_text": planner_res.assistant_text,
                        "tool_trace": planner_res.tool_trace,
                    },
                )
            finally:
                try:
                    planner_agent.close()
                except Exception:
                    pass

        prompt = build_model_candidate_prompt(
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
            candidate_type=candidate_type,
            allowed_model_families=allowed_families,
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            feature_file=str(getattr(config, "model_feature_file", "user_data/freqai_features_real.json")),
            expressions_file=getattr(config, "model_expressions_file", None),
            pairs=pairs,
            history=state.history,
            planner_notes=planner_notes or None,
            previous_suggestions=_prev_suggestions,
            base_timeframe=str(objective_profile.get("base_timeframe") or "1h"),
            max_strategy_timeframe=str(objective_profile.get("max_strategy_timeframe") or objective_profile.get("base_timeframe") or "1h"),
            target_trades=int(objective_profile.get("target_trades") or 20),
            min_acceptable_trades=int(objective_profile.get("min_acceptable_trades") or 10),
            roi_target_min_pct=float(objective_profile.get("roi_target_min_pct") or 5.0),
            roi_target_max_pct=float(objective_profile.get("roi_target_max_pct") or 10.0),
            stoploss_min_pct=float(objective_profile.get("stoploss_min_pct") or 5.0),
            stoploss_max_pct=float(objective_profile.get("stoploss_max_pct") or 10.0),
            training_validation_ratio=float(getattr(config, "training_validation_ratio", 0.2) or 0.2),
            training_rolling_splits=int(getattr(config, "training_rolling_splits", 3) or 3),
            training_scaler=str(getattr(config, "training_scaler", "robust") or "robust"),
            rl_total_timesteps=int(getattr(config, "rl_total_timesteps", 5000) or 5000),
        )

        local_agent = build_strategy_agent(config, sandbox)
        try:
            res = local_agent.run_result(prompt)
            generation_provider = str(getattr(res, "provider", "") or "")
            generation_model = getattr(res, "model", None)
            _write_trace(
                "coder",
                {
                    "provider": generation_provider,
                    "model": generation_model,
                    "assistant_text": res.assistant_text,
                    "tool_trace": res.tool_trace,
                },
            )
            payload_raw = _json_block_or_none(res.assistant_text or "")
        finally:
            try:
                local_agent.close()
            except Exception:
                pass

        if not payload_raw:
            logger.warning("Candidate %d returned invalid model spec JSON", candidate_idx)
            return None

        payload = _normalize_model_candidate_payload(
            raw_spec=payload_raw,
            candidate_type=candidate_type,
            config=config,
            run_dir=run_dir,
            iteration=state.iteration,
            candidate_idx=candidate_idx,
        )
        reserved_name = _reserve_candidate_name(
            payload.get("name_hint") or "",
            candidate_idx,
            fallback=f"{candidate_type.title()}Candidate",
        )
        payload["name_hint"] = reserved_name

        spec_dir = sandbox / "user_data" / "model_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"{reserved_name}.json"
        spec_text = json.dumps(payload, ensure_ascii=False, indent=2)
        spec_path.write_text(spec_text, encoding="utf-8")

        cand = StrategyCandidate(
            name=reserved_name,
            code=spec_text,
            strategy_path=spec_path,
            iteration=state.iteration,
        )
        cand.candidate_slot = int(candidate_idx)
        cand.candidate_type = candidate_type
        cand.model_family = str(payload.get("model_family") or "")
        cand.candidate_payload = payload
        cand.training_config = payload.get("training_config")
        cand.planner_notes = planner_notes
        cand.agent_traces = dict(trace_paths)
        cand.generation_provider = generation_provider
        cand.generation_model = generation_model
        cand.source_provider = generation_provider
        cand.source_model = generation_model

        try:
            from .artifacts import write_candidate_snapshot

            write_candidate_snapshot(run_dir, cand)
        except Exception:
            logger.debug("Candidate snapshot write failed", exc_info=True)

        return cand

    def _gen_one(candidate_idx: int) -> Optional[StrategyCandidate]:
        cand_label = f"cand_{candidate_idx:02d}"
        candidate_type = _candidate_type_for_slot(
            config,
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
        )

        if candidate_type != "rule":
            return _gen_model_candidate(candidate_idx, candidate_type)

        if not bool(getattr(config, "multiagent_enabled", False)):
            variant = None if n == 1 else cand_label
            sandbox = prepare_sandbox(
                config,
                run_dir,
                state.iteration,
                variant=variant,
            )

            # Diversity: assign role based on candidate index
            _div = _INDICATOR_SETS[candidate_idx % len(_INDICATOR_SETS)]
            prompt = build_strategy_gen_prompt(
                iteration=state.iteration,
                candidate_idx=candidate_idx,
                candidates_per_iteration=n,
                sandbox_path=str(sandbox),
                freqtrade_config=config.freqtrade_config,
                timerange=config.timerange,
                history=state.history,
                best_score=state.best_score,
                best_strategy_code=best_code if _div["role"] == "exploit" else None,
                elite_summaries=elite_summaries,
                failure_summary=failure_summary,
                provider=config.provider,
                market_profile=_build_market_profile(config.freqtrade_config),
                market_context=_market_context_str,
                research_insights=research_insights,
                strategy_blueprints=strategy_blueprints,
                diversity_instructions=_div["instructions"],
                previous_suggestions=_prev_suggestions,
                **objective_profile,
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
                cand.candidate_type = "rule"
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

        _div = _INDICATOR_SETS[candidate_idx % len(_INDICATOR_SETS)]
        coder_prompt = build_strategy_gen_prompt(
            iteration=state.iteration,
            candidate_idx=candidate_idx,
            candidates_per_iteration=n,
            sandbox_path=str(sandbox_coder),
            freqtrade_config=config.freqtrade_config,
            timerange=config.timerange,
            history=state.history,
            best_score=state.best_score,
            best_strategy_code=best_code if _div["role"] == "exploit" else None,
            elite_summaries=elite_summaries,
            failure_summary=failure_summary,
            provider=config.provider,
            market_profile=_build_market_profile(config.freqtrade_config),
            market_context=_market_context_str,
            research_insights=research_insights,
            strategy_blueprints=strategy_blueprints,
            diversity_instructions=_div["instructions"],
            previous_suggestions=_prev_suggestions,
            **objective_profile,
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
            **objective_profile,
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
            **objective_profile,
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
        cand.candidate_type = "rule"

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

    first_candidate = _pick_active_candidate(state)
    state.phase = _phase_for_candidate(first_candidate)
    logger.info("Generated %d candidates for iteration %d", len(new_idxs), state.iteration)
