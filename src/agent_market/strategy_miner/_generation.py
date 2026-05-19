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
    prompt_metadata,
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
    _load_freqtrade_payload,
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
    safe_transition,
    update_candidate_stage,
)
from ._rendering import _normalize_model_candidate_payload
from ._scheduler import BanditScheduler, resolve_family_weights

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


def _accumulate_tokens(state: MinerState, result: Any) -> None:
    """D13: Accumulate token usage from AgentRunResult into state.economics."""
    usage = getattr(result, "usage", None)
    if not usage or not isinstance(usage, dict):
        # Fallback: try extracting from raw response (OpenAI format)
        raw = getattr(result, "raw", None)
        if isinstance(raw, dict):
            usage = raw.get("usage")
    if not usage or not isinstance(usage, dict):
        return
    total = int(usage.get("total_tokens", 0) or 0)
    if not hasattr(state, "economics"):
        return
    if total > 0:
        state.economics["total_tokens"] = (
            state.economics.get("total_tokens", 0) + total
        )

    direct_cost = usage.get("cost_usd")
    try:
        direct_cost_value = float(direct_cost) if direct_cost is not None else 0.0
    except Exception:
        direct_cost_value = 0.0

    if direct_cost_value > 0:
        state.economics["total_cost_usd"] = round(
            state.economics.get("total_cost_usd", 0.0) + direct_cost_value,
            6,
        )
        return

    if total > 0:
        # Estimate cost: $0.01 per 1K tokens (conservative average)
        # Real cost depends on model; this provides a rough budget signal.
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        # Use differentiated pricing if available, else flat rate
        est_cost = (prompt_tokens * 0.003 + completion_tokens * 0.012) / 1000.0
        if est_cost <= 0:
            est_cost = total * 0.01 / 1000.0
        state.economics["total_cost_usd"] = round(
            state.economics.get("total_cost_usd", 0.0) + est_cost, 6
        )


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
    payload = _load_freqtrade_payload(freqtrade_config_path)
    if not payload:
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
    reserved_name: str | None = None,
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
        if reserved_name:
            names_seen.discard(reserved_name)
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
    global_kb: Optional["KnowledgeBase"] = None,
    research_insights: Optional[str] = None,
    strategy_blueprints: Optional[str] = None,
) -> None:
    """Generate strategies via the LLM agent (multi-candidate, parallel)."""

    # Reset candidate queue for this iteration.
    state.pending_candidate_idxs = []
    state.active_candidate_idx = None

    n = max(1, int(getattr(config, "candidates_per_iteration", 1) or 1))

    # --- Adaptive family allocation via Thompson Sampling ---
    _bandit: BanditScheduler | None = getattr(state, "_bandit", None)
    if _bandit is None:
        # Try restore from checkpoint
        if state.bandit_state:
            _bandit = BanditScheduler.from_dict(state.bandit_state)
        else:
            configured_families = list(getattr(config, "search_families", None) or [])
            all_families = configured_families or [
                "rule/mean-reversion",
                "rule/trend-pullback",
                "rule/trend-following",
                "rule/breakout",
                "rule/basket-rotation",
                "rule/pair-relative",
                "rule/dca-grid",
                "rule/martingale",
                "ml/lightgbm",
                "ml/xgboost",
                "dl/pytorch_mlp",
                "rl/ppo",
            ]
            # Filter by enabled types
            from ._helpers import _configured_candidate_types
            enabled_types = set(_configured_candidate_types(config))
            families = [f for f in all_families if f.split("/")[0] in enabled_types]
            _bandit = BanditScheduler(families or ["rule/mean-reversion", "ml/lightgbm"])
        state._bandit = _bandit  # type: ignore[attr-defined]

    active_family_weights = resolve_family_weights(
        list(getattr(config, "family_weight_schedule", None) or []),
        iteration=state.iteration,
    )
    selected_families = _bandit.select_families(n, family_weights=active_family_weights)
    # Persist bandit state for checkpoint
    state.bandit_state = _bandit.to_dict()
    logger.info(
        "Bandit selected families: %s%s",
        selected_families,
        f" weights={active_family_weights}" if active_family_weights else "",
    )

    best_code = state.best_candidate.code if state.best_candidate is not None else None

    # Inject knowledge base context
    elite_summaries = None
    failure_summary = None
    kb_entries: list[tuple[str, "KnowledgeBase"]] = []
    seen_kb_paths: set[str] = set()
    if kb is not None:
        kb_path = str(getattr(kb, "_path", ""))
        kb_entries.append((kb_path or "local", kb))
        if kb_path:
            seen_kb_paths.add(kb_path)
    if global_kb is not None:
        gkb_path = str(getattr(global_kb, "_path", ""))
        if not gkb_path or gkb_path not in seen_kb_paths:
            kb_entries.append((gkb_path or "global", global_kb))

    if kb_entries:
        merged_elites: dict[str, dict[str, Any]] = {}
        failure_lines: list[str] = []
        seen_failure_lines: set[str] = set()
        for _, entry in kb_entries:
            for elite in list(entry.elites or []):
                name = str(elite.get("name") or "").strip()
                if not name:
                    continue
                existing = merged_elites.get(name)
                elite_reward = float(elite.get("reward", 0) or 0.0)
                existing_reward = float((existing or {}).get("reward", 0) or 0.0)
                if existing is None or elite_reward > existing_reward:
                    merged_elites[name] = dict(elite)
            summary = entry.failure_summary(5)
            if summary and summary != "No recorded failures.":
                for line in [item.strip() for item in summary.splitlines() if item.strip()]:
                    if line not in seen_failure_lines:
                        failure_lines.append(line)
                        seen_failure_lines.add(line)
        if merged_elites:
            elite_summaries = sorted(
                merged_elites.values(),
                key=lambda item: float(item.get("reward", 0) or 0.0),
                reverse=True,
            )[:3]
        if failure_lines:
            failure_summary = "\n".join(failure_lines)

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
    market_context = _freqtrade_market_context(config.freqtrade_config)
    market_pairs = market_context.pairs
    factor_store_entries: list[tuple[str, Any]] = []
    factor_store_lock = threading.Lock()
    factor_memory_path = str(getattr(config, "factor_memory_path", "") or "").strip()
    global_factor_memory_path = str(getattr(config, "global_factor_memory_path", "") or "").strip()
    if not global_factor_memory_path and bool(getattr(config, "use_global_memory", True)):
        global_factor_memory_path = str(paths.global_factor_memory_path())
    seen_factor_paths: set[str] = set()

    def _load_factor_store(raw_path: str) -> None:
        raw = str(raw_path or "").strip()
        if not raw:
            return
        try:
            from agent_market.factor_memory import FactorMemoryStore  # noqa: WPS433

            resolved = paths.resolve_repo_path(raw)
            resolved_key = str(resolved)
            if resolved_key in seen_factor_paths:
                return
            if resolved.exists():
                factor_store_entries.append((resolved_key, FactorMemoryStore(resolved)))
                seen_factor_paths.add(resolved_key)
                logger.info("Loaded factor memory for strategy generation: %s", resolved)
            else:
                logger.debug("Factor memory path does not exist yet: %s", resolved)
        except Exception:
            logger.warning("Failed to load factor memory for strategy generation", exc_info=True)

    _load_factor_store(factor_memory_path)
    _load_factor_store(global_factor_memory_path)

    def _factor_retrieval_for_candidate(
        *,
        selected_family: str | None,
        candidate_type: str,
    ) -> tuple[str, dict[str, Any] | None]:
        if not factor_store_entries:
            return "", None
        from agent_market.factor_memory import format_factor_retrieval_context  # noqa: WPS433

        family = str(selected_family or candidate_type or "").strip()
        timeframe = str(objective_profile.get("base_timeframe") or "")
        universe = list(getattr(config, "model_training_pairs", None) or []) or list(market_pairs or [])
        try:
            factor_cards: list[dict[str, Any]] = []
            failure_cards: list[dict[str, Any]] = []
            seen_card_ids: set[str] = set()
            seen_failure_ids: set[str] = set()
            with factor_store_lock:
                for memory_path, store in factor_store_entries:
                    result = store.retrieve_for_strategy(
                        family=family,
                        timeframe=timeframe,
                        universe=universe,
                        top_n=max(1, int(getattr(config, "factor_retrieval_top_n", 3) or 3)),
                    )
                    for card in list(result.factor_cards or []):
                        card_id = str(card.get("card_id") or "").strip()
                        dedupe_key = card_id or json.dumps(card, ensure_ascii=False, sort_keys=True)
                        if dedupe_key in seen_card_ids:
                            continue
                        seen_card_ids.add(dedupe_key)
                        factor_cards.append(dict(card))
                    for card in list(result.failure_cards or []):
                        failure_id = str(card.get("failure_id") or "").strip()
                        dedupe_key = failure_id or json.dumps(card, ensure_ascii=False, sort_keys=True)
                        if dedupe_key in seen_failure_ids:
                            continue
                        seen_failure_ids.add(dedupe_key)
                        failure_cards.append(dict(card))
        except Exception:
            logger.warning("Factor retrieval failed for family=%s", family, exc_info=True)
            return "", None
        context = format_factor_retrieval_context(
            factor_cards=factor_cards,
            failure_cards=failure_cards,
        )
        snapshot = {
            "memory_paths": [item[0] for item in factor_store_entries],
            "query": {
                "family": family,
                "timeframe": timeframe,
                "universe": universe,
                "top_n": max(1, int(getattr(config, "factor_retrieval_top_n", 3) or 3)),
            },
            "factor_cards": factor_cards,
            "failure_cards": failure_cards,
        }
        return context, snapshot

    def _strategy_retrieval_for_candidate(
        *,
        selected_family: str | None,
        candidate_type: str,
    ) -> tuple[str, dict[str, Any] | None]:
        if not kb_entries:
            return "", None
        from .knowledge_base import format_strategy_retrieval_context  # noqa: WPS433

        family = str(selected_family or candidate_type or "").strip()
        timeframe = str(objective_profile.get("base_timeframe") or "")
        universe = list(getattr(config, "model_training_pairs", None) or []) or list(market_pairs or [])
        strategy_cards: list[dict[str, Any]] = []
        failure_cards: list[dict[str, Any]] = []
        seen_strategy_ids: set[str] = set()
        seen_failure_ids: set[str] = set()
        try:
            for kb_path, entry in kb_entries:
                result = entry.retrieve_for_generation(
                    family=family,
                    timeframe=timeframe,
                    universe=universe,
                    top_n=max(1, int(getattr(config, "strategy_retrieval_top_n", 3) or 3)),
                    recent_n=max(0, int(getattr(config, "strategy_retrieval_recent_n", 0) or 0)),
                )
                for card in list(result.strategy_cards or []):
                    card_id = str(card.get("card_id") or "").strip()
                    dedupe_key = card_id or f"{card.get('run_id','')}::{card.get('name','')}"
                    if dedupe_key in seen_strategy_ids:
                        continue
                    seen_strategy_ids.add(dedupe_key)
                    strategy_cards.append(dict(card))
                for card in list(result.failure_cards or []):
                    failure_id = str(card.get("failure_id") or "").strip()
                    dedupe_key = failure_id or f"{card.get('run_id','')}::{card.get('name','')}::{card.get('category','')}"
                    if dedupe_key in seen_failure_ids:
                        continue
                    seen_failure_ids.add(dedupe_key)
                    failure_cards.append(dict(card))
        except Exception:
            logger.warning("Strategy retrieval failed for family=%s", family, exc_info=True)
            return "", None
        context = format_strategy_retrieval_context(
            strategy_cards=strategy_cards,
            failure_cards=failure_cards,
        )
        snapshot = {
            "knowledge_base_paths": [item[0] for item in kb_entries],
            "query": {
                "family": family,
                "timeframe": timeframe,
                "universe": universe,
                "top_n": max(1, int(getattr(config, "strategy_retrieval_top_n", 3) or 3)),
                "recent_n": max(0, int(getattr(config, "strategy_retrieval_recent_n", 0) or 0)),
            },
            "strategy_cards": strategy_cards,
            "failure_cards": failure_cards,
        }
        return context, snapshot

    def _register_factor_references(cand: StrategyCandidate, factor_snapshot: dict[str, Any] | None) -> None:
        if not factor_snapshot:
            return
        card_ids = [item.get("card_id") for item in factor_snapshot.get("factor_cards", []) if item.get("card_id")]
        if not card_ids:
            return
        try:
            with factor_store_lock:
                for _, store in factor_store_entries:
                    store.register_strategy_references(
                        card_ids=card_ids,
                        strategy_name=cand.name,
                        strategy_run_id=state.run_id,
                        candidate_family=str(cand.candidate_family or cand.candidate_type or ""),
                    )
        except Exception:
            logger.debug("Failed to register factor references for candidate", exc_info=True)

    def _register_strategy_memory_references(cand: StrategyCandidate, strategy_snapshot: dict[str, Any] | None) -> None:
        if not strategy_snapshot:
            return
        card_ids = [item.get("card_id") for item in strategy_snapshot.get("strategy_cards", []) if item.get("card_id")]
        if not card_ids:
            return
        try:
            for _, entry in kb_entries:
                entry.register_strategy_references(
                    card_ids=card_ids,
                    strategy_name=cand.name,
                    strategy_run_id=state.run_id,
                    candidate_family=str(cand.candidate_family or cand.candidate_type or ""),
                )
        except Exception:
            logger.debug("Failed to register strategy references for candidate", exc_info=True)

    def _reserve_candidate_name(name_hint: str, candidate_idx: int, fallback: str) -> str:
        name = _sanitize_candidate_name(name_hint, fallback=fallback)
        with names_seen_lock:
            if name in names_seen:
                name = f"{name}_cand{candidate_idx}"
            names_seen.add(name)
        return name

    def _gen_model_candidate(candidate_idx: int, candidate_type: str, selected_family: str | None = None) -> Optional[StrategyCandidate]:
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

        pairs = list(getattr(config, "model_training_pairs", None) or []) or market_context.pairs
        factor_context, factor_snapshot = _factor_retrieval_for_candidate(
            selected_family=selected_family,
            candidate_type=candidate_type,
        )
        strategy_memory_context, strategy_snapshot = _strategy_retrieval_for_candidate(
            selected_family=selected_family,
            candidate_type=candidate_type,
        )

        planner_notes = ""
        trace_paths: dict[str, str] = {}
        generation_provider = ""
        generation_model = None

        def _write_trace(role: str, payload: dict[str, Any],
                         prompt_meta: Optional[dict] = None) -> None:
            try:
                from .artifacts import write_agent_trace

                p = write_agent_trace(
                    run_dir,
                    iteration=state.iteration,
                    candidate_idx=candidate_idx,
                    role=role,
                    payload=payload,
                    prompt_meta=prompt_meta,
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
                factor_context=factor_context or None,
                strategy_memory_context=strategy_memory_context or None,
                target_trades=int(objective_profile.get("target_trades") or 20),
                min_acceptable_trades=int(objective_profile.get("min_acceptable_trades") or 10),
            )
            planner_agent = build_strategy_agent(config, sandbox)
            try:
                planner_res = planner_agent.run_result(planner_prompt)
                _accumulate_tokens(state, planner_res)
                planner_notes = (planner_res.assistant_text or "").strip()[:3000]
                _write_trace(
                    "planner",
                    {
                        "provider": planner_res.provider,
                        "model": planner_res.model,
                        "assistant_text": planner_res.assistant_text,
                        "tool_trace": planner_res.tool_trace,
                    },
                    prompt_meta=prompt_metadata("model_planner", planner_prompt),
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
            factor_context=factor_context or None,
            strategy_memory_context=strategy_memory_context or None,
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
            _accumulate_tokens(state, res)
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
                prompt_meta=prompt_metadata("model_candidate", prompt),
            )
            payload_raw = _json_block_or_none(res.assistant_text or "")
        finally:
            try:
                local_agent.close()
            except Exception:
                pass

        if not payload_raw:
            logger.warning("Candidate %d returned invalid model spec JSON", candidate_idx)
            _write_trace(
                "failure",
                {
                    "failure_category": "invalid_json",
                    "reason": "model_candidate_missing_json_payload",
                    "candidate_idx": candidate_idx,
                },
            )
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
        if selected_family:
            cand.candidate_family = selected_family  # e.g. "ml/lightgbm"
        cand.candidate_payload = payload
        if factor_snapshot:
            cand.candidate_payload["factor_retrieval"] = factor_snapshot
        if strategy_snapshot:
            cand.candidate_payload["strategy_retrieval"] = strategy_snapshot
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
        _register_factor_references(cand, factor_snapshot)
        _register_strategy_memory_references(cand, strategy_snapshot)

        return cand

    def _gen_one(candidate_idx: int) -> Optional[StrategyCandidate]:
        cand_label = f"cand_{candidate_idx:02d}"
        # Use bandit-selected family when available, fall back to fixed rotation
        selected_family: str | None = None
        if candidate_idx < len(selected_families):
            selected_family = selected_families[candidate_idx]
            candidate_type = selected_family.split("/")[0]
        else:
            candidate_type = _candidate_type_for_slot(
                config,
                iteration=state.iteration,
                candidate_idx=candidate_idx,
                candidates_per_iteration=n,
            )

        if candidate_type != "rule":
            return _gen_model_candidate(candidate_idx, candidate_type, selected_family=selected_family)

        if not bool(getattr(config, "multiagent_enabled", False)):
            variant = None if n == 1 else cand_label
            sandbox = prepare_sandbox(
                config,
                run_dir,
                state.iteration,
                variant=variant,
            )
            factor_context, factor_snapshot = _factor_retrieval_for_candidate(
                selected_family=selected_family,
                candidate_type="rule",
            )
            strategy_memory_context, strategy_snapshot = _strategy_retrieval_for_candidate(
                selected_family=selected_family,
                candidate_type="rule",
            )

            # Diversity: use bandit-selected family when available, else slot rotation
            if selected_family and "/" in selected_family:
                _family_archetype = selected_family.split("/", 1)[1]  # e.g. "mean-reversion"
                _div = {"role": _family_archetype, "instructions": f"Build a {_family_archetype} strategy."}
            else:
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
                factor_context=factor_context or None,
                strategy_memory_context=strategy_memory_context or None,
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
                if selected_family:
                    cand.candidate_family = selected_family  # e.g. "rule/mean-reversion"
                if gen_provider:
                    cand.source_provider = gen_provider
                    cand.generation_provider = gen_provider
                if gen_model is not None:
                    cand.source_model = gen_model
                    cand.generation_model = gen_model
                cand.candidate_payload = {
                    "selected_family": selected_family or "",
                    "factor_retrieval": factor_snapshot or {},
                    "strategy_retrieval": strategy_snapshot or {},
                }

                try:
                    from .artifacts import write_candidate_snapshot

                    write_candidate_snapshot(run_dir, cand)
                except Exception:
                    logger.debug("Candidate snapshot write failed", exc_info=True)
                _register_factor_references(cand, factor_snapshot)
                _register_strategy_memory_references(cand, strategy_snapshot)

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
        factor_context, factor_snapshot = _factor_retrieval_for_candidate(
            selected_family=selected_family,
            candidate_type="rule",
        )
        strategy_memory_context, strategy_snapshot = _strategy_retrieval_for_candidate(
            selected_family=selected_family,
            candidate_type="rule",
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
            factor_context=factor_context or None,
            strategy_memory_context=strategy_memory_context or None,
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
            factor_context=factor_context or None,
            strategy_memory_context=strategy_memory_context or None,
            **objective_profile,
        )

        trace_paths: dict[str, str] = {}
        planner_notes = ""
        reviewer_notes = ""
        backtester_notes = ""

        generation_provider = ""
        generation_model = None

        def _write_trace(role: str, payload: dict[str, Any],
                         prompt_meta: Optional[dict] = None) -> None:
            try:
                from .artifacts import write_agent_trace

                p = write_agent_trace(
                    run_dir,
                    iteration=state.iteration,
                    candidate_idx=candidate_idx,
                    role=role,
                    payload=payload,
                    prompt_meta=prompt_meta,
                )
                trace_paths[role] = str(p)
            except Exception:
                logger.debug("Agent trace write failed (%s)", role, exc_info=True)

        def _planner_job() -> None:
            nonlocal planner_notes
            local_agent = build_strategy_agent(config, sandbox_planner)
            try:
                res = local_agent.run_result(planner_prompt)
                _accumulate_tokens(state, res)
                planner_notes = (res.assistant_text or "").strip()[:4000]
                _write_trace(
                    "planner",
                    {
                        "provider": res.provider,
                        "model": res.model,
                        "assistant_text": res.assistant_text,
                        "tool_trace": res.tool_trace,
                    },
                    prompt_meta=prompt_metadata("planner", planner_prompt),
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
                    prompt_meta=prompt_metadata("strategy_gen", prompt),
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
                _write_trace(
                    "failure",
                    {
                        "failure_category": "illegal_code",
                        "reason": "no_strategy_file_produced",
                        "candidate_idx": candidate_idx,
                    },
                )
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
                _accumulate_tokens(state, res)
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
                    prompt_meta=prompt_metadata("reviewer", reviewer_prompt),
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
                _accumulate_tokens(state, res)
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
                    prompt_meta=prompt_metadata("backtester", backtester_prompt),
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
                    reserved_name=name,
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
        cand.candidate_payload = {
            "selected_family": selected_family or "",
            "factor_retrieval": factor_snapshot or {},
            "strategy_retrieval": strategy_snapshot or {},
        }
        if selected_family:
            cand.candidate_family = selected_family  # e.g. "rule/mean-reversion"

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
        _register_factor_references(cand, factor_snapshot)
        _register_strategy_memory_references(cand, strategy_snapshot)

        return cand

    results: list[tuple[int, StrategyCandidate]] = []
    max_workers = min(8, n)
    max_parallel = int(getattr(config, "max_parallel_candidates", 0) or 0)
    if max_parallel > 0:
        max_workers = max(1, min(max_workers, max_parallel))
    roles_workers = max(1, min(2, int(getattr(config, "max_parallel_roles", 1) or 1)))

    logger.info(
        "Phase STRATEGY_GEN: iteration=%d generating %d candidates (candidate_workers=%d role_workers=%d)",
        state.iteration,
        n,
        max_workers,
        roles_workers,
    )

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
        safe_transition(state, Phase.ANALYSIS)
        return

    start_idx = len(state.candidates)
    state.candidates.extend([c for _, c in results])

    new_idxs = list(range(start_idx, len(state.candidates)))
    state.pending_candidate_idxs = new_idxs
    state.active_candidate_idx = new_idxs[0]

    first_candidate = _pick_active_candidate(state)
    safe_transition(state, _phase_for_candidate(first_candidate))
    # D2: Mark generated candidates
    for c in state.candidates[len(state.candidates) - len(new_idxs):]:
        update_candidate_stage(c, "generated")
    logger.info("Generated %d candidates for iteration %d", len(new_idxs), state.iteration)
