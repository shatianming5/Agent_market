"""Main runner loop for strategy mining with checkpoint support."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from agent_market import paths

from .agent_factory import build_strategy_agent
from .dtypes import MinerConfig, MinerState, Phase, StrategyCandidate
from ._helpers import safe_transition
from .knowledge_base import KnowledgeBase
from .phases import (
    phase_analysis,
    phase_backtest,
    phase_evaluation,
    phase_strategy_gen,
    phase_train_model,
)

logger = logging.getLogger(__name__)

_TIMEFRAME_RE = re.compile(r"^\s*timeframe\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


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


def _update_knowledge_base(
    kb: KnowledgeBase,
    state: MinerState,
    candidate: Optional[StrategyCandidate] = None,
    *,
    config: Optional[MinerConfig] = None,
) -> None:
    """Update knowledge base (best-effort) based on the just-evaluated candidate.

    Args:
        kb: Knowledge base instance.
        state: Current miner state.
        candidate: The specific candidate that was just evaluated. If None,
            falls back to active_candidate_idx then to state.candidates[-1].
    """

    if not state.candidates:
        return

    cand = candidate
    if cand is None:
        idx = getattr(state, "active_candidate_idx", None)
        if idx is not None and isinstance(idx, int) and 0 <= idx < len(state.candidates):
            cand = state.candidates[idx]
        else:
            cand = state.candidates[-1]

    timeframe = ""
    universe: list[str] = []
    if config is not None:
        try:
            from ._helpers import _freqtrade_market_context

            timeframe, market_pairs, _, _ = _freqtrade_market_context(config.freqtrade_config)
            universe = list(getattr(config, "model_training_pairs", None) or []) or list(market_pairs or [])
        except Exception:
            logger.debug("Failed to infer market context for strategy card", exc_info=True)

    if cand.reward is not None and cand.backtest_summary is not None:
        if getattr(cand, "constraints_ok", True):
            kb.add_elite(
                name=cand.name,
                code=cand.code,
                reward=cand.reward,  # stores Sharpe
                backtest_summary=cand.backtest_summary,
                iteration=state.iteration,
            )
            # D7: Write strategy card with provenance + metrics
            try:
                kb.add_strategy_card(
                    name=cand.name,
                    code=cand.code,
                    iteration=state.iteration,
                    candidate_type=getattr(cand, "candidate_type", "rule"),
                    candidate_family=getattr(cand, "candidate_family", ""),
                    metrics={
                        "sharpe": cand.reward,
                        "profit_pct": (cand.backtest_summary or {}).get("profit_total_pct"),
                        "trades": (cand.backtest_summary or {}).get("trades"),
                        "trace_grade": (cand.candidate_payload or {}).get("trace_grade", {}).get("overall_grade"),
                    },
                    parent_name=state.best_candidate.name if state.best_candidate and state.best_candidate.name != cand.name else "",
                    run_id=state.run_id,
                    timeframe=timeframe,
                    universe=universe,
                )
            except Exception:
                pass
        else:
            kb.add_failure(
                name=cand.name,
                iteration=state.iteration,
                failure_type="constraint_violation",
                detail=f"Constraint violations: {', '.join(cand.constraint_violations or [])}",
            )
            # D6/D7: Write failure card
            try:
                kb.add_failure_card(
                    name=cand.name,
                    iteration=state.iteration,
                    phase="evaluation",
                    category="constraint_violation",
                    detail=f"Constraint violations: {', '.join(cand.constraint_violations or [])}",
                    run_id=state.run_id,
                    candidate_family=getattr(cand, "candidate_family", ""),
                    candidate_type=getattr(cand, "candidate_type", "rule"),
                    timeframe=timeframe,
                    universe=universe,
                )
            except Exception:
                pass
    elif cand.diagnosis:
        failure_type = "validation" if not cand.validation_passed else "backtest"
        kb.add_failure(
            name=cand.name,
            iteration=state.iteration,
            failure_type=failure_type,
            detail=cand.diagnosis,
        )
        # D6/D7: Write failure card
        try:
            kb.add_failure_card(
                name=cand.name,
                iteration=state.iteration,
                phase=failure_type,
                category=getattr(cand, "failure_category", "") or failure_type,
                detail=cand.diagnosis,
                run_id=state.run_id,
                candidate_family=getattr(cand, "candidate_family", ""),
                candidate_type=getattr(cand, "candidate_type", "rule"),
                timeframe=timeframe,
                universe=universe,
            )
        except Exception:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_timeframe(*texts: Any) -> str:
    for raw in texts:
        text = str(raw or "")
        if not text:
            continue
        match = _TIMEFRAME_RE.search(text)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _infer_candidate_family(
    *,
    candidate_type: Any,
    model_family: Any,
    name: Any,
    existing: Any = "",
) -> str:
    existing_text = str(existing or "").strip()
    if existing_text:
        return existing_text

    ctype = str(candidate_type or "").strip().lower()
    model = str(model_family or "").strip().lower()
    blob = str(name or "").strip().lower()

    if ctype == "ml" or any(token in blob for token in ("lgbm", "lightgbm", "xgb", "xgboost")):
        model_name = model or ("xgboost" if ("xgb" in blob or "xgboost" in blob) else "lightgbm")
        return f"ml/{model_name}"
    if ctype == "dl" or any(token in blob for token in ("mlp", "torch", "deep")):
        return f"dl/{model or 'pytorch_mlp'}"
    if ctype == "rl" or any(token in blob for token in ("ppo", "policy", "reinforcement")):
        return f"rl/{model or 'ppo'}"
    if any(token in blob for token in ("breakout", "donchian", "channel", "squeeze")):
        return "rule/breakout"
    if any(token in blob for token in ("pullback", "retest", "dip")):
        return "rule/trend-pullback"
    if any(token in blob for token in ("meanrev", "mean_revert", "revert", "vwap", "boll", "keltner", "stoch", "cci", "wr")):
        return "rule/mean-reversion"
    if any(token in blob for token in ("basket", "rotation")):
        return "rule/basket-rotation"
    if ctype == "rule":
        return "rule/trend-following"
    return existing_text


def _run_strategy_meta(miner_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    summary = _read_json(miner_dir / "strategy_miner_summary.json")
    leaderboard = _read_json(miner_dir / "leaderboard.json")
    proposal = _read_json(miner_dir / "proposal.json")
    cfg = proposal.get("config") or {}
    universe = list(cfg.get("model_training_pairs") or []) or list(cfg.get("quick_backtest_pairs") or [])

    by_name: dict[str, dict[str, Any]] = {}
    for row in list(summary.get("candidates") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        by_name[name] = dict(row)

    for row in list(summary.get("history") or []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        current = dict(by_name.get(name) or {})
        for key in (
            "name",
            "candidate_type",
            "model_family",
            "candidate_family",
            "constraint_violations",
            "constraints_ok",
            "analysis_structured",
            "diagnosis",
        ):
            if key not in current and key in row:
                current[key] = row.get(key)
        if "reward" not in current and "sharpe" in row:
            current["reward"] = row.get("sharpe")
        by_name[name] = current

    leaderboard_rows = list(leaderboard.get("items") or [])
    best_row = leaderboard.get("best")
    if isinstance(best_row, dict):
        leaderboard_rows.append(best_row)
    for row in leaderboard_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        current = dict(by_name.get(name) or {})
        for key in (
            "name",
            "candidate_type",
            "model_family",
            "candidate_family",
            "training_summary",
            "constraints_ok",
            "constraint_violations",
        ):
            if key not in current and key in row:
                current[key] = row.get(key)
        if "reward" not in current and "sharpe" in row:
            current["reward"] = row.get("sharpe")
        training_data = ((row.get("training_summary") or {}).get("data") or {})
        if "inferred_timeframe" not in current and training_data.get("timeframe"):
            current["inferred_timeframe"] = training_data.get("timeframe")
        if not universe:
            pairs = training_data.get("pairs")
            if isinstance(pairs, list) and pairs:
                universe = list(pairs)
        by_name[name] = current

    return by_name, universe


def _strategy_memory_payload_from_run(miner_dir: Path, run_id: str) -> dict[str, Any]:
    kb = KnowledgeBase(miner_dir / "knowledge_base.json")
    by_name, universe = _run_strategy_meta(miner_dir)

    payload: dict[str, Any] = {
        "elites": kb.elites,
        "failures": kb.failures,
        "strategy_cards": [],
        "failure_cards": [],
        "edges": kb.edges,
    }
    existing_strategy_keys: set[tuple[str, int]] = set()
    existing_failure_keys: set[tuple[str, int, str]] = set()

    for raw_card in kb.strategy_cards:
        card = dict(raw_card)
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        meta = dict(by_name.get(name) or {})
        metrics = dict(card.get("metrics") or {})
        trace_grade = ((meta.get("trace_grade") or {}).get("overall_grade"))
        if trace_grade is None:
            trace_grade = (((meta.get("candidate_payload") or {}).get("trace_grade") or {}).get("overall_grade"))
        if trace_grade is not None and "trace_grade" not in metrics:
            metrics["trace_grade"] = trace_grade
        card["metrics"] = metrics
        if not card.get("candidate_type"):
            card["candidate_type"] = meta.get("candidate_type") or "rule"
        card["candidate_family"] = _infer_candidate_family(
            candidate_type=card.get("candidate_type") or meta.get("candidate_type"),
            model_family=meta.get("model_family"),
            name=name,
            existing=card.get("candidate_family") or meta.get("candidate_family"),
        )
        if not card.get("timeframe"):
            card["timeframe"] = (
                str(meta.get("inferred_timeframe") or "").strip()
                or _infer_timeframe(meta.get("code"), meta.get("reviewer_notes"), meta.get("planner_notes"))
            )
        if not card.get("universe"):
            card["universe"] = list(universe)
        payload["strategy_cards"].append(card)
        existing_strategy_keys.add((name, int(card.get("iteration") or 0)))

    for raw_card in kb.failure_cards:
        card = dict(raw_card)
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        meta = dict(by_name.get(name) or {})
        if not card.get("candidate_type"):
            card["candidate_type"] = meta.get("candidate_type") or "rule"
        card["candidate_family"] = _infer_candidate_family(
            candidate_type=card.get("candidate_type") or meta.get("candidate_type"),
            model_family=meta.get("model_family"),
            name=name,
            existing=card.get("candidate_family") or meta.get("candidate_family"),
        )
        if not card.get("timeframe"):
            card["timeframe"] = (
                str(meta.get("inferred_timeframe") or "").strip()
                or _infer_timeframe(meta.get("code"), meta.get("reviewer_notes"), meta.get("planner_notes"))
            )
        if not card.get("universe"):
            card["universe"] = list(universe)
        payload["failure_cards"].append(card)
        existing_failure_keys.add(
            (
                name,
                int(card.get("iteration") or 0),
                str(card.get("category") or card.get("subcategory") or ""),
            )
        )

    for elite in kb.elites:
        if not isinstance(elite, dict):
            continue
        name = str(elite.get("name") or "").strip()
        iteration = int(elite.get("iteration") or 0)
        if not name or (name, iteration) in existing_strategy_keys:
            continue
        meta = dict(by_name.get(name) or {})
        reward = elite.get("reward")
        card = {
            "run_id": run_id,
            "name": name,
            "iteration": iteration,
            "candidate_type": meta.get("candidate_type") or "rule",
            "candidate_family": _infer_candidate_family(
                candidate_type=meta.get("candidate_type") or "rule",
                model_family=meta.get("model_family"),
                name=name,
                existing=meta.get("candidate_family"),
            ),
            "timeframe": str(meta.get("inferred_timeframe") or "").strip()
            or _infer_timeframe(meta.get("code"), elite.get("code_snippet"), meta.get("reviewer_notes")),
            "universe": list(universe),
            "indicators": [],
            "metrics": {
                "sharpe": reward,
                "profit_pct": elite.get("profit_pct"),
                "trades": elite.get("trades"),
                "winrate": elite.get("winrate"),
                "max_drawdown_abs": elite.get("max_drawdown"),
                "trace_grade": ((meta.get("trace_grade") or {}).get("overall_grade"))
                or (((meta.get("candidate_payload") or {}).get("trace_grade") or {}).get("overall_grade")),
            },
            "parent_name": "",
            "mutation_description": "",
            "reuse_count": 0,
            "downstream_strategy_references": [],
        }
        payload["strategy_cards"].append(card)
        existing_strategy_keys.add((name, iteration))

    for failure in kb.failures:
        if not isinstance(failure, dict):
            continue
        name = str(failure.get("name") or "").strip()
        iteration = int(failure.get("iteration") or 0)
        category = str(failure.get("failure_type") or "")
        if not name or (name, iteration, category) in existing_failure_keys:
            continue
        meta = dict(by_name.get(name) or {})
        payload["failure_cards"].append(
            {
                "run_id": run_id,
                "name": name,
                "iteration": iteration,
                "phase": category,
                "category": category,
                "subcategory": "",
                "candidate_family": _infer_candidate_family(
                    candidate_type=meta.get("candidate_type") or "rule",
                    model_family=meta.get("model_family"),
                    name=name,
                    existing=meta.get("candidate_family"),
                ),
                "candidate_type": meta.get("candidate_type") or "rule",
                "timeframe": str(meta.get("inferred_timeframe") or "").strip()
                or _infer_timeframe(meta.get("code"), meta.get("reviewer_notes")),
                "universe": list(universe),
                "detail": str(failure.get("detail") or "")[:500],
                "fix_applied": "",
                "attempt": 0,
            }
        )
        existing_failure_keys.add((name, iteration, category))

    return payload


def backfill_global_strategy_knowledge_base(
    global_kb: KnowledgeBase,
    *,
    runs_root: Optional[Path] = None,
    current_run_id: str = "",
) -> dict[str, int]:
    root = Path(runs_root or paths.runs_root()).resolve()
    stats = {
        "runs_scanned": 0,
        "runs_merged": 0,
        "strategy_cards_before": len(global_kb.strategy_cards),
        "failure_cards_before": len(global_kb.failure_cards),
    }

    if not root.exists():
        stats["strategy_cards_after"] = stats["strategy_cards_before"]
        stats["failure_cards_after"] = stats["failure_cards_before"]
        stats["strategy_cards_added"] = 0
        stats["failure_cards_added"] = 0
        return stats

    for kb_path in sorted(root.glob("*/strategy_miner/knowledge_base.json")):
        run_id = str(kb_path.parent.parent.name or "")
        if not run_id or run_id == str(current_run_id or "").strip():
            continue
        stats["runs_scanned"] += 1
        payload = _strategy_memory_payload_from_run(kb_path.parent, run_id)
        if not any(payload.get(key) for key in ("elites", "failures", "strategy_cards", "failure_cards", "edges")):
            continue
        global_kb.merge_payload(payload, run_id_hint=run_id, save=False)
        stats["runs_merged"] += 1

    global_kb.save()
    stats["strategy_cards_after"] = len(global_kb.strategy_cards)
    stats["failure_cards_after"] = len(global_kb.failure_cards)
    stats["strategy_cards_added"] = (
        stats["strategy_cards_after"] - stats["strategy_cards_before"]
    )
    stats["failure_cards_added"] = (
        stats["failure_cards_after"] - stats["failure_cards_before"]
    )
    return stats


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
    final_artifacts: dict[str, str] = {}
    final_status: dict[str, object] = {}

    # Persist a stable proposal artifact for API/audit.
    from .artifacts import write_proposal, write_goal_contract, write_run_meta, append_event

    write_proposal(miner_dir, run_id=state.run_id, config=config, overwrite=False)

    # D1: Goal contract snapshot (backward compatible — auto-extract from config)
    _goal_contract = None
    try:
        from .goal_contract import GoalContract
        _goal_contract = GoalContract.from_miner_config(config)
        _gc_errors = _goal_contract.validate()
        if _gc_errors:
            logger.warning("Goal contract validation issues: %s", _gc_errors)
        write_goal_contract(miner_dir, _goal_contract)
        logger.debug("Goal contract snapshot: sha=%s", _goal_contract.sha256())
    except Exception as exc:
        logger.debug("Goal contract snapshot skipped: %s", exc)

    # D9: Initialize run metadata
    write_run_meta(miner_dir, run_id=state.run_id, phase=state.phase.value,
                   iteration=state.iteration)
    append_event(miner_dir, "run_start", {"run_id": state.run_id})

    kb = KnowledgeBase(miner_dir / "knowledge_base.json")
    global_kb: Optional[KnowledgeBase] = None
    if bool(getattr(config, "use_global_memory", True)):
        global_kb_path_raw = str(getattr(config, "global_strategy_knowledge_base_path", "") or "").strip()
        global_kb_path = (
            paths.resolve_repo_path(global_kb_path_raw)
            if global_kb_path_raw
            else paths.global_strategy_knowledge_base_path()
        )
        try:
            global_kb = KnowledgeBase(global_kb_path)
            logger.info("Loaded global strategy knowledge base: %s", global_kb_path)
            backfill_stats = backfill_global_strategy_knowledge_base(
                global_kb,
                current_run_id=state.run_id,
            )
            logger.info(
                "Backfilled global strategy knowledge base: runs=%d merged=%d strategy_cards +%d => %d, failure_cards +%d => %d",
                int(backfill_stats.get("runs_scanned", 0)),
                int(backfill_stats.get("runs_merged", 0)),
                int(backfill_stats.get("strategy_cards_added", 0)),
                int(backfill_stats.get("strategy_cards_after", 0)),
                int(backfill_stats.get("failure_cards_added", 0)),
                int(backfill_stats.get("failure_cards_after", 0)),
            )
        except Exception:
            logger.warning("Failed to load global strategy knowledge base", exc_info=True)

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

    # D13: Track total wall time
    import time as _run_time
    _run_start_wall = _run_time.time()

    try:
        while state.phase != Phase.COMPLETE:
            # D1: Check budget exhaustion from GoalContract
            if _goal_contract is not None and _goal_contract.is_budget_exhausted(
                state.economics, state.iteration, len(state.candidates)
            ):
                logger.info("Budget exhausted per GoalContract — completing run")
                append_event(miner_dir, "budget_exhausted", {
                    "iteration": state.iteration,
                    "candidates": len(state.candidates),
                    "economics": state.economics,
                })
                safe_transition(state, Phase.COMPLETE)
                break
            logger.info(
                "=== Iteration %d | Phase %s ===",
                state.iteration,
                state.phase.value,
            )

            # Capture phase name BEFORE execution (phase may advance inside)
            _phase_before = state.phase.value

            if state.phase == Phase.STRATEGY_GEN:
                # Track stagnation
                if state.best_score <= _last_best_score:
                    _stagnation_count += 1
                else:
                    _stagnation_count = 0
                    _last_best_score = state.best_score

                # D1: Stop on stagnation if GoalContract declares it
                if (
                    _goal_contract is not None
                    and "stagnation" in (_goal_contract.stop_conditions or [])
                    and _stagnation_count >= max(3, _goal_contract.max_iterations() // 2)
                ):
                    logger.info(
                        "Stagnation stop: no improvement for %d iterations (limit=%d)",
                        _stagnation_count,
                        _goal_contract.max_iterations() // 2,
                    )
                    append_event(miner_dir, "stagnation_stop", {
                        "stagnation_count": _stagnation_count,
                        "best_score": state.best_score,
                    })
                    safe_transition(state, Phase.COMPLETE)
                    break

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
                phase_strategy_gen(state, config, miner_dir, kb=kb, global_kb=global_kb, **_extra_kw)

            elif state.phase == Phase.BACKTEST:
                phase_backtest(state, config, miner_dir, kb=kb)

            elif state.phase == Phase.TRAIN_MODEL:
                phase_train_model(state, config, miner_dir, kb=kb)

            elif state.phase == Phase.EVALUATION:
                # Capture the candidate being evaluated BEFORE phase advances
                _eval_idx = getattr(state, "active_candidate_idx", None)
                _eval_cand = None
                if _eval_idx is not None and isinstance(_eval_idx, int) and 0 <= _eval_idx < len(state.candidates):
                    _eval_cand = state.candidates[_eval_idx]
                phase_evaluation(state, config, run_dir=miner_dir, kb=kb,
                                 goal_contract=_goal_contract)
                # D7: Update knowledge base after evaluation — pass the exact candidate
                _update_knowledge_base(kb, state, candidate=_eval_cand, config=config)
                if global_kb is not None and getattr(global_kb, "_path", None) != getattr(kb, "_path", None):
                    _update_knowledge_base(global_kb, state, candidate=_eval_cand, config=config)

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

            # D9: Update run metadata + event timeline after each phase
            append_event(miner_dir, "phase_complete", {
                "iteration": state.iteration,
                "phase_completed": _phase_before,
                "phase_next": state.phase.value,
            })
            write_run_meta(miner_dir, run_id=state.run_id,
                           phase=state.phase.value, iteration=state.iteration)

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
                    final_status["holdout_passed"] = not holdout_result.get("overfitting_flag", False)
                    # D10: Enforce holdout gate — demote champion if overfitting detected
                    holdout_passed = not holdout_result.get("overfitting_flag", False)
                    from ._helpers import update_candidate_stage
                    update_candidate_stage(state.best_candidate, "holdout_tested")
                    if holdout_passed:
                        # D1/D10: Check promotion conditions from GoalContract
                        _promotion_ok = True
                        if _goal_contract is not None:
                            _delta_pct = holdout_result.get("delta_pct")
                            # Walk-forward folds are stored in backtest_summary["walkforward"]
                            _wf_data = (state.best_candidate.backtest_summary or {}).get("walkforward") or {}
                            _wf_passed = int(_wf_data.get("folds_completed", 0) or 0)
                            _obs_days = int((state.best_candidate.backtest_summary or {}).get("observation_days", 0) or 0)
                            _promotion_ok = _goal_contract.is_promotable(
                                sharpe=float(state.best_candidate.reward or 0),
                                holdout_delta_pct=float(_delta_pct) if _delta_pct is not None else None,
                                wf_folds_passed=_wf_passed,
                                constraints_ok=bool(getattr(state.best_candidate, "constraints_ok", True)),
                                observation_days=_obs_days,
                            )
                        benchmark_ok = True
                        benchmark_result = None
                        benchmark_suite = ""
                        if _goal_contract is not None:
                            benchmark_suite = str(
                                (_goal_contract.evaluation_protocol or {}).get("benchmark_suite") or ""
                            ).strip()
                        if not benchmark_suite:
                            benchmark_suite = str(getattr(config, "benchmark_suite", "") or "").strip()
                        if benchmark_suite:
                            try:
                                from ._benchmark import run_benchmark_suite
                                from .artifacts import write_benchmark_verdict

                                benchmark_result = run_benchmark_suite(
                                    state.best_candidate,
                                    suite_path=benchmark_suite,
                                    holdout_result=holdout_result,
                                )
                                state.best_candidate.funnel_state["benchmark"] = benchmark_result
                                benchmark_ok = bool(benchmark_result.get("passed", False))
                                final_status["benchmark_passed"] = benchmark_ok
                                _benchmark_path = write_benchmark_verdict(miner_dir, benchmark_result)
                                final_artifacts["benchmark_verdict"] = str(_benchmark_path.resolve())
                                append_event(miner_dir, "benchmark_complete", {
                                    "passed": benchmark_ok,
                                    "failed_ids": benchmark_result.get("failed_ids", []),
                                })
                            except Exception as exc:
                                benchmark_ok = False
                                final_status["benchmark_passed"] = False
                                logger.warning("Frozen benchmark suite failed: %s", exc)
                        if _promotion_ok:
                            if benchmark_ok:
                                update_candidate_stage(state.best_candidate, "promoted")
                                logger.info(
                                    "Holdout PASSED + promoted: %s (delta=%.2f%%)",
                                    state.best_candidate.name,
                                    float(holdout_result.get("delta_pct") or 0),
                                )
                                final_status["promotion_status"] = "promoted"
                            else:
                                logger.warning(
                                    "Holdout PASSED but frozen benchmark FAILED: %s",
                                    state.best_candidate.name,
                                )
                                final_status["promotion_status"] = "benchmark_failed"
                        else:
                            logger.warning(
                                "Holdout PASSED but promotion conditions NOT met: %s",
                                state.best_candidate.name,
                            )
                            final_status["promotion_status"] = "promotion_conditions_failed"
                        try:
                            from .artifacts import append_promotion_log

                            _factor_retrieval = ((state.best_candidate.candidate_payload or {}).get("factor_retrieval") or {})
                            append_promotion_log(miner_dir, {
                                "candidate": state.best_candidate.name,
                                "holdout_passed": holdout_passed,
                                "holdout_delta_pct": holdout_result.get("delta_pct"),
                                "benchmark_passed": benchmark_result.get("passed") if isinstance(benchmark_result, dict) else None,
                                "benchmark_failed_ids": benchmark_result.get("failed_ids", []) if isinstance(benchmark_result, dict) else [],
                                "promotion_ok": _promotion_ok and benchmark_ok,
                                "factor_references": [item.get("card_id") for item in _factor_retrieval.get("factor_cards", []) if item.get("card_id")],
                                "factor_query": _factor_retrieval.get("query"),
                            })
                        except Exception:
                            pass
                    else:
                        _failed_name = state.best_candidate.name
                        logger.warning(
                            "Holdout FAILED (overfitting): %s demoted (delta=%.2f%%)",
                            state.best_candidate.name,
                            float(holdout_result.get("delta_pct") or 0),
                        )
                        # Demote: mark and clear best_candidate so downstream
                        # consumers know no strategy passed the full pipeline.
                        state.best_candidate.constraints_ok = False
                        state.best_candidate.constraint_violations = list(
                            state.best_candidate.constraint_violations or []
                        ) + ["holdout_overfitting"]
                        # Nullify best so API/export never sees a failed champion
                        state.best_candidate = None
                        state.best_score = float("-inf")
                        final_status["promotion_status"] = "holdout_failed"
                        try:
                            from .artifacts import append_promotion_log

                            _factor_retrieval = ((state.best_candidate.candidate_payload or {}).get("factor_retrieval") or {})
                            append_promotion_log(miner_dir, {
                                "candidate": _failed_name,
                                "holdout_passed": False,
                                "holdout_delta_pct": holdout_result.get("delta_pct"),
                                "promotion_ok": False,
                                "factor_references": [item.get("card_id") for item in _factor_retrieval.get("factor_cards", []) if item.get("card_id")],
                                "factor_query": _factor_retrieval.get("query"),
                            })
                        except Exception:
                            pass
                    _save_checkpoint(state, miner_dir)
                    # D10: Write holdout gate artifact
                    try:
                        from .artifacts import write_holdout_gate
                        _holdout_path = write_holdout_gate(miner_dir, holdout_result)
                        final_artifacts["holdout_gate"] = str(_holdout_path.resolve())
                    except Exception:
                        pass
                    # D9: Event
                    append_event(miner_dir, "holdout_complete", {
                        "passed": not holdout_result.get("overfitting_flag", False),
                        "delta_pct": holdout_result.get("delta_pct"),
                    })
            except Exception as exc:
                logger.warning("Sealed holdout failed: %s", exc)

        try:
            if bool(getattr(config, "portfolio_enabled", True)):
                from ._portfolio import build_candidate_portfolio_report
                from .artifacts import write_portfolio_plan

                portfolio_report = build_candidate_portfolio_report(
                    state.candidates,
                    top_k=int(getattr(config, "portfolio_top_k", 3) or 3),
                    min_candidates=int(getattr(config, "portfolio_min_candidates", 2) or 2),
                    correlation_threshold=float(
                        getattr(config, "portfolio_correlation_threshold", 0.85) or 0.85
                    ),
                    max_weight=float(getattr(config, "portfolio_max_weight", 0.60) or 0.60),
                )
                if portfolio_report is not None:
                    _portfolio_path = write_portfolio_plan(miner_dir, portfolio_report)
                    final_artifacts["portfolio_plan"] = str(_portfolio_path.resolve())
                    final_status["portfolio_method"] = portfolio_report.get("method")
                    append_event(miner_dir, "portfolio_complete", {
                        "passed": portfolio_report.get("passed"),
                        "method": portfolio_report.get("method"),
                        "weights": portfolio_report.get("weights", {}),
                    })
        except Exception as exc:
            logger.warning("Candidate portfolio construction failed: %s", exc)

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
        # D13: Accumulate total wall time (don't overwrite on resume)
        _session_wall = round(_run_time.time() - _run_start_wall, 1)
        state.economics["total_wall_seconds"] = round(
            state.economics.get("total_wall_seconds", 0.0) + _session_wall, 1
        )
        _save_checkpoint(state, miner_dir)
        # D9: Final event
        try:
            append_event(miner_dir, "run_complete", {
                "iterations": state.iteration,
                "best_score": state.best_score,
                "best_name": state.best_candidate.name if state.best_candidate else None,
            })
            write_run_meta(miner_dir, run_id=state.run_id,
                           phase="complete", iteration=state.iteration,
                           extra={
                               "best_score": state.best_score,
                               "artifacts": final_artifacts,
                               **final_status,
                           })
        except Exception:
            pass
        # D13: Persist economics rollup
        try:
            from .artifacts import write_economics
            write_economics(miner_dir, state.economics)
        except Exception:
            pass

    return state
