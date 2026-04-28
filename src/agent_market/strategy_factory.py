from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_market import paths
from agent_market.run_artifacts import RunArtifacts


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _resolve_artifact(path_like: str | None) -> Path | None:
    raw = str(path_like or "").strip()
    if not raw:
        return None
    try:
        return paths.resolve_repo_path(raw)
    except Exception:
        return None


def _load_json(path_like: str | Path | None) -> dict[str, Any]:
    if path_like is None:
        return {}
    path = Path(path_like) if isinstance(path_like, Path) else _resolve_artifact(path_like)
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path_like: str | Path | None) -> list[dict[str, Any]]:
    if path_like is None:
        return []
    path = Path(path_like) if isinstance(path_like, Path) else _resolve_artifact(path_like)
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    return paths.relpath_for_meta(path.resolve())


def _artifact_map(arts: RunArtifacts) -> dict[str, Any]:
    return {
        "expression_output": arts.expression_output,
        "expression_scored_output": arts.expression_scored_output,
        "factor_spec_json": arts.factor_spec_json,
        "factor_eval_meta": arts.factor_eval_meta,
        "factor_scores_json": arts.factor_scores_json,
        "factor_memory_json": arts.factor_memory_json,
        "factor_cards_json": arts.factor_cards_json,
        "factor_failure_cards_json": arts.factor_failure_cards_json,
        "factor_lineage_json": arts.factor_lineage_json,
        "global_factor_memory_json": arts.global_factor_memory_json,
        "global_factor_cards_json": arts.global_factor_cards_json,
        "global_factor_failure_cards_json": arts.global_factor_failure_cards_json,
        "global_factor_lineage_json": arts.global_factor_lineage_json,
        "training_summary_json": arts.training_summary_json,
        "feedback_summary_json": arts.feedback_summary_json,
        "bundle_zip": arts.bundle_zip,
        "bundle_manifest": arts.bundle_manifest,
        "strategy_miner_summary": arts.strategy_miner_summary,
        "strategy_miner_dir": arts.strategy_miner_dir,
        "global_strategy_knowledge_base_json": arts.global_strategy_knowledge_base_json,
    }


def _step_artifacts(step_name: str, arts: RunArtifacts) -> dict[str, Any]:
    mapping = {
        "factor_compile": {
            "factor_spec_json": arts.factor_spec_json,
            "factor_expression_txt": arts.factor_expression_txt,
            "factor_expression_json": arts.factor_expression_json,
            "factor_ast_json": arts.factor_ast_json,
        },
        "expression": {
            "expression_output": arts.expression_output,
            "expression_scored_output": arts.expression_scored_output,
            "factor_eval_meta": arts.factor_eval_meta,
            "factor_scores_json": arts.factor_scores_json,
            "factor_memory_json": arts.factor_memory_json,
            "factor_cards_json": arts.factor_cards_json,
            "factor_failure_cards_json": arts.factor_failure_cards_json,
            "factor_lineage_json": arts.factor_lineage_json,
            "global_factor_memory_json": arts.global_factor_memory_json,
        },
        "factor_eval": {
            "factor_eval_meta": arts.factor_eval_meta,
            "factor_scores_json": arts.factor_scores_json,
            "factor_pareto_csv": arts.factor_pareto_csv,
            "factor_memory_json": arts.factor_memory_json,
            "factor_cards_json": arts.factor_cards_json,
            "factor_failure_cards_json": arts.factor_failure_cards_json,
            "factor_lineage_json": arts.factor_lineage_json,
            "global_factor_memory_json": arts.global_factor_memory_json,
        },
        "ml": {
            "training_summary_json": arts.training_summary_json,
        },
        "backtest": {
            "feedback_summary_json": arts.feedback_summary_json,
        },
        "strategy_miner": {
            "strategy_miner_summary": arts.strategy_miner_summary,
            "strategy_miner_dir": arts.strategy_miner_dir,
            "global_strategy_knowledge_base_json": arts.global_strategy_knowledge_base_json,
        },
        "report": {
            "bundle_zip": arts.bundle_zip,
            "bundle_manifest": arts.bundle_manifest,
        },
        "portfolio": {
            "portfolio_weights": arts.portfolio_weights,
            "portfolio_report": arts.portfolio_report,
            "portfolio_returns": arts.portfolio_returns,
        },
    }
    return {k: v for k, v in (mapping.get(step_name) or {}).items() if v}


def _representative_candidate(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_candidate")
    if isinstance(best, dict) and best:
        return best
    candidates = list(summary.get("candidates") or [])
    if not candidates:
        return {}
    candidates.sort(
        key=lambda item: float(item.get("reward") or item.get("sharpe") or item.get("score_sharpe") or float("-inf")),
        reverse=True,
    )
    return dict(candidates[0]) if candidates else {}


def append_experiment_registry(
    *,
    run_id: str,
    run_dir: Path,
    experiment_cfg: dict[str, Any],
    status: str,
    started_at: str,
    steps_meta: list[dict[str, Any]],
    error_info: dict[str, Any] | None,
    arts: RunArtifacts,
) -> Path:
    registry_path = paths.experiment_registry_path()
    parent_run_id = str(experiment_cfg.get("parent_run_id") or "")
    parent_experiment_id = str(experiment_cfg.get("parent_experiment_id") or "")
    root_experiment_id = str(experiment_cfg.get("experiment_id") or f"{run_id}:root")
    budget = dict(experiment_cfg.get("budget") or {})
    owner = str(experiment_cfg.get("owner") or "")
    hypothesis = str(experiment_cfg.get("hypothesis") or "")

    root_entry = {
        "schema_version": "1.0",
        "experiment_id": root_experiment_id,
        "run_id": run_id,
        "layer": "strategy_factory",
        "status": status,
        "started_at": started_at,
        "ended_at": _iso_now(),
        "owner": owner,
        "hypothesis": hypothesis,
        "budget": budget,
        "parent_run_id": parent_run_id,
        "parent_experiment_id": parent_experiment_id,
        "artifacts": _artifact_map(arts),
        "verdict": {
            "status": status,
            "error": error_info or None,
        },
        "run_dir": _rel(run_dir),
    }
    _append_jsonl(registry_path, root_entry)

    for step in steps_meta:
        step_name = str(step.get("name") or "").strip()
        if not step_name:
            continue
        if step_name in {"expression", "factor_compile", "factor_eval"}:
            layer = "factor"
        elif step_name in {"ml", "backtest"} and arts.factor_cards_json and not arts.strategy_miner_dir:
            layer = "factor_validation"
        elif step_name == "strategy_miner":
            layer = "strategy"
        else:
            layer = "workflow"
        _append_jsonl(
            registry_path,
            {
                "schema_version": "1.0",
                "experiment_id": f"{run_id}:{step_name}",
                "run_id": run_id,
                "layer": layer,
                "step": step_name,
                "status": step.get("status") or "unknown",
                "started_at": step.get("started_at"),
                "ended_at": step.get("ended_at"),
                "owner": owner,
                "hypothesis": hypothesis,
                "budget": budget.get(step_name) if isinstance(budget.get(step_name), dict) else {},
                "parent_run_id": run_id,
                "parent_experiment_id": root_experiment_id,
                "artifacts": _step_artifacts(step_name, arts),
                "verdict": step.get("error") or {"status": step.get("status") or "unknown"},
                "run_dir": _rel(run_dir / step_name),
            },
        )
    return registry_path


def write_budget_plan(
    *,
    run_dir: Path,
    experiment_cfg: dict[str, Any],
    arts: RunArtifacts,
) -> Path:
    factor_cards = (_load_json(arts.factor_cards_json).get("items") or [])
    miner_dir = _resolve_artifact(arts.strategy_miner_dir)
    leaderboard = _load_json(miner_dir / "leaderboard.json" if miner_dir else None)
    promotion_rows = _load_jsonl(miner_dir / "promotion_log.jsonl" if miner_dir else None)

    factor_pass = sum(1 for item in factor_cards if bool(item.get("gate_pass")))
    surviving_strategies = len(leaderboard.get("items") or [])
    promoted = sum(1 for row in promotion_rows if bool(row.get("promotion_ok")))

    allocations = {"factor": 0.30, "strategy": 0.50, "benchmark": 0.10, "portfolio": 0.10}
    rationale: list[str] = []
    if factor_pass <= 1:
        allocations["factor"] += 0.10
        allocations["strategy"] -= 0.05
        allocations["portfolio"] -= 0.05
        rationale.append("Few factors passed gates; bias next-run budget toward factor discovery.")
    if surviving_strategies == 0:
        allocations["strategy"] += 0.10
        allocations["benchmark"] -= 0.05
        allocations["portfolio"] -= 0.05
        rationale.append("No strategies survived evaluation; reserve more budget for strategy search.")
    if promoted > 0:
        allocations["benchmark"] += 0.05
        allocations["portfolio"] += 0.05
        allocations["strategy"] -= 0.10
        rationale.append("A strategy reached promotion; shift next-run budget to benchmark and portfolio validation.")

    requested_budget = dict(experiment_cfg.get("budget") or {})
    payload = {
        "schema_version": "1.0",
        "generated_at": _iso_now(),
        "requested_budget": requested_budget,
        "observed": {
            "factor_cards": len(factor_cards),
            "factor_gate_pass": factor_pass,
            "surviving_strategies": surviving_strategies,
            "promoted_strategies": promoted,
        },
        "recommended_allocations": allocations,
        "rationale": rationale or ["Using default balanced allocation because no strong signal was observed yet."],
    }
    return _atomic_write_json(run_dir / "budget_plan.json", payload)


def write_lineage_graph(
    *,
    run_dir: Path,
    arts: RunArtifacts,
) -> Path | None:
    if not arts.factor_lineage_json and not arts.strategy_miner_summary:
        return None
    factor_lineage = _load_json(arts.factor_lineage_json)
    summary = _load_json(arts.strategy_miner_summary)
    edges = list(factor_lineage.get("edges") or [])

    for candidate in summary.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_name = str(candidate.get("name") or "")
        payload = candidate.get("candidate_payload") or {}
        retrieval = payload.get("factor_retrieval") or {}
        for card in retrieval.get("factor_cards") or []:
            card_id = str((card or {}).get("card_id") or "")
            if not card_id or not candidate_name:
                continue
            edges.append(
                {
                    "parent": card_id,
                    "child": candidate_name,
                    "edge_type": "factor_to_strategy_candidate",
                    "run_id": summary.get("run_id"),
                }
            )

    best_candidate = summary.get("best_candidate") or {}
    best_name = str(best_candidate.get("name") or "")
    if best_name:
        miner_dir = _resolve_artifact(arts.strategy_miner_dir)
        portfolio = _load_json(miner_dir / "portfolio_plan.json" if miner_dir else None)
        weights = portfolio.get("weights") or {}
        for slot_name, weight in weights.items():
            if str(slot_name) == best_name:
                edges.append(
                    {
                        "parent": best_name,
                        "child": f"portfolio:{slot_name}",
                        "edge_type": "strategy_to_portfolio_slot",
                        "run_id": summary.get("run_id"),
                        "weight": weight,
                    }
                )
    return _atomic_write_json(run_dir / "lineage_graph.json", {"edges": edges})


def write_promotion_chain(
    *,
    run_dir: Path,
    arts: RunArtifacts,
) -> Path | None:
    miner_dir = _resolve_artifact(arts.strategy_miner_dir)
    if miner_dir is None:
        factor_cards = (_load_json(arts.factor_cards_json).get("items") or [])
        backtest_summary = _load_json(arts.feedback_summary_json)
        best_factor = max(
            factor_cards,
            key=lambda item: float(((item.get("metrics") or {}).get("weighted_score")) or 0.0),
            default={},
        )
        payload = {
            "schema_version": "1.0",
            "generated_at": _iso_now(),
            "run_id": run_dir.name,
            "best_candidate": None,
            "best_factor": best_factor.get("card_id") or best_factor.get("name"),
            "factor_references": [best_factor.get("card_id")] if best_factor.get("card_id") else [],
            "promotion_log": [],
            "offline_stage": {
                "factor_memory": arts.factor_memory_json,
                "factor_scores": arts.factor_scores_json,
                "training_summary": arts.training_summary_json,
                "backtest_summary": arts.feedback_summary_json,
                "backtest_profit_pct": backtest_summary.get("profit_total_pct"),
            },
            "online_stage": {
                "shadow": "pending",
                "champion_challenger": "pending",
                "rollback": "pending",
            },
        }
        return _atomic_write_json(run_dir / "promotion_chain.json", payload)
    summary = _load_json(arts.strategy_miner_summary)
    promotion_rows = _load_jsonl(miner_dir / "promotion_log.jsonl")
    best_candidate = _representative_candidate(summary)
    candidate_payload = best_candidate.get("candidate_payload") or {}
    retrieval = candidate_payload.get("factor_retrieval") or {}
    strategy_retrieval = candidate_payload.get("strategy_retrieval") or {}
    payload = {
        "schema_version": "1.0",
        "generated_at": _iso_now(),
        "run_id": summary.get("run_id"),
        "best_candidate": best_candidate.get("name"),
        "factor_references": [item.get("card_id") for item in retrieval.get("factor_cards", []) if item.get("card_id")],
        "strategy_references": [item.get("card_id") for item in strategy_retrieval.get("strategy_cards", []) if item.get("card_id")],
        "promotion_log": promotion_rows,
        "offline_stage": {
            "holdout_gate": _rel(miner_dir / "holdout_gate.json") if (miner_dir / "holdout_gate.json").exists() else None,
            "benchmark_verdict": _rel(miner_dir / "benchmark_verdict.json") if (miner_dir / "benchmark_verdict.json").exists() else None,
            "portfolio_plan": _rel(miner_dir / "portfolio_plan.json") if (miner_dir / "portfolio_plan.json").exists() else None,
        },
        "online_stage": {
            "shadow": "pending",
            "champion_challenger": "pending",
            "rollback": "pending",
        },
    }
    return _atomic_write_json(run_dir / "promotion_chain.json", payload)


def write_replay_manifest(
    *,
    run_id: str,
    run_dir: Path,
    experiment_cfg: dict[str, Any],
    arts: RunArtifacts,
) -> Path:
    miner_dir = _resolve_artifact(arts.strategy_miner_dir)
    summary = _load_json(arts.strategy_miner_summary)
    best_candidate = _representative_candidate(summary)
    payload = {
        "schema_version": "1.0",
        "generated_at": _iso_now(),
        "run_id": run_id,
        "experiment": {
            "experiment_id": experiment_cfg.get("experiment_id") or f"{run_id}:root",
            "hypothesis": experiment_cfg.get("hypothesis"),
            "owner": experiment_cfg.get("owner"),
            "budget": experiment_cfg.get("budget") or {},
            "parent_run_id": experiment_cfg.get("parent_run_id") or "",
            "parent_experiment_id": experiment_cfg.get("parent_experiment_id") or "",
        },
        "factor_inputs": {
            "expression_output": arts.expression_output,
            "expression_scored_output": arts.expression_scored_output,
            "factor_spec_json": arts.factor_spec_json,
            "factor_eval_meta": arts.factor_eval_meta,
            "factor_scores_json": arts.factor_scores_json,
            "factor_memory_json": arts.factor_memory_json,
            "factor_cards_json": arts.factor_cards_json,
            "factor_failure_cards_json": arts.factor_failure_cards_json,
            "factor_lineage_json": arts.factor_lineage_json,
            "global_factor_memory_json": arts.global_factor_memory_json,
            "global_factor_cards_json": arts.global_factor_cards_json,
            "global_factor_failure_cards_json": arts.global_factor_failure_cards_json,
            "global_factor_lineage_json": arts.global_factor_lineage_json,
            "training_summary_json": arts.training_summary_json,
            "feedback_summary_json": arts.feedback_summary_json,
        },
        "strategy_inputs": {
            "strategy_miner_summary": arts.strategy_miner_summary,
            "strategy_miner_dir": arts.strategy_miner_dir,
            "global_strategy_knowledge_base_json": arts.global_strategy_knowledge_base_json,
            "proposal_json": _rel(miner_dir / "proposal.json" if miner_dir and (miner_dir / "proposal.json").exists() else None),
            "leaderboard_json": _rel(miner_dir / "leaderboard.json" if miner_dir and (miner_dir / "leaderboard.json").exists() else None),
            "economics_json": _rel(miner_dir / "economics.json" if miner_dir and (miner_dir / "economics.json").exists() else None),
        },
        "retrieval_snapshot": {
            "factor_retrieval": ((best_candidate.get("candidate_payload") or {}).get("factor_retrieval") or {}),
            "strategy_retrieval": ((best_candidate.get("candidate_payload") or {}).get("strategy_retrieval") or {}),
        },
        "validation_outputs": {
            "holdout_gate": _rel(miner_dir / "holdout_gate.json" if miner_dir and (miner_dir / "holdout_gate.json").exists() else None),
            "benchmark_verdict": _rel(miner_dir / "benchmark_verdict.json" if miner_dir and (miner_dir / "benchmark_verdict.json").exists() else None),
            "portfolio_plan": _rel(miner_dir / "portfolio_plan.json" if miner_dir and (miner_dir / "portfolio_plan.json").exists() else None),
            "promotion_log": _rel(miner_dir / "promotion_log.jsonl" if miner_dir and (miner_dir / "promotion_log.jsonl").exists() else None),
            "backtest_summary": arts.feedback_summary_json,
        },
    }
    return _atomic_write_json(run_dir / "replay_manifest.json", payload)


def write_resource_dashboard(
    *,
    run_dir: Path,
    arts: RunArtifacts,
) -> Path:
    factor_cards = (_load_json(arts.factor_cards_json).get("items") or [])
    miner_dir = _resolve_artifact(arts.strategy_miner_dir)
    summary = _load_json(arts.strategy_miner_summary)
    leaderboard = _load_json(miner_dir / "leaderboard.json" if miner_dir else None)
    economics = _load_json(miner_dir / "economics.json" if miner_dir else None)
    promotion_rows = _load_jsonl(miner_dir / "promotion_log.jsonl" if miner_dir else None)
    expression_payload = _load_json(arts.expression_output)

    surviving = len(leaderboard.get("items") or [])
    promoted = sum(1 for row in promotion_rows if bool(row.get("promotion_ok")))
    total_cost = float(economics.get("total_cost_usd") or 0.0)
    total_tokens = int(economics.get("total_tokens") or 0)
    if total_tokens == 0:
        llm_usage = expression_payload.get("llm_usage") or {}
        total_tokens = int(llm_usage.get("total_tokens") or 0)
        total_cost = float(llm_usage.get("cost_usd") or total_cost or 0.0)
    retrieval_hits = sum(
        1
        for candidate in (summary.get("candidates") or [])
        if ((candidate.get("candidate_payload") or {}).get("factor_retrieval") or {}).get("factor_cards")
    )
    strategy_retrieval_hits = sum(
        1
        for candidate in (summary.get("candidates") or [])
        if ((candidate.get("candidate_payload") or {}).get("strategy_retrieval") or {}).get("strategy_cards")
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": _iso_now(),
        "totals": {
            "factor_cards": len(factor_cards),
            "surviving_strategies": surviving,
            "promoted_strategies": promoted,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "retrieval_hits": retrieval_hits,
            "strategy_retrieval_hits": strategy_retrieval_hits,
        },
        "efficiency": {
            "cost_per_factor_card": round(total_cost / len(factor_cards), 6) if factor_cards else None,
            "cost_per_surviving_strategy": round(total_cost / surviving, 6) if surviving else None,
            "cost_per_promoted_strategy": round(total_cost / promoted, 6) if promoted else None,
        },
    }
    return _atomic_write_json(run_dir / "resource_dashboard.json", payload)


def finalize_strategy_factory_artifacts(
    *,
    run_id: str,
    run_dir: Path,
    experiment_cfg: dict[str, Any],
    status: str,
    started_at: str,
    steps_meta: list[dict[str, Any]],
    error_info: dict[str, Any] | None,
    arts: RunArtifacts,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    registry_path = append_experiment_registry(
        run_id=run_id,
        run_dir=run_dir,
        experiment_cfg=experiment_cfg,
        status=status,
        started_at=started_at,
        steps_meta=steps_meta,
        error_info=error_info,
        arts=arts,
    )
    outputs["experiment_registry"] = _rel(registry_path) or str(registry_path)

    budget_plan = write_budget_plan(run_dir=run_dir, experiment_cfg=experiment_cfg, arts=arts)
    outputs["budget_plan_json"] = _rel(budget_plan) or str(budget_plan)

    lineage_path = write_lineage_graph(run_dir=run_dir, arts=arts)
    if lineage_path is not None:
        outputs["lineage_graph_json"] = _rel(lineage_path) or str(lineage_path)

    promotion_path = write_promotion_chain(run_dir=run_dir, arts=arts)
    if promotion_path is not None:
        outputs["promotion_chain_json"] = _rel(promotion_path) or str(promotion_path)

    replay_path = write_replay_manifest(
        run_id=run_id,
        run_dir=run_dir,
        experiment_cfg=experiment_cfg,
        arts=arts,
    )
    outputs["replay_manifest_json"] = _rel(replay_path) or str(replay_path)

    dashboard_path = write_resource_dashboard(run_dir=run_dir, arts=arts)
    outputs["resource_dashboard_json"] = _rel(dashboard_path) or str(dashboard_path)
    return outputs


__all__ = [
    "append_experiment_registry",
    "finalize_strategy_factory_artifacts",
]
