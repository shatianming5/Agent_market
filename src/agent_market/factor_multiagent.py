"""Lightweight multi-role diagnostics for factor expression mining.

The roles here are intentionally deterministic sidecars. They review and tag
factor candidates, but they never mark anything as promotion-ready.
"""
from __future__ import annotations

import ast
import json
import re
import time
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from agent_market.freqai.expression_engine import (
    ExpressionValidationError,
    allowed_expression_functions,
    safe_eval_expression,
)


DEFAULT_FACTOR_MULTIAGENT_ROLES: tuple[str, ...] = (
    "discoverer",
    "critic",
    "transfer_auditor",
    "curator",
)

ALLOWED_FACTOR_ROLES: set[str] = set(DEFAULT_FACTOR_MULTIAGENT_ROLES)

ALLOWED_EXPR_FUNCS: set[str] = allowed_expression_functions()
IMPLICIT_EXPR_FIELDS: set[str] = {"open", "high", "low", "close", "volume", "date", "ts"}
TIME_KEY_FIELDS: set[str] = {"date", "ts"}
XS_GROUP_ARG_INDEXES: dict[str, set[int]] = {
    "rank_xs": {1},
    "zscore_xs": {1},
    "corr_xs": {2},
}

_ILLEGAL_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Attribute,
    ast.Subscript,
    ast.Dict,
    ast.List,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.NamedExpr,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_multiagent_roles(raw: str | Sequence[str] | None) -> list[str]:
    """Parse and sanitize the factor role list from CLI/config values."""
    if raw is None:
        tokens = list(DEFAULT_FACTOR_MULTIAGENT_ROLES)
    elif isinstance(raw, str):
        tokens = [part.strip() for part in re.split(r"[,;\s]+", raw) if part.strip()]
    else:
        tokens = [str(part).strip() for part in raw if str(part).strip()]

    roles: list[str] = []
    for token in tokens:
        role = token.lower()
        if role in ALLOWED_FACTOR_ROLES and role not in roles:
            roles.append(role)
    return roles or list(DEFAULT_FACTOR_MULTIAGENT_ROLES)


def empty_factor_agent_traces(
    *,
    enabled: bool,
    roles: Sequence[str],
    parallelism: int,
) -> dict[str, Any]:
    return {
        "version": "factor-expression-agent-traces-v1",
        "saved_at": iso_now(),
        "enabled": bool(enabled),
        "roles": list(roles),
        "parallelism": max(1, int(parallelism or 1)),
        "events": [],
        "failure_taxonomy": {},
    }


def record_factor_agent_event(
    traces: dict[str, Any],
    *,
    role: str,
    event: str,
    status: str = "ok",
    detail: Mapping[str, Any] | None = None,
    failure_category: str = "",
) -> None:
    """Append a bounded diagnostic event and update failure counts."""
    payload = {
        "timestamp": iso_now(),
        "role": str(role or "unknown"),
        "event": str(event or "event"),
        "status": str(status or "ok"),
        "detail": dict(detail or {}),
    }
    if failure_category:
        payload["failure_category"] = str(failure_category)
        failures = traces.setdefault("failure_taxonomy", {})
        failures[str(failure_category)] = int(failures.get(str(failure_category), 0) or 0) + 1
    traces.setdefault("events", []).append(payload)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _names_used(expr: str) -> set[str]:
    try:
        tree = ast.parse(str(expr or ""), mode="eval")
    except SyntaxError:
        return set()
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in called and node.id not in {"True", "False"}
    }


def _probe_expression_frame(feature_cols: Iterable[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(str(col) for col in feature_cols if str(col)))
    base_cols = list(
        dict.fromkeys([*cols, "date", "ts", "open", "high", "low", "close", "volume", "mid", "spread", "ofi_10", "imbalance_10"])
    )
    rows = 16
    data: dict[str, list[float | int]] = {}
    for offset, col in enumerate(base_cols):
        if col in {"date", "ts"}:
            data[col] = [idx // 4 for idx in range(rows)]
        elif "spread" in col:
            data[col] = [0.01 + 0.001 * idx for idx in range(rows)]
        elif "imbalance" in col:
            data[col] = [((idx % 5) - 2) / 5.0 for idx in range(rows)]
        else:
            data[col] = [float(idx + 1 + offset) for idx in range(rows)]
    return pd.DataFrame(data, index=range(rows))


def _engine_validation_reasons(expr: str, feature_cols: Iterable[str]) -> list[str]:
    try:
        safe_eval_expression(expr, _probe_expression_frame(feature_cols))
    except ExpressionValidationError as exc:
        return [f"engine_validation:{str(exc)}"]
    except Exception as exc:
        return [f"engine_runtime:{exc.__class__.__name__}:{str(exc)}"]
    return []


def _allowed_time_key_node_ids(tree: ast.AST) -> set[int]:
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        for idx in XS_GROUP_ARG_INDEXES.get(node.func.id, set()):
            if idx >= len(node.args):
                continue
            arg = node.args[idx]
            if isinstance(arg, ast.Name) and arg.id in TIME_KEY_FIELDS:
                allowed.add(id(arg))
    return allowed


def critic_audit_expression(expr: str, feature_cols: Iterable[str]) -> dict[str, Any]:
    """Static critic for DSL safety, leakage-prone constructs, and fields."""
    text = str(expr or "").strip()
    reasons: list[str] = []
    warnings: list[str] = []
    feature_set = {str(col) for col in feature_cols if str(col)}
    allowed_fields = set(feature_set) | set(IMPLICIT_EXPR_FIELDS)

    if not text:
        return {"ok": False, "reasons": ["empty_expression"], "warnings": [], "fields": []}

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        return {
            "ok": False,
            "reasons": ["invalid_expression_syntax"],
            "warnings": [],
            "fields": [],
            "error": str(exc),
        }

    for node in ast.walk(tree):
        if isinstance(node, _ILLEGAL_AST_NODES):
            reasons.append(f"illegal_ast_node:{node.__class__.__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                reasons.append("illegal_call_target")
            elif node.func.id not in ALLOWED_EXPR_FUNCS:
                reasons.append(f"unknown_function:{node.func.id}")

    allowed_time_key_ids = _allowed_time_key_node_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in TIME_KEY_FIELDS and id(node) not in allowed_time_key_ids:
            reasons.append(f"time_key_not_factor:{node.id}")

    fields = sorted(_names_used(text))
    for name in fields:
        if name not in allowed_fields:
            reasons.append(f"unknown_field:{name}")
        lowered = name.lower()
        if any(token in lowered for token in ("future", "label", "target", "return_fwd")):
            reasons.append(f"leakage_field:{name}")

    reasons.extend(_engine_validation_reasons(text, feature_set))

    if len(text) > 320:
        warnings.append("expression_too_long")
    if text.count("shift(") >= 4:
        warnings.append("many_shift_calls")

    deduped_reasons = list(dict.fromkeys(reasons))
    return {
        "ok": not deduped_reasons,
        "reasons": deduped_reasons,
        "warnings": list(dict.fromkeys(warnings)),
        "fields": fields,
    }


def transfer_audit_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Score whether a factor is useful as a strategy/rank-portfolio input."""
    score_raw = _float_or_none(item.get("score"))
    ic = _float_or_none(item.get("metric_abs_ic"))
    if ic is None:
        ic = _float_or_none(item.get("ic"))
    turnover = _float_or_none(item.get("turnover"))
    if turnover is None:
        ac = _float_or_none(item.get("autocorr_1"))
        turnover = max(0.05, 1.0 - min(max(ac if ac is not None else 0.5, 0.0), 0.99))
    complexity = _float_or_none(item.get("complexity"))
    if complexity is None:
        complexity = max(1.0, len(str(item.get("expression") or "")) / 12.0)

    ic_proxy = abs(ic if ic is not None else score_raw if score_raw is not None else 0.0)
    turnover_proxy = max(0.0, turnover if turnover is not None else 0.5)
    capacity_slippage_proxy = round(1.0 / (1.0 + turnover_proxy + max(complexity, 1.0) / 50.0), 6)
    rank_portfolio_transfer_score = round(
        float(ic_proxy) - 0.05 * turnover_proxy - 0.001 * max(complexity, 1.0),
        6,
    )
    reasons: list[str] = []
    if ic_proxy < 0.01:
        reasons.append("weak_ic_proxy")
    if turnover_proxy > 5.0:
        reasons.append("high_turnover_proxy")
    if max(complexity, 1.0) > 40:
        reasons.append("high_complexity")
    if capacity_slippage_proxy < 0.15:
        reasons.append("low_capacity_proxy")
    return {
        "status": "candidate" if rank_portfolio_transfer_score > 0 and not reasons else "needs_review",
        "rank_ic_proxy": round(float(ic_proxy), 6),
        "turnover_proxy": round(float(turnover_proxy), 6),
        "capacity_slippage_proxy": capacity_slippage_proxy,
        "rank_portfolio_transfer_score": rank_portfolio_transfer_score,
        "strategy_transfer_score": rank_portfolio_transfer_score,
        "reasons": reasons,
        "promotion_eligible": False,
    }


def run_factor_multiagent_review(
    *,
    expressions: Sequence[Mapping[str, Any]],
    feature_cols: Sequence[str],
    enabled: bool,
    roles: Sequence[str],
    parallelism: int,
    traces: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Review, transfer-audit, and curate expression candidates.

    Returns ``(curated_expressions, traces, transfer_audit, summary)``.
    """
    roles_list = parse_multiagent_roles(list(roles))
    out_traces = dict(traces or empty_factor_agent_traces(enabled=enabled, roles=roles_list, parallelism=parallelism))
    out_traces["saved_at"] = iso_now()
    out_traces["enabled"] = bool(enabled)
    out_traces["roles"] = roles_list
    out_traces["parallelism"] = max(1, int(parallelism or 1))
    out_traces.setdefault("events", [])
    out_traces.setdefault("failure_taxonomy", {})

    active_roles = set(roles_list)
    run_discoverer = "discoverer" in active_roles
    run_critic = "critic" in active_roles
    run_transfer = "transfer_auditor" in active_roles
    run_curator = "curator" in active_roles

    source_items = [dict(item) for item in expressions if isinstance(item, Mapping)]
    if run_discoverer:
        record_factor_agent_event(
            out_traces,
            role="discoverer",
            event="candidate_pool_received",
            detail={"candidate_count": len(source_items)},
        )

    curated: list[dict[str, Any]] = []
    seen_exprs: set[str] = set()
    audit_items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    duplicate_count = 0

    for idx, item in enumerate(source_items, start=1):
        expr = str(item.get("expression") or "").strip()
        name = str(item.get("name") or f"factor_{idx:03d}")
        safety_audit = critic_audit_expression(expr, feature_cols)
        audit = (
            safety_audit
            if run_critic
            else {
                "ok": bool(safety_audit.get("ok")),
                "skipped": True,
                "reason": "critic_role_not_enabled",
                "safety_guard": safety_audit,
            }
        )
        transfer = (
            transfer_audit_item(item)
            if run_transfer
            else {
                "status": "not_run",
                "reason": "transfer_auditor_role_not_enabled",
                "promotion_eligible": False,
            }
        )
        record = {
            "name": name,
            "expression": expr,
            "critic": audit,
            "transfer_audit": transfer,
        }
        audit_items.append(record)

        if not safety_audit.get("ok"):
            reject_role = "critic" if run_critic else "safety_guard"
            failure = {
                "name": name,
                "expression": expr,
                "category": "critic_reject" if run_critic else "safety_reject",
                "subcategory": ",".join(safety_audit.get("reasons") or []) or "invalid_expression",
                "role": reject_role,
                "promotion_eligible": False,
            }
            failures.append(failure)
            record_factor_agent_event(
                out_traces,
                role=reject_role,
                event="candidate_rejected",
                status="failed",
                detail={"name": name, "reasons": safety_audit.get("reasons") or []},
                failure_category="invalid_expression",
            )
            continue

        if run_curator and expr in seen_exprs:
            duplicate_count += 1
            failure = {
                "name": name,
                "expression": expr,
                "category": "curator_reject",
                "subcategory": "duplicate_expression",
                "role": "curator",
                "promotion_eligible": False,
            }
            failures.append(failure)
            record_factor_agent_event(
                out_traces,
                role="curator",
                event="candidate_deduped",
                detail={"name": name},
                failure_category="duplicate_expression",
            )
            continue

        if run_curator:
            seen_exprs.add(expr)
        category = str(item.get("category") or "other").strip() or "other"
        category_counts[category] += 1
        enriched = dict(item)
        enriched["critic_review"] = audit
        if not run_critic:
            enriched["safety_review"] = safety_audit
        if run_transfer:
            enriched["transfer_audit"] = transfer
        if run_curator:
            tags = ["multiagent_reviewed", f"category:{category}"]
            if run_transfer:
                tags.append(f"transfer:{transfer.get('status')}")
            enriched["agent_tags"] = list(dict.fromkeys(tags))
            enriched["memory_scope"] = "pending_review"
        enriched["promotion_eligible"] = False
        curated.append(enriched)

    transfer_audit = {
        "version": "factor-transfer-audit-v1",
        "saved_at": iso_now(),
        "items": audit_items,
        "summary": {
            "candidate_count": len(source_items),
            "accepted_count": len(curated),
            "rejected_count": len(failures),
            "duplicate_count": duplicate_count,
        },
    }
    summary = {
        "version": "factor-multiagent-summary-v1",
        "saved_at": iso_now(),
        "enabled": bool(enabled),
        "roles": roles_list,
        "parallelism": max(1, int(parallelism or 1)),
        "promotion_controller": "agent_market.factor_lab.strategy-loop",
        "promotion_policy": "multiagent_outputs_are_search_only",
        "safety_guard_always_on": True,
        "promotion_eligible": False,
        "role_execution": {
            "discoverer": run_discoverer,
            "critic": run_critic,
            "transfer_auditor": run_transfer,
            "curator": run_curator,
        },
        "counts": {
            "input_candidates": len(source_items),
            "accepted_candidates": len(curated),
            "rejected_candidates": len(failures),
            "duplicates_removed": duplicate_count,
        },
        "category_counts": dict(category_counts),
        "failure_taxonomy": dict(out_traces.get("failure_taxonomy") or {}),
        "failures": failures[:100],
    }
    if run_curator:
        record_factor_agent_event(
            out_traces,
            role="curator",
            event="curation_complete",
            detail=summary["counts"],
        )
    return curated, out_traces, transfer_audit, summary


def write_factor_multiagent_artifacts(
    *,
    output_dir: Any,
    output_stem: str | None = None,
    traces: Mapping[str, Any],
    transfer_audit: Mapping[str, Any],
    summary: Mapping[str, Any],
    manifest_extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write sidecar JSON artifacts and a run-local manifest."""
    from pathlib import Path

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = ""
    if output_stem:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(output_stem).strip()).strip("._-")
    prefix = f"{safe_stem}_" if safe_stem else ""
    paths = {
        "factor_agent_traces": out_dir / f"{prefix}factor_agent_traces.json",
        "factor_transfer_audit": out_dir / f"{prefix}factor_transfer_audit.json",
        "multiagent_summary": out_dir / f"{prefix}multiagent_summary.json",
    }
    payloads = {
        "factor_agent_traces": dict(traces),
        "factor_transfer_audit": dict(transfer_audit),
        "multiagent_summary": dict(summary),
    }
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key], ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "version": "factor-expression-manifest-v1",
        "saved_at": iso_now(),
        "promotion_controller": "agent_market.factor_lab.strategy-loop",
        "promotion_policy": "search_and_review_only",
        "promotion_eligible": False,
        "artifact_refs": {key: str(path) for key, path in paths.items()},
    }
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    manifest_path = out_dir / f"{prefix}manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()} | {"manifest": str(manifest_path)}
