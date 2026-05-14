#!/usr/bin/env python3
"""wq_brain unified CLI — human and hermes-agent entry point.

Human commands: auth, agent, scan, report.
Hermes-agent commands (called from inside agent session via terminal):
    auth, validate, simulate, submit, pool list, corr, search-arxiv,
    search-papers, math, docs.
All agent-facing commands emit JSON to stdout (for parseable output).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _ensure_dotenv() -> None:
    dotenv = REPO_ROOT / ".env"
    if not dotenv.exists():
        return
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _emit(data: Any, *, code: int = 0) -> None:
    """Print JSON to stdout and exit. Used by hermes-agent-facing commands."""
    print(json.dumps(data, indent=2, default=str))
    sys.exit(code)


# ── Hermes-agent-facing commands ──────────────────────────────────────────

def cmd_auth(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.client import session_from_env
    try:
        sess = session_from_env()
        sess.login()
        _emit({"ok": True, "msg": "WQ login successful"})
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)


def cmd_validate(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.operators import validate_expression
    errors = validate_expression(args.expr, strict=not args.lax)
    _emit({
        "ok": not errors,
        "errors": errors,
        "expr": args.expr,
        "mode": "lax" if args.lax else "strict",
    })


def cmd_mutate(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.mutation import FailureContext, MutationEngine
    ctx = FailureContext(
        expr=args.expr,
        sharpe=args.sharpe,
        fitness=args.fitness,
        turnover=args.turnover,
        returns=args.returns,
        status=args.status,
        error=args.error,
    )
    engine = MutationEngine(ctx)
    diag = engine.diagnose()
    _emit({
        "ok": True,
        "expr": args.expr,
        "quick_score": engine.score,
        "diagnosis": diag.to_dict(),
        "prompt_hints": engine.format_for_prompt(),
    })


def cmd_simulate(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.dtypes import AlphaPoolEntry, AlphaSettings
    from agent_market.wq_brain.paths import alpha_pool_path, tried_exprs_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.quota_monitor import release_action, reserve_action
    from agent_market.wq_brain.tried_log import (
        append_tried,
        classify_altitude,
        read_tried,
        utility_score,
    )
    settings = AlphaSettings(
        region=args.region, universe=args.universe, decay=args.decay,
        neutralization=args.neutralization, truncation=args.truncation,
    )
    local_gate_error = _agent_local_sim_gate_error(args.expr)
    if local_gate_error:
        _emit({
            "ok": False,
            "expr": args.expr,
            "rejected_by": "agent_local_sim_gate",
            "error": local_gate_error,
        }, code=2)
        return
    if args.tag and not getattr(args, "skip_cooldown", False):
        from agent_market.wq_brain.gates import slot_cooldown_block
        tried_rows = read_tried(tried_exprs_path(args.tag), tail=200)
        block = slot_cooldown_block(
            args.expr,
            tried_rows,
            window=int(getattr(args, "cooldown_window", 8)),
            delta_flip=float(getattr(args, "cooldown_delta_flip", 0.05)),
        )
        if block is not None:
            _emit(
                {
                    "ok": False,
                    "expr": args.expr,
                    "rejected_by": "slot_cooldown",
                    "error": block.reason,
                    "cooldown": block.to_dict(),
                },
                code=2,
            )
            return
    # Atomic reserve+commit closes the TOCTOU window between
    # "is there capacity?" and "record the call". The reservation is rolled
    # back via release_action below if the simulate call never reaches WQ.
    quota = reserve_action("simulate")
    if quota["status"] == "block":
        _emit({"ok": False, "expr": args.expr, "rejected_by": "quota",
               **quota}, code=2)
        return
    # Pin the reservation to its UTC day so a refund crossing midnight
    # decrements the SAME bucket we incremented, not "today".
    reserved_day = quota["day"]
    network_called = False
    try:
        sess = session_from_env()
        # session_from_env is a local cred read — no network yet. Mark the
        # network as "called" only when we hand off to WQ.
        network_called = True
        result = sess.simulate_and_parse(args.expr, settings, timeout=args.timeout)
        if args.tag:
            parent_row = None
            parent_alpha_id = getattr(args, "parent_alpha_id", None) or None
            if parent_alpha_id:
                for row in reversed(read_tried(tried_exprs_path(args.tag), tail=2000)):
                    if (row.get("alpha_id") or "") == parent_alpha_id:
                        parent_row = row
                        break
            altitude = None
            delta_u = None
            child_view = {
                "expr": args.expr,
                "region": args.region,
                "universe": args.universe,
            }
            if parent_row is not None:
                altitude = classify_altitude(parent_row, child_view)
                if result.sharpe is not None and result.fitness is not None:
                    delta_u = (
                        utility_score(
                            sharpe=result.sharpe,
                            fitness=result.fitness,
                            turnover=result.turnover,
                        )
                        - utility_score(
                            sharpe=parent_row.get("sharpe"),
                            fitness=parent_row.get("fitness"),
                            turnover=parent_row.get("turnover"),
                        )
                    )
            elif parent_alpha_id:
                # Parent id given but not found in log → still tag the row so
                # we can see the *intended* parent in the prompt trail.
                altitude = "L1_region_universe"
            evidence_type = (
                getattr(args, "evidence_type", None)
                or ("manual" if not parent_alpha_id else "mutation")
            )
            if altitude is None:
                # Parent-less rows still need an altitude so they propagate
                # through the shared cache. Infer from evidence_type using
                # the same convention as classify_altitude:
                #   seed / manual / region_swap        → L1_region_universe
                #   op_swap / crossover                → L2_op_family
                #   param_shift / decay_shift /        → L3_slot_param
                #     neutralization_swap
                #   numeric_tweak                       → L4_numeric_tweak
                _altitude_by_evidence = {
                    "seed": "L1_region_universe",
                    "manual": "L1_region_universe",
                    "region_swap": "L1_region_universe",
                    "op_swap": "L2_op_family",
                    "crossover": "L2_op_family",
                    "param_shift": "L3_slot_param",
                    "decay_shift": "L3_slot_param",
                    "neutralization_swap": "L3_slot_param",
                    "numeric_tweak": "L4_numeric_tweak",
                }
                altitude = _altitude_by_evidence.get(
                    evidence_type, "L3_slot_param"
                )
            append_tried(
                tried_exprs_path(args.tag),
                expr=args.expr,
                sharpe=result.sharpe,
                fitness=result.fitness,
                turnover=result.turnover,
                alpha_id=result.alpha_id,
                status=result.status,
                error=result.error,
                region=args.region,
                universe=args.universe,
                decay=args.decay,
                evidence_type=evidence_type,
                altitude=altitude,
                parent_alpha_id=parent_alpha_id,
                delta_U=delta_u,
            )
            colony_tag = getattr(args, "colony_tag", None) or ""
            if colony_tag:
                from agent_market.wq_brain.colony_state import update_best_so_far
                _, bsf_updated = update_best_so_far(
                    colony_tag,
                    args.tag,
                    alpha_id=result.alpha_id,
                    expr=args.expr,
                    sharpe=result.sharpe,
                    fitness=result.fitness,
                    turnover=result.turnover,
                    delta_U=delta_u,
                )
                if bsf_updated:
                    # Let callers know they've set a new high-water mark.
                    # Surfaced in the simulate response so the colony log can
                    # promote the candidate immediately.
                    setattr(args, "_bsf_updated", True)
        out: dict[str, Any] = {"ok": True, "expr": args.expr, **result.to_dict()}
        if quota["status"] == "throttle":
            out["quota_advisory"] = quota
        if getattr(args, "_bsf_updated", False):
            out["best_so_far_updated"] = True

        # Auto-persist passing candidates as UNSUBMITTED in the pool so a
        # later salvage / submit pass can reach them. Pre-fix, 70% of
        # high-fi candidates (fi≥1.0) computed by the agent were lost
        # because the LLM session abandoned them or the agent ran out of
        # turns before submitting. This guarantees we keep them.
        sh_min = float(getattr(args, "auto_persist_sharpe", 1.25) or 1.25)
        fi_min = float(getattr(args, "auto_persist_fitness", 1.0) or 1.0)
        if (args.tag and result.status == "COMPLETE"
                and result.alpha_id
                and result.sharpe is not None and result.fitness is not None
                and float(result.sharpe) >= sh_min
                and float(result.fitness) >= fi_min):
            try:
                pool = AlphaPool(alpha_pool_path(args.tag))
                already = any(e.alpha_id == result.alpha_id for e in pool.entries)
                if not already:
                    entry = AlphaPoolEntry(
                        alpha_id=result.alpha_id,
                        expr=args.expr,
                        settings_dict={
                            "region": args.region,
                            "universe": args.universe,
                            "decay": args.decay,
                            "neutralization": args.neutralization,
                            "truncation": args.truncation,
                        },
                        sharpe=float(result.sharpe),
                        fitness=float(result.fitness),
                        returns=float(result.returns or 0.0),
                        turnover=float(result.turnover or 0.0),
                        tag=args.tag,
                        source="auto_persist",
                        verified_status="UNSUBMITTED",
                        verified_at=0.0,
                        rejection_reasons=[],
                    )
                    if pool.add(entry):
                        out["auto_persisted"] = True
            except OSError as exc:
                out["auto_persist_error"] = str(exc)
        _emit(out)
    except Exception as exc:
        # If the call never reached WQ, refund the reserved slot so the
        # daily quota reflects actual usage, not failed local setup.
        if not network_called:
            try:
                release_action("simulate", day=reserved_day)
            except OSError:
                pass
        if args.tag:
            try:
                append_tried(
                    tried_exprs_path(args.tag), expr=args.expr,
                    sharpe=None, fitness=None, turnover=None, alpha_id=None,
                    status="ERROR", error=str(exc),
                    region=args.region, universe=args.universe, decay=args.decay,
                )
            except OSError:
                pass
        _emit({"ok": False, "expr": args.expr, "error": str(exc)}, code=1)


def cmd_submit(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.quota_monitor import release_action, reserve_action

    sess = session_from_env()

    # Pre-check 1 (LOCAL): jaccard token similarity vs ACTIVE pool — fast,
    # no API calls, no quota cost. Run BEFORE reserving the submit slot so
    # a candidate that fails this gate doesn't temporarily occupy quota.
    if not args.no_pre_check and args.tag:
        from agent_market.wq_brain.submit_gates import _finite_float, auto_fill_metrics
        # Codex review R2-#5: use _finite_float at both ingress and egress so
        # NaN from a stale tried_log row never reaches the gate or the
        # emitted JSON (json.dumps of NaN is non-standard).
        cand_meta = auto_fill_metrics(args.tag, args.alpha_id) if args.alpha_id else {}
        expr_for_check = args.expr or cand_meta.get("expr") or ""
        cand_sh = _finite_float(cand_meta.get("sharpe"))
        cand_fi = _finite_float(cand_meta.get("fitness"))
        if expr_for_check:
            local_check = _check_local_jaccard_vs_active(
                args.tag, expr_for_check,
                threshold=args.jaccard_max,
                semantic_threshold=args.semantic_max,
                candidate_sharpe=cand_sh,
                candidate_fitness=cand_fi,
                sharpe_margin=getattr(args, "sharpe_margin", 0.10),
                override_mode=getattr(args, "override_mode", "sharpe_and_fitness"),
                absolute_fitness_floor=getattr(args, "absolute_fitness_floor", 1.0),
            )
            # Strip non-finite numerics from emitted JSON (defensive)
            for k in ("candidate_sharpe", "candidate_fitness"):
                local_check[k] = _finite_float(local_check.get(k))
            if not local_check["accept"]:
                _emit({
                    "ok": False,
                    "rejected_by": "local_jaccard_pre_check",
                    "alpha_id": args.alpha_id,
                    **local_check,
                    "hint": "Mutate to a different family. Check Cross-Over Candidates and try ts_corr_pv / vwap_dev / intraday_range / open_gap / sector_relative — anything NOT structurally similar to the ACTIVE alpha above.",
                }, code=2)
                return
            if local_check.get("override_applied"):
                # Surface the override in stderr so operators see it in tmux logs
                print(
                    f"INFO local_jaccard_pre_check OVERRIDE for {args.alpha_id}: "
                    f"{local_check.get('reason', '')}",
                    file=sys.stderr,
                )

    # Now that the free local gate is past, reserve a submit slot. Refunds
    # below all pin to ``reserved_day`` so a UTC-midnight crossing decrements
    # the SAME bucket we incremented.
    submit_quota = reserve_action("submit")
    if submit_quota["status"] == "block":
        _emit({"ok": False, "alpha_id": args.alpha_id,
               "rejected_by": "quota", **submit_quota}, code=2)
        return
    reserved_day = submit_quota["day"]

    # Pre-check 2 (REMOTE): WQ-aligned self-correlation + sharpe-margin rule
    if not args.no_pre_check:
        from agent_market.wq_brain.submit_gates import GateInfraError
        try:
            check = _check_self_correlation(
                sess, args.alpha_id,
                corr_max=args.corr_max,
                sharpe_margin=getattr(args, "sharpe_margin", 0.10),
                tag=args.tag,
            )
            if not check["accept"]:
                try: release_action("submit", day=reserved_day)
                except OSError: pass
                _emit({
                    "ok": False,
                    "rejected_by": "wq_pre_check",
                    "alpha_id": args.alpha_id,
                    **check,
                    "hint": "Use --no-pre-check to override (will likely be rejected by WQ submit step).",
                }, code=2)
                return
        except GateInfraError as exc:
            # Fail-CLOSED on infra failure: the gate itself is broken, so
            # we can't tell whether this would pass policy. Continuing to
            # submit would burn the daily quota on a maybe-rejected alpha.
            # The agent can opt out with `--force-submit-on-precheck-error`.
            if not getattr(args, "force_submit_on_precheck_error", False):
                try: release_action("submit", day=reserved_day)
                except OSError: pass
                _emit({
                    "ok": False,
                    "rejected_by": "wq_pre_check_infra",
                    "alpha_id": args.alpha_id,
                    "error": str(exc),
                    "hint": (
                        "Pre-check infrastructure failure (network/auth/5xx). "
                        "Default is fail-closed to avoid wasting submit quota. "
                        "Pass --force-submit-on-precheck-error to override."
                    ),
                }, code=2)
                return
            print(
                f"WARN: pre_check infra error ({exc}); --force-submit-on-precheck-error "
                "set, proceeding anyway",
                file=sys.stderr,
            )

    try:
        wq_resp = sess.submit_alpha(args.alpha_id, verify_after_sec=args.verify_after_sec)
    except Exception as exc:
        # submit never reached WQ — refund the reservation pinned to its day
        try: release_action("submit", day=reserved_day)
        except OSError: pass
        _emit({"ok": False, "error": str(exc)}, code=1)
        return

    # If WQ async-rejected, surface the rejection prominently
    if wq_resp.get("verified_status") in ("REJECTED", "UNSUBMITTED"):
        # Still update pool with rejection so agent learns
        pool_added = False
        if args.tag:
            try:
                metrics = sess.fetch_alpha_metrics(args.alpha_id)
                if metrics.alpha_id:
                    actual_expr = args.expr or _auto_fill_expr(args.tag, metrics.alpha_id) \
                                  or "(submitted via CLI)"
                    entry = AlphaPoolEntry(
                        alpha_id=metrics.alpha_id,
                        expr=actual_expr,
                        settings_dict={},
                        sharpe=float(metrics.sharpe or 0.0),
                        fitness=float(metrics.fitness or 0.0),
                        returns=float(metrics.returns or 0.0),
                        turnover=float(metrics.turnover or 0.0),
                        tag=args.tag, source="agent",
                        verified_status=wq_resp.get("verified_status", "REJECTED"),
                        verified_at=time.time(),
                        rejection_reasons=wq_resp.get("rejection_reasons") or [],
                    )
                    pool = AlphaPool(alpha_pool_path(args.tag))
                    # Codex review R1-#1: must use upsert. The auto-persist
                    # path stamps the alpha as UNSUBMITTED first; pool.add()
                    # would return False (duplicate alpha_id) and the new
                    # REJECTED status would be silently dropped.
                    upsert_result = pool.upsert(entry)
                    pool_added = upsert_result in ("inserted", "updated")
            except Exception:
                pass
        _emit({
            "ok": False,
            "rejected_by": "wq_review",
            "alpha_id": args.alpha_id,
            "verified_status": wq_resp.get("verified_status"),
            "rejection_reasons": wq_resp.get("rejection_reasons", []),
            "summary": _summarize_rejection(wq_resp.get("rejection_reasons") or []),
            "recorded_to_pool": pool_added,
            "hint": "Try a structurally different alpha family — check Cross-Over Candidates.",
        }, code=2)
        return

    # Successful path: verified ACTIVE
    pool_added = False
    if args.tag:
        try:
            metrics = sess.fetch_alpha_metrics(args.alpha_id)
            if metrics.alpha_id and metrics.sharpe is not None:
                actual_expr = args.expr or _auto_fill_expr(args.tag, metrics.alpha_id) \
                              or "(submitted via CLI)"
                entry = AlphaPoolEntry(
                    alpha_id=metrics.alpha_id,
                    expr=actual_expr,
                    settings_dict={},
                    sharpe=float(metrics.sharpe),
                    fitness=float(metrics.fitness or 0.0),
                    returns=float(metrics.returns or 0.0),
                    turnover=float(metrics.turnover or 0.0),
                    tag=args.tag,
                    source="agent",
                    verified_status=wq_resp.get("verified_status", "ACTIVE"),
                    verified_at=time.time(),
                    rejection_reasons=[],
                )
                pool = AlphaPool(alpha_pool_path(args.tag))
                # Codex review R1-#1: see above — must upsert so ACTIVE
                # status persists when the alpha was previously stamped
                # UNSUBMITTED by the auto-persist path.
                upsert_result = pool.upsert(entry)
                pool_added = upsert_result in ("inserted", "updated")
        except Exception as exc:
            _emit({"ok": True, "alpha_id": args.alpha_id, "wq_response": wq_resp,
                   "pool_recording_error": str(exc)})
            return

    _emit({"ok": True, "alpha_id": args.alpha_id, "pool_added": pool_added,
           "wq_response": wq_resp})


def cmd_pool_list(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    entries = [e.to_dict() for e in pool]
    _emit({"ok": True, "tag": args.tag, "size": len(entries), "entries": entries})


def cmd_corr(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.client import session_from_env
    try:
        sess = session_from_env()
        corrs = sess.get_alpha_correlations(args.alpha_id)
        max_corr = max((float(c.get("correlation", 0)) for c in corrs), default=0.0)
        _emit({"ok": True, "alpha_id": args.alpha_id, "max_correlation": max_corr,
               "correlations": corrs[:20]})
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)


# Thin shims — the real implementations live in
# agent_market.wq_brain.submit_gates so notebooks/integration-tests can
# call the same gate chain without depending on the CLI.

def _check_local_jaccard_vs_active(
    tag: str,
    expr: str,
    *,
    threshold: float = 0.7,
    semantic_threshold: float = 0.85,
    candidate_sharpe: Optional[float] = None,
    candidate_fitness: Optional[float] = None,
    sharpe_margin: float = 0.10,
    override_mode: str = "sharpe_and_fitness",
    absolute_fitness_floor: float = 1.0,
) -> dict[str, Any]:
    from agent_market.wq_brain.submit_gates import local_jaccard_gate
    return local_jaccard_gate(
        tag, expr,
        threshold=threshold,
        semantic_threshold=semantic_threshold,
        candidate_sharpe=candidate_sharpe,
        candidate_fitness=candidate_fitness,
        sharpe_margin=sharpe_margin,
        override_mode=override_mode,
        absolute_fitness_floor=absolute_fitness_floor,
    )


def _summarize_rejection(reasons: list) -> str:
    from agent_market.wq_brain.submit_gates import summarize_rejection
    return summarize_rejection(reasons)


def _auto_fill_expr(tag: str, alpha_id: str) -> str:
    from agent_market.wq_brain.submit_gates import auto_fill_expr
    return auto_fill_expr(tag, alpha_id)


def _check_self_correlation(
    sess: Any,
    alpha_id: str,
    *,
    corr_max: float = 0.7,
    sharpe_margin: float = 0.10,
    tag: str = "",
) -> dict[str, Any]:
    from agent_market.wq_brain.submit_gates import self_correlation_gate
    return self_correlation_gate(
        sess, alpha_id,
        corr_max=corr_max, sharpe_margin=sharpe_margin, tag=tag,
    )


def cmd_pre_check(args: argparse.Namespace) -> None:
    """Pre-submission gate: WQ-aligned self-correlation + sharpe-margin rule."""
    from agent_market.wq_brain.client import session_from_env
    try:
        sess = session_from_env()
        result = _check_self_correlation(
            sess, args.alpha_id,
            corr_max=args.corr_max,
            sharpe_margin=args.sharpe_margin,
            tag=args.tag,
        )
        _emit({"ok": True, "alpha_id": args.alpha_id, **result},
              code=0 if result["accept"] else 2)
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)


def cmd_pre_check_local(args: argparse.Namespace) -> None:
    """Fast local pre-check: token + semantic jaccard vs ACTIVE pool entries.

    Use BEFORE simulating to avoid generating near-duplicates that WQ will
    reject. No API calls — purely local read of pool.json.

    Codex review R1-#6 + R2-#4: accepts the same candidate-metric /
    override flags as ``submit`` and ``pool submit-worker`` so the
    operator preview matches production. ``--alpha-id`` (optional) auto-
    fills the expr + candidate_sharpe + candidate_fitness from
    ``tried_log.jsonl`` — same lookup ``cmd_submit`` uses.
    """
    from agent_market.wq_brain.submit_gates import _finite_float, auto_fill_metrics

    alpha_id = getattr(args, "alpha_id", "") or ""
    expr = getattr(args, "expr", "") or ""
    cand_sh = _finite_float(getattr(args, "candidate_sharpe", None))
    cand_fi = _finite_float(getattr(args, "candidate_fitness", None))

    # Auto-fill from tried_log when --alpha-id is supplied
    if alpha_id and args.tag:
        meta = auto_fill_metrics(args.tag, alpha_id)
        if not expr:
            expr = meta.get("expr") or ""
        if cand_sh is None:
            cand_sh = _finite_float(meta.get("sharpe"))
        if cand_fi is None:
            cand_fi = _finite_float(meta.get("fitness"))
    if not expr:
        _emit({"ok": False, "error": "no expr provided and --alpha-id lookup empty"}, code=1)
        return

    result = _check_local_jaccard_vs_active(
        args.tag, expr,
        threshold=args.jaccard_max,
        semantic_threshold=args.semantic_max,
        candidate_sharpe=cand_sh,
        candidate_fitness=cand_fi,
        sharpe_margin=getattr(args, "sharpe_margin", 0.10),
        override_mode=getattr(args, "override_mode", "sharpe_and_fitness"),
        absolute_fitness_floor=getattr(args, "absolute_fitness_floor", 1.0),
    )
    # Strip non-finite numerics from emitted JSON (defensive — see R2-#5)
    for k in ("candidate_sharpe", "candidate_fitness"):
        result[k] = _finite_float(result.get(k))
    _emit({"ok": True, "alpha_id": alpha_id, **result, "expr": expr},
          code=0 if result["accept"] else 2)


def cmd_pool_resubmit(args: argparse.Namespace) -> None:
    """POST /alphas/{id}/submit for every pool entry that's not already ACTIVE.

    Codex review R4-#2 / R2 (project-quality loop): this is a LEGACY path
    that bypasses the gate stack used by ``pool submit-worker``
    (no quota reservation, no local-jaccard, no self-corr override, no
    outcome upsert). It is now refused by default — operator must pass
    ``--legacy-unsafe`` to opt in. Production should run
    ``pool submit-worker --tag <tag>`` instead.

    Even with ``--legacy-unsafe`` set, by default LOCAL_BLOCKED /
    SELF_CORR_BLOCKED entries are skipped so this command can't burn
    quota on entries the gate stack already rejected. Use
    ``--include-blocked`` to opt into retrying them after raising
    thresholds. ``--status`` filters by exact verified_status.
    """
    import time as _time
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    if not getattr(args, "legacy_unsafe", False):
        _emit({
            "ok": False,
            "rejected_by": "legacy_unsafe_gate",
            "hint": (
                "pool resubmit-all is a legacy path that bypasses quota / "
                "local-jaccard / self-corr / outcome persistence. Refused "
                "by default. Either: (a) run `pool submit-worker --tag "
                f"{getattr(args, 'tag', '<tag>')}` (recommended); or "
                "(b) re-run with `--legacy-unsafe` to opt into the bypass."
            ),
        }, code=2)
        return

    pool = AlphaPool(alpha_pool_path(args.tag))
    # Codex review R4-#2: filter blocked states unless --include-blocked.
    BLOCKED_STATES = {"LOCAL_BLOCKED", "SELF_CORR_BLOCKED"}
    raw_entries = list(pool.entries)
    skipped_blocked = 0
    if not getattr(args, "include_blocked", False):
        before = len(raw_entries)
        raw_entries = [
            e for e in raw_entries
            if getattr(e, "verified_status", "") not in BLOCKED_STATES
        ]
        skipped_blocked = before - len(raw_entries)
    # Optional --status filter (e.g. only retry UNSUBMITTED with reasons)
    status_filter = getattr(args, "status_filter", "") or ""
    if status_filter:
        raw_entries = [
            e for e in raw_entries
            if getattr(e, "verified_status", "") == status_filter
        ]
    # Sort by fitness desc — submit best alphas first; later near-duplicates
    # will be rejected by WQ self-corr but won't displace earlier winners.
    from agent_market.wq_brain.submit_gates import _finite_float as _fff
    entries = sorted(raw_entries, key=lambda e: -(_fff(getattr(e, "fitness", None)) or float("-inf")))
    sess = session_from_env()
    submitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    max_submit = getattr(args, "max", 0) or 0
    print(f"Resubmitting up to {max_submit or 'unlimited'} of {len(entries)} pool alphas "
          f"for tag={args.tag}", file=sys.stderr)
    for i, entry in enumerate(entries):
        # Codex review R2-#3: cap successful submit attempts.
        if max_submit > 0 and len(submitted) >= max_submit:
            break
        # Quick status pre-check — 1 GET, no correlations
        try:
            status = sess.get_alpha_status(entry.alpha_id)
        except Exception as exc:
            rejected.append({"alpha_id": entry.alpha_id, "error": f"status fetch: {exc}"[:200]})
            _time.sleep(args.polite_sleep)
            continue
        if status.get("status") == "ACTIVE" or status.get("date_submitted"):
            skipped.append({"alpha_id": entry.alpha_id, "reason": "already submitted"})
            _time.sleep(args.polite_sleep)
            continue
        # POST /submit — let WQ do the self-corr check
        try:
            resp = sess.submit_alpha(entry.alpha_id)
            submitted.append({
                "alpha_id": entry.alpha_id,
                "fitness": entry.fitness,
                "sharpe": entry.sharpe,
                "response": resp,
            })
        except Exception as exc:
            msg = str(exc)[:300]
            rejected.append({
                "alpha_id": entry.alpha_id,
                "fitness": entry.fitness,
                "sharpe": entry.sharpe,
                "error": msg,
                "is_self_corr": "Self-correlation" in msg or "correlation" in msg.lower(),
            })
        _time.sleep(args.polite_sleep)
        if (i + 1) % 5 == 0:
            print(f"... {i+1}/{len(entries)} (submitted={len(submitted)}, "
                  f"rejected={len(rejected)}, skipped={len(skipped)})", file=sys.stderr)
    _emit({
        "ok": True,
        "tag": args.tag,
        "pool_size": len(pool),
        "scanned_count": len(entries),
        "skipped_blocked_count": skipped_blocked,
        "status_filter": status_filter,
        "submitted_count": len(submitted),
        "rejected_count": len(rejected),
        "skipped_count": len(skipped),
        "submitted": submitted,
        "rejected": rejected[:20],
        "skipped": skipped[:5],
    })


def cmd_pool_status(args: argparse.Namespace) -> None:
    """Query WQ for each pool alpha's current status (UNSUBMITTED / ACTIVE / ...)."""
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    sess = session_from_env()
    by_status: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    print(f"Checking {len(pool)} pool alphas...", file=sys.stderr)
    for i, entry in enumerate(pool.entries):
        try:
            s = sess.get_alpha_status(entry.alpha_id)
            status = s.get("status") or "UNKNOWN"
            by_status[status] = by_status.get(status, 0) + 1
            rows.append({
                "alpha_id": entry.alpha_id, "status": status,
                "stage": s.get("stage"),
                "date_submitted": s.get("date_submitted"),
                "fitness": entry.fitness, "sharpe": entry.sharpe,
                "expr": entry.expr[:80],
            })
        except Exception as exc:
            rows.append({"alpha_id": entry.alpha_id, "error": str(exc)[:200]})
        if (i + 1) % 10 == 0:
            print(f"... {i+1}/{len(pool)}", file=sys.stderr)
    _emit({"ok": True, "tag": args.tag, "pool_size": len(pool),
           "summary_by_status": by_status, "rows": rows})


def cmd_pool_sync_status(args: argparse.Namespace) -> None:
    """Query WQ for each pool entry's actual status + IS check failures,
    write verified_status + rejection_reasons back to pool.json.

    For UNSUBMITTED entries with empty rejection_reasons, also probe
    POST /alphas/{id}/submit (which returns 403 + cached IS check failures
    for previously-rejected alphas) to populate the rejection details.
    GET /alphas/{id} does NOT include check results, only metrics.

    Codex review R4-#1: terminal local states (LOCAL_BLOCKED /
    SELF_CORR_BLOCKED) are NOT overwritten by WQ's view of the alpha
    (which would normally show "UNSUBMITTED" since the alpha never
    actually hit the submit endpoint). Pass ``--reset-local-blocks`` to
    force the overwrite when you've genuinely retired those gates and
    want to retry the entries.
    """
    import time as _t
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    PRESERVED_LOCAL_STATES = {"LOCAL_BLOCKED", "SELF_CORR_BLOCKED"}
    pool = AlphaPool(alpha_pool_path(args.tag))
    sess = session_from_env()
    by_status: dict[str, int] = {}
    rej_probed = 0
    preserved_count = 0
    # Codex review R3-CRIT: track every alpha_id this command authoritatively
    # touched so the final _save() doesn't silently revert intentional
    # demotions via the precedence merge (e.g. --reset-local-blocks moving
    # LOCAL_BLOCKED → UNSUBMITTED based on what WQ reports).
    authoritative_ids: set[str] = set()
    print(f"Syncing {len(pool)} pool alphas with WQ...", file=sys.stderr)
    for i, entry in enumerate(pool.entries):
        try:
            prior = getattr(entry, "verified_status", "")
            # Codex review R4-#1: preserve terminal local-blocked states
            # unless explicitly reset. WQ would tell us "UNSUBMITTED" for
            # these (they never reached the submit endpoint), and that
            # would erase the gate's terminal verdict and let the next
            # `submit-worker --status UNSUBMITTED` replay them.
            preserve = (
                prior in PRESERVED_LOCAL_STATES
                and not getattr(args, "reset_local_blocks", False)
            )
            if preserve:
                entry.verified_at = _t.time()
                by_status[prior] = by_status.get(prior, 0) + 1
                preserved_count += 1
                continue

            # Step 1: GET status
            url = f"{sess._api_base}/alphas/{entry.alpha_id}"
            data = sess._request_with_retry("GET", url, timeout=20).json()
            actual_status = data.get("status") or "UNKNOWN"
            entry.verified_status = actual_status
            entry.verified_at = _t.time()
            authoritative_ids.add(entry.alpha_id)
            by_status[actual_status] = by_status.get(actual_status, 0) + 1

            # Step 2: For UNSUBMITTED with no recorded reasons, probe submit
            existing_rj = entry.rejection_reasons or []
            if (actual_status == "UNSUBMITTED"
                and not existing_rj
                and args.probe_rejections):
                _t.sleep(args.polite_sleep)
                try:
                    submit_url = f"{sess._api_base}/alphas/{entry.alpha_id}/submit"
                    with sess._global_sem:
                        sub_resp = sess.post(submit_url, timeout=20)
                    if sub_resp.status_code in (400, 403):
                        try:
                            body = sub_resp.json()
                        except (ValueError, AttributeError):
                            body = {}
                        checks = (body.get("is") or {}).get("checks") or []
                        failed = [{"name": c.get("name"), "result": c.get("result"),
                                   "limit": c.get("limit"), "value": c.get("value")}
                                  for c in checks if c.get("result") == "FAIL"]
                        if failed:
                            entry.rejection_reasons = failed
                            rej_probed += 1
                except Exception as exc:
                    print(f"WARN: {entry.alpha_id} probe failed: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"WARN: {entry.alpha_id} sync failed: {exc}", file=sys.stderr)
        if (i + 1) % 5 == 0:
            print(f"... {i+1}/{len(pool)} ({rej_probed} rejections probed)", file=sys.stderr)
        _t.sleep(args.polite_sleep)
    # Codex R3-CRIT: pass authoritative_ids so precedence merge can't
    # silently revert intentional demotions from WQ's actual status.
    pool._save(authoritative_ids=authoritative_ids)  # type: ignore[attr-defined]
    _emit({
        "ok": True, "tag": args.tag,
        "pool_size": len(pool),
        "summary_by_status": by_status,
        "rejections_probed": rej_probed,
        "local_states_preserved": preserved_count,
        "authoritative_writes": len(authoritative_ids),
    })


def cmd_pool_dedup(args: argparse.Namespace) -> None:
    """Greedy dedup: keep highest-fitness alpha per token-similarity cluster.

    Two entries are considered the "same cluster" if their expression
    token-set Jaccard similarity ≥ threshold (default 0.85). For each cluster,
    keep the entry with highest fitness, drop the rest. Useful before
    re-submission campaigns to avoid burning slots on near-duplicates that
    WQ self-correlation will reject.
    """
    import re
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    # Sort: ACTIVE alphas FIRST (always-keep), then by fitness desc.
    # ACTIVE = WQ-confirmed earning alphas; never drop these in favor of
    # an unverified higher-fitness sibling.
    def _priority(e):
        is_active = getattr(e, "verified_status", "") == "ACTIVE"
        return (0 if is_active else 1, -e.fitness)
    entries = sorted(pool.entries, key=_priority)

    def tokens(expr: str) -> frozenset:
        return frozenset(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", expr.lower()))

    def jaccard(a: frozenset, b: frozenset) -> float:
        return len(a & b) / max(1, len(a | b))

    kept: list[Any] = []  # AlphaPoolEntry
    kept_tokens: list[frozenset] = []
    dropped: list[dict[str, Any]] = []
    for entry in entries:
        toks = tokens(entry.expr)
        max_sim = 0.0
        winner_id = ""
        for k_toks, k_entry in zip(kept_tokens, kept):
            sim = jaccard(toks, k_toks)
            if sim > max_sim:
                max_sim = sim
                winner_id = k_entry.alpha_id
        if max_sim >= args.threshold:
            dropped.append({
                "alpha_id": entry.alpha_id,
                "fitness": entry.fitness,
                "max_jaccard": round(max_sim, 3),
                "winner_id": winner_id,
            })
        else:
            kept.append(entry)
            kept_tokens.append(toks)

    if not args.dry_run:
        # Codex review R2-CRIT: must use replace_all (not _save with default
        # merge_missing=True), or the dropped duplicates would be re-read
        # from disk inside _save and silently re-appended.
        pool.replace_all(kept)

    _emit({
        "ok": True,
        "tag": args.tag,
        "threshold": args.threshold,
        "before": len(entries),
        "after_kept": len(kept),
        "dropped_count": len(dropped),
        "dry_run": args.dry_run,
        "kept_top_5": [{"alpha_id": k.alpha_id, "fi": k.fitness, "expr": k.expr[:80]}
                       for k in kept[:5]],
        "dropped_sample": dropped[:10],
    })


def cmd_pool_backfill(args: argparse.Namespace) -> None:
    """Backfill pool.json `expr` field from tried_exprs.jsonl by alpha_id."""
    from agent_market.wq_brain.paths import alpha_pool_path, tried_exprs_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.tried_log import read_tried
    pool = AlphaPool(alpha_pool_path(args.tag))
    tried = read_tried(tried_exprs_path(args.tag), tail=10000)
    by_id: dict[str, str] = {}
    for r in tried:
        aid = r.get("alpha_id")
        expr = r.get("expr")
        if aid and expr and aid not in by_id:
            by_id[aid] = expr

    fixed = 0
    skipped = 0
    for entry in pool.entries:
        if entry.expr and entry.expr != "(submitted via CLI)":
            skipped += 1
            continue
        new_expr = by_id.get(entry.alpha_id)
        if new_expr:
            entry.expr = new_expr
            fixed += 1
    pool._save()  # type: ignore[attr-defined]
    _emit({
        "ok": True, "tag": args.tag,
        "pool_size": len(pool),
        "fixed": fixed, "skipped": skipped,
        "still_missing": sum(1 for e in pool.entries if not e.expr or e.expr == "(submitted via CLI)"),
    })


def cmd_pool_salvage(args: argparse.Namespace) -> None:
    """Backfill the pool with high-fitness candidates from tried_exprs.jsonl
    that were never submitted (or never recorded as ACTIVE/REJECTED).

    Defaults: sharpe ≥ 1.25 AND fitness ≥ 1.0 (the WQ ACTIVE quality gate).

    The agent's LLM session can drop high-fi candidates if it runs out of
    turns / quits early / abandons after a pre-check warning. Production
    data showed 70% loss rate of fi≥1.0 candidates. This CLI reads the
    tried_log, finds alpha_ids that meet quality but aren't in the pool,
    and writes them as UNSUBMITTED so a later submit pass can attempt them.
    """
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path, tried_exprs_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.tried_log import read_tried

    pool = AlphaPool(alpha_pool_path(args.tag))
    pool_ids = {e.alpha_id for e in pool.entries}
    tried = read_tried(tried_exprs_path(args.tag), tail=20000)

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r in tried:
        if r.get("status") != "COMPLETE":
            continue
        aid = r.get("alpha_id") or ""
        if not aid or aid in pool_ids or aid in seen_ids:
            continue
        sh = r.get("sharpe")
        fi = r.get("fitness")
        if sh is None or fi is None:
            continue
        if float(sh) < args.sharpe_min or float(fi) < args.fitness_min:
            continue
        seen_ids.add(aid)
        candidates.append(r)

    candidates.sort(key=lambda r: -float(r["fitness"]))
    if args.top_n > 0:
        candidates = candidates[: args.top_n]

    if args.dry_run:
        _emit({
            "ok": True, "tag": args.tag, "dry_run": True,
            "pool_before": len(pool),
            "would_add": len(candidates),
            "thresholds": {"sharpe_min": args.sharpe_min,
                            "fitness_min": args.fitness_min},
            "top_5_preview": [
                {"alpha_id": r["alpha_id"], "sh": r["sharpe"],
                 "fi": r["fitness"], "to": r.get("turnover"),
                 "expr": (r.get("expr") or "")[:90]}
                for r in candidates[:5]
            ],
        })
        return

    added = 0
    for r in candidates:
        try:
            entry = AlphaPoolEntry(
                alpha_id=r["alpha_id"],
                expr=r.get("expr") or "",
                settings_dict={
                    "region": r.get("region", "USA"),
                    "universe": r.get("universe", "TOP3000"),
                    "decay": r.get("decay", 6),
                },
                sharpe=float(r["sharpe"]),
                fitness=float(r["fitness"]),
                returns=float(r.get("returns") or 0.0),
                turnover=float(r.get("turnover") or 0.0),
                tag=args.tag,
                source="salvage",
                verified_status="UNSUBMITTED",
                verified_at=0.0,
                rejection_reasons=[],
            )
            if pool.add(entry):
                added += 1
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("salvage skip %s: %s", r.get("alpha_id"), exc)
            continue

    _emit({
        "ok": True, "tag": args.tag,
        "pool_before": len(pool) - added,
        "pool_after": len(pool),
        "salvaged": added,
        "skipped_already_in_pool": len(seen_ids) - added,
        "thresholds": {"sharpe_min": args.sharpe_min,
                        "fitness_min": args.fitness_min},
        "hint": (
            # Codex review R2-#3: was pointing at `resubmit-all --status/--max`
            # which never existed. The replacement is `pool submit-worker`,
            # which has the full local + remote gate stack and quota
            # accounting; resubmit-all is now legacy.
            f"Now run `wq_brain pool submit-worker --tag {args.tag} "
            f"--status UNSUBMITTED --max N --one-per-cluster` to attempt "
            "WQ submission of the salvaged candidates with the full gate stack."
        ),
    })


def cmd_pool_submit_worker(args: argparse.Namespace) -> None:
    """Sharpe-clustered submit worker for UNSUBMITTED pool entries.

    Design (per Codex Round 3 review):

      1. Filter pool by ``--status`` (default UNSUBMITTED).
      2. Optionally cluster candidates by operator skeleton; keep the
         highest-fitness representative per cluster (avoids burning quota
         on N near-duplicates of the same structural template).
      3. For each pick: run self-correlation pre-check (fail-CLOSED on
         infra error), then submit if accepted.
      4. Upsert pool with the verified outcome (ACTIVE / REJECTED /
         UNSUBMITTED + rejection_reasons).

    Each successful submit consumes 1 submit-quota slot via reserve_action.
    Reservation is refunded on infra error / pre-check policy reject.
    """
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    from agent_market.wq_brain.prompt_builder import _operator_skeleton
    from agent_market.wq_brain.quota_monitor import release_action, reserve_action
    from agent_market.wq_brain.submit_gates import GateInfraError

    from agent_market.wq_brain.crossover import infer_family
    from agent_market.wq_brain.prompt_builder import (
        _FIELD_KIND, _extract_field_set,
    )
    from agent_market.wq_brain.submit_gates import _finite_float as _finite_float_for_sort

    pool = AlphaPool(alpha_pool_path(args.tag))
    targets = [e for e in pool.entries
               if getattr(e, "verified_status", "") == args.status]
    if not targets:
        _emit({
            "ok": True, "tag": args.tag, "status": args.status,
            "n_targets": 0, "submitted": 0, "active": 0, "rejected": 0,
            "infra_blocked": 0, "policy_blocked": 0,
            "hint": f"No pool entries with verified_status={args.status!r}",
        })
        return

    # Codex review R3-#5: sort defends against dirty pool rows (None/NaN/
    # string fitness). _finite_float returns None on bad input, so we fall
    # back to -inf to keep them at the bottom of priority.
    targets.sort(key=lambda e: -(_finite_float_for_sort(getattr(e, "fitness", None)) or float("-inf")))

    # Codex review R3-#2: cluster key now mirrors the slot logic without
    # exact fields — `(family, skeleton, field_kinds)` keeps the structural
    # diversity the agent prompt encourages. Bare `skeleton` would collapse
    # `rank(close)`, `rank(sales/assets)`, and `rank(open-high)` all under
    # `rankx1`, suppressing the family/field signal we want to surface.
    if args.one_per_cluster:
        seen_clusters: set[tuple] = set()
        clustered: list[AlphaPoolEntry] = []
        for e in targets:
            skel = _operator_skeleton(e.expr or "") or f"_unique_{e.alpha_id}"
            family = infer_family(e.expr or "") if e.expr else "_unknown"
            kinds = frozenset(_FIELD_KIND.get(f, "other")
                              for f in _extract_field_set(e.expr or ""))
            cluster_key = (family, skel, kinds)
            if cluster_key in seen_clusters:
                continue
            seen_clusters.add(cluster_key)
            clustered.append(e)
        targets = clustered

    # Codex review R2-#1 + R3-#3: --max is a SUBMIT budget, not a scan
    # budget. --scan-limit is the upper bound on candidates evaluated.
    # When operator passes --scan-limit explicitly we honor it exactly
    # (never override with max*5). Auto-bump only when the flag is unset.
    scan_limit_arg = getattr(args, "scan_limit", None)
    if scan_limit_arg is None:
        scan_limit = max(200, args.max * 5) if args.max > 0 else 200
    else:
        scan_limit = scan_limit_arg
    if scan_limit > 0:
        targets = targets[:scan_limit]

    # Gate-shaped args read defensively so test fixtures with minimal
    # Namespace can still drive the worker.
    from agent_market.wq_brain.submit_gates import _finite_float
    jaccard_max = getattr(args, "jaccard_max", 0.7)
    semantic_max = getattr(args, "semantic_max", 0.85)
    sharpe_margin_arg = getattr(args, "sharpe_margin", 0.10)
    override_mode_arg = getattr(args, "override_mode", "sharpe_and_fitness")
    fitness_floor_arg = getattr(args, "absolute_fitness_floor", 1.0)

    # Codex review R2-#4: dry-run now actually evaluates the local gate so
    # operators can preview block/override/projected-submit counts without
    # spending quota or hitting WQ.
    if args.dry_run:
        # Codex review R3-#4: evaluate ALL scanned targets for aggregate
        # counts so projected_submit / would_block reflect run-level
        # projections, not just the first 20 preview rows. Render only the
        # first --dry-run-limit (default 20) into the preview list.
        dry_run_limit = getattr(args, "dry_run_limit", 20)
        previews: list[dict[str, Any]] = []
        would_local_block = 0
        would_override = 0
        projected_submit = 0
        max_submit_target = args.max if args.max > 0 else len(targets)
        for idx, e in enumerate(targets):
            local_check = _check_local_jaccard_vs_active(
                args.tag, e.expr or "",
                threshold=jaccard_max,
                semantic_threshold=semantic_max,
                candidate_sharpe=_finite_float(getattr(e, "sharpe", None)),
                candidate_fitness=_finite_float(getattr(e, "fitness", None)),
                sharpe_margin=sharpe_margin_arg,
                override_mode=override_mode_arg,
                absolute_fitness_floor=fitness_floor_arg,
            )
            if not local_check["accept"]:
                would_local_block += 1
                outcome = "local_block"
            else:
                if local_check.get("override_applied"):
                    would_override += 1
                if projected_submit < max_submit_target:
                    projected_submit += 1
                outcome = ("local_accept_override" if local_check.get("override_applied")
                           else "local_accept")
            if idx < dry_run_limit:
                previews.append({
                    "alpha_id": e.alpha_id, "fi": e.fitness, "sh": e.sharpe,
                    "skeleton": _operator_skeleton(e.expr or ""),
                    "expr": (e.expr or "")[:90],
                    "would": outcome,
                    "blocker_count": local_check.get("blocker_count", 0),
                    "override_applied": bool(local_check.get("override_applied")),
                })
        _emit({
            "ok": True, "tag": args.tag, "dry_run": True,
            "status_filter": args.status, "one_per_cluster": args.one_per_cluster,
            "n_targets": len(targets),
            "would_local_block": would_local_block,
            "would_override": would_override,
            "projected_submit_attempts": projected_submit,
            "scan_limit": scan_limit,
            "max_submit": args.max,
            "dry_run_limit": dry_run_limit,
            "preview": previews,
        })
        return

    sess = session_from_env()
    submitted = 0
    active = 0
    rejected = 0
    infra_blocked = 0
    policy_blocked = 0
    local_blocked = 0
    local_overrides = 0
    outcomes: list[dict[str, Any]] = []

    for e in targets:
        # Codex review R2-#1: stop the loop once we've actually submitted
        # `--max` candidates. `submitted` increments only on successful WQ
        # submit_alpha; quota_block / infra_block break out separately.
        if args.max > 0 and submitted >= args.max:
            break
        # Pre-check 0 (LOCAL): structural-proxy jaccard vs ACTIVE pool.
        # Cheap, no quota cost. Sharpe-margin override lets strictly-better
        # alphas through structural near-duplicates — heuristic, NOT WQ's
        # signal-correlation rule. Codex review #5: coerce metrics defensively
        # so a stale pool entry with NaN/None/string sharpe never crashes
        # the worker.
        local_check = _check_local_jaccard_vs_active(
            args.tag, e.expr or "",
            threshold=jaccard_max,
            semantic_threshold=semantic_max,
            candidate_sharpe=_finite_float(getattr(e, "sharpe", None)),
            candidate_fitness=_finite_float(getattr(e, "fitness", None)),
            sharpe_margin=sharpe_margin_arg,
            override_mode=override_mode_arg,
            absolute_fitness_floor=fitness_floor_arg,
        )
        if not local_check["accept"]:
            # Codex review #3: stamp LOCAL_BLOCKED so the default
            # `--status UNSUBMITTED` filter doesn't re-pick this entry on
            # the next run. Operators can retry with `--status LOCAL_BLOCKED`
            # after raising thresholds or relaxing the override rule.
            e.verified_status = "LOCAL_BLOCKED"
            e.rejection_reasons = [{
                "name": "local_jaccard",
                "value": local_check.get("max_jaccard"),
                "limit": jaccard_max,
                "reason": local_check.get("reason", ""),
            }]
            e.verified_at = time.time()
            pool.upsert(e)
            local_blocked += 1
            outcomes.append({"alpha_id": e.alpha_id, "result": "local_block",
                              "reason": local_check.get("reason", "")})
            continue
        if local_check.get("override_applied"):
            local_overrides += 1

        # Reserve quota up-front; refund on early-out
        q = reserve_action("submit")
        if q["status"] == "block":
            outcomes.append({"alpha_id": e.alpha_id, "result": "quota_block",
                              "remaining": q.get("remaining")})
            break
        reserved_day = q["day"]

        # Pre-check 1 (REMOTE): WQ self-correlation gate (with sharpe-margin override)
        try:
            check = _check_self_correlation(
                sess, e.alpha_id,
                corr_max=args.corr_max,
                sharpe_margin=args.sharpe_margin,
                tag=args.tag,
            )
            if not check["accept"]:
                # POLICY reject — refund slot + persist new state.
                # Codex review R2-#2: stamp SELF_CORR_BLOCKED (not UNSUBMITTED)
                # so the next worker run with default --status UNSUBMITTED
                # filter doesn't replay the same self-corr block. Operators
                # can opt-in retry via --status SELF_CORR_BLOCKED.
                try: release_action("submit", day=reserved_day)
                except OSError: pass
                e.verified_status = "SELF_CORR_BLOCKED"
                e.rejection_reasons = [{"name": "self_correlation",
                                        "value": check.get("max_correlation"),
                                        "limit": args.corr_max,
                                        "reason": check.get("reason", "")}]
                e.verified_at = time.time()
                pool.upsert(e)
                policy_blocked += 1
                outcomes.append({"alpha_id": e.alpha_id, "result": "policy_block",
                                  "reason": check.get("reason", "")})
                continue
        except GateInfraError as exc:
            try: release_action("submit", day=reserved_day)
            except OSError: pass
            infra_blocked += 1
            outcomes.append({"alpha_id": e.alpha_id, "result": "infra_block",
                              "error": str(exc)})
            if not args.continue_on_infra:
                break
            continue

        # Actual submit
        try:
            wq_resp = sess.submit_alpha(e.alpha_id,
                                          verify_after_sec=args.verify_after_sec)
            submitted += 1
        except Exception as exc:
            try: release_action("submit", day=reserved_day)
            except OSError: pass
            outcomes.append({"alpha_id": e.alpha_id, "result": "submit_error",
                              "error": str(exc)})
            continue

        # Persist outcome via upsert
        new_status = wq_resp.get("verified_status") or "UNSUBMITTED"
        e.verified_status = new_status
        e.verified_at = time.time()
        e.rejection_reasons = wq_resp.get("rejection_reasons") or []
        pool.upsert(e)

        if new_status == "ACTIVE":
            active += 1
        else:
            rejected += 1
        outcomes.append({"alpha_id": e.alpha_id, "result": new_status,
                          "fi": e.fitness, "sh": e.sharpe})

    _emit({
        "ok": True, "tag": args.tag,
        "status_filter": args.status,
        "one_per_cluster": args.one_per_cluster,
        "n_targets": len(targets),
        "submitted": submitted, "active": active, "rejected": rejected,
        "infra_blocked": infra_blocked, "policy_blocked": policy_blocked,
        "local_blocked": local_blocked, "local_overrides": local_overrides,
        "outcomes_sample": outcomes[:20],
    })


def cmd_fetch_data(args: argparse.Namespace) -> None:
    """Bulk-fetch US stock OHLCV + sectors into local parquet cache.

    Default backend is Stooq (free, no API key); set --backend yfinance|auto
    or env WQB_DATA_BACKEND to override.
    """
    from agent_market.wq_brain.data_loader import fetch_data, load_tickers
    tickers = load_tickers(Path(args.tickers_file) if args.tickers_file else None)
    backend = args.backend or os.environ.get("WQB_DATA_BACKEND") or "stooq"
    print(f"Fetching {len(tickers)} tickers from {args.start} to {args.end} "
          f"(backend={backend}, polite_sleep={args.polite_sleep}s)", file=sys.stderr)
    summary = fetch_data(
        tickers, args.start, args.end,
        skip_sectors=args.skip_sectors,
        polite_sleep=args.polite_sleep,
        backend=args.backend,
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_audit_data(args: argparse.Namespace) -> None:
    """Audit OHLCV cache for split/dividend/survivor/outlier issues.

    Five checks: ohlc_invariant, split_sanity, ticker_reuse, survivor_bias,
    outliers. Writes _audit.json + _audit.md to the cache dir; never mutates
    the OHLCV data itself.
    """
    from agent_market.wq_brain.data_audit import run_audit, write_audit_artifacts
    from agent_market.wq_brain.data_loader import load_cached_ohlcv

    df = load_cached_ohlcv()
    if df is None or len(df) == 0:
        _emit({"ok": False, "error": "no cached OHLCV; run kaggle-fetch + kaggle-import first"}, code=1)
        return
    if args.ticker:
        df = df[df.index.get_level_values("ticker") == args.ticker.upper()]
        if len(df) == 0:
            _emit({"ok": False, "error": f"ticker {args.ticker!r} not in cache"}, code=1)
            return
    report = run_audit(df, sample_size=args.sample_size)
    paths = write_audit_artifacts(report)
    out = report.to_dict()
    out["ok"] = True
    out["artifacts"] = paths
    _emit(out)


def cmd_calibrate_local(args: argparse.Namespace) -> None:
    """Calibrate local-simulate's wq_fitness gate against the tried_log ledger.

    Picks the top-N COMPLETE remote results, re-runs each through
    simulate_expression_locally on the cached OHLCV, computes
    Pearson/Spearman/RMSE between local & remote fitness, and saves the
    F1-maximising threshold to artifacts/wq_brain/calibration/{tag}/.
    """
    from agent_market.wq_brain.calibration import (
        calibrate_local_threshold,
        report_path,
        threshold_path,
    )

    def _progress(i, total, expr):
        print(f"  [{i}/{total}] local-simulate: {(expr or '')[:80]}", file=sys.stderr, flush=True)

    try:
        result = calibrate_local_threshold(
            args.tag,
            top_n=args.top_n,
            max_time_sec=args.max_time_sec,
            save=not args.no_save,
            progress_cb=_progress,
            min_samples=args.min_samples,
        )
    except Exception as exc:
        _emit({"ok": False, "tag": args.tag, "error": str(exc)}, code=1)
        return

    out = result.to_dict()
    out["ok"] = True
    out["threshold_path"] = str(threshold_path(args.tag))
    out["report_path"] = str(report_path(args.tag))
    _emit(out)


def cmd_seed_calibration(args: argparse.Namespace) -> None:
    """Seed calibration/{tag}/threshold.json from a JSONL of pre-computed samples.

    Bypasses the slow local-simulate loop — useful when the caller has 30+
    remote alphas with both local_fitness (e.g. from a previous local run)
    and passes_remote (remote fitness ≥ 1.0) already collected, and just
    wants the F1-maximising threshold.
    """
    from agent_market.wq_brain.calibration import (
        report_path, seed_calibration_from_samples, threshold_path,
    )

    src = Path(args.from_samples)
    if not src.exists():
        _emit({"ok": False, "error": f"file not found: {src}"}, code=2)
        return
    samples: list[dict[str, Any]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except ValueError as exc:
            _emit({"ok": False, "error": f"bad JSONL row: {exc}"}, code=2)
            return

    try:
        result = seed_calibration_from_samples(
            args.tag, samples,
            save=not args.no_save,
            min_samples=args.min_samples,
        )
    except Exception as exc:
        _emit({"ok": False, "tag": args.tag, "error": str(exc)}, code=1)
        return

    out = result.to_dict()
    out["ok"] = True
    out["threshold_path"] = str(threshold_path(args.tag))
    out["report_path"] = str(report_path(args.tag))
    _emit(out)


def cmd_kaggle_fetch(args: argparse.Namespace) -> None:
    """Download a Kaggle stock dataset ZIP (and auto-extract by default)."""
    from agent_market.wq_brain.kaggle_loader import (
        KaggleCredentialsError,
        download_kaggle_dataset,
        extract_kaggle_zip,
    )
    from agent_market.wq_brain.paths import wq_brain_root

    dest_dir = Path(args.dest_dir) if args.dest_dir else wq_brain_root() / "data" / "kaggle"
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = download_kaggle_dataset(args.dataset, dest_dir, force=args.force)
    except KaggleCredentialsError as exc:
        _emit({"ok": False, "error": str(exc)}, code=2)
        return

    out: dict[str, Any] = {
        "ok": True,
        "dataset": args.dataset,
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
    }
    if not args.no_extract:
        extract_dir = extract_kaggle_zip(zip_path, dest_dir)
        out["extract_dir"] = str(extract_dir)
        out["files"] = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
    _emit(out)


def cmd_kaggle_import(args: argparse.Namespace) -> None:
    """Parse an already-fetched Kaggle dataset and merge into ohlcv.parquet."""
    from agent_market.wq_brain.kaggle_loader import import_kaggle_to_cache
    from agent_market.wq_brain.paths import wq_brain_root

    dest_dir = Path(args.dest_dir) if args.dest_dir else wq_brain_root() / "data" / "kaggle"
    safe_name = args.dataset.replace("/", "__")
    extract_dir = dest_dir / safe_name
    if not extract_dir.is_dir():
        _emit({
            "ok": False,
            "error": f"extract dir not found: {extract_dir}; run kaggle-fetch first",
        }, code=2)
        return

    column_map: Optional[dict[str, str]] = None
    if args.column_map:
        try:
            column_map = json.loads(args.column_map)
        except ValueError as exc:
            _emit({"ok": False, "error": f"--column-map invalid JSON: {exc}"}, code=2)
            return

    ticker_col = args.ticker_col or None  # empty string → None
    try:
        summary = import_kaggle_to_cache(
            extract_dir,
            files_glob=args.files_glob,
            ticker_col=ticker_col,
            date_col=args.date_col,
            ticker_from_filename=args.ticker_from_filename,
            column_map=column_map,
            split_adjust_from_col=args.split_adjust_from or None,
            audit=not getattr(args, "no_audit", False),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)
        return
    _emit(summary)


def cmd_update_data(args: argparse.Namespace) -> None:
    """Incremental daily update of the OHLCV cache."""
    import time as _t
    from agent_market.wq_brain.data_loader import (
        fetch_data, load_cached_ohlcv, load_tickers, metadata_path,
    )
    tickers = load_tickers(Path(args.tickers_file) if args.tickers_file else None)

    # Determine start = last cached date + 1, end = today
    cached = load_cached_ohlcv()
    if not cached.empty:
        last_date = cached.index.get_level_values("date").max()
        start = (last_date + _t.timedelta(days=1) if hasattr(last_date, "year")
                 else args.start).strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else args.start
    else:
        start = args.start

    end = args.end or _t.strftime("%Y-%m-%d", _t.gmtime())
    print(f"Updating {len(tickers)} tickers from {start} to {end}", file=sys.stderr)
    summary = fetch_data(tickers, start, end, skip_sectors=True, backend=args.backend)
    print(json.dumps(summary, indent=2, default=str))


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_local_sim_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"claims": [], "active": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"claims": [], "active": []}
    return {
        "claims": list(raw.get("claims") or []),
        "active": list(raw.get("active") or []),
    }


def _write_local_sim_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _locked_local_sim_state_update(run_dir: Path, fn):
    lock_path = run_dir / ".local_sim_budget.lock"
    state_path = run_dir / "local_sim_budget.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+", encoding="utf-8") as lockf:
        try:
            import fcntl
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        state = _read_local_sim_state(state_path)
        state["active"] = [
            row for row in state.get("active", [])
            if _pid_alive(int(row.get("pid") or 0))
        ]
        result = fn(state)
        _write_local_sim_state(state_path, state)
        try:
            import fcntl
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        return result


def _mark_agent_local_sim_status(
    expr: str,
    status: str,
    *,
    error: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> None:
    run_dir_raw = os.environ.get("WQB_RUN_DIR", "")
    if not run_dir_raw:
        return
    pid = os.getpid()

    def _mark(state: dict[str, Any]) -> None:
        claims = state.get("claims", [])
        for row in reversed(claims):
            if str(row.get("expr") or "") == expr and int(row.get("pid") or 0) == pid:
                row["status"] = status
                row["finished_ts"] = time.time()
                if error:
                    row["error"] = error
                if extra:
                    row.update(extra)
                break
        state["claims"] = claims

    _locked_local_sim_state_update(Path(run_dir_raw), _mark)


def _agent_local_sim_gate_error(expr: str) -> str:
    if os.environ.get("WQB_AGENT_REQUIRE_LOCAL_SIM", "").lower() not in {"1", "true", "yes"}:
        return ""
    run_dir_raw = os.environ.get("WQB_RUN_DIR", "")
    if not run_dir_raw:
        return ""

    def _check(state: dict[str, Any]) -> str:
        for row in reversed(state.get("claims", [])):
            if str(row.get("expr") or "") == expr:
                status = str(row.get("status") or "")
                if status == "passed":
                    return ""
                return (
                    "agent local-simulate gate not passed for this expression "
                    f"(status={status or 'running'}); do not call remote simulate"
                )
        return (
            "agent local-simulate gate missing for this expression; run "
            "`local-simulate` first and only simulate the exact expression "
            "that passes the local gate"
        )

    return _locked_local_sim_state_update(Path(run_dir_raw), _check)


@contextlib.contextmanager
def _agent_local_sim_slot(expr: str):
    """Bound local-simulate fan-out inside compact autonomous agent runs."""
    run_dir_raw = os.environ.get("WQB_RUN_DIR", "")
    limit = _env_int("WQB_AGENT_LOCAL_SIM_LIMIT")
    max_concurrent = _env_int("WQB_AGENT_LOCAL_SIM_MAX_CONCURRENT")
    if not run_dir_raw or (limit <= 0 and max_concurrent <= 0):
        yield
        return

    run_dir = Path(run_dir_raw)
    pid = os.getpid()

    def _claim(state: dict[str, Any]) -> None:
        active = state.get("active", [])
        if max_concurrent > 0 and len(active) >= max_concurrent:
            raise RuntimeError(
                "agent local-simulate concurrency limit reached; wait for the "
                "running local-simulate to finish instead of launching another"
            )
        claims = state.get("claims", [])
        claimed_exprs = {str(row.get("expr") or "") for row in claims}
        if limit > 0 and expr not in claimed_exprs and len(claims) >= limit:
            raise RuntimeError(
                f"agent local-simulate budget exhausted for this compact loop "
                f"(limit={limit}); write summary.md or proceed with prior results"
            )
        if expr not in claimed_exprs:
            claims.append({"expr": expr, "pid": pid, "ts": time.time(), "status": "running"})
        active.append({"expr": expr, "pid": pid, "ts": time.time()})
        state["claims"] = claims
        state["active"] = active

    def _release(state: dict[str, Any]) -> None:
        state["active"] = [
            row for row in state.get("active", [])
            if int(row.get("pid") or 0) != pid
        ]

    _locked_local_sim_state_update(run_dir, _claim)
    try:
        yield
    finally:
        _locked_local_sim_state_update(run_dir, _release)


@contextlib.contextmanager
def _agent_local_sim_time_limit():
    seconds = _env_float("WQB_AGENT_LOCAL_SIM_TIMEOUT_SEC")
    if seconds <= 0:
        yield
        return
    try:
        import signal
    except ImportError:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):
        raise TimeoutError(f"agent local-simulate timed out after {seconds:g}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def cmd_local_simulate(args: argparse.Namespace) -> None:
    """Run wq_simulate against cached OHLCV — no WQ API call, pure local.

    Pass ``--tag X`` to use the calibrated fitness gate for that tag (if
    a calibration has been run); otherwise the legacy 0.5 default applies.
    """
    from agent_market.wq_brain.local_sim import simulate_expression_locally
    try:
        with _agent_local_sim_slot(args.expr):
            with _agent_local_sim_time_limit():
                result = simulate_expression_locally(
                    args.expr,
                    rebalance_freq=args.rebalance_freq,
                    tag=args.tag or None,
                    fitness_gate=args.fitness_gate,
                )
        passes_local_gate = bool(result.raw.get("passes_local_gate"))
        _mark_agent_local_sim_status(
            args.expr,
            "passed" if passes_local_gate else "failed_gate",
            extra={
                "passes_local_gate": passes_local_gate,
                "wq_sharpe": result.wq_sharpe,
                "wq_fitness": result.wq_fitness,
                "wq_turnover": result.wq_turnover,
            },
        )
        _emit({
            "ok": True,
            "expr": result.expr,
            "wq_sharpe": result.wq_sharpe,
            "wq_fitness": result.wq_fitness,
            "wq_turnover": result.wq_turnover,
            "wq_returns": result.wq_returns,
            "submittable": result.submittable,
            "passes_local_gate": result.raw.get("passes_local_gate"),
            "fitness_gate": result.raw.get("fitness_gate"),
            "rating": result.rating,
            "raw": result.raw,
        })
    except Exception as exc:
        _mark_agent_local_sim_status(args.expr, "error", error=str(exc))
        _emit({"ok": False, "expr": args.expr, "error": str(exc)}, code=1)


def cmd_anti_overfit(args: argparse.Namespace) -> None:
    """Run 4-layer anti-overfit detection against cached OHLCV."""
    from agent_market.wq_brain.anti_overfit import run_anti_overfit_for_expression
    try:
        result = run_anti_overfit_for_expression(args.expr, holding_period=args.holding_period)
        _emit({"ok": True, "expr": args.expr, **result})
    except Exception as exc:
        _emit({"ok": False, "expr": args.expr, "error": str(exc)}, code=1)


def cmd_score(args: argparse.Namespace) -> None:
    """Score a single SimulationResult-like input."""
    from agent_market.wq_brain.dtypes import SimulationResult
    from agent_market.wq_brain.scoring import score_simulation_result
    sim = SimulationResult(
        sharpe=args.sharpe,
        fitness=args.fitness,
        turnover=args.turnover,
        returns=args.returns,
        status=args.status,
    )
    out = score_simulation_result(sim)
    _emit({"ok": True, "expr": args.expr or "", **out.to_dict()})


def cmd_search_arxiv(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.research_tools import search_arxiv
    try:
        out = search_arxiv(
            args.query,
            max_results=args.max,
            categories=args.category,
            sort_by=args.sort,
            sort_order=args.sort_order,
            raw_query=args.raw_query,
        )
    except Exception as exc:
        _emit({"ok": False, "error": f"arxiv search failed: {exc}"}, code=1)
    _emit(out)


def cmd_search_papers(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.research_tools import search_papers
    try:
        out = search_papers(
            args.query,
            max_results=args.max,
            sources=args.source,
            arxiv_categories=args.category,
            arxiv_sort_by=args.arxiv_sort,
            year=args.year,
            fields_of_study=args.fields_of_study,
            min_citation_count=args.min_citations,
        )
    except Exception as exc:
        _emit({"ok": False, "error": f"paper search failed: {exc}"}, code=1)
    _emit(out, code=0 if out.get("ok") else 1)


def cmd_math(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.symbolic_math import symbolic_math
    try:
        out = symbolic_math(
            args.operation,
            args.expr,
            var=args.var,
            solve_for=args.solve_for,
            point=args.point,
            order=args.order,
        )
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)
    _emit(out)


def cmd_docs(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.operators import operators_prompt_block
    if args.topic == "operators":
        _emit({"ok": True, "topic": "operators", "content": operators_prompt_block()})
    else:
        _emit({"ok": False, "error": f"unknown topic: {args.topic}"}, code=1)


def cmd_colony_status(args: argparse.Namespace) -> None:
    """Print best-so-far + cache summary for a colony tag."""
    from agent_market.wq_brain.colony import colony_run_dir
    from agent_market.wq_brain.colony_state import list_panel_bests
    from agent_market.wq_brain.pheromone_cache import (
        cache_path,
        classify_uncertainty,
        read_cache,
    )

    manifest_path = colony_run_dir(args.colony_tag) / "manifest.json"
    manifest: Any = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError:
            manifest = None
    bests = [b.to_dict() for b in list_panel_bests(args.colony_tag)]
    cache = read_cache(args.colony_tag)
    cache_summary = {
        alt: {
            "size": len(bucket),
            "executed": sum(
                1 for l in bucket
                if classify_uncertainty(l) == "executed_verified"
            ),
            "low_support": sum(
                1 for l in bucket
                if classify_uncertainty(l) == "low_support"
            ),
            "disagreement_prone": sum(
                1 for l in bucket
                if classify_uncertainty(l) == "disagreement_prone"
            ),
        }
        for alt, bucket in cache.items()
    }
    _emit(
        {
            "ok": True,
            "colony_tag": args.colony_tag,
            "cache_path": str(cache_path(args.colony_tag)),
            "manifest": manifest,
            "panel_bests": bests,
            "cache_summary": cache_summary,
        }
    )


def cmd_colony_pheromones_list(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.pheromone_cache import (
        classify_uncertainty,
        read_cache,
        score,
    )

    cache = read_cache(args.colony_tag)
    rows: list[dict[str, Any]] = []
    for alt, bucket in cache.items():
        for link in bucket:
            rows.append(
                {
                    "altitude": alt,
                    "alpha_id": link.alpha_id,
                    "expr": link.expr,
                    "sharpe": link.sharpe,
                    "fitness": link.fitness,
                    "turnover": link.turnover,
                    "delta_U": link.delta_U,
                    "evidence_type": link.evidence_type,
                    "support": link.support,
                    "conflicts": link.conflicts,
                    "uncertainty": classify_uncertainty(link),
                    "score": round(score(link), 4),
                    "source_panel_tag": link.source_panel_tag,
                    "ts": link.ts,
                }
            )
    rows.sort(key=lambda r: -r["score"])
    _emit(
        {
            "ok": True,
            "colony_tag": args.colony_tag,
            "count": len(rows),
            "links": rows[: args.limit] if args.limit > 0 else rows,
        }
    )


def cmd_colony_pheromones_show(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.pheromone_cache import (
        classify_uncertainty,
        read_cache,
        score,
    )

    for bucket in read_cache(args.colony_tag).values():
        for link in bucket:
            if (link.alpha_id or "") == args.alpha_id:
                payload = link.to_dict()
                payload["uncertainty"] = classify_uncertainty(link)
                payload["score"] = round(score(link), 4)
                _emit({"ok": True, "link": payload})
    _emit(
        {"ok": False, "error": f"alpha_id={args.alpha_id} not in cache"},
        code=1,
    )


def cmd_colony_train_policy(args: argparse.Namespace) -> None:
    """Train the colony's learned routing policy from logged tried_log rows."""
    from agent_market.wq_brain.colony import colony_run_dir
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.routing_policy import (
        LearnedPolicy,
        policy_path,
        samples_from_history,
    )
    from agent_market.wq_brain.tried_log import read_tried

    panel_tags: list[str] = args.panel_tags.split(",") if args.panel_tags else []
    if not panel_tags:
        manifest_path = colony_run_dir(args.colony_tag) / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            for p in data.get("panels", []) or []:
                tag = p.get("tag") if isinstance(p, dict) else None
                if tag:
                    panel_tags.append(tag)
    rows: list[dict[str, Any]] = []
    for tag in panel_tags:
        path = tried_exprs_path(tag.strip())
        if path.exists():
            rows.extend(read_tried(path, tail=5000))
    samples = samples_from_history(rows)
    if not samples:
        _emit(
            {
                "ok": False,
                "error": "no enriched (state, action, reward) samples — "
                         "ensure tried_log rows carry altitude + delta_U "
                         "metadata before training",
            },
            code=1,
        )
    pp = policy_path(args.colony_tag)
    policy = LearnedPolicy.load(pp) or LearnedPolicy.empty()
    policy.train(samples, epochs=args.epochs, lr=args.lr)
    policy.save(pp)
    _emit({
        "ok": True,
        "colony_tag": args.colony_tag,
        "policy_path": str(pp),
        "training_samples": policy.training_samples,
        "training_epochs": policy.training_epochs,
        "training_lr": policy.training_lr,
    })


def cmd_colony_reset(args: argparse.Namespace) -> None:
    """Delete shared cache + routing advisories + best-so-far for a colony.

    Manifest and per-panel run dirs are preserved so analysis remains
    possible; only the *advisory layer* is wiped.
    """
    from agent_market.wq_brain.colony import colony_run_dir
    from agent_market.wq_brain.pheromone_cache import cache_path

    removed: list[str] = []
    cp = cache_path(args.colony_tag)
    if cp.exists():
        cp.unlink()
        removed.append(str(cp))
    run_dir = colony_run_dir(args.colony_tag)
    for sub in ("routing", "best_so_far"):
        sub_dir = run_dir / sub
        if sub_dir.exists():
            for f in sub_dir.glob("*"):
                if f.is_file():
                    f.unlink()
                    removed.append(str(f))
    _emit({"ok": True, "colony_tag": args.colony_tag, "removed": removed})


def cmd_endpoint_failover(args: argparse.Namespace) -> None:
    """Probe LLM endpoint candidates and pin the first healthy one."""
    from agent_market.wq_brain.endpoint_probe import (
        EndpointCandidate,
        first_healthy,
        load_candidates_from_env,
        load_candidates_from_file,
        probe_candidates,
        write_env_local,
    )
    from agent_market.wq_brain.paths import repo_root

    candidates: list[EndpointCandidate] = []
    if args.candidates_file:
        candidates = load_candidates_from_file(Path(args.candidates_file))
    elif args.base_url and args.model:
        candidates = [
            EndpointCandidate(
                base_url=args.base_url, model=args.model,
                api_key=args.api_key, label="cli_override",
            )
        ]
    else:
        candidates = load_candidates_from_env()
    if not candidates:
        _emit(
            {
                "ok": False,
                "error": "no candidates supplied — provide --candidates-file, "
                         "--base-url/--model, or OPENAI_FALLBACK_ENDPOINTS env",
            },
            code=2,
        )
    probes = probe_candidates(candidates, timeout=args.timeout)
    winner = first_healthy(probes)
    payload: dict[str, Any] = {
        "ok": winner is not None,
        "candidates_count": len(candidates),
        "probes": [p.to_dict() for p in probes],
    }
    if winner is None:
        _emit(payload | {"error": "no healthy candidate"}, code=1)
    if not args.dry_run:
        env_local = repo_root() / ".env.local"
        write_env_local(env_local, winner.candidate)
        payload["env_local"] = str(env_local)
    payload["chosen"] = {
        "base_url": winner.candidate.base_url,
        "model": winner.candidate.model,
        "label": winner.candidate.label,
        "elapsed_ms": winner.elapsed_ms,
    }
    _emit(payload)


def cmd_colony_run(args: argparse.Namespace) -> None:
    """Run a wq_brain colony — one ant per (region, universe) panel.

    Each panel runs sequentially with the same model/CLI settings. After
    every panel completes, its high-altitude (L1 / L2) pheromone rows fan
    out to the next panel's tried log under ``evidence_type=colony_shared``.
    """
    from agent_market.wq_brain.colony import (
        ColonyConfig,
        PanelSpec,
        parse_panels,
        run_colony,
    )

    try:
        regions = parse_panels(args.panels)
    except ValueError as exc:
        _emit({"ok": False, "error": str(exc)}, code=2)
    if args.panel_tag_prefix:
        prefix = args.panel_tag_prefix
    else:
        prefix = f"{args.colony_tag}_panel"
    panels = [
        PanelSpec(
            tag=f"{prefix}_{i}_{region}_{universe}".lower(),
            region=region,
            universe=universe,
            decay=args.decay,
            neutralization=args.neutralization,
            truncation=args.truncation,
            max_turns=args.max_turns,
            quality_sharpe_min=args.quality_sharpe_min,
            quality_fitness_min=args.quality_fitness_min,
            auto_submit=not args.no_auto_submit,
        )
        for i, (region, universe) in enumerate(regions)
    ]
    cfg = ColonyConfig(
        colony_tag=args.colony_tag,
        panels=panels,
        cli=args.cli,
        model=args.model,
        timeout_sec=args.timeout_sec,
        provider=args.provider,
        toolsets=args.toolsets,
        yolo=not args.no_yolo,
        reasoning_effort=args.reasoning_effort,
        workers=int(getattr(args, "workers", 1) or 1),
    )
    manifest = run_colony(cfg)
    _emit({"ok": True, "manifest": manifest})


def cmd_ping_llm(args: argparse.Namespace) -> None:
    """Health-check the configured LLM endpoint before launching an agent loop.

    Sends a minimal chat-completions request and reports latency + token usage,
    so 'opencode run' / 'hermes' loops do not silently die at the title-gen step.
    """
    import os
    import time
    import urllib.error
    import urllib.request

    base = (args.base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or ""
    model = args.model or os.environ.get("OPENAI_MODEL") or ""
    if not base or not api_key or not model:
        _emit(
            {
                "ok": False,
                "error": "missing OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL — "
                         "set via env or --base-url / --api-key / --model",
            },
            code=2,
        )
    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    # OPENAI_BASE_URL is allowed to end with /v1 or be the host root; normalise
    # so we never end up with /v1/v1/chat/completions (404 on most providers).
    if base.endswith("/v1") or base.endswith("/v1/"):
        url = base.rstrip("/") + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            elapsed_ms = int((time.monotonic() - t0) * 1000)
    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit(
            {
                "ok": False,
                "url": url,
                "model": model,
                "elapsed_ms": elapsed_ms,
                "http_status": exc.code,
                "error": exc.reason,
                "body": exc.read().decode("utf-8", errors="replace")[:1500],
            },
            code=1,
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _emit(
            {
                "ok": False,
                "url": url,
                "model": model,
                "elapsed_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            },
            code=1,
        )
    usage = payload.get("usage") or {}
    _emit(
        {
            "ok": True,
            "url": url,
            "model": model,
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "first_choice_role": (
                ((payload.get("choices") or [{}])[0].get("message") or {}).get("role")
            ),
        }
    )


def cmd_web_search(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.web_search import web_search
    sources = tuple(s.strip() for s in args.source.split(",") if s.strip())
    out = web_search(args.query, max_results=args.max, sources=sources or ("auto",))
    _emit({"ok": True, **out})


def cmd_fetch_url(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.web_search import fetch_url
    out = fetch_url(args.url, timeout=args.timeout, max_chars=args.max_chars)
    if not out.get("ok"):
        _emit(out, code=1)
    else:
        _emit(out)


def cmd_skill_search(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.skill_search import search_skill
    out = search_skill(args.query, top_k=args.top_k)
    _emit(out, code=0 if out.get("ok") else 1)


def cmd_skill_list(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.skill_search import list_skill_files
    out = list_skill_files()
    _emit(out, code=0 if out.get("ok") else 1)


# ── Human-facing commands ─────────────────────────────────────────────────

def cmd_scan(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.scan_runner import ScanConfig, run_scan
    config = ScanConfig(
        tag=args.tag,
        seed_file=Path(args.seed_file),
        region=args.region,
        universe=args.universe,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
        max_candidates=args.max_candidates,
        auto_submit=args.auto_submit,
        legacy_unsafe_auto_submit=getattr(args, "legacy_unsafe_auto_submit", False),
        dry_run=args.dry_run,
    )
    summary = run_scan(config)
    print(json.dumps(summary, indent=2, default=str))


def cmd_agent(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.agent_runner import AgentConfig, run_agent
    config = AgentConfig(
        tag=args.tag,
        region=args.region,
        universe=args.universe,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
        max_turns=args.max_turns,
        cli=args.cli,
        model=args.model,
        provider=args.provider,
        yolo=args.yolo,
        toolsets=args.toolsets,
        reasoning_effort=args.reasoning_effort,
        auto_submit=args.auto_submit,
        timeout_sec=args.timeout_sec,
    )
    summary = run_agent(config)
    print(json.dumps(summary, indent=2, default=str))
    effective_rc = int(summary.get("agent_effective_returncode") or 0)
    if effective_rc != 0:
        sys.exit(effective_rc)


def cmd_report(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    print(json.dumps({
        "tag": args.tag,
        "pool_size": len(pool),
        "top_5_by_fitness": [e.to_dict() for e in pool.top_n_by_fitness(5)],
    }, indent=2, default=str))


def cmd_review(args: argparse.Namespace) -> None:
    """Show iter_review.json across all loop iterations for a tag."""
    from agent_market.wq_brain.paths import alpha_pool_path, wq_brain_root
    from agent_market.wq_brain.pool import AlphaPool

    runs_root = wq_brain_root() / "runs"
    if not runs_root.exists():
        print(json.dumps({"tag": args.tag, "iters": [], "note": "no runs/ directory"}, indent=2))
        return

    prefix = f"wqbrain_agent_{args.tag}_"
    iters: list[dict] = []
    for d in sorted(runs_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        review_path = d / "iter_review.json"
        if not review_path.exists():
            continue
        try:
            r = json.loads(review_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        iters.append({
            "run_id": d.name,
            "duration_sec": r.get("iter_window", {}).get("duration_sec"),
            "simulated": r.get("iter_simulated"),
            "completed": r.get("iter_completed"),
            "passed": r.get("iter_passed"),
            "errored": r.get("iter_errored"),
            "top_3": r.get("top_3_by_fitness", []),
            "pool_size_after": r.get("pool_size_after"),
        })

    if args.last and args.last > 0:
        iters = iters[-args.last:]

    pool = AlphaPool(alpha_pool_path(args.tag))
    totals = {
        "simulated": sum((i.get("simulated") or 0) for i in iters),
        "completed": sum((i.get("completed") or 0) for i in iters),
        "passed": sum((i.get("passed") or 0) for i in iters),
        "errored": sum((i.get("errored") or 0) for i in iters),
    }
    print(json.dumps({
        "tag": args.tag,
        "iter_count": len(iters),
        "totals": totals,
        "pool_size_now": len(pool),
        "iters": iters,
    }, indent=2, default=str))


# ── Parser ────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wq_brain", description="wq_brain unified CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("auth", help="verify WQ credentials")
    sp.set_defaults(func=cmd_auth)

    sp = sub.add_parser("validate", help="locally validate a FASTEXPR")
    sp.add_argument("expr")
    sp.add_argument("--lax", action="store_true",
                    help="use legacy token scan only (skip arity / unknown-op / nesting checks)")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("mutate", help="diagnose a failure + recommend mutation strategy")
    sp.add_argument("expr")
    sp.add_argument("--sharpe", type=float, default=None)
    sp.add_argument("--fitness", type=float, default=None)
    sp.add_argument("--turnover", type=float, default=None)
    sp.add_argument("--returns", type=float, default=None)
    sp.add_argument("--status", default="COMPLETE")
    sp.add_argument("--error", default=None)
    sp.set_defaults(func=cmd_mutate)

    sp = sub.add_parser("simulate", help="run a single WQ simulation")
    sp.add_argument("expr")
    sp.add_argument("--region", default="USA")
    sp.add_argument("--universe", default="TOP3000")
    sp.add_argument("--decay", type=int, default=6)
    sp.add_argument("--neutralization", default="SUBINDUSTRY")
    sp.add_argument("--truncation", type=float, default=0.08)
    sp.add_argument("--timeout", type=float, default=600.0)
    sp.add_argument("--tag", default="", help="when set, append result to tried_exprs.jsonl")
    sp.add_argument("--auto-persist-sharpe", type=float, default=1.25,
                    help="when set with --tag, candidates with sharpe ≥ X AND fitness ≥ "
                         "--auto-persist-fitness are written to the pool as UNSUBMITTED. "
                         "Prevents the agent's LLM session from losing high-fi candidates. "
                         "Set to 999 to disable auto-persist.")
    sp.add_argument("--auto-persist-fitness", type=float, default=1.0,
                    help="see --auto-persist-sharpe")
    sp.add_argument("--parent-alpha-id", default=None,
                    help="alpha_id of the parent expression that this candidate "
                         "was derived from; if supplied, altitude and ΔU are "
                         "auto-computed against the parent's tried_log row")
    sp.add_argument("--evidence-type", default=None,
                    help="override the evidence_type label written to tried_log "
                         "(e.g. seed/mutation/crossover/region_swap/op_swap/"
                         "param_shift/decay_shift/neutralization_swap/"
                         "numeric_tweak); defaults to 'manual' when no parent "
                         "is supplied, 'mutation' otherwise")
    sp.add_argument("--skip-cooldown", action="store_true",
                    help="bypass the slot cool-down hard gate "
                         "(use only to recover from false positives)")
    sp.add_argument("--cooldown-window", type=int, default=8,
                    help="cool-down window: how many recent rows to scan "
                         "for the same slot (default 8)")
    sp.add_argument("--cooldown-delta-flip", type=float, default=0.05,
                    help="cool-down ΔU threshold: same-slot retries are "
                         "allowed when the prior ΔU > this (default 0.05)")
    sp.add_argument("--colony-tag", default=None,
                    help="when set, update the panel's best-so-far record "
                         "inside artifacts/wq_brain/colony/<colony-tag>/")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("submit", help="submit alpha to WQ PROD pool")
    sp.add_argument("alpha_id")
    sp.add_argument("--tag", default="", help="local pool tag")
    sp.add_argument("--expr", default="", help="optional expr to record locally")
    sp.add_argument("--corr-max", type=float, default=0.7,
                    help="reject if max correlation with existing pool >= this (default 0.7)")
    sp.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="WQ override rule: high-corr submission still allowed if our sharpe ≥ (1+margin) × theirs (default 0.10 = 10%%)")
    sp.add_argument("--jaccard-max", type=float, default=0.7,
                    help="local pre-check: reject if token-jaccard vs any ACTIVE pool alpha >= this (default 0.7)")
    sp.add_argument("--semantic-max", type=float, default=0.85,
                    help="local pre-check: reject if multiset semantic-jaccard (operators+fields) vs any ACTIVE alpha >= this (default 0.85). Tightened to 0.65 if you want more diversity.")
    sp.add_argument("--no-pre-check", action="store_true",
                    help="skip the correlation pre-check")
    sp.add_argument("--force-submit-on-precheck-error", action="store_true",
                    help="on pre-check INFRASTRUCTURE failure (network / 5xx / auth), "
                         "submit anyway. Default is fail-CLOSED to avoid burning submit "
                         "quota on a candidate the gate could not actually evaluate.")
    sp.add_argument("--verify-after-sec", type=float, default=30.0,
                    help="seconds to wait before re-fetching alpha to verify ACTIVE/REJECTED (default 30, 0 to skip)")
    # Codex review R4-#3: same override controls as `pool submit-worker` so
    # operator behavior is consistent across single-submit and batch-submit.
    sp.add_argument("--override-mode", choices=("sharpe_and_fitness", "sharpe_only"),
                    default="sharpe_and_fitness",
                    help="sharpe_and_fitness (default, strict): candidate must clear sharpe-margin "
                         "AND fitness ≥ each blocker's fitness. sharpe_only (looser, closer to WQ's "
                         "documented sharpe-margin clause).")
    sp.add_argument("--absolute-fitness-floor", type=float, default=1.0,
                    help="absolute fitness bar in --override-mode sharpe_only (default 1.0 = WQ "
                         "ACTIVE bar)")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("pre-check", help="check if alpha would be rejected by WQ self-correlation gate")
    sp.add_argument("alpha_id")
    sp.add_argument("--corr-max", type=float, default=0.7)
    sp.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="WQ allows high_corr submission if sharpe ≥ (1+margin) × correlated.sharpe")
    sp.add_argument("--tag", default="",
                    help="local pool tag (used to look up correlated alphas' sharpes)")
    sp.set_defaults(func=cmd_pre_check)

    sp = sub.add_parser("pre-check-local",
                        help="fast local jaccard + semantic check vs ACTIVE pool (no API calls)")
    sp.add_argument("expr", nargs="?", default="",
                    help="FASTEXPR to check. If omitted, --alpha-id is used to look "
                         "up the expr from tried_log.jsonl.")
    sp.add_argument("--tag", required=True)
    sp.add_argument("--alpha-id", default="",
                    help="Codex review R2-#4: when provided, auto-fill expr + "
                         "candidate sharpe/fitness from tried_log so the preview "
                         "matches `submit` / `pool submit-worker` exactly.")
    sp.add_argument("--jaccard-max", type=float, default=0.7,
                    help="reject if token-jaccard vs any ACTIVE alpha >= this (default 0.7)")
    sp.add_argument("--semantic-max", type=float, default=0.85,
                    help="reject if multiset semantic-jaccard (operators+fields) >= this (default 0.85). Catches 'same skeleton, different fields' impostors that token jaccard misses.")
    # Codex review R1-#6: accept the same override controls as `submit` so
    # the operator preview matches what production would do.
    sp.add_argument("--candidate-sharpe", type=float, default=None,
                    help="candidate's measured sharpe (enables sharpe-margin override)")
    sp.add_argument("--candidate-fitness", type=float, default=None,
                    help="candidate's measured fitness (enables sharpe-margin override)")
    sp.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="sharpe override: blocked candidate accepted if "
                         "sharpe ≥ (1+margin) × blocking.sharpe (default 0.10)")
    sp.add_argument("--override-mode", choices=("sharpe_and_fitness", "sharpe_only"),
                    default="sharpe_and_fitness",
                    help="see `submit --help` — default strict (relative-fitness clause active)")
    sp.add_argument("--absolute-fitness-floor", type=float, default=1.0,
                    help="absolute fitness bar in --override-mode sharpe_only (default 1.0)")
    sp.set_defaults(func=cmd_pre_check_local)

    sp = sub.add_parser(
        "audit-data",
        help="run 5-check OHLCV cache audit (writes _audit.json + _audit.md)",
    )
    sp.add_argument("--ticker", default=None,
                    help="restrict audit to a single ticker (default: all)")
    sp.add_argument("--sample-size", type=int, default=10,
                    help="rows-per-finding to include in the report (default 10)")
    sp.set_defaults(func=cmd_audit_data)

    sp = sub.add_parser(
        "calibrate-local",
        help="calibrate local-simulate fitness gate against tried_log history",
    )
    sp.add_argument("--tag", required=True)
    sp.add_argument("--top-n", type=int, default=30,
                    help="how many COMPLETE samples to re-simulate (default 30)")
    sp.add_argument("--max-time-sec", type=float, default=None,
                    help="hard time budget; stop early if exceeded (default unbounded)")
    sp.add_argument("--no-save", action="store_true",
                    help="compute report only; don't write threshold.json")
    sp.add_argument("--min-samples", type=int, default=5,
                    help="minimum tried_log COMPLETE rows required (default 5; "
                         "lower to 3 to bootstrap calibration on a fresh tag)")
    sp.set_defaults(func=cmd_calibrate_local)

    sp = sub.add_parser(
        "seed-calibration",
        help="seed calibration/{tag}/threshold.json from a JSONL of pre-computed samples",
    )
    sp.add_argument("--tag", required=True)
    sp.add_argument("--from-samples", required=True,
                    help='JSONL file; each row needs local_fitness + passes_remote (bool). '
                         'Optional: remote_fitness/remote_sharpe/remote_turnover for the report.')
    sp.add_argument("--min-samples", type=int, default=5)
    sp.add_argument("--no-save", action="store_true",
                    help="compute report only; don't write threshold.json")
    sp.set_defaults(func=cmd_seed_calibration)

    sp = sub.add_parser("fetch-data", help="bulk-fetch US stock OHLCV + sectors into local cache")
    sp.add_argument("--tickers-file", default=None,
                    help="path to file with one ticker per line; default uses bundled ~300-name list")
    sp.add_argument("--start", default="2021-01-01")
    sp.add_argument("--end", default="2026-05-05")
    sp.add_argument("--skip-sectors", action="store_true",
                    help="skip the slow per-ticker sector lookup")
    sp.add_argument("--polite-sleep", type=float, default=2.0,
                    help="seconds between single-ticker calls (default 2)")
    sp.add_argument("--backend", choices=("stooq", "yfinance", "auto"), default=None,
                    help="data backend (default: env WQB_DATA_BACKEND or stooq)")
    sp.set_defaults(func=cmd_fetch_data)

    sp = sub.add_parser("update-data", help="incremental daily OHLCV update")
    sp.add_argument("--tickers-file", default=None)
    sp.add_argument("--start", default="2021-01-01",
                    help="fallback start when cache is empty")
    sp.add_argument("--end", default=None,
                    help="default: today UTC")
    sp.add_argument("--backend", choices=("stooq", "yfinance", "auto"), default=None)
    sp.set_defaults(func=cmd_update_data)

    sp = sub.add_parser(
        "kaggle-fetch",
        help="download a Kaggle stock dataset ZIP (one HTTP roundtrip, ~tens of MB)",
    )
    sp.add_argument("--dataset", required=True,
                    help="Kaggle slug, e.g. nelgiriyewithana/world-stock-prices-daily-updating")
    sp.add_argument("--dest-dir", default=None,
                    help="default: artifacts/wq_brain/data/kaggle")
    sp.add_argument("--force", action="store_true",
                    help="re-download even if ZIP already present")
    sp.add_argument("--no-extract", action="store_true",
                    help="just download, skip auto-extract")
    sp.set_defaults(func=cmd_kaggle_fetch)

    sp = sub.add_parser(
        "kaggle-import",
        help="parse an extracted Kaggle dataset into ohlcv.parquet",
    )
    sp.add_argument("--dataset", required=True,
                    help="Kaggle slug — used to locate the extract dir")
    sp.add_argument("--dest-dir", default=None,
                    help="default: artifacts/wq_brain/data/kaggle (must match kaggle-fetch)")
    sp.add_argument("--files-glob", default="*.csv",
                    help="glob inside the extract dir (default *.csv)")
    sp.add_argument("--ticker-col", default="Ticker",
                    help='CSV column carrying the ticker (set to "" to use filename)')
    sp.add_argument("--date-col", default="Date",
                    help="CSV column carrying the date")
    sp.add_argument("--ticker-from-filename", action="store_true",
                    help="derive ticker from each CSV's filename stem instead of a column")
    sp.add_argument("--column-map", default=None,
                    help='JSON dict, e.g. \'{"adj_close":"close"}\' — applied on top of defaults')
    sp.add_argument("--split-adjust-from", default=None,
                    help='Name of the *raw* close column (after column-map rename) used to back out '
                         'a split factor and apply it to open/high/low. Use this when the dataset '
                         'has separate raw + adjusted close (e.g. close + adjusted) and you want '
                         'a fully split-consistent OHLC bar.')
    sp.add_argument("--no-audit", action="store_true",
                    help="skip the post-import data_audit hook (default: run + write _audit.json/_audit.md)")
    sp.set_defaults(func=cmd_kaggle_import)

    sp = sub.add_parser("local-simulate", help="local WQ-aligned simulation against cached OHLCV (no WQ API)")
    sp.add_argument("expr")
    sp.add_argument("--rebalance-freq", type=int, default=5,
                    help="rebalance every N trading days (default: 5)")
    sp.add_argument("--tag", default="",
                    help="calibration tag — if set, reads calibration/{tag}/threshold.json")
    sp.add_argument("--fitness-gate", type=float, default=None,
                    help="explicit local-fitness gate override (default: per-tag calibration or 0.5)")
    sp.set_defaults(func=cmd_local_simulate)

    sp = sub.add_parser("anti-overfit", help="4-layer anti-overfit detection on cached OHLCV")
    sp.add_argument("expr")
    sp.add_argument("--holding-period", type=int, default=5)
    sp.set_defaults(func=cmd_anti_overfit)

    sp = sub.add_parser("score", help="multi-dim score a single alpha result")
    sp.add_argument("--sharpe", type=float, default=None)
    sp.add_argument("--fitness", type=float, default=None)
    sp.add_argument("--turnover", type=float, default=None)
    sp.add_argument("--returns", type=float, default=None)
    sp.add_argument("--status", default="COMPLETE")
    sp.add_argument("--expr", default="")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("pool", help="local alpha pool ops")
    pool_sub = sp.add_subparsers(dest="pool_cmd", required=True)
    pl = pool_sub.add_parser("list", help="list pool entries")
    pl.add_argument("--tag", required=True)
    pl.set_defaults(func=cmd_pool_list)
    bf = pool_sub.add_parser("backfill-exprs",
                             help="fill in missing expr fields from tried_exprs.jsonl by alpha_id")
    bf.add_argument("--tag", required=True)
    bf.set_defaults(func=cmd_pool_backfill)
    dd = pool_sub.add_parser("dedup",
                             help="merge near-duplicate pool entries (jaccard ≥ threshold), keep highest fitness")
    dd.add_argument("--tag", required=True)
    dd.add_argument("--threshold", type=float, default=0.85,
                    help="jaccard similarity threshold for clustering (default 0.85)")
    dd.add_argument("--dry-run", action="store_true",
                    help="show what would be dropped without modifying pool")
    dd.set_defaults(func=cmd_pool_dedup)
    rs = pool_sub.add_parser("resubmit-all",
                             help="POST /alphas/{id}/submit for every pool entry still UNSUBMITTED. "
                                  "Codex review R4-#2: blocked states are skipped by default; "
                                  "prefer `pool submit-worker` for the full local + remote gate stack.")
    rs.add_argument("--tag", required=True)
    rs.add_argument("--polite-sleep", type=float, default=3.0,
                    help="seconds to sleep between alphas (default 3.0)")
    rs.add_argument("--include-blocked", action="store_true",
                    help="include LOCAL_BLOCKED and SELF_CORR_BLOCKED entries (default: skip — "
                         "these were already rejected by the local gate stack and re-submitting "
                         "burns quota)")
    rs.add_argument("--status-filter", default="UNSUBMITTED",
                    help="only attempt entries with this verified_status "
                         "(default UNSUBMITTED — Codex review R2-#3 changed "
                         "from broad scan to UNSUBMITTED-only). Pass an empty "
                         "string to include every non-ACTIVE entry.")
    rs.add_argument("--max", type=int, default=20,
                    help="cap successful WQ submit attempts per run (default 20). "
                         "Codex review R2-#3: was previously absent — could submit "
                         "the entire pool unboundedly.")
    rs.add_argument("--legacy-unsafe", action="store_true",
                    help="Codex R2 gate (project-quality loop): opt into the legacy "
                         "bypass (no quota / no local-jaccard / no self-corr / no "
                         "outcome persistence). Without this flag, the command is "
                         "refused — see `pool submit-worker` for the production path.")
    rs.set_defaults(func=cmd_pool_resubmit)
    ps = pool_sub.add_parser("status",
                             help="query WQ for each pool alpha's current submission status")
    ps.add_argument("--tag", required=True)
    ps.set_defaults(func=cmd_pool_status)
    ss = pool_sub.add_parser("sync-status",
                             help="query WQ for each entry's actual status + IS checks; write verified_status + rejection_reasons into pool.json")
    ss.add_argument("--tag", required=True)
    ss.add_argument("--polite-sleep", type=float, default=2.0)
    ss.add_argument("--probe-rejections", action="store_true", default=False,
                    help="DANGER (Codex review R1-CRIT): when set, POSTs /alphas/{id}/submit "
                         "for every UNSUBMITTED entry to capture WQ's cached IS-check failure "
                         "details. This actually SUBMITS the alpha and BURNS WQ submit quota. "
                         "Default is now OFF — `sync-status` only reads. Use this flag only "
                         "after you've already exhausted submit quota and want WQ to re-emit "
                         "the cached rejection reasons for diagnosis.")
    ss.add_argument("--no-probe-rejections", dest="probe_rejections", action="store_false",
                    help="explicit no-op (default behaviour) — kept for backward compat")
    ss.add_argument("--reset-local-blocks", action="store_true",
                    help="Codex review R4-#1: by default, terminal LOCAL_BLOCKED / "
                         "SELF_CORR_BLOCKED states are PRESERVED (WQ would report them "
                         "as UNSUBMITTED, which would erase the gate verdict). Pass this "
                         "flag to overwrite them — e.g. after raising thresholds or "
                         "retiring a gate, when you want to retry blocked entries.")
    ss.set_defaults(func=cmd_pool_sync_status)
    sv = pool_sub.add_parser(
        "salvage",
        help="backfill pool with high-fi candidates from tried_exprs.jsonl that "
             "the agent's LLM session forgot to submit (production showed 70%% loss rate)",
    )
    sv.add_argument("--tag", required=True)
    sv.add_argument("--sharpe-min", type=float, default=1.25,
                    help="only salvage candidates with sharpe ≥ this (default 1.25 = WQ ACTIVE bar)")
    sv.add_argument("--fitness-min", type=float, default=1.0,
                    help="only salvage candidates with fitness ≥ this (default 1.0)")
    sv.add_argument("--top-n", type=int, default=0,
                    help="if > 0, only salvage the top-N highest-fitness misses; 0 = all matches")
    sv.add_argument("--dry-run", action="store_true",
                    help="preview what would be salvaged without writing pool")
    sv.set_defaults(func=cmd_pool_salvage)

    sw = pool_sub.add_parser(
        "submit-worker",
        help="cluster + submit worker for UNSUBMITTED entries; runs WQ self-corr "
             "precheck, submits accepted, upserts pool with outcome",
    )
    sw.add_argument("--tag", required=True)
    sw.add_argument("--status", default="UNSUBMITTED",
                    help="filter pool by verified_status (default UNSUBMITTED)")
    sw.add_argument("--max", type=int, default=20,
                    help="max successful WQ submissions in this run (default 20). "
                         "Codex review R2-#1: this is the SUBMIT budget — blocked "
                         "candidates do NOT consume it. Set 0 for unbounded.")
    sw.add_argument("--scan-limit", type=int, default=None,
                    help="upper bound on candidates scanned. When unset, "
                         "auto-bumps to max(200, --max × 5). Codex review R3-#3: "
                         "an explicit value is honored exactly — never overridden.")
    sw.add_argument("--dry-run-limit", type=int, default=20,
                    help="how many candidates to render in the dry-run preview "
                         "table (default 20). Aggregate counts evaluate ALL scanned "
                         "targets regardless of this cap.")
    sw.add_argument("--one-per-cluster", action="store_true",
                    help="cluster by operator skeleton; submit only the highest-"
                         "fitness candidate per cluster (avoids burning quota on "
                         "structural near-duplicates)")
    sw.add_argument("--corr-max", type=float, default=0.7,
                    help="self-correlation threshold for WQ pre-check (default 0.7)")
    sw.add_argument("--jaccard-max", type=float, default=0.7,
                    help="local token-jaccard threshold (default 0.7); "
                         "candidates above this are blocked unless sharpe-margin override fires")
    sw.add_argument("--semantic-max", type=float, default=0.85,
                    help="local semantic-jaccard threshold (default 0.85); "
                         "operator-skeleton overlap. Same override as jaccard-max")
    sw.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="sharpe override threshold: high-corr or high-jaccard alphas allowed "
                         "if candidate sharpe ≥ (1+margin) × blocking.sharpe (Codex review R2-#3: "
                         "fitness clause behavior controlled by --override-mode)")
    sw.add_argument("--override-mode", choices=("sharpe_and_fitness", "sharpe_only"),
                    default="sharpe_and_fitness",
                    help="sharpe_and_fitness (default, strict): candidate must clear sharpe-margin "
                         "AND have fitness ≥ each blocker's fitness. sharpe_only (looser, closer "
                         "to WQ's documented sharpe-margin clause): only sharpe-margin per blocker, "
                         "plus candidate fitness ≥ --absolute-fitness-floor")
    sw.add_argument("--absolute-fitness-floor", type=float, default=1.0,
                    help="absolute fitness bar used in --override-mode sharpe_only (default 1.0 "
                         "= WQ ACTIVE bar)")
    sw.add_argument("--verify-after-sec", type=float, default=30.0)
    sw.add_argument("--continue-on-infra", action="store_true",
                    help="keep going even if pre-check infra error fires "
                         "(default: stop the worker on first infra error)")
    sw.add_argument("--dry-run", action="store_true",
                    help="preview clustered targets without submitting")
    sw.set_defaults(func=cmd_pool_submit_worker)

    sp = sub.add_parser("corr", help="get correlations of alpha with WQ pool")
    sp.add_argument("alpha_id")
    sp.set_defaults(func=cmd_corr)

    sp = sub.add_parser("search-arxiv", help="search arxiv abstracts with quant-finance category anchors")
    sp.add_argument("query")
    sp.add_argument("--max", type=int, default=5)
    sp.add_argument("--category", default="q-fin.*,stat.ML,cs.CE",
                    help="comma-separated arXiv categories/aliases; default q-fin.*,stat.ML,cs.CE")
    sp.add_argument("--sort", choices=("relevance", "submittedDate", "lastUpdatedDate"),
                    default="relevance")
    sp.add_argument("--sort-order", choices=("ascending", "descending"), default="descending")
    sp.add_argument("--raw-query", action="store_true",
                    help="treat query as a raw arXiv search_query fragment")
    sp.set_defaults(func=cmd_search_arxiv)

    sp = sub.add_parser("search-papers", help="multi-source paper search (arXiv/S2/OpenAlex/SSRN web)")
    sp.add_argument("query")
    sp.add_argument("--max", type=int, default=5,
                    help="maximum results per source")
    sp.add_argument("--source", default="arxiv,semantic_scholar,openalex,ssrn",
                    help="comma-separated: arxiv, semantic_scholar, openalex, ssrn")
    sp.add_argument("--category", default="q-fin.*,stat.ML,cs.CE",
                    help="arXiv category filter used for the arxiv source")
    sp.add_argument("--arxiv-sort", choices=("relevance", "submittedDate", "lastUpdatedDate"),
                    default="relevance")
    sp.add_argument("--year", default=None,
                    help="Semantic Scholar year/range filter, e.g. 2024 or 2023-")
    sp.add_argument("--fields-of-study", default="Computer Science,Economics,Business",
                    help="Semantic Scholar fieldsOfStudy filter")
    sp.add_argument("--min-citations", type=int, default=None)
    sp.set_defaults(func=cmd_search_papers)

    sp = sub.add_parser("math", help="symbolic math via SymPy")
    sp.add_argument("operation", choices=("simplify", "expand", "factor", "diff", "integrate", "solve", "series", "latex"))
    sp.add_argument("expr")
    sp.add_argument("--var", default=None,
                    help="variable for diff/integrate/series; defaults to first free symbol")
    sp.add_argument("--solve-for", default=None,
                    help="symbol to solve for; defaults to --var or first free symbol")
    sp.add_argument("--point", default="0",
                    help="series expansion point")
    sp.add_argument("--order", type=int, default=6,
                    help="series order")
    sp.set_defaults(func=cmd_math)

    sp = sub.add_parser("docs", help="show local FASTEXPR docs")
    sp.add_argument("topic", choices=["operators"])
    sp.set_defaults(func=cmd_docs)

    sp = sub.add_parser("ping-llm",
                        help="health-check the configured LLM endpoint before "
                             "launching an agent loop")
    sp.add_argument("--base-url", default=None,
                    help="override OPENAI_BASE_URL")
    sp.add_argument("--api-key", default=None,
                    help="override OPENAI_API_KEY (avoid; prefer env)")
    sp.add_argument("--model", default=None,
                    help="override OPENAI_MODEL")
    sp.add_argument("--timeout", type=float, default=15.0,
                    help="request timeout in seconds (default 15)")
    sp.set_defaults(func=cmd_ping_llm)

    sp = sub.add_parser(
        "endpoint",
        help="LLM endpoint health probe + failover (writes .env.local)",
    )
    esub = sp.add_subparsers(dest="endpoint_cmd", required=True)
    ef = esub.add_parser("failover",
                         help="probe candidates and pin the first healthy one")
    ef.add_argument("--candidates-file",
                    help="JSON file listing {base_url, api_key, model, label}")
    ef.add_argument("--base-url", default=None,
                    help="override candidate base URL (one-shot)")
    ef.add_argument("--api-key", default=None,
                    help="override candidate API key (one-shot)")
    ef.add_argument("--model", default=None,
                    help="override candidate model (one-shot)")
    ef.add_argument("--timeout", type=float, default=10.0)
    ef.add_argument("--dry-run", action="store_true",
                    help="probe only; do not write .env.local")
    ef.set_defaults(func=cmd_endpoint_failover)

    sp = sub.add_parser(
        "colony",
        help="multi-ant colony: one ant per (region, universe) panel with "
             "L1/L2 pheromone fan-out",
    )
    csub = sp.add_subparsers(dest="colony_cmd", required=True)
    sc = csub.add_parser("run", help="run a colony sequentially")
    sc.add_argument("--colony-tag", required=True,
                    help="unique tag for this colony run (used in artifact paths)")
    sc.add_argument("--panels", required=True,
                    help="REGION:UNIVERSE,REGION:UNIVERSE,...  e.g. "
                         "USA:TOP500,USA:TOP1000,USA:TOP3000")
    sc.add_argument("--panel-tag-prefix", default="",
                    help="tag prefix per panel; defaults to <colony-tag>_panel")
    sc.add_argument("--cli", default="opencode",
                    help="agent CLI: opencode | hermes | auto")
    sc.add_argument("--model", default="",
                    help="LLM model id for the agent CLI (required)")
    sc.add_argument("--provider", default="",
                    help="optional provider hint for the agent CLI")
    sc.add_argument("--toolsets", default="terminal,file")
    sc.add_argument("--reasoning-effort", default="")
    sc.add_argument("--no-yolo", action="store_true",
                    help="disable opencode --yolo")
    sc.add_argument("--max-turns", type=int, default=12)
    sc.add_argument("--decay", type=int, default=6)
    sc.add_argument("--neutralization", default="SUBINDUSTRY")
    sc.add_argument("--truncation", type=float, default=0.08)
    sc.add_argument("--quality-sharpe-min", type=float, default=1.25)
    sc.add_argument("--quality-fitness-min", type=float, default=1.0)
    sc.add_argument("--no-auto-submit", action="store_true")
    sc.add_argument("--timeout-sec", type=float, default=900.0)
    sc.add_argument("--workers", type=int, default=1,
                    help="thread-pool size; >1 enables parallel panels "
                         "(default 1 — sequential)")
    sc.set_defaults(func=cmd_colony_run)

    ss = csub.add_parser("status",
                         help="show manifest + cache summary + panel bests")
    ss.add_argument("--colony-tag", required=True)
    ss.set_defaults(func=cmd_colony_status)

    sph = csub.add_parser(
        "pheromones",
        help="inspect or manage entries in the shared pheromone cache",
    )
    psub = sph.add_subparsers(dest="pheromones_cmd", required=True)
    pls = psub.add_parser("list",
                          help="list cache entries sorted by score desc")
    pls.add_argument("--colony-tag", required=True)
    pls.add_argument("--limit", type=int, default=20,
                     help="0 = no limit (default 20)")
    pls.set_defaults(func=cmd_colony_pheromones_list)
    psh = psub.add_parser("show", help="show full detail of one link")
    psh.add_argument("--colony-tag", required=True)
    psh.add_argument("--alpha-id", required=True)
    psh.set_defaults(func=cmd_colony_pheromones_show)

    sr = csub.add_parser("reset", help="wipe cache + routing + best-so-far")
    sr.add_argument("--colony-tag", required=True)
    sr.set_defaults(func=cmd_colony_reset)

    sp_train = csub.add_parser(
        "train-policy",
        help="train the learned routing policy μ_θ from tried_log history",
    )
    sp_train.add_argument("--colony-tag", required=True)
    sp_train.add_argument(
        "--panel-tags",
        default="",
        help="comma-separated panel tags; defaults to all panels in the "
             "colony manifest",
    )
    sp_train.add_argument("--epochs", type=int, default=20)
    sp_train.add_argument("--lr", type=float, default=0.1)
    sp_train.set_defaults(func=cmd_colony_train_policy)

    sp = sub.add_parser("web-search", help="general web search (Brave/Wikipedia/GitHub fallback)")
    sp.add_argument("query")
    sp.add_argument("--max", type=int, default=5)
    sp.add_argument("--source", default="auto",
                    help="comma-separated: auto | brave | wikipedia | bing | github")
    sp.set_defaults(func=cmd_web_search)

    sp = sub.add_parser("fetch-url", help="fetch URL → plain text")
    sp.add_argument("url")
    sp.add_argument("--timeout", type=float, default=20.0)
    sp.add_argument("--max-chars", type=int, default=6000)
    sp.set_defaults(func=cmd_fetch_url)

    sp = sub.add_parser("skill-search", help="search the vendored worldquant-skill knowledge base")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=5)
    sp.set_defaults(func=cmd_skill_search)

    sp = sub.add_parser("skill-list", help="list files in the vendored worldquant-skill")
    sp.set_defaults(func=cmd_skill_list)

    sp = sub.add_parser("scan", help="non-LLM batch scan from seed templates")
    sp.add_argument("--tag", required=True)
    sp.add_argument("--seed-file", required=True)
    sp.add_argument("--region", default="USA")
    sp.add_argument("--universe", default="TOP3000")
    sp.add_argument("--decay", type=int, default=6)
    sp.add_argument("--neutralization", default="SUBINDUSTRY")
    sp.add_argument("--truncation", type=float, default=0.08)
    sp.add_argument("--max-candidates", type=int, default=200)
    sp.add_argument("--auto-submit", action="store_true",
                    help="DEPRECATED legacy bypass — see --legacy-unsafe. "
                         "Production: drop this flag, then run `pool submit-worker --tag <tag>`.")
    sp.add_argument("--legacy-unsafe", dest="legacy_unsafe_auto_submit",
                    action="store_true",
                    help="Codex R2 gate: opt into the legacy auto-submit bypass "
                         "(no quota / no local-jaccard / no self-corr / no persistence). "
                         "Without this flag, --auto-submit is refused.")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("agent", help="launch LLM CLI for autonomous WQ research")
    sp.add_argument("--tag", required=True)
    sp.add_argument("--region", default="USA")
    sp.add_argument("--universe", default="TOP3000")
    sp.add_argument("--decay", type=int, default=6)
    sp.add_argument("--neutralization", default="SUBINDUSTRY")
    sp.add_argument("--truncation", type=float, default=0.08)
    sp.add_argument("--max-turns", type=int, default=100)
    sp.add_argument("--cli", default="auto", choices=["auto", "opencode", "hermes"],
                    help="agentic LLM CLI to spawn")
    sp.add_argument("--model", default="")
    sp.add_argument("--provider", default="")
    sp.add_argument("--yolo", dest="yolo", action="store_true", default=True)
    sp.add_argument("--no-yolo", dest="yolo", action="store_false")
    sp.add_argument("--toolsets", default="terminal,file", help="hermes-only")
    sp.add_argument("--reasoning-effort", default="", help="hermes-only")
    sp.add_argument("--auto-submit", action="store_true")
    sp.add_argument("--timeout-sec", type=float, default=7200.0)
    sp.set_defaults(func=cmd_agent)

    sp = sub.add_parser("report", help="show pool report")
    sp.add_argument("--tag", required=True)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("review", help="show per-iter reviews across loop iterations")
    sp.add_argument("--tag", required=True)
    sp.add_argument("--last", type=int, default=0,
                    help="show only the last N iterations (0 = all)")
    sp.set_defaults(func=cmd_review)

    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _ensure_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
