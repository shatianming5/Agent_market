#!/usr/bin/env python3
"""wq_brain unified CLI — human and hermes-agent entry point.

Human commands: auth, agent, scan, report.
Hermes-agent commands (called from inside agent session via terminal):
    auth, validate, simulate, submit, pool list, corr, search-arxiv, docs.
All agent-facing commands emit JSON to stdout (for parseable output).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import time

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
    from agent_market.wq_brain.dtypes import AlphaSettings
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import append_tried
    settings = AlphaSettings(
        region=args.region, universe=args.universe, decay=args.decay,
        neutralization=args.neutralization, truncation=args.truncation,
    )
    try:
        sess = session_from_env()
        result = sess.simulate_and_parse(args.expr, settings, timeout=args.timeout)
        if args.tag:
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
            )
        _emit({"ok": True, "expr": args.expr, **result.to_dict()})
    except Exception as exc:
        if args.tag:
            try:
                append_tried(
                    tried_exprs_path(args.tag), expr=args.expr,
                    sharpe=None, fitness=None, turnover=None, alpha_id=None,
                    status="ERROR", error=str(exc),
                    region=args.region, universe=args.universe, decay=args.decay,
                )
            except Exception:
                pass
        _emit({"ok": False, "expr": args.expr, "error": str(exc)}, code=1)


def cmd_submit(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.dtypes import AlphaPoolEntry
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    sess = session_from_env()

    # Pre-check: WQ-aligned self-correlation + sharpe-margin rule
    if not args.no_pre_check:
        try:
            check = _check_self_correlation(
                sess, args.alpha_id,
                corr_max=args.corr_max,
                sharpe_margin=getattr(args, "sharpe_margin", 0.10),
                tag=args.tag,
            )
            if not check["accept"]:
                _emit({
                    "ok": False,
                    "rejected_by": "pre_check",
                    "alpha_id": args.alpha_id,
                    **check,
                    "hint": "Use --no-pre-check to override (will likely be rejected by WQ submit step).",
                }, code=2)
                return
        except Exception as exc:
            print(f"WARN: pre_check failed ({exc}); proceeding to submit", file=sys.stderr)

    try:
        wq_resp = sess.submit_alpha(args.alpha_id, verify_after_sec=args.verify_after_sec)
    except Exception as exc:
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
                    pool_added = pool.add(entry)
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
                pool_added = pool.add(entry)
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


def _summarize_rejection(reasons: list) -> str:
    """One-line human summary of failed IS checks."""
    if not reasons:
        return "no specific check failures captured"
    parts = []
    for r in reasons[:5]:
        n = r.get("name", "?")
        v = r.get("value", "?")
        lim = r.get("limit", "?")
        parts.append(f"{n}={v} (limit={lim})")
    return "; ".join(parts)


def _auto_fill_expr(tag: str, alpha_id: str) -> str:
    """Look up the most-recent tried_exprs.jsonl row for alpha_id."""
    if not tag or not alpha_id:
        return ""
    from agent_market.wq_brain.paths import tried_exprs_path
    from agent_market.wq_brain.tried_log import read_tried
    rows = read_tried(tried_exprs_path(tag), tail=2000)
    for r in reversed(rows):
        if r.get("alpha_id") == alpha_id and r.get("expr"):
            return r["expr"]
    return ""


def _check_self_correlation(
    sess: Any,
    alpha_id: str,
    *,
    corr_max: float = 0.7,
    sharpe_margin: float = 0.10,
    tag: str = "",
) -> dict[str, Any]:
    """WQ-aligned self-correlation gate.

    WQ rejects a submission if any pool alpha has correlation ≥ corr_max AND
    the new alpha's sharpe is NOT ≥ (1+sharpe_margin) × correlated.sharpe.

    Returns a dict with 'accept' bool + diagnostic fields.
    """
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool

    corrs = sess.get_alpha_correlations(alpha_id)
    if not corrs:
        return {"accept": True, "reason": "no correlation data — accepting",
                "max_correlation": 0.0, "high_corr_count": 0}

    abs_max = max((abs(float(c.get("correlation", 0))) for c in corrs), default=0.0)
    high_corr = [c for c in corrs if abs(float(c.get("correlation", 0))) >= corr_max]
    if not high_corr:
        return {"accept": True, "reason": f"max_corr={abs_max:.3f} < {corr_max:.3f}",
                "max_correlation": abs_max, "high_corr_count": 0}

    # Need new alpha's sharpe — fetch if not already known
    new_sharpe = None
    try:
        m = sess.fetch_alpha_metrics(alpha_id)
        new_sharpe = m.sharpe
    except Exception as exc:
        return {"accept": False,
                "reason": f"high_corr count={len(high_corr)} but sharpe unknown: {exc}",
                "max_correlation": abs_max, "high_corr_count": len(high_corr)}

    if new_sharpe is None:
        return {"accept": False,
                "reason": f"high_corr count={len(high_corr)} but our sharpe missing",
                "max_correlation": abs_max, "high_corr_count": len(high_corr)}

    # Look up correlated alphas' sharpes (local pool first, WQ fallback)
    pool_by_id: dict[str, Any] = {}
    if tag:
        try:
            pool = AlphaPool(alpha_pool_path(tag))
            pool_by_id = {e.alpha_id: e for e in pool.entries}
        except Exception:
            pass

    blocking: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    for c in high_corr:
        corr_id = c.get("alpha") or c.get("id") or c.get("alphaId") or ""
        corr_value = float(c.get("correlation", 0))
        other_sharpe: Optional[float] = None
        if corr_id and corr_id in pool_by_id:
            other_sharpe = pool_by_id[corr_id].sharpe
        elif corr_id:
            try:
                om = sess.fetch_alpha_metrics(corr_id)
                other_sharpe = om.sharpe
            except Exception:
                other_sharpe = None

        entry: dict[str, Any] = {
            "id": corr_id, "correlation": round(corr_value, 4),
        }
        if other_sharpe is None:
            entry["reason"] = "unknown sharpe — assumed blocking"
            blocking.append(entry)
            continue
        required = (1.0 + sharpe_margin) * other_sharpe
        entry["other_sharpe"] = round(other_sharpe, 3)
        entry["required_sharpe"] = round(required, 3)
        entry["our_sharpe"] = round(new_sharpe, 3)
        if new_sharpe >= required:
            entry["status"] = "override (sharpe ≥ 110% of correlated)"
            overrides.append(entry)
        else:
            entry["status"] = f"BLOCK: short by {required - new_sharpe:.3f}"
            blocking.append(entry)

    accept = not blocking
    reason = (
        f"all {len(overrides)} high_corr alphas overridden by sharpe-margin"
        if accept else
        f"BLOCK: {len(blocking)} high_corr alphas with insufficient sharpe-margin"
    )
    return {
        "accept": accept,
        "reason": reason,
        "max_correlation": abs_max,
        "high_corr_count": len(high_corr),
        "new_sharpe": round(new_sharpe, 3),
        "blocking": blocking[:10],
        "overrides": overrides[:5],
    }


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


def cmd_pool_resubmit(args: argparse.Namespace) -> None:
    """POST /alphas/{id}/submit for every pool entry that's not already ACTIVE.

    Strategy: skip the local pre_check (it triple-burns API calls under rate
    limit). Let WQ's submit endpoint do the self-corr gating itself; capture
    the rejection message so we know why each alpha was rejected.
    """
    import time as _time
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    # Sort by fitness desc — submit best alphas first; later near-duplicates
    # will be rejected by WQ self-corr but won't displace earlier winners.
    entries = sorted(pool.entries, key=lambda e: -e.fitness)
    sess = session_from_env()
    submitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    print(f"Resubmitting {len(entries)} pool alphas for tag={args.tag}", file=sys.stderr)
    for i, entry in enumerate(entries):
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
    write verified_status + rejection_reasons back to pool.json."""
    import time as _t
    from agent_market.wq_brain.client import session_from_env
    from agent_market.wq_brain.paths import alpha_pool_path
    from agent_market.wq_brain.pool import AlphaPool
    pool = AlphaPool(alpha_pool_path(args.tag))
    sess = session_from_env()
    by_status: dict[str, int] = {}
    print(f"Syncing {len(pool)} pool alphas with WQ...", file=sys.stderr)
    for i, entry in enumerate(pool.entries):
        try:
            url = f"{sess._api_base}/alphas/{entry.alpha_id}"
            data = sess._request_with_retry("GET", url, timeout=20).json()
            actual_status = data.get("status") or "UNKNOWN"
            checks = (data.get("is") or {}).get("checks") or []
            failed = [{"name": c.get("name"), "result": c.get("result"),
                       "limit": c.get("limit"), "value": c.get("value")}
                      for c in checks if c.get("result") == "FAIL"]
            entry.verified_status = actual_status
            entry.verified_at = _t.time()
            entry.rejection_reasons = failed
            by_status[actual_status] = by_status.get(actual_status, 0) + 1
        except Exception as exc:
            print(f"WARN: {entry.alpha_id} sync failed: {exc}", file=sys.stderr)
        if (i + 1) % 5 == 0:
            print(f"... {i+1}/{len(pool)}", file=sys.stderr)
        _t.sleep(args.polite_sleep)
    pool._save()  # type: ignore[attr-defined]
    _emit({
        "ok": True, "tag": args.tag,
        "pool_size": len(pool),
        "summary_by_status": by_status,
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
    entries = sorted(pool.entries, key=lambda e: -e.fitness)

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
        pool._entries = kept  # type: ignore[attr-defined]
        pool._save()  # type: ignore[attr-defined]

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


def cmd_fetch_data(args: argparse.Namespace) -> None:
    """Bulk-fetch US stock OHLCV + sectors via yfinance into local parquet cache."""
    from agent_market.wq_brain.data_loader import fetch_data, load_tickers
    tickers = load_tickers(Path(args.tickers_file) if args.tickers_file else None)
    print(f"Fetching {len(tickers)} tickers from {args.start} to {args.end}", file=sys.stderr)
    summary = fetch_data(
        tickers, args.start, args.end,
        skip_sectors=args.skip_sectors,
    )
    print(json.dumps(summary, indent=2, default=str))


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
    summary = fetch_data(tickers, start, end, skip_sectors=True)
    print(json.dumps(summary, indent=2, default=str))


def cmd_local_simulate(args: argparse.Namespace) -> None:
    """Run wq_simulate against cached OHLCV — no WQ API call, pure local."""
    from agent_market.wq_brain.local_sim import simulate_expression_locally
    try:
        result = simulate_expression_locally(args.expr, rebalance_freq=args.rebalance_freq)
        _emit({
            "ok": True,
            "expr": result.expr,
            "wq_sharpe": result.wq_sharpe,
            "wq_fitness": result.wq_fitness,
            "wq_turnover": result.wq_turnover,
            "wq_returns": result.wq_returns,
            "submittable": result.submittable,
            "rating": result.rating,
            "raw": result.raw,
        })
    except Exception as exc:
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
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    q = urllib.parse.quote(args.query)
    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{q}"
        f"&start=0&max_results={args.max}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read()
    except Exception as exc:
        _emit({"ok": False, "error": f"arxiv fetch failed: {exc}"}, code=1)
        return
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        _emit({"ok": False, "error": f"arxiv parse failed: {exc}"}, code=1)
        return
    papers = []
    for entry in root.findall("atom:entry", ns):
        papers.append({
            "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
            "id": (entry.findtext("atom:id", default="", namespaces=ns) or "").strip(),
            "abstract": (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()[:1500],
            "published": (entry.findtext("atom:published", default="", namespaces=ns) or "").strip(),
        })
    _emit({"ok": True, "query": args.query, "count": len(papers), "papers": papers})


def cmd_docs(args: argparse.Namespace) -> None:
    from agent_market.wq_brain.operators import operators_prompt_block
    if args.topic == "operators":
        _emit({"ok": True, "topic": "operators", "content": operators_prompt_block()})
    else:
        _emit({"ok": False, "error": f"unknown topic: {args.topic}"}, code=1)


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
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("submit", help="submit alpha to WQ PROD pool")
    sp.add_argument("alpha_id")
    sp.add_argument("--tag", default="", help="local pool tag")
    sp.add_argument("--expr", default="", help="optional expr to record locally")
    sp.add_argument("--corr-max", type=float, default=0.7,
                    help="reject if max correlation with existing pool >= this (default 0.7)")
    sp.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="WQ override rule: high-corr submission still allowed if our sharpe ≥ (1+margin) × theirs (default 0.10 = 10%%)")
    sp.add_argument("--no-pre-check", action="store_true",
                    help="skip the correlation pre-check")
    sp.add_argument("--verify-after-sec", type=float, default=30.0,
                    help="seconds to wait before re-fetching alpha to verify ACTIVE/REJECTED (default 30, 0 to skip)")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("pre-check", help="check if alpha would be rejected by WQ self-correlation gate")
    sp.add_argument("alpha_id")
    sp.add_argument("--corr-max", type=float, default=0.7)
    sp.add_argument("--sharpe-margin", type=float, default=0.10,
                    help="WQ allows high_corr submission if sharpe ≥ (1+margin) × correlated.sharpe")
    sp.add_argument("--tag", default="",
                    help="local pool tag (used to look up correlated alphas' sharpes)")
    sp.set_defaults(func=cmd_pre_check)

    sp = sub.add_parser("fetch-data", help="bulk-fetch US stock OHLCV + sectors into local cache")
    sp.add_argument("--tickers-file", default=None,
                    help="path to file with one ticker per line; default uses bundled ~300-name list")
    sp.add_argument("--start", default="2021-01-01")
    sp.add_argument("--end", default="2026-05-05")
    sp.add_argument("--skip-sectors", action="store_true",
                    help="skip the slow per-ticker sector lookup")
    sp.set_defaults(func=cmd_fetch_data)

    sp = sub.add_parser("update-data", help="incremental daily OHLCV update")
    sp.add_argument("--tickers-file", default=None)
    sp.add_argument("--start", default="2021-01-01",
                    help="fallback start when cache is empty")
    sp.add_argument("--end", default=None,
                    help="default: today UTC")
    sp.set_defaults(func=cmd_update_data)

    sp = sub.add_parser("local-simulate", help="local WQ-aligned simulation against cached OHLCV (no WQ API)")
    sp.add_argument("expr")
    sp.add_argument("--rebalance-freq", type=int, default=5,
                    help="rebalance every N trading days (default: 5)")
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
                             help="POST /alphas/{id}/submit for every pool entry still UNSUBMITTED")
    rs.add_argument("--tag", required=True)
    rs.add_argument("--polite-sleep", type=float, default=3.0,
                    help="seconds to sleep between alphas (default 3.0)")
    rs.set_defaults(func=cmd_pool_resubmit)
    ps = pool_sub.add_parser("status",
                             help="query WQ for each pool alpha's current submission status")
    ps.add_argument("--tag", required=True)
    ps.set_defaults(func=cmd_pool_status)
    ss = pool_sub.add_parser("sync-status",
                             help="query WQ for each entry's actual status + IS checks; write verified_status + rejection_reasons into pool.json")
    ss.add_argument("--tag", required=True)
    ss.add_argument("--polite-sleep", type=float, default=2.0)
    ss.set_defaults(func=cmd_pool_sync_status)

    sp = sub.add_parser("corr", help="get correlations of alpha with WQ pool")
    sp.add_argument("alpha_id")
    sp.set_defaults(func=cmd_corr)

    sp = sub.add_parser("search-arxiv", help="search arxiv abstracts")
    sp.add_argument("query")
    sp.add_argument("--max", type=int, default=5)
    sp.set_defaults(func=cmd_search_arxiv)

    sp = sub.add_parser("docs", help="show local FASTEXPR docs")
    sp.add_argument("topic", choices=["operators"])
    sp.set_defaults(func=cmd_docs)

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
    sp.add_argument("--auto-submit", action="store_true")
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
