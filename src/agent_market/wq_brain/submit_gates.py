"""Pre-submit gates — local jaccard + remote self-correlation.

Extracted from ``scripts/wq_brain.py`` so the same gate chain is reusable
from notebooks, integration tests, and any future "pre-flight" tool that
wants to score a candidate before spending WQ submit quota.

Three gates layered front-to-back:

  1. :func:`local_jaccard_gate` — token + semantic jaccard vs the ACTIVE
     pool. Catches >90% of self-correlation rejections without any API
     call. Cheap and offline.
  2. :func:`self_correlation_gate` — WQ ``get_alpha_correlations`` with the
     sharpe-margin override rule. One API call.
  3. :func:`summarize_rejection` — humanise WQ async-verify failures.

The CLI entry-point ``cmd_submit`` simply wires these three together.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .diversity import semantic_jaccard
from .paths import alpha_pool_path, tried_exprs_path
from .pool import AlphaPool
from .tried_log import read_tried


# ── Token helper ───────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _tokens(s: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(s.lower()))


# ── Gate 1: local jaccard ──────────────────────────────────────────────


def local_jaccard_gate(
    tag: str,
    expr: str,
    *,
    threshold: float = 0.7,
    semantic_threshold: float = 0.85,
) -> dict[str, Any]:
    """Block-or-accept a candidate by comparing it to the ACTIVE pool.

    WQ self-correlation gate (server-side) measures *signal* correlation.
    We use two cheap proxies:

      * **token jaccard** — catches literal duplicates / repeated structures.
      * **semantic jaccard** — multiset over (operators, fields) so
        ``rank(ts_rank(close,N))`` ≈ ``rank(ts_rank(vwap,N))`` correctly.

    Either threshold breached → BLOCK. Returns a dict carrying both scores
    and the blocking ACTIVE alpha (alpha_id, fitness, expr) so the agent
    can re-mutate intelligently.
    """
    if not tag or not expr:
        return {"accept": True, "reason": "no tag/expr — skipped"}

    pool = AlphaPool(alpha_pool_path(tag))
    active = [
        e for e in pool.entries
        if getattr(e, "verified_status", "") == "ACTIVE"
    ]
    if not active:
        return {"accept": True, "reason": "no ACTIVE alphas in pool yet"}

    new_t = _tokens(expr)
    if not new_t:
        return {"accept": True, "reason": "empty token set"}

    max_jac = 0.0
    max_sem = 0.0
    block_jac_id = ""
    block_sem_id = ""
    block_jac_fi = 0.0
    block_sem_fi = 0.0
    block_jac_expr = ""
    block_sem_expr = ""
    for a in active:
        a_expr = a.expr or ""
        a_t = _tokens(a_expr)
        if a_t:
            union = len(new_t | a_t)
            jac = (len(new_t & a_t) / union) if union else 0.0
            if jac > max_jac:
                max_jac = jac
                block_jac_id = a.alpha_id
                block_jac_fi = a.fitness
                block_jac_expr = a_expr
        sem = semantic_jaccard(expr, a_expr) if a_expr else 0.0
        if sem > max_sem:
            max_sem = sem
            block_sem_id = a.alpha_id
            block_sem_fi = a.fitness
            block_sem_expr = a_expr

    jac_block = max_jac >= threshold
    sem_block = max_sem >= semantic_threshold
    accept = not (jac_block or sem_block)

    if jac_block:
        reason = (
            f"BLOCK token-jaccard={max_jac:.3f} ≥ {threshold:.3f} vs ACTIVE "
            f"{block_jac_id} (fi={block_jac_fi:.2f})"
        )
    elif sem_block:
        reason = (
            f"BLOCK semantic-jaccard={max_sem:.3f} ≥ {semantic_threshold:.3f} "
            f"vs ACTIVE {block_sem_id} (fi={block_sem_fi:.2f}) — operator "
            f"skeleton near-identical even after field swap"
        )
    else:
        reason = (
            f"jaccard={max_jac:.3f} < {threshold:.3f}, "
            f"semantic={max_sem:.3f} < {semantic_threshold:.3f}"
        )
    return {
        "accept": accept,
        "max_jaccard": round(max_jac, 3),
        "max_semantic": round(max_sem, 3),
        "jaccard_threshold": threshold,
        "semantic_threshold": semantic_threshold,
        "vs_alpha_id": block_jac_id if jac_block else block_sem_id,
        "vs_alpha_fitness": block_jac_fi if jac_block else block_sem_fi,
        "vs_alpha_expr": (block_jac_expr if jac_block else block_sem_expr)[:120],
        "reason": reason,
    }


# ── Gate 2: WQ self-correlation ────────────────────────────────────────


class GateInfraError(RuntimeError):
    """Raised when a gate cannot complete because of *infrastructure*
    failure (network down, WQ session expired, 5xx) rather than because
    the candidate violated policy. Callers should fail-CLOSED on this
    rather than continuing to burn quota under the assumption that the
    pre-check passed.
    """


def self_correlation_gate(
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

    Returns a dict with 'accept' bool + diagnostic fields. Raises
    :class:`GateInfraError` when the gate cannot be evaluated due to
    *infrastructure* failure (network / 5xx / auth) — distinct from a
    legitimate policy BLOCK (correlation threshold breached). The CLI
    layer should default fail-closed on GateInfraError to avoid burning
    submit quota when the pre-check itself is broken.
    """
    try:
        corrs = sess.get_alpha_correlations(alpha_id)
    except Exception as exc:
        # WQ returns 404 on /alphas/{id}/correlations for alphas that
        # haven't been submitted yet (no peer correlations stored). That's
        # *information* ("no data to gate on"), NOT infrastructure failure.
        # Treat 404 as empty correlations and let the submit attempt
        # through; WQ will run its own self-corr check at submit time.
        msg = str(exc)
        if "404" in msg or "Not Found" in msg:
            return {
                "accept": True,
                "reason": "no correlation data (404 — alpha not yet in WQ pool); accepting",
                "max_correlation": 0.0, "high_corr_count": 0,
            }
        raise GateInfraError(
            f"get_alpha_correlations({alpha_id!r}) failed: {exc}"
        ) from exc

    if not corrs:
        return {
            "accept": True,
            "reason": "no correlation data — accepting",
            "max_correlation": 0.0, "high_corr_count": 0,
        }

    abs_max = max((abs(float(c.get("correlation", 0))) for c in corrs), default=0.0)
    high_corr = [c for c in corrs if abs(float(c.get("correlation", 0))) >= corr_max]
    if not high_corr:
        return {
            "accept": True,
            "reason": f"max_corr={abs_max:.3f} < {corr_max:.3f}",
            "max_correlation": abs_max, "high_corr_count": 0,
        }

    try:
        m = sess.fetch_alpha_metrics(alpha_id)
        new_sharpe = m.sharpe
    except Exception as exc:
        raise GateInfraError(
            f"fetch_alpha_metrics({alpha_id!r}) failed during self-corr "
            f"evaluation: {exc}"
        ) from exc
    if new_sharpe is None:
        return {
            "accept": False,
            "reason": (
                f"high_corr count={len(high_corr)} but our sharpe missing — "
                "policy BLOCK"
            ),
            "max_correlation": abs_max, "high_corr_count": len(high_corr),
        }

    pool_by_id: dict[str, Any] = {}
    if tag:
        try:
            pool = AlphaPool(alpha_pool_path(tag))
            pool_by_id = {e.alpha_id: e for e in pool.entries}
        except OSError:
            # pool file missing/corrupt is non-fatal for the gate — we
            # just fall back to remote sharpe lookups
            pool_by_id = {}

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


# ── Helpers used by cmd_submit ─────────────────────────────────────────


def summarize_rejection(reasons: list) -> str:
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


def auto_fill_expr(tag: str, alpha_id: str) -> str:
    """Look up the most-recent tried_exprs.jsonl row for ``alpha_id``."""
    if not tag or not alpha_id:
        return ""
    rows = read_tried(tried_exprs_path(tag), tail=2000)
    for r in reversed(rows):
        if r.get("alpha_id") == alpha_id and r.get("expr"):
            return r["expr"]
    return ""
