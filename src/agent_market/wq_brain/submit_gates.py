"""Pre-submit gates — local structural proxy + remote self-correlation.

Extracted from ``scripts/wq_brain.py`` so the same gate chain is reusable
from notebooks, integration tests, and any future "pre-flight" tool that
wants to score a candidate before spending WQ submit quota.

Three gates layered front-to-back:

  1. :func:`local_jaccard_gate` — token + semantic jaccard vs the ACTIVE
     pool. **Heuristic structural proxy** — NOT WQ's signal-correlation
     rule. Catches obvious near-duplicates without any API call. Cheap
     and offline. Calibrate against real submit outcomes before treating
     it as a binding gate.
  2. :func:`self_correlation_gate` — WQ ``get_alpha_correlations`` with the
     documented sharpe-margin override rule. One API call.
  3. :func:`summarize_rejection` — humanise WQ async-verify failures.

The CLI entry-point ``cmd_submit`` simply wires these three together.
"""
from __future__ import annotations

import math
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


# ── Defensive numeric helpers (Codex review #4 + #5) ──────────────────


def _finite_float(x: Any) -> Optional[float]:
    """Coerce ``x`` to a finite ``float`` or return ``None``.

    Used to defend the override path (and any caller that pulls metrics
    from JSON dumps) against ``None`` / ``NaN`` / non-numeric strings —
    historically the pool can carry these for failed-simulate rows. Bare
    ``float(x)`` would crash the worker on a single dirty entry.
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite_positive(x: Any) -> Optional[float]:
    """Like :func:`_finite_float` but additionally requires ``> 0``."""
    v = _finite_float(x)
    return v if (v is not None and v > 0.0) else None


# ── Gate 1: local jaccard ──────────────────────────────────────────────


def local_jaccard_gate(
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
    """Local structural-proxy gate vs the ACTIVE pool.

    **NOT WQ's submit-time rule.** WQ enforces signal/PnL self-correlation
    (≥ 0.7 against any pool alpha → reject unless sharpe-margin clause
    overrides). We approximate that with two cheap text-level proxies:

      * **token jaccard** — catches literal duplicates / repeated structures.
      * **semantic jaccard** — multiset over (operators, fields) so
        ``rank(ts_rank(close,N))`` ≈ ``rank(ts_rank(vwap,N))`` even though
        token jaccard is partial.

    Override semantics: if the caller passes ``candidate_sharpe`` and
    ``candidate_fitness``, the candidate may bypass *every* blocker iff
    ``candidate_sharpe ≥ (1+sharpe_margin) × b.sharpe`` AND
    ``candidate_fitness ≥ b.fitness`` for **all** blockers ``b``. A blocker
    is any ACTIVE alpha with ``token_jaccard ≥ threshold`` OR
    ``semantic_jaccard ≥ semantic_threshold``. Blockers (or candidates)
    with non-finite or non-positive sharpe/fitness disqualify the
    override — fail-closed, never wide-open. Calibrate by comparing
    override accept-rate vs WQ ACTIVE conversion before treating this as
    a binding pre-submit gate.
    """
    if not tag or not expr:
        return {
            "accept": True, "reason": "no tag/expr — skipped",
            "blockers": [], "blocker_count": 0,
            "override_applied": False, "vs_alpha_id": "",
            "vs_alpha_fitness": 0.0, "vs_alpha_sharpe": 0.0,
            "vs_alpha_expr": "",
            "max_jaccard": 0.0, "max_semantic": 0.0,
            "jaccard_threshold": threshold, "semantic_threshold": semantic_threshold,
        }

    pool = AlphaPool(alpha_pool_path(tag))
    active = [
        e for e in pool.entries
        if getattr(e, "verified_status", "") == "ACTIVE"
    ]
    if not active:
        return {
            "accept": True, "reason": "no ACTIVE alphas in pool yet",
            "blockers": [], "blocker_count": 0,
            "override_applied": False, "vs_alpha_id": "",
            "vs_alpha_fitness": 0.0, "vs_alpha_sharpe": 0.0,
            "vs_alpha_expr": "",
            "max_jaccard": 0.0, "max_semantic": 0.0,
            "jaccard_threshold": threshold, "semantic_threshold": semantic_threshold,
        }

    new_t = _tokens(expr)
    if not new_t:
        return {
            "accept": True, "reason": "empty token set",
            "blockers": [], "blocker_count": 0,
            "override_applied": False, "vs_alpha_id": "",
            "vs_alpha_fitness": 0.0, "vs_alpha_sharpe": 0.0,
            "vs_alpha_expr": "",
            "max_jaccard": 0.0, "max_semantic": 0.0,
            "jaccard_threshold": threshold, "semantic_threshold": semantic_threshold,
        }

    # Fix #1 (multi-blocker): score every ACTIVE alpha; collect EVERY
    # blocker that exceeds either threshold. The override path must clear
    # all of them, not just the max-jaccard one.
    blockers: list[dict[str, Any]] = []
    max_jac = 0.0
    max_sem = 0.0
    for a in active:
        a_expr = a.expr or ""
        a_t = _tokens(a_expr)
        jac = 0.0
        if a_t:
            union = len(new_t | a_t)
            jac = (len(new_t & a_t) / union) if union else 0.0
        sem = semantic_jaccard(expr, a_expr) if a_expr else 0.0
        if jac > max_jac:
            max_jac = jac
        if sem > max_sem:
            max_sem = sem
        if jac >= threshold or sem >= semantic_threshold:
            blockers.append({
                "alpha_id": a.alpha_id,
                "fitness": _finite_float(a.fitness) or 0.0,
                "sharpe":  _finite_float(a.sharpe)  or 0.0,
                "fitness_raw": a.fitness,
                "sharpe_raw":  a.sharpe,
                "jaccard":  round(jac, 3),
                "semantic": round(sem, 3),
                "blocked_by": "token" if jac >= threshold else "semantic",
                "expr": a_expr[:120],
            })
    # Sort blockers by required_sharpe desc so blockers[0] is the strictest.
    # When sharpe is non-positive the override path will reject the blocker
    # outright, so the sort key only needs to disambiguate finite cases.
    blockers.sort(key=lambda b: (
        -((1.0 + sharpe_margin) * (b["sharpe"] if b["sharpe"] > 0 else 0.0)),
        -(b["fitness"] if b["fitness"] > 0 else 0.0),
    ))

    accept = not blockers

    # Build a block_reason that PRESERVES the legacy single-blocker phrasing
    # ("token-jaccard" / "semantic-jaccard") so existing callers and tests
    # keep working, while reporting blocker count when there is more than one.
    if not blockers:
        block_reason = (
            f"jaccard={max_jac:.3f} < {threshold:.3f}, "
            f"semantic={max_sem:.3f} < {semantic_threshold:.3f}"
        )
    else:
        primary = blockers[0]
        if primary["blocked_by"] == "token":
            primary_reason = (
                f"BLOCK token-jaccard={primary['jaccard']:.3f} ≥ {threshold:.3f} "
                f"vs ACTIVE {primary['alpha_id']} "
                f"(fi={primary['fitness']:.2f}, sh={primary['sharpe']:.2f})"
            )
        else:
            primary_reason = (
                f"BLOCK semantic-jaccard={primary['semantic']:.3f} ≥ "
                f"{semantic_threshold:.3f} vs ACTIVE {primary['alpha_id']} "
                f"(fi={primary['fitness']:.2f}, sh={primary['sharpe']:.2f}) "
                f"— operator skeleton near-identical even after field swap"
            )
        if len(blockers) > 1:
            block_reason = f"{primary_reason} (+{len(blockers) - 1} more blocker(s))"
        else:
            block_reason = primary_reason

    # Fix #4 (defensive override): require finite candidate metrics AND
    # finite-positive blocker metrics. Otherwise override declined — never
    # let "blocker.sharpe = 0.0" silently grant unconditional accept.
    #
    # Codex review R2-#3: override_mode controls whether the relative
    # fitness clause is enforced.
    #   * "sharpe_and_fitness" (default, strict, original behavior):
    #       cand_sh ≥ (1+m) × b.sh  AND  cand_fi ≥ b.fi  for every blocker.
    #     This is a *local* heuristic — NOT WQ's submit-time rule.
    #   * "sharpe_only" (looser, closer to WQ's documented sharpe-margin
    #     clause): cand_sh ≥ (1+m) × b.sh for every blocker, and
    #     cand_fi ≥ ``absolute_fitness_floor`` (default 1.0 = WQ's IS bar).
    #     Use this when calibration shows the strict clause overblocks.
    override_applied = False
    decline_note = ""
    cand_sh = _finite_float(candidate_sharpe)
    cand_fi = _finite_float(candidate_fitness)

    if override_mode not in ("sharpe_and_fitness", "sharpe_only"):
        raise ValueError(
            f"override_mode must be 'sharpe_and_fitness' or 'sharpe_only', "
            f"got {override_mode!r}"
        )

    if blockers and cand_sh is not None and cand_fi is not None:
        worst_short_sh = 0.0
        worst_short_fi = 0.0
        cleared_all = True
        # In sharpe_only mode the absolute fitness floor is checked once,
        # not per blocker.
        if override_mode == "sharpe_only" and cand_fi < absolute_fitness_floor:
            cleared_all = False
            decline_note = (
                f"sharpe_only mode: candidate fi={cand_fi:.2f} < "
                f"absolute_fitness_floor={absolute_fitness_floor:.2f}"
            )
        else:
            for b in blockers:
                b_sh = _finite_positive(b["sharpe_raw"])
                b_fi = _finite_positive(b["fitness_raw"])
                if b_sh is None or (b_fi is None and override_mode == "sharpe_and_fitness"):
                    cleared_all = False
                    decline_note = (
                        f"blocker {b['alpha_id']} has non-positive/non-finite metrics "
                        f"(sh={b['sharpe_raw']!r}, fi={b['fitness_raw']!r}); "
                        "override fail-closed"
                    )
                    break
                required_sh = (1.0 + sharpe_margin) * b_sh
                sh_short = max(0.0, required_sh - cand_sh)
                fi_short = (
                    max(0.0, (b_fi or 0.0) - cand_fi)
                    if override_mode == "sharpe_and_fitness" else 0.0
                )
                if sh_short > 0 or fi_short > 0:
                    cleared_all = False
                    if sh_short + fi_short > worst_short_sh + worst_short_fi:
                        worst_short_sh = sh_short
                        worst_short_fi = fi_short
                        decline_note = (
                            f"failed vs {b['alpha_id']}: "
                            f"sh shortfall={sh_short:.2f}, fi shortfall={fi_short:.2f}"
                        )
        if cleared_all:
            accept = True
            override_applied = True

    if override_applied:
        reason = (
            f"{block_reason} → OVERRIDE ({override_mode}): candidate sh={cand_sh:.2f} "
            f"fi={cand_fi:.2f} clears all {len(blockers)} blocker(s) at "
            f"sharpe-margin={sharpe_margin:.2f}"
        )
    elif blockers and decline_note:
        reason = f"{block_reason} — override declined: {decline_note}"
    else:
        reason = block_reason

    primary = blockers[0] if blockers else None
    return {
        "accept": accept,
        "max_jaccard": round(max_jac, 3),
        "max_semantic": round(max_sem, 3),
        "jaccard_threshold": threshold,
        "semantic_threshold": semantic_threshold,
        # Multi-blocker fields (Codex review fix #1)
        "blockers": [
            {k: v for k, v in b.items() if k not in ("fitness_raw", "sharpe_raw")}
            for b in blockers[:5]
        ],
        "blocker_count": len(blockers),
        # Backward-compat single-blocker fields point to the strictest blocker
        "vs_alpha_id":      primary["alpha_id"] if primary else "",
        "vs_alpha_fitness": primary["fitness"] if primary else 0.0,
        "vs_alpha_sharpe":  primary["sharpe"]  if primary else 0.0,
        "vs_alpha_expr":    primary["expr"]    if primary else "",
        "override_applied": override_applied,
        "override_mode": override_mode,
        "candidate_sharpe": candidate_sharpe,
        "candidate_fitness": candidate_fitness,
        "sharpe_margin": sharpe_margin,
        "absolute_fitness_floor": absolute_fitness_floor,
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


def auto_fill_metrics(tag: str, alpha_id: str) -> dict[str, Any]:
    """Look up most-recent ``(expr, sharpe, fitness)`` for ``alpha_id``.

    Used by :func:`local_jaccard_gate`'s sharpe-margin override path so
    the gate can compare candidate vs blocking ACTIVE alpha without
    spending an extra ``fetch_alpha_metrics`` API call. Returns a dict
    with whatever fields are available; missing keys are absent rather
    than ``None`` so callers can ``.get(...)`` cleanly.
    """
    if not tag or not alpha_id:
        return {}
    rows = read_tried(tried_exprs_path(tag), tail=2000)
    for r in reversed(rows):
        if r.get("alpha_id") != alpha_id:
            continue
        out: dict[str, Any] = {}
        for k in ("expr", "sharpe", "fitness", "turnover"):
            v = r.get(k)
            if v is not None:
                out[k] = v
        if out:
            return out
    return {}
