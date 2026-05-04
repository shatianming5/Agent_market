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

    # Pre-check: reject if too correlated with existing pool (unless --no-pre-check)
    if not args.no_pre_check:
        try:
            corrs = sess.get_alpha_correlations(args.alpha_id)
            max_corr = max((float(c.get("correlation", 0)) for c in corrs), default=0.0)
            if max_corr >= args.corr_max:
                _emit({
                    "ok": False,
                    "rejected_by": "pre_check",
                    "alpha_id": args.alpha_id,
                    "max_correlation": max_corr,
                    "corr_max_threshold": args.corr_max,
                    "reason": (
                        f"max_corr={max_corr:.3f} >= threshold={args.corr_max:.3f} "
                        f"— too similar to existing pool. Use --no-pre-check to override."
                    ),
                }, code=2)
                return
        except Exception as exc:
            # If pre_check itself fails, log but don't block submission
            print(f"WARN: pre_check failed ({exc}); proceeding to submit", file=sys.stderr)

    try:
        wq_resp = sess.submit_alpha(args.alpha_id)
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)
        return

    pool_added = False
    if args.tag:
        try:
            metrics = sess.fetch_alpha_metrics(args.alpha_id)
            if metrics.alpha_id and metrics.sharpe is not None:
                entry = AlphaPoolEntry(
                    alpha_id=metrics.alpha_id,
                    expr=args.expr or "(submitted via CLI)",
                    settings_dict={},
                    sharpe=float(metrics.sharpe),
                    fitness=float(metrics.fitness or 0.0),
                    returns=float(metrics.returns or 0.0),
                    turnover=float(metrics.turnover or 0.0),
                    tag=args.tag,
                    source="agent",
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


def cmd_pre_check(args: argparse.Namespace) -> None:
    """Pre-submission gate: query WQ correlations and decide submit/reject."""
    from agent_market.wq_brain.client import session_from_env
    try:
        sess = session_from_env()
        corrs = sess.get_alpha_correlations(args.alpha_id)
        max_corr = max((float(c.get("correlation", 0)) for c in corrs), default=0.0)
        accept = max_corr < args.corr_max
        _emit({
            "ok": True,
            "alpha_id": args.alpha_id,
            "max_correlation": max_corr,
            "corr_max_threshold": args.corr_max,
            "accept": accept,
            "reason": (
                f"max_corr={max_corr:.3f} < threshold={args.corr_max:.3f}"
                if accept else
                f"REJECT: max_corr={max_corr:.3f} >= threshold={args.corr_max:.3f} "
                f"(too similar to existing pool)"
            ),
            "top_5_correlations": sorted(
                corrs, key=lambda c: -abs(float(c.get("correlation", 0)))
            )[:5],
        }, code=0 if accept else 2)
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)}, code=1)


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
    sp.add_argument("--no-pre-check", action="store_true",
                    help="skip the correlation pre-check")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("pre-check", help="check if alpha is similar enough to existing pool to reject")
    sp.add_argument("alpha_id")
    sp.add_argument("--corr-max", type=float, default=0.7)
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
