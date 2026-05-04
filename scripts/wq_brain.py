#!/usr/bin/env python3
"""WorldQuant BRAIN automated alpha mining CLI.

Subcommands:
  auth      — verify WQ credentials
  simulate  — run a single alpha expression through WebSim
  mine      — main LLM-driven iterative mining loop
  mine-multi — run mining across multiple regions in parallel
  mutate    — genetic mutation of top pool alphas (no LLM)
  pool      — manage the local alpha pool (list / clean)
  kb        — knowledge base management (search / list / rebuild / stats)
  report    — print run statistics from the registry
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── sys.path bootstrap ────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
for _p in (_REPO, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load .env into os.environ (same approach as factor_lab.py)
_ENV_PATH = _REPO / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        _item = _line.strip()
        if not _item or _item.startswith("#"):
            continue
        if _item.lower().startswith("export "):
            _item = _item[7:].strip()
        if "=" not in _item:
            continue
        _k, _v = _item.split("=", 1)
        _k = _k.strip()
        _v = _v.strip()
        if len(_v) >= 2 and _v[0] == _v[-1] and _v[0] in ('"', "'"):
            _v = _v[1:-1]
        if _k:
            os.environ.setdefault(_k, _v)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wq_brain")

# ── Lazy imports (after sys.path setup) ───────────────────────────────────
from agent_market.wq_brain.client import WQSession, session_from_env
from agent_market.wq_brain.dtypes import AlphaSettings, WQBrainConfig
from agent_market.wq_brain.knowledge_base import KnowledgeBase
from agent_market.wq_brain.multiregion import MultiRegionConfig, MultiRegionRunner, parse_regions
from agent_market.wq_brain.mutation import generate_mutations
from agent_market.wq_brain.operators import validate_expression
from agent_market.wq_brain.paths import alpha_pool_path, kb_index_path
from agent_market.wq_brain.pool import AlphaPool
from agent_market.wq_brain.registry import load_registry
from agent_market.wq_brain.runner import WQBrainRunner, make_wqb_run_id


# ── Subcommand implementations ────────────────────────────────────────────


def cmd_auth(args: argparse.Namespace) -> None:
    """Verify WQ credentials."""
    email = args.email or os.environ.get("WQ_EMAIL", "")
    password = args.password or os.environ.get("WQ_PASSWORD", "")
    if not email or not password:
        print("ERROR: provide --email and --password, or set WQ_EMAIL / WQ_PASSWORD in .env")
        sys.exit(1)
    sess = WQSession(email, password)
    try:
        sess.login()
        alphas = sess.list_my_alphas(limit=5)
        print(f"✓ Login successful. You have {len(alphas)} recent alphas.")
    except Exception as exc:
        print(f"✗ Login failed: {exc}")
        sys.exit(1)


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run a single alpha expression and print its metrics."""
    sess = session_from_env()
    sess.login()
    settings = AlphaSettings(
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
    )
    print(f"Submitting: {args.expr}")
    result = sess.simulate_and_parse(args.expr, settings, timeout=300.0)
    if result.status in ("ERROR", "FAILED"):
        print(f"✗ Simulation failed: {result.error}")
        sys.exit(1)
    print(f"Status     : {result.status}")
    print(f"Sharpe     : {result.sharpe}")
    print(f"Fitness    : {result.fitness}")
    print(f"Returns    : {result.returns}")
    print(f"Turnover   : {result.turnover}")
    print(f"Drawdown   : {result.drawdown}")
    print(f"Alpha ID   : {result.alpha_id}")
    passes = result.passes_quality
    print(f"Passes QC  : {'✓ YES' if passes else '✗ NO'} (Sharpe>1.25 & Fitness>1)")


def cmd_mine(args: argparse.Namespace) -> None:
    """Main LLM-driven iterative mining loop."""
    run_id = make_wqb_run_id(args.tag)
    config = WQBrainConfig(
        tag=args.tag,
        run_id=run_id,
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        auto_submit=args.auto_submit,
        model=args.model or "",
        hermes_provider=args.hermes_provider or "",
        hermes_yolo=args.hermes_yolo,
        hermes_toolsets=args.hermes_toolsets,
        hermes_reasoning_effort=args.hermes_reasoning_effort or "",
        max_turns=args.max_turns,
        quality_sharpe_min=args.sharpe_min,
        quality_fitness_min=args.fitness_min,
        corr_max=args.corr_max,
        dry_run=args.dry_run,
    )
    print(f"Starting WQ Brain run: {run_id}")
    print(f"  tag={args.tag}  region={args.region}  universe={args.universe}")
    print(f"  max_iterations={args.max_iterations}  batch_size={args.batch_size}")
    print(f"  auto_submit={args.auto_submit}  dry_run={args.dry_run}")

    runner = WQBrainRunner(config)
    runner.run()


def cmd_mine_multi(args: argparse.Namespace) -> None:
    """Multi-region parallel alpha mining."""
    try:
        regions = parse_regions(args.regions)
    except Exception as exc:
        print(f"ERROR parsing --regions: {exc}")
        sys.exit(1)

    config = MultiRegionConfig(
        base_tag=args.base_tag,
        regions=regions,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        delay=args.delay,
        decay=args.decay,
        global_max_concurrent=args.global_max_concurrent,
        auto_submit=args.auto_submit,
        dry_run=args.dry_run,
        model=args.model or "",
        hermes_provider=args.hermes_provider or "",
        hermes_yolo=args.hermes_yolo,
        hermes_toolsets=args.hermes_toolsets,
        hermes_reasoning_effort=args.hermes_reasoning_effort or "",
        max_turns=args.max_turns,
        quality_sharpe_min=args.sharpe_min,
        quality_fitness_min=args.fitness_min,
        corr_max=args.corr_max,
    )
    region_strs = ", ".join(f"{s.region}:{s.universe}" for s in regions)
    print(f"Starting multi-region run: base_tag={args.base_tag}")
    print(f"  regions     : {region_strs}")
    print(f"  global_sem  : {args.global_max_concurrent}")
    print(f"  iterations  : {args.max_iterations}  batch_size={args.batch_size}")
    print(f"  auto_submit : {args.auto_submit}  dry_run={args.dry_run}")

    runner = MultiRegionRunner(config)
    result = runner.run()

    print("\n=== Results ===")
    for tag, summary in result["results"].items():
        print(f"  [ok]    {tag}: {summary}")
    for tag, err in result["errors"].items():
        print(f"  [error] {tag}: {err}")


def cmd_mutate(args: argparse.Namespace) -> None:
    """Standalone genetic mutation of top pool alphas — no LLM needed."""
    source_tag = args.source_tag
    target_tag = args.target_tag or source_tag

    src_pool = AlphaPool(alpha_pool_path(source_tag))
    if len(src_pool) == 0:
        print(f"ERROR: source pool '{source_tag}' is empty — run `mine` first.")
        sys.exit(1)

    settings = AlphaSettings(
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        decay=args.decay,
        neutralization=args.neutralization,
        truncation=args.truncation,
    )
    parents = src_pool.top_n_by_fitness(args.top_n)
    print(
        f"Mutating top {len(parents)} alphas from pool '{source_tag}' "
        f"(pool size={len(src_pool)})"
    )

    candidates = generate_mutations(
        parents,
        settings,
        top_n=args.top_n,
        variants_per_parent=args.variants_per_parent,
        crossover=not args.no_crossover,
    )
    if not candidates:
        print("Mutation produced 0 candidates. Try a larger pool or different parents.")
        sys.exit(0)
    print(f"Generated {len(candidates)} mutation candidates")

    # Pre-filter invalid expressions
    valid = []
    for c in candidates:
        errs = validate_expression(c.expr)
        if errs:
            print(f"  [skip-invalid] {c.expr[:70]}  ({errs[0]})")
        elif src_pool.is_local_duplicate(c.expr):
            print(f"  [skip-dup]     {c.expr[:70]}")
        else:
            valid.append(c)
    print(f"After filter: {len(valid)} valid candidates to simulate")
    if not valid:
        sys.exit(0)

    if args.dry_run:
        print("dry-run: skipping simulation")
        for c in valid:
            print(f"  would simulate: {c.expr}")
        sys.exit(0)

    sess = session_from_env()
    sess.login()
    max_concurrent = int(os.environ.get("WQ_MAX_CONCURRENT", "3"))
    sess.batch_simulate(valid, max_concurrent=max_concurrent, timeout=300.0)

    target_pool = AlphaPool(alpha_pool_path(target_tag))
    passed = []
    print("\n=== Mutation Results ===")
    for c in valid:
        r = c.sim_result
        if r is None:
            print(f"  [no-result]  {c.expr[:70]}")
            continue
        if r.status in ("ERROR", "FAILED", "INVALID_EXPR"):
            print(f"  [error]      {c.expr[:60]}  ({r.status})")
            continue
        passes = r.passes_quality
        tag_str = "PASS" if passes else "fail"
        print(
            f"  [{tag_str}]  sh={r.sharpe or 0:.2f}  fi={r.fitness or 0:.2f}  "
            f"to={r.turnover or 0:.2f}  {c.expr[:60]}"
        )
        if passes:
            passed.append(c)

    print(f"\nPassed QC: {len(passed)} / {len(valid)}")
    if not passed:
        sys.exit(0)

    if args.auto_submit:
        submitted = 0
        for c in passed:
            if not c.sim_result or not c.sim_result.alpha_id:
                continue
            try:
                corr_list = sess.get_alpha_correlations(c.sim_result.alpha_id)
                max_corr = max(
                    (abs(float(x.get("value", 0))) for x in corr_list), default=0.0
                )
                if max_corr >= args.corr_max:
                    print(f"  [corr-skip]  {c.expr[:60]}  max_corr={max_corr:.3f}")
                    continue
                sess.submit_alpha(c.sim_result.alpha_id)
                entry = target_pool.add_from_candidate(c, tag=target_tag)
                if entry:
                    submitted += 1
                    print(
                        f"  [submitted]  alpha_id={c.sim_result.alpha_id}  "
                        f"sh={entry.sharpe:.2f}  fi={entry.fitness:.2f}"
                    )
            except Exception as exc:
                print(f"  [submit-error] {c.sim_result.alpha_id}: {exc}")
        print(f"Submitted to WQ pool: {submitted}")
    else:
        print("(--auto-submit not set; skipping WQ pool submission)")
        print("Passing alpha IDs (submit manually if desired):")
        for c in passed:
            if c.sim_result:
                print(f"  {c.sim_result.alpha_id}  {c.expr[:70]}")


def cmd_kb(args: argparse.Namespace) -> None:
    """Knowledge base management: search / list / rebuild / stats."""
    kb = KnowledgeBase(kb_index_path())

    if args.kb_cmd == "search":
        query = args.query
        top_k = args.top_k
        results = kb.search(query, top_k=top_k)
        if not results:
            print("No results.")
            return
        print(f"Top {len(results)} results for query: '{query}'")
        print("-" * 80)
        for i, e in enumerate(results, 1):
            sh = e.metadata.get("sharpe", "?")
            fi = e.metadata.get("fitness", e.metadata.get("turnover", "?"))
            to = e.metadata.get("turnover", "?")
            desc = e.metadata.get("desc", "")
            print(f"  {i:2d}. [{e.source}] sh={sh}  to={to}  {e.text}")
            if desc:
                print(f"       → {desc}")

    elif args.kb_cmd == "list":
        source_filter = args.source  # None means all
        entries = [e for e in kb._entries if source_filter is None or e.source == source_filter]
        if not entries:
            print(f"No entries{' for source=' + source_filter if source_filter else ''}.")
            return
        print(f"Knowledge base entries ({len(entries)}):")
        print("-" * 80)
        for i, e in enumerate(entries, 1):
            sh = e.metadata.get("sharpe", "?")
            to = e.metadata.get("turnover", "?")
            print(f"  {i:3d}. [{e.source:<10}] sh={sh}  to={to}  {e.text[:70]}")

    elif args.kb_cmd == "rebuild":
        tags = [t.strip() for t in (args.from_tags or "").split(",") if t.strip()]
        added = 0
        for tag in tags:
            pool = AlphaPool(alpha_pool_path(tag))
            for entry in pool._entries:
                kb.add_alpha(
                    entry.expr,
                    sharpe=entry.sharpe,
                    fitness=entry.fitness,
                    turnover=entry.turnover,
                )
                added += 1
            print(f"  Added {len(pool)} entries from pool '{tag}'")
        kb.save()
        print(f"KB rebuilt: {len(kb)} total entries ({added} added from pools, seeds always present)")

    elif args.kb_cmd == "stats":
        from collections import Counter
        counts: Counter = Counter(e.source for e in kb._entries)
        print(f"Knowledge base: {len(kb)} total entries")
        for src, cnt in sorted(counts.items()):
            print(f"  {src:<15}: {cnt}")

    else:
        print(f"Unknown kb subcommand: {args.kb_cmd}")
        sys.exit(1)


def cmd_pool(args: argparse.Namespace) -> None:
    """Manage the local alpha pool."""
    pool = AlphaPool(alpha_pool_path(args.tag))
    if args.pool_cmd == "list":
        entries = pool.top_n_by_fitness(args.top_n)
        print(f"Pool '{args.tag}': {len(pool)} entries. Top {len(entries)}:")
        for i, e in enumerate(entries, 1):
            print(
                f"  {i:3d}. sharpe={e.sharpe:.2f}  fitness={e.fitness:.2f}  "
                f"id={e.alpha_id}  {e.expr[:70]}"
            )
    elif args.pool_cmd == "clean":
        removed = pool.clean(keep_top_n=args.keep_top_n)
        print(f"Removed {removed} entries. Pool now has {len(pool)} entries.")
    else:
        print(f"Unknown pool subcommand: {args.pool_cmd}")
        sys.exit(1)


def cmd_report(args: argparse.Namespace) -> None:
    """Print run statistics from registry."""
    entries = load_registry()
    if args.tag:
        entries = [e for e in entries if e.get("tag") == args.tag]
    if not entries:
        print("No runs found.")
        return
    print(f"{'run_id':<50} {'tag':<20} {'submitted':>9} {'pool':>5} {'ts'}")
    print("-" * 100)
    for e in entries[-args.last_n:]:
        print(
            f"{e.get('run_id','?'):<50} "
            f"{e.get('tag','?'):<20} "
            f"{e.get('total_submitted', 0):>9} "
            f"{e.get('pool_size', '?'):>5} "
            f"{e.get('ts','')}"
        )


# ── Argument parser ───────────────────────────────────────────────────────


def _add_common_sim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--region", default=os.environ.get("WQ_DEFAULT_REGION", "USA"))
    p.add_argument("--universe", default=os.environ.get("WQ_DEFAULT_UNIVERSE", "TOP3000"))
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--decay", type=int, default=0)
    p.add_argument("--neutralization", default="SUBINDUSTRY")
    p.add_argument("--truncation", type=float, default=0.08)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wq_brain",
        description="WorldQuant BRAIN automated alpha mining",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # auth
    pa = sub.add_parser("auth", help="Verify WQ credentials")
    pa.add_argument("--email", default="")
    pa.add_argument("--password", default="")

    # simulate
    ps = sub.add_parser("simulate", help="Simulate a single alpha expression")
    ps.add_argument("--expr", required=True, help="FASTEXPR expression to simulate")
    _add_common_sim_args(ps)

    # mine
    pm = sub.add_parser("mine", help="Run LLM-driven alpha mining loop")
    pm.add_argument("--tag", required=True, help="Run tag (used for pool naming)")
    _add_common_sim_args(pm)
    pm.add_argument("--max-iterations", type=int, default=50)
    pm.add_argument("--batch-size", type=int, default=5, help="Alphas per LLM call")
    pm.add_argument("--max-concurrent", type=int,
                    default=int(os.environ.get("WQ_MAX_CONCURRENT", "3")))
    pm.add_argument("--auto-submit", action="store_true", help="Submit passing alphas to WQ pool")
    pm.add_argument("--dry-run", action="store_true", help="Skip actual WQ API submission")
    pm.add_argument("--model", default="")
    pm.add_argument("--hermes-provider", default="")
    pm.add_argument("--hermes-yolo", action="store_true", default=True)
    pm.add_argument("--no-hermes-yolo", dest="hermes_yolo", action="store_false")
    pm.add_argument("--hermes-toolsets", default="terminal,file")
    pm.add_argument("--hermes-reasoning-effort", default="")
    pm.add_argument("--max-turns", type=int, default=30)
    pm.add_argument("--sharpe-min", type=float, default=1.25)
    pm.add_argument("--fitness-min", type=float, default=1.0)
    pm.add_argument("--corr-max", type=float, default=0.70)

    # mine-multi
    pmm = sub.add_parser("mine-multi", help="Run LLM-driven mining across multiple regions in parallel")
    pmm.add_argument("--base-tag", required=True, help="Base tag; each region appends its suffix")
    pmm.add_argument(
        "--regions",
        required=True,
        help="Comma-separated region specs, e.g. 'USA:TOP3000,CHN:TOP2000,IND:TOP500'",
    )
    pmm.add_argument("--delay", type=int, default=1)
    pmm.add_argument("--decay", type=int, default=0)
    pmm.add_argument("--max-iterations", type=int, default=50)
    pmm.add_argument("--batch-size", type=int, default=5)
    pmm.add_argument(
        "--global-max-concurrent",
        type=int,
        default=int(os.environ.get("WQ_MAX_CONCURRENT", "3")),
        help="Total concurrent simulations across all regions",
    )
    pmm.add_argument("--auto-submit", action="store_true")
    pmm.add_argument("--dry-run", action="store_true")
    pmm.add_argument("--model", default="")
    pmm.add_argument("--hermes-provider", default="")
    pmm.add_argument("--hermes-yolo", action="store_true", default=True)
    pmm.add_argument("--no-hermes-yolo", dest="hermes_yolo", action="store_false")
    pmm.add_argument("--hermes-toolsets", default="terminal,file")
    pmm.add_argument("--hermes-reasoning-effort", default="")
    pmm.add_argument("--max-turns", type=int, default=30)
    pmm.add_argument("--sharpe-min", type=float, default=1.25)
    pmm.add_argument("--fitness-min", type=float, default=1.0)
    pmm.add_argument("--corr-max", type=float, default=0.70)

    # mutate
    pmt = sub.add_parser("mutate", help="Genetic mutation of top pool alphas (no LLM)")
    pmt.add_argument("--source-tag", required=True, help="Pool tag to take parents from")
    pmt.add_argument("--target-tag", default="", help="Pool tag to save results (default: same as source)")
    _add_common_sim_args(pmt)
    pmt.add_argument("--top-n", type=int, default=10, help="Number of top pool entries to use as parents")
    pmt.add_argument("--variants-per-parent", type=int, default=4)
    pmt.add_argument("--no-crossover", action="store_true", help="Disable crossover mutation")
    pmt.add_argument("--auto-submit", action="store_true")
    pmt.add_argument("--dry-run", action="store_true")
    pmt.add_argument("--corr-max", type=float, default=0.70)

    # kb
    pkb = sub.add_parser("kb", help="Knowledge base management")
    kb_sub = pkb.add_subparsers(dest="kb_cmd", required=True)

    kb_search = kb_sub.add_parser("search", help="Search the KB")
    kb_search.add_argument("--query", required=True, help="Search query text")
    kb_search.add_argument("--top-k", type=int, default=5)

    kb_list = kb_sub.add_parser("list", help="List KB entries")
    kb_list.add_argument("--source", default=None, choices=["seed", "pool"],
                         help="Filter by source (seed/pool). Omit for all.")

    kb_rebuild = kb_sub.add_parser("rebuild", help="Rebuild KB from pool(s) + seeds")
    kb_rebuild.add_argument("--from-tags", default="",
                            help="Comma-separated pool tags to import (e.g. 'wqb_v1,wqb_v2')")

    kb_sub.add_parser("stats", help="Show KB entry counts by source")

    # pool
    pp = sub.add_parser("pool", help="Manage local alpha pool")
    pp.add_argument("--tag", required=True)
    pool_sub = pp.add_subparsers(dest="pool_cmd", required=True)
    pool_list = pool_sub.add_parser("list")
    pool_list.add_argument("--top-n", type=int, default=20)
    pool_clean = pool_sub.add_parser("clean")
    pool_clean.add_argument("--keep-top-n", type=int, default=100)

    # report
    pr = sub.add_parser("report", help="Print run statistics")
    pr.add_argument("--tag", default="")
    pr.add_argument("--last-n", type=int, default=20)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "auth": cmd_auth,
        "simulate": cmd_simulate,
        "mine": cmd_mine,
        "mine-multi": cmd_mine_multi,
        "mutate": cmd_mutate,
        "kb": cmd_kb,
        "pool": cmd_pool,
        "report": cmd_report,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
