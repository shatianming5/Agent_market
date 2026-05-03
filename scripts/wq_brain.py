#!/usr/bin/env python3
"""WorldQuant BRAIN automated alpha mining CLI.

Subcommands:
  auth      — verify WQ credentials
  simulate  — run a single alpha expression through WebSim
  mine      — main LLM-driven iterative mining loop
  pool      — manage the local alpha pool (list / clean)
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
from agent_market.wq_brain.multiregion import MultiRegionConfig, MultiRegionRunner, parse_regions
from agent_market.wq_brain.paths import alpha_pool_path
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
        "pool": cmd_pool,
        "report": cmd_report,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
