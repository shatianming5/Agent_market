#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_market import paths
from agent_market.strategy_miner.knowledge_base import KnowledgeBase
from agent_market.strategy_miner.runner import backfill_global_strategy_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill global strategy memory from local strategy_miner runs")
    parser.add_argument("--runs-root", default="", help="Override runs root (defaults to AGENT_MARKET_RUNS_ROOT)")
    parser.add_argument("--target", default="", help="Override target global knowledge base path")
    parser.add_argument("--current-run-id", default="", help="Skip this run id while backfilling")
    parser.add_argument("--reset", action="store_true", help="Delete target KB before rebuilding it")
    args = parser.parse_args()

    target = (
        paths.resolve_repo_path(args.target)
        if str(args.target or "").strip()
        else paths.global_strategy_knowledge_base_path()
    )
    if args.reset and target.exists():
        target.unlink()

    runs_root = Path(args.runs_root).resolve() if str(args.runs_root or "").strip() else None
    kb = KnowledgeBase(target)
    stats = backfill_global_strategy_knowledge_base(
        kb,
        runs_root=runs_root,
        current_run_id=str(args.current_run_id or "").strip(),
    )
    payload = {
        **stats,
        "target": str(target.resolve()),
        "top_strategy_cards": [
            {
                "card_id": card.get("card_id"),
                "name": card.get("name"),
                "family": card.get("candidate_family"),
                "metrics": card.get("metrics"),
            }
            for card in kb.strategy_cards[:10]
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
