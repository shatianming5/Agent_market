#!/usr/bin/env python3
"""Run startup preflight checks for the ws_production workspace."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent_market.runtime_preflight import run_ws_production_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ws_production startup preflight.")
    parser.add_argument("--workspace", default="ws_production", help="Workspace directory to validate")
    parser.add_argument("--model", default="custom/gpt-5.2", help="OpenCode model to validate")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    report = run_ws_production_preflight(
        workspace=Path(args.workspace),
        model=str(args.model),
        raise_on_error=False,
    )
    print(f"ok={report['ok']} warnings={report['warnings']} errors={report['errors']}")
    print(f"report={Path(args.workspace).resolve() / 'results' / 'preflight.json'}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
