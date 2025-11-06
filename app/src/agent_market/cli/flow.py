from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from agent_market.agent_flow import AgentFlow, load_agent_flow_config
from agent_market.paths import USER_DATA_DIR, PROJECT_ROOT, CONFIG_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="am-flow",
        description="Agent Market end-to-end orchestrator",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Optional subset of steps to run (download, feature, expression, ml, rl, backtest).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(USER_DATA_DIR / "agent_logs"),
        help="Directory to store agent flow log files.",
    )
    return parser


def _normalise_steps(steps: Sequence[str] | None) -> list[str] | None:
    if not steps:
        return None
    return [step.lower() for step in steps]


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_flow_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger().info("Agent Flow log file: %s", log_file)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidates = [
            (CONFIG_ROOT / config_path).resolve(),
            (PROJECT_ROOT / config_path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
    cfg = load_agent_flow_config(config_path)
    flow = AgentFlow(cfg)
    flow.run(_normalise_steps(args.steps))


if __name__ == "__main__":
    main()
