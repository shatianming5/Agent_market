#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent_market.agent_flow import AgentFlow, load_agent_flow_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Market end-to-end orchestrator")
    parser.add_argument('--config', required=True, help='Path to JSON configuration file')
    parser.add_argument('--steps', nargs='*', help='Optional subset of steps to run (feature, expression, ml, rl, backtest)')
    parser.add_argument('--log-dir', default='user_data/agent_logs', help='Directory to store agent flow log files')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"agent_flow_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    handlers = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=handlers,
    )
    logging.getLogger().info("Agent Flow log file: %s", log_file)

    cfg = load_agent_flow_config(Path(args.config))
    flow = AgentFlow(cfg)
    flow.run(args.steps)


if __name__ == '__main__':
    main()
