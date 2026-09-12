"""Allow running as ``python -m agent_market.strategy_miner``."""
from __future__ import annotations

import sys

from agent_market.strategy_miner.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
