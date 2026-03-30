#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_market import paths
from agent_market.demo_data import ensure_demo_ohlcv_feather
from agent_market.strategy_miner.artifacts import leaderboard_path
from agent_market.strategy_miner.dtypes import MinerConfig
from agent_market.strategy_miner.runner import miner_run_dir, run_strategy_miner


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_demo_data(cfg: MinerConfig) -> None:
    ft_cfg_path = paths.resolve_repo_path(cfg.freqtrade_config)
    if not ft_cfg_path.exists():
        raise FileNotFoundError(f"freqtrade_config not found: {ft_cfg_path}")

    ft_payload = _load_json(ft_cfg_path)
    datadir = paths.resolve_repo_path(str(ft_payload.get("datadir") or "user_data/data"))
    exchange = str((ft_payload.get("exchange") or {}).get("name") or "unknown")
    pairs = (ft_payload.get("exchange") or {}).get("pair_whitelist") or []
    timeframe = str(ft_payload.get("timeframe") or "1h")

    if not isinstance(pairs, list) or not pairs:
        raise ValueError("freqtrade_config.exchange.pair_whitelist is empty")

    created = ensure_demo_ohlcv_feather(
        datadir=datadir,
        exchange=exchange,
        pairs=[str(p) for p in pairs],
        timeframe=timeframe,
        timerange=str(cfg.timerange),
    )

    logging.getLogger(__name__).info(
        "Demo data ready: datadir=%s created=%d", datadir, len(created)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strategy miner in recovery mode")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/strategy_miner_recovery.json"),
        help="Path to recovery config JSON",
    )
    parser.add_argument("--run-id", default=None, help="Override run_id")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        import freqtrade  # noqa: F401
    except Exception as exc:
        logging.getLogger(__name__).error(
            "freqtrade is required for real backtesting. Install with: pip install freqtrade (or requirements-full) (%s)",
            exc,
        )
        return 2

    cfg_dict = _load_json(args.config)
    cfg = MinerConfig.from_dict(cfg_dict)

    # Ensure minimal demo OHLCV exists so backtesting can run offline.
    _ensure_demo_data(cfg)

    state = run_strategy_miner(cfg, run_id=args.run_id)

    miner_dir = miner_run_dir(state.run_id)
    lb = leaderboard_path(miner_dir)

    print(f"run_id: {state.run_id}")
    print(f"miner_dir: {miner_dir}")
    print(f"best_reward: {state.best_reward}")
    print(f"leaderboard: {lb}")

    if lb.exists():
        payload = _load_json(lb)
        items = payload.get("items") or []
        top3 = items[:3]
        print("\nTop3:")
        for i, row in enumerate(top3, 1):
            print(
                f"{i}. {row.get('name')} reward={row.get('reward')} profit={row.get('profit_total_pct')}% "
                f"trades={row.get('trades')} winrate={row.get('winrate')} max_dd={row.get('max_drawdown_abs')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
