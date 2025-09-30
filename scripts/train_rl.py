from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    # Ensure src import
    repo = Path(__file__).resolve().parents[1]
    src = repo / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from agent_market.freqai.rl.trainer import RLTrainer  # noqa: WPS433

    parser = argparse.ArgumentParser(description='Train RL trading agent (PPO) on TradingEnv')
    parser.add_argument('--config', type=str, help='JSON config path for RLTrainer')
    # quick args if no config supplied
    parser.add_argument('--feature-file', type=str, help='freqai_features.json')
    parser.add_argument('--data-dir', type=str, default='freqtrade/user_data/data')
    parser.add_argument('--exchange', type=str, default='binanceus')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--pairs', type=str, default='BTC/USDT')
    parser.add_argument('--total-timesteps', type=int, default=10000)
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--algo-params', type=str, default='{}', help='JSON of PPO kwargs')
    parser.add_argument('--model-dir', type=str, default='artifacts/models/rl_ppo_demo')

    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    else:
        try:
            algo_params = json.loads(args.algo_params)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f'Invalid --algo-params JSON: {exc}')
        cfg = {
            'data': {
                'feature_file': args.feature_file,
                'data_dir': args.data_dir,
                'exchange': args.exchange,
                'timeframe': args.timeframe,
                'pairs': args.pairs.split(),
            },
            'training': {
                'total_timesteps': int(args.total_timesteps),
            },
            'policy': args.policy,
            'algo': algo_params,
            'output': {
                'model_dir': args.model_dir,
            },
        }

    print('[rl] config:', json.dumps(cfg, ensure_ascii=False))
    trainer = RLTrainer(cfg)
    try:
        result = trainer.train()
    except ImportError as exc:  # stable-baselines3 not installed
        print('[rl] ERROR:', exc)
        print('[rl] Hint: pip install "stable-baselines3[extra]" gymnasium')
        return 2
    print('[rl] done. model_path=', result.model_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

