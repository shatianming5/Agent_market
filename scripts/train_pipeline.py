from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    # Ensure src is importable
    root = Path(__file__).resolve().parents[1]
    src = root / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from agent_market.freqai.training.pipeline import TrainingPipeline  # noqa: WPS433

    parser = argparse.ArgumentParser(description='Run unified ML training pipeline')
    parser.add_argument('--config', type=str, help='JSON config file for training pipeline')
    parser.add_argument('--feature-file', type=str, help='Path to freqai_features.json')
    parser.add_argument('--data-dir', type=str, default='freqtrade/user_data/data')
    parser.add_argument('--exchange', type=str, default='binanceus')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--pairs', type=str, default='BTC/USDT', help='Space separated pairs')
    parser.add_argument('--model', type=str, default='lightgbm', help='lightgbm|xgboost|catboost')
    parser.add_argument('--params', type=str, default='{}', help='JSON string of model params')
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--model-dir', type=str, default='artifacts/models/auto')
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    else:
        try:
            model_params = json.loads(args.params)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f'Invalid --params JSON: {exc}')
        cfg = {
            'data': {
                'feature_file': args.feature_file,
                'data_dir': args.data_dir,
                'exchange': args.exchange,
                'timeframe': args.timeframe,
                'pairs': args.pairs.split(),
            },
            'model': {
                'name': args.model,
                'params': model_params,
            },
            'training': {
                'validation_ratio': args.validation_ratio,
            },
            'output': {
                'model_dir': args.model_dir,
            },
        }

    # Basic validation
    if 'data' not in cfg or 'feature_file' not in cfg['data']:
        raise SystemExit('Missing data.feature_file in config')

    print('[train] config:', json.dumps(cfg, ensure_ascii=False))
    pipe = TrainingPipeline(cfg)
    result = pipe.run()
    print('[train] done. model_path=', result.model_path)
    print('[train] metrics=', json.dumps(result.metrics, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

