from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_market.freqai.training.pipeline import TrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ML training pipeline from JSON config")
    parser.add_argument('--config', required=True, help='Path to training config JSON')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
        return 2
    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse config JSON: {exc}", file=sys.stderr)
        return 3

    print("[STEP] 00:00:00 [1/3] load dataset")
    pipe = TrainingPipeline(cfg)
    # Feature building happens inside run()
    print("[STEP] 00:00:01 [2/3] train model")
    result = pipe.run()
    print("[STEP] 00:00:02 [3/3] write artifacts")
    print(json.dumps({
        'model_path': str(result.model_path),
        'metrics': result.metrics,
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

