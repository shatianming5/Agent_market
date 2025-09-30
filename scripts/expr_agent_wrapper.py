#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _is_json_ok(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        # minimal schema check
        return isinstance(data, dict) and 'expressions' in data and isinstance(data['expressions'], list)
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description='Wrapper for freqai_expression_agent with retry/backoff/fallback')
    parser.add_argument('--config', required=True)
    parser.add_argument('--feature-file', required=True)
    parser.add_argument('--output', default='user_data/freqai_expressions.json')
    parser.add_argument('--timeframe', default='1h')
    # pass-through LLM args
    parser.add_argument('--llm-model')
    parser.add_argument('--llm-count', type=int)
    parser.add_argument('--llm-loops', type=int)
    parser.add_argument('--llm-timeout', type=float)
    parser.add_argument('--feedback-top', type=int, default=0)
    parser.add_argument('--feedback')
    # wrapper controls
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--backoff', type=float, default=2.0)

    args, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    expr_script = root / 'freqtrade' / 'scripts' / 'freqai_expression_agent.py'
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = root / 'user_data' / 'llm_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, str(expr_script),
        '--config', args.config,
        '--feature-file', args.feature_file,
        '--output', str(out),
        '--timeframe', args.timeframe,
        '--feedback-top', str(args.feedback_top),
    ]
    if args.llm_model:
        base_cmd += ['--llm-model', args.llm_model]
    if args.llm_count is not None:
        base_cmd += ['--llm-count', str(args.llm_count)]
    if args.llm_loops is not None:
        base_cmd += ['--llm-loops', str(args.llm_loops)]
    if args.llm_timeout is not None:
        base_cmd += ['--llm-timeout', str(args.llm_timeout)]
    if args.feedback:
        base_cmd += ['--feedback', args.feedback]
    base_cmd += list(extra)

    # try with retries
    for i in range(max(1, args.retries)):
        print(f'[wrapper] run #{i+1}:', ' '.join(base_cmd))
        proc = subprocess.run(base_cmd, cwd=root)
        # keep raw copy
        ts = time.strftime('%Y%m%d-%H%M%S')
        raw_copy = raw_dir / f'expr_{ts}_try{i+1}.json'
        if out.exists():
            try:
                shutil.copy(out, raw_copy)
                print('[wrapper] saved raw:', raw_copy)
            except Exception:
                pass
        if proc.returncode == 0 and out.exists() and _is_json_ok(out):
            print('[wrapper] success')
            return 0
        # backoff
        time.sleep(max(0.0, args.backoff))

    print('[wrapper] retries exhausted, falling back to classic expressions', file=sys.stderr)
    # fallback: remove LLM args and re-run
    classic_cmd = [
        sys.executable, str(expr_script),
        '--config', args.config,
        '--feature-file', args.feature_file,
        '--output', str(out),
        '--timeframe', args.timeframe,
        '--feedback-top', str(args.feedback_top),
    ]
    if args.feedback:
        classic_cmd += ['--feedback', args.feedback]
    print('[wrapper] classic:', ' '.join(classic_cmd))
    proc = subprocess.run(classic_cmd, cwd=root)
    return proc.returncode


if __name__ == '__main__':
    raise SystemExit(main())

