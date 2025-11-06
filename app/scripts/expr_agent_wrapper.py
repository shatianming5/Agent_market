#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Only add src/ to sys.path, prefer installed packages for freqtrade
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if SRC.exists():
    sp = str(SRC)
    if sp not in sys.path:
        sys.path.append(sp)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[STEP] {ts} {msg}", flush=True)


def _import_expr_agent():
    import importlib
    return importlib.import_module('agent_market.freqai.expression_agent')
def _inject_step_wrappers(expr):
    # Monkeypatch internal steps to emit fine-grained [STEP] logs
    steps = [
        ('resolve_feature_context','resolve'),
        ('load_features_config','load cfg'),
        ('load_data','load data'),
        ('build_feature_dataframe','build features'),
        ('generate_llm_expressions','generate'),
        ('evaluate_expressions','evaluate'),
    ]
    total = len(steps) + 1

    def _wrap(name, label, func, idx):
        def inner(*a, **kw):
            log(f"[{idx}/{total}] {label} start")
            t0 = time.time()
            try:
                return func(*a, **kw)
            finally:
                log(f"[{idx}/{total}] {label} done ({time.time()-t0:.2f}s)")
        return inner

    for i, (attr, label) in enumerate(steps, start=1):
        if hasattr(expr, attr):
            setattr(expr, attr, _wrap(attr, label, getattr(expr, attr), i))


def main() -> None:
    parser = argparse.ArgumentParser(description="Expression Agent Wrapper with step logs")
    # passthrough args (only collect; delegate to module)
    parser.add_argument("--feature-file")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--llm-model", default="gpt-3.5-turbo")
    parser.add_argument("--llm-count", type=int, default=1)
    parser.add_argument("--llm-loops", type=int, default=1)
    parser.add_argument("--llm-timeout", type=float, default=10)
    parser.add_argument("--feedback-top", type=int, default=0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--combo-top", type=int, default=0)
    parser.add_argument("--stability-windows", type=int, default=1)
    parser.add_argument("--stability-min-samples", type=int, default=50)
    parser.add_argument("--backtest-weight", type=float, default=0.0)
    parser.add_argument("--no-llm", dest="llm_enabled", action="store_false")
    parser.add_argument("--gp-enabled", action="store_true")
    parser.add_argument("--feedback")
    args, unknown = parser.parse_known_args()

    log("1/3 Validating inputs")
    if args.config and not Path(args.config).exists():
        print(f"[ERROR] config not found: {args.config}", flush=True)
    if args.feature_file and not Path(args.feature_file).exists():
        print(f"[ERROR] feature file not found: {args.feature_file}", flush=True)

    log("2/3 Running expression agent")
    expr = _import_expr_agent()
    _inject_step_wrappers(expr)

    # Rebuild argv for downstream parser
    argv = [
        "--feature-file", args.feature_file,
        "--config", args.config,
        "--output", args.output,
        "--timeframe", args.timeframe,
        "--llm-model", args.llm_model,
        "--llm-count", str(args.llm_count),
        "--llm-loops", str(args.llm_loops),
        "--llm-timeout", str(args.llm_timeout),
        "--feedback-top", str(args.feedback_top),
        "--top", str(args.top),
        "--combo-top", str(args.combo_top),
        "--stability-windows", str(args.stability_windows),
        "--stability-min-samples", str(args.stability_min_samples),
        "--backtest-weight", str(args.backtest_weight),
    ]
    if args.feedback:
        argv += ["--feedback", args.feedback]
    if not args.llm_enabled:
        argv += ["--no-llm"]
    if args.gp_enabled:
        argv += ["--gp-enabled"]
    # Append unknown passthrough flags
    argv += unknown

    # Temporarily replace sys.argv for downstream argparser
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        expr.main()  # type: ignore[attr-defined]
    finally:
        sys.argv = old_argv

    log("3/3 Done")


if __name__ == "__main__":
    main()




