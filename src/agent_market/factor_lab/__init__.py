"""Factor Lab — unified factor mining / backtest / deployment framework.

Usage (from project root):
  python scripts/factor_lab.py <command> [options]

Commands:
  data        download raw OHLCV / funding data
  features    merge engineered features (mtf4h, xs, pair, funding, micro) into feathers
  mine        iterative factor mining (IC + model-driven + LLM + composition)
  validate    sub-period stability + random-baseline + correlation audit
  backtest    walk-forward with a factor set
  deploy      list / switch / describe factor libraries

Artifacts:
  Factor library:  user_data/freqai_expressions.json (canonical)
  Snapshots:       user_data/freqai_expressions_<tag>.json
  State:           artifacts/factor_lab/<subcmd>/*

See docs: scripts/factor_lab.py --help
"""
from __future__ import annotations

__version__ = "1.0.0"
