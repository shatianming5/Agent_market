#!/bin/bash
# Start OpenCode agent in this workspace
cd "$(dirname "$0")"
opencode run -m custom/gpt-5.2 \
  "Read GUIDE.md in this directory. You are an autonomous quant researcher. \
Your goal: find profitable trading strategies using the tools documented in GUIDE.md. \
Start by scanning cointegrated pairs, then backtest, then walk-forward validate. \
Record all experiments. Iterate until you find a strategy with positive mean profit \
across walk-forward windows."
