#!/bin/bash
# Start OpenCode agent in this workspace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

if [ -z "${LLM_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && { [ -n "${LLM_BASE_URL:-}" ] || [ -n "${OPENAI_BASE_URL:-}" ] || [ -n "${OPENAI_API_BASE:-}" ]; }; then
  export LLM_API_KEY=_
fi

cd "$SCRIPT_DIR"
python3 "$REPO_ROOT/scripts/ws_production_preflight.py" --workspace "$SCRIPT_DIR" --model "custom/gpt-5.2"

opencode run -m custom/gpt-5.2 \
  "Read GUIDE.md in this directory. You are an autonomous quant researcher. \
Your goal: find profitable trading strategies using the tools documented in GUIDE.md. \
Start by scanning cointegrated pairs, then backtest, then walk-forward validate. \
Record all experiments. Iterate until you find a strategy with positive mean profit \
across walk-forward windows."
