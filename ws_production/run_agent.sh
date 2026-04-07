#!/bin/bash
# Start OpenCode agent in this workspace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "$REPO_ROOT/.opencode.json" ] && [ -z "${OPENCODE_CONFIG:-}" ]; then
  export OPENCODE_CONFIG="$REPO_ROOT/.opencode.json"
fi

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

if [ -z "${OPENAI_BASE_URL:-}" ] && [ -n "${LLM_BASE_URL:-}" ]; then
  export OPENAI_BASE_URL="$LLM_BASE_URL"
elif [ -z "${OPENAI_BASE_URL:-}" ] && [ -n "${OPENAI_API_BASE:-}" ]; then
  export OPENAI_BASE_URL="$OPENAI_API_BASE"
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="$LLM_API_KEY"
fi
if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
  export OPENAI_API_KEY=_
fi
if [ -z "${LLM_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
  export LLM_API_KEY="$OPENAI_API_KEY"
fi
if [ -z "${LLM_BASE_URL:-}" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
  export LLM_BASE_URL="$OPENAI_BASE_URL"
fi

MODEL="${OPENCODE_MODEL:-custom/gpt-5.2}"
case "$MODEL" in
  */*) ;;
  *) MODEL="custom/$MODEL" ;;
esac
export OPENCODE_MODEL="$MODEL"

cd "$SCRIPT_DIR"
python3 "$REPO_ROOT/scripts/ws_production_preflight.py" --workspace "$SCRIPT_DIR" --model "$MODEL"

opencode run -m "$MODEL" \
  "Read GUIDE.md in this directory. You are an autonomous quant researcher. \
Your goal: find profitable trading strategies using the tools documented in GUIDE.md. \
Start by scanning cointegrated pairs, then backtest, then walk-forward validate. \
Record all experiments. Iterate until you find a strategy with positive mean profit \
across walk-forward windows."
