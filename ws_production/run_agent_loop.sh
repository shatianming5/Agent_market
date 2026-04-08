#!/bin/bash
# OpenCode Agent Continuous Loop
# Runs opencode in repeated cycles, each time feeding back the previous results.
#
# Usage:
#   cd ws_NNN
#   ./run_agent_loop.sh           # default 5 iterations
#   ./run_agent_loop.sh 10        # 10 iterations
#   ./run_agent_loop.sh 0         # infinite loop (Ctrl+C to stop)

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

MAX_ITERATIONS=${1:-5}
ITERATION=0
MODEL="${OPENCODE_MODEL:-custom/gpt-5.2}"
case "$MODEL" in
    */*) ;;
    *) MODEL="custom/$MODEL" ;;
esac
export OPENCODE_MODEL="$MODEL"

cd "$SCRIPT_DIR"
python3 "$REPO_ROOT/scripts/ws_production_preflight.py" --workspace "$SCRIPT_DIR" --model "$MODEL"

echo "======================================"
echo "OpenCode Agent Continuous Loop"
echo "Model: $MODEL"
echo "Max iterations: $MAX_ITERATIONS (0=infinite)"
echo "======================================"

while true; do
    ITERATION=$((ITERATION + 1))

    if [ "$MAX_ITERATIONS" -gt 0 ] && [ "$ITERATION" -gt "$MAX_ITERATIONS" ]; then
        echo "Max iterations ($MAX_ITERATIONS) reached. Stopping."
        break
    fi

    echo ""
    echo "======================================"
    echo "ITERATION $ITERATION / $MAX_ITERATIONS"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"

    FACTOR_ARGS=(--workspace "$SCRIPT_DIR")
    if [ "${WS_FACTOR_LLM_ENABLED:-0}" = "1" ]; then
        FACTOR_ARGS+=(--llm-enabled --model "$MODEL")
    else
        FACTOR_ARGS+=(--no-llm)
    fi

    echo "Running factor mining/evaluation cycle..."
    python3 "$REPO_ROOT/scripts/ws_production_factor_cycle.py" \
        "${FACTOR_ARGS[@]}" 2>&1 | tee -a results/agent_loop.log || true

    # Build context from previous results
    CONTEXT=""
    if [ -f results/last_cycle.json ]; then
        CONTEXT="Previous cycle results: $(cat results/last_cycle.json | python3 -c 'import json,sys; r=json.load(sys.stdin); print(f"strategies={r.get(\"summary\",{}).get(\"total_strategies\",\"?\")}, phases={r.get(\"summary\",{}).get(\"cycle_phases_completed\",\"?\")}")' 2>/dev/null || echo 'available'). "
    fi
    FACTOR_CONTEXT=""
    if [ -f results/factor_cycle_latest.json ]; then
        FACTOR_CONTEXT="$(cat results/factor_cycle_latest.json | python3 -c 'import json,sys; r=json.load(sys.stdin); top=r.get("top_expression") or {}; print(f"Latest factor cycle ok={r.get(\"ok\")}, top={top.get(\"name\") or \"?\"}, score={top.get(\"score\")}, report={r.get(\"validation_report\") or \"n/a\"}, eval={((r.get(\"factor_eval_artifacts\") or {}).get(\"factor_scores_json\") or \"n/a\")}")' 2>/dev/null || echo 'Latest factor cycle available.'). "
    fi

    # Run opencode
    opencode run -m "$MODEL" \
        "${CONTEXT}${FACTOR_CONTEXT}Read GUIDE.md. You are iteration $ITERATION of a continuous research loop.

Your tasks for this iteration:
1. Review the latest factor mining/evaluation outputs in results/factor_cycle_latest.json and use them as research input.
2. Run continuous_runner.ContinuousRunner(exchange='gate').run_cycle(skip_download=True)
3. Check if any strategies need parameter recalibration (adaptive_params.AdaptiveEngine)
4. If fewer than 3 strategies in paper/active, discover new ones
5. Run gate_pipeline on any new strategies
6. Generate daily report (report_generator.generate_daily_report)
7. Save cycle summary to results/last_cycle.json

Execute Python code for each step. Be thorough but efficient." 2>&1 | tee -a results/agent_loop.log

    if [ -f "$REPO_ROOT/workspace/results/last_cycle.json" ]; then
        cp "$REPO_ROOT/workspace/results/last_cycle.json" "$SCRIPT_DIR/results/last_cycle.json"
    fi
    if [ -f "$REPO_ROOT/workspace/results/paper_returns_latest.json" ]; then
        cp "$REPO_ROOT/workspace/results/paper_returns_latest.json" "$SCRIPT_DIR/results/paper_returns_latest.json"
    fi

    # Wait between iterations
    if [ "$MAX_ITERATIONS" -eq 0 ] || [ "$ITERATION" -lt "$MAX_ITERATIONS" ]; then
        echo "Waiting 10 seconds before next iteration..."
        sleep 10
    fi
done

echo ""
echo "======================================"
echo "Loop complete: $ITERATION iterations"
echo "======================================"
