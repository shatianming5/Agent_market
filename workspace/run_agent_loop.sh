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
if [ -f "$REPO_ROOT/.opencode.json" ] && [ -z "${OPENCODE_CONFIG:-}" ]; then
    export OPENCODE_CONFIG="$REPO_ROOT/.opencode.json"
fi

MAX_ITERATIONS=${1:-5}
ITERATION=0
MODEL="${OPENCODE_MODEL:-custom/gpt-5.2}"
export OPENCODE_MODEL="$MODEL"

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

    # Build context from previous results
    CONTEXT=""
    if [ -f results/last_cycle.json ]; then
        CONTEXT="Previous cycle results: $(cat results/last_cycle.json | python3 -c 'import json,sys; r=json.load(sys.stdin); print(f"strategies={r.get(\"summary\",{}).get(\"total_strategies\",\"?\")}, phases={r.get(\"summary\",{}).get(\"cycle_phases_completed\",\"?\")}")' 2>/dev/null || echo 'available'). "
    fi

    # Run opencode
    opencode run -m "$MODEL" \
        "${CONTEXT}Read GUIDE.md. You are iteration $ITERATION of a continuous research loop.

Your tasks for this iteration:
1. Run continuous_runner.ContinuousRunner(exchange='gate').run_cycle(skip_download=True)
2. Check if any strategies need parameter recalibration (adaptive_params.AdaptiveEngine)
3. If fewer than 3 strategies in paper/active, discover new ones
4. Run gate_pipeline on any new strategies
5. Generate daily report (report_generator.generate_daily_report)
6. Save cycle summary to results/last_cycle.json

Execute Python code for each step. Be thorough but efficient." 2>&1 | tee -a results/agent_loop.log

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
