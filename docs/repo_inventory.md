# Repo Inventory

## Tree

```text
Agent_market-1/
  analysis/                  # Security/architecture notes (audit only, not runtime)
  artifacts/                 # Unified output root (models/ + runs/<run_id>/)
  configs/                   # Agent flow, miner, freqtrade, RL/ML templates
  data/                      # Supplemental offline data and download scripts
  docs/                      # Plan/experiment/mohu/verify/real_backtest_material
  scripts/                   # Flow wrappers, backtest helpers, smoke tests, strategy miner CLIs
  server/                    # FastAPI job manager + aggregator that wires scripts to HTTP jobs
  src/                       # `agent_market` package (flow, backtest, runner_fsm) + strategy miner internals
  tests/                     # pytest suites and fixtures that hit the API and sandboxed flows
  user_data/                 # Live workspace (data/<exchange>, strategies/, backtest_results/, logs, feedback)
  web/                       # Frontend static bundle (served by `server/`)
  Makefile/README+/plan.md/requirements*.txt  # common entry docs/config
```

## Entry Points

- `python scripts/agent_flow.py --config <file> --steps ...`
  - Orchestrates the capture → expression → ML → backtest → report pipeline described in `plan.md`; every step emits artifacts via `RunArtifacts` and the `flow_ext` handlers before writing `artifacts/runs/<run_id>/run_meta.json`.
- `python scripts/strategy_miner.py --config configs/strategy_miner_default.json`
  - Kicks off the full mining state machine (`strategy_miner.runner.run_strategy_miner`) that loops through generation/backtest/evaluation/analysis (planner/coder/reviewer/backtester roles by default).
- `python scripts/strategy_miner_backtest.py --run-id <run_id> [--candidate <name>]`
  - Job-manager helper that replays `phase_backtest`/`phase_evaluation` against a saved checkpoint to validate or rank a spot candidate without regenerating others.
- `uvicorn server.main:app --host 0.0.0.0 --port 8000`
  - FastAPI service that exposes job submission endpoints, surfaces artifacts, and can enqueue the miner/backtest scripts via `server.jobs`.
- `pytest -q` / `python scripts/smoke_test.py`
  - Exercise API fixtures, agent flow endpoints, and the lightweight smoke path described in `scripts/smoke_test.py`.

## Core Modules

- `src/agent_market/agent_flow.py`
  - Driver for the configurable pipeline. It hydrates `AgentFlowConfig`, selects requested steps (capture → backtest → report), writes run metadata (`artifacts/runs/<run_id>/run_meta.json`), and resolves artifact paths via `paths.py` and `RunArtifacts`.
- `src/agent_market/flow_ext/step_dispatch.py` + `src/agent_market/flow_ext/steps.py`
  - Each pipeline phase (feature/expression/ml/rl/backtest/tca/report/strategy_miner) delegates to `flow_steps` helpers that run freqtrade commands, RL signal prep, TCA, and the miner step; `step_dispatch` wires outputs back into `RunArtifacts` for metadata and feedback writes.
- `src/agent_market/backtest_results.py`
  - Reads `backtest-result-*.zip` archives, computes a sanitized summary (profit%, winrate, sharpe/sortino/calmar/profit factor/cagr/max drawdown) and writes `feedback.json` for downstream loops (`write_latest_backtest_summary`).
- `src/agent_market/strategy_miner/runner.py`
  - Maintains `MinerState` checkpoints, writes proposals, stitches in the `KnowledgeBase`, and advances the phases defined in `phases.py` until `state.phase == Phase.COMPLETE`.
- `src/agent_market/strategy_miner/phases.py`
  - Implements the core feedback loop: `phase_strategy_gen` spins up sandboxed planner/coder/reviewer/backtester agents (parallel when `max_parallel_roles` > 1) and appends candidates; `phase_backtest` runs freqtrade backtesting + repair loops; `phase_evaluation` scores candidates (Sharpe primary, risk gating via min trades/winrate/max drawdown); `phase_analysis` produces LLM explanations, updates history, and increments `state.iteration`/`state.phase` for the next generation.
- `src/agent_market/strategy_miner/knowledge_base.py`
  - Persists elite candidates and failure buckets (`KnowledgeBase.add_elite`/`add_failure`) so subsequent iterations can reuse proven code or avoid bad patterns.
- `src/agent_market/run_artifacts.py` and `src/agent_market/paths.py`
  - Centralize path resolution for `artifacts/`, `runs/`, `user_data/`, and ensure flow steps and miner share the same roots.

## Config & Data

- Flow configs under `configs/agent_flow_*.json` define which steps run, the freqtrade config to use, and artifact paths; `configs/agent_flow_kucoin_cpu_nollm*.json` are the gold paths referenced in `plan.md`.
- Strategy miner configs (`configs/strategy_miner_default.json`, `_maxpower.json`, `_recovery.json`) map to `MinerConfig` fields (budget/tool policy/backtest settings) and point to `user_data/config_freqai.json` for freqtrade.
- Freqtrade/LLM configs: `configs/config_freqai*.json`, `configs/rl_config_real.json`, `configs/ml_config_real.json`, plus `configs/feeds.yaml`/`symbols.yaml` control data sources.
- `user_data/` stores workspace artifacts:
  - `data/<exchange>/<pair>-<timeframe>.feather` (OHLCV) used by `scripts` and flow steps.
  - `strategies/` contains generated/candidate strategy files loaded by freqtrade.
  - `backtest_results/` holds `backtest-result-*.zip` that `backtest_results.build_backtest_summary` ingests.
  - `job_logs/`, `logs/`, `llm_feedback/`, `freqai_*` JSONs, and `reports/` accumulate run telemetry and feedback for expression/backtest loops.
- Environment overrides: `AGENT_MARKET_ARTIFACTS_ROOT`, `AGENT_MARKET_RUNS_ROOT`, `AGENT_MARKET_USER_DATA_ROOT`, and LLM endpoints (`OPENAI_MODEL`/`LLM_MODEL`, `OPENCODE_URL`/`OPENCODE_MODEL`) influence where artifacts land and which providers the miner uses.

## How To Run

```bash
python scripts/agent_flow.py --config configs/agent_flow_example.json --steps capture expression backtest report
python scripts/strategy_miner.py --config configs/strategy_miner_default.json
python scripts/strategy_miner_backtest.py --run-id <run_id> --candidate <name>
uvicorn server.main:app --host 0.0.0.0 --port 8000
pytest -q
python scripts/smoke_test.py
```

## Risks / Unknowns

- `freqtrade` dependencies (ccxt, talib/pandas_ta, pandas) must match `requirements-full.txt`; missing packages break `phase_backtest` before the summary layer can run.
- Historical OHLCV (`user_data/data/…`) and `user_data/config_freqai.json` need to cover the miner timerange (`configs/.../timerange`), otherwise `phase_backtest` will bail with `backtest.data_missing`.
- LLM/backtester agents rely on external OpenCode/OpenAI endpoints plus tool allowlists from `configs`; outages or rate limits stall candidate generation/repair.
- RL signal generation (in `_maybe_generate_rl_signals_for_backtest`/`scripts/rl_generate_signals.py`) adds run-time when enabled and relies on `user_data/freqaimodels/*/training_summary.json` existing.
- Repair loops in `phase_backtest` (validation/backtest failure → `_repair_candidate`) can replay freqtrade dozens of times per candidate, so candidate count × repairs × RL make iteration latency the main bottleneck.
