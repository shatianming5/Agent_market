# Repo Inventory

## Tree

```text
.
  artifacts/               # Runs, model outputs, factor_lab/rank_portfolio artifacts, control-plane state
  benchmark_pack/          # Benchmark/challenge pack inputs
  configs/                 # Agent Flow, strategy miner, feed/symbol/rule configs
  docs/                    # Project plans, experiments, evidence mapping, inventories
  freqtrade/               # Vendored Freqtrade snapshot; pip freqtrade may also be used
  logs/                    # Mining/backtest/data logs
  runtime_configs/         # Runtime config snapshots from server/jobs
  runtime_logs/            # Runtime logs from server/jobs
  runtime_manifests/       # Job manifests and metadata
  scripts/                 # CLI entrypoints for flow, Factor Lab, mining, training, data, maintenance
  server/                  # FastAPI app, API routes, auth, job manager
  src/                     # Python packages: agent_market, runner_fsm
  strategies/              # Legacy/template strategy files
  tests/                   # Pytest suite
  user_data/               # Freqtrade configs, OHLCV data, strategies, reports, backtest outputs
  web/                     # Static frontend served at /web
  ws_production/           # Standalone production-workspace style helpers and paper-loop tools
  README.md / Makefile / plan.md / requirements*.txt / constraints.txt
```

## Entry Points

- `uvicorn server.main:app --host 127.0.0.1 --port 8000`
  - Starts the FastAPI service from `server/app.py`.
  - Serves `web/` at `/web` and wires `/run/*`, `/flow/*`, `/jobs/*`, `/results/*`, `/strategy-miner/*`, settings and root routes.
  - `AGENT_MARKET_API_KEY` enables `X-API-Key` protection for protected API surfaces.
- `python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest`
  - Runs the main offline pipeline through `src/agent_market/agent_flow.py`.
  - Step handlers live in `src/agent_market/flow_ext/step_dispatch.py` and mostly delegate to `src/agent_market/flow_steps.py`.
  - Writes latest/run-scoped metadata under `artifacts/run_meta.json` and `artifacts/runs/<run_id>/run_meta.json`.
- `python scripts/factor_lab.py <subcommand>`
  - Main research CLI for data download, feature merges, factor mining, validation, walk-forward backtests, rank-portfolio export/backtest/sweep, RL, combo GA, deploy, and Factor Hub.
  - Important subcommands: `data`, `features`, `mine`, `mine-export`, `factor-report`, `exposure-report`, `validate`, `backtest`, `rank-export`, `rank-backtest`, `rank-sweep`, `rl`, `combo`, `deploy`, `hub`.
- `python scripts/strategy_miner.py --config configs/strategy_miner_default.json`
  - Wrapper for LLM strategy mining via `src/agent_market/strategy_miner/runner.py`.
  - Alternative module entrypoint: `python -m agent_market.strategy_miner --config ...`.
- `freqtrade backtesting --config user_data/config_okx_futures_rank_backtest.json --strategy ELRankPortfolioLeverageStrategy --strategy-path user_data/strategies`
  - Consumes precomputed rank-portfolio signals from `artifacts/rank_portfolio/<tag>/signals`.
  - Strategy file is `user_data/strategies/ELRankPortfolioLeverageStrategy.py`.
- `python -m agent_market.factor_hub.server --host 127.0.0.1 --port 8765`
  - Starts the Factor Hub registry/API for factor lifecycle, evaluation, deployment and live event streaming.
- `make install-full`, `make run`, `make flow`, `make e2e`, `make test`
  - Convenience wrappers around install, server, flow, e2e and pytest commands.

## Core Modules

- `server/app.py`
  - FastAPI app factory, CORS, API-key middleware, static frontend mount, router wiring.
- `server/job_manager.py`, `server/runtime.py`, `server/api/routes/*`
  - Background job queue/runtime roots and HTTP API domains for runs, flow, results, jobs, features, settings, strategy miner, analytics and microstructure.
- `src/agent_market/agent_flow.py`
  - End-to-end orchestrator: loads JSON config, runs preflight, dispatches ordered steps, records run metadata.
- `src/agent_market/flow_steps.py`, `src/agent_market/flow_ext/*`
  - Step execution layer for feature generation, expression generation, ML/RL training, backtest, TCA, reports, capture/LOB/microstructure and strategy miner.
- `src/agent_market/paths.py`
  - Central root/path resolver with `AGENT_MARKET_*_ROOT` overrides and `safe_resolve()` for API-supplied paths.
- `src/agent_market/config.py`
  - JSON/YAML helpers and `FreqAISettings` for reading Freqtrade/FreqAI configs and validating OHLCV data layout.
- `src/agent_market/freqai/*`
  - Feature construction, expression engine/safe eval, LLM expression helpers, ML model wrappers, stacking, training pipeline and RL environments.
- `src/agent_market/factor_lab/*`
  - Factor research subsystem: canonical paths, data loaders/downloaders, feature merges, IC mining, purification, fitness, validation, backtesting, combo GA, RL/BC, deploy and reporting.
- `src/agent_market/factor_lab/rank_portfolio.py`
  - Cross-sectional rank portfolio engine: factor selection, ensemble scoring, rolling IC/regime filters, pair exclusions, dynamic per-pair leverage, liquidation-distance guards, account kill modes, signal export and research backtest.
- `user_data/strategies/ELRankPortfolioLeverageStrategy.py`
  - Freqtrade futures strategy that loads rank signals, supports shorts, dynamic leverage, custom stake and custom stoploss.
- `src/agent_market/factor_compiler/*`
  - Factor DSL/parser/AST/operators plus leakage/time-safety/complexity/schema checks and scoring utilities.
- `src/agent_market/factor_hub/*`
  - SQLite-backed factor registry, API server, client, migration and UI helpers.
- `src/agent_market/strategy_miner/*`
  - LLM multi-agent strategy miner: config/state types, generation, sandbox execution, Freqtrade backtest integration, scoring, holdout/benchmark gates and knowledge base updates.
- `src/runner_fsm/opencode/*`
  - OpenCode/cli-proxy integration: client, proxy, tool parsing/execution and monitor utilities for streamed LLM/tool loops.
- `src/agent_market/microstructure/*`
  - KuCoin capture, REST/WS collectors, LOB rebuild/checksum, OHLCV/L2/trade microstructure feature libraries and parquet schemas.
- `src/agent_market/tca/*`
  - Transaction cost analysis schemas, metrics, Freqtrade adapters and report generation.
- `src/agent_market/portfolio_opt.py`
  - Portfolio return loading and HRP-style allocation support.

## Config & Data

- Agent Flow configs: `configs/agent_flow_*.json`
  - Blocks map to flow steps (`feature`, `expression`, `ml_training`, `rl_training`, `backtest`, `portfolio`, `tca`, `report`, `strategy_miner`, etc.).
- Strategy miner configs: `configs/strategy_miner_*.json`
  - Key sections: `budget`, `backtest`, `tools`, `evaluation`; maps into `MinerConfig`.
- Freqtrade/FreqAI configs: `user_data/config_*.json`
  - Examples: `user_data/config_freqai_kucoin.json`, `user_data/config_okx_futures_rank_backtest.json`.
- Factor libraries: `user_data/freqai_expressions*.json`
  - Deployed/selected factor expression sets used by miners, training and rank portfolio.
- Data roots:
  - KuCoin spot OHLCV: `user_data/data/kucoin/*.feather`
  - OKX futures OHLCV: `user_data/data/okx/futures/*-futures.feather`
  - Funding/micro/captured data: `user_data/data/funding`, `user_data/micro_capture`, `artifacts/runs/*/micro_feature`
- Rank portfolio outputs:
  - `artifacts/rank_portfolio/<tag>/selected_factors.json`
  - `artifacts/rank_portfolio/<tag>/signals/*.feather`
  - `artifacts/rank_portfolio/<tag>/rank_export.json`
  - `artifacts/rank_portfolio/<tag>/backtest.json`
  - `artifacts/rank_portfolio/<tag>/sweep.json`
- Common environment variables:
  - Service/paths: `AGENT_MARKET_API_KEY`, `AGENT_MARKET_CORS_ORIGINS`, `AGENT_MARKET_ARTIFACTS_ROOT`, `AGENT_MARKET_USER_DATA_ROOT`, `AGENT_MARKET_RUNS_ROOT`, `AGENT_MARKET_MODELS_ROOT`
  - LLM: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, plus compatible `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
  - Rank portfolio: `RP_SIGNAL_DIR`, `RP_TAG`, `RP_MAX_LEVERAGE`, `RP_GROSS_CAP`, `RP_RISK_PER_TRADE`, `RP_REBALANCE_HOURS`, `RP_EDGE_*`, `RP_REGIME_*`, `RP_SHORT_*`

## How To Run

```bash
# Full dependencies for research + backtest
pip install -c constraints.txt -r requirements-full.txt
```

```bash
# Minimal backend/test dependencies
pip install -r server/requirements.txt -r requirements-dev.txt
```

```bash
# API + web UI
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

```bash
# Main flow
python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest
```

```bash
# Factor Lab help and common research path
python scripts/factor_lab.py --help
python scripts/factor_lab.py data okx-futures
python scripts/factor_lab.py features all
python scripts/factor_lab.py mine --tag exp1 --rounds 50
python scripts/factor_lab.py mine-export --tag exp1 --n 30
python scripts/factor_lab.py validate user_data/freqai_expressions_exp1.json
```

```bash
# Rank portfolio research + signal export
python scripts/factor_lab.py rank-export --tag gpt54_purealpha_v2_full1000_fix1 --n 50 --risk-profile aggressive
python scripts/factor_lab.py rank-backtest --tag gpt54_purealpha_v2_full1000_fix1 --venue okx --top-k 3 --gross-cap 10
python scripts/factor_lab.py rank-sweep --tag gpt54_purealpha_v2_full1000_fix1 --venue okx
```

```bash
# Strategy miner
python scripts/strategy_miner.py --config configs/strategy_miner_default.json
python -m agent_market.strategy_miner --config configs/strategy_miner_default.json
```

```bash
# Factor Hub
python scripts/factor_lab.py hub server --host 127.0.0.1 --port 8765
python -m agent_market.factor_hub.server --host 127.0.0.1 --port 8765
```

```bash
# Tests / checks
pytest -q
pytest tests/test_rank_portfolio.py -q
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
```

## Risks / Unknowns

- The worktree is currently very dirty: many tracked files are modified/deleted and many research files are untracked. Treat `git status` as required context before any cleanup, commit or refactor.
- `README.md` and `Makefile` still reference `scripts/smoke_test.py`, but that file is currently deleted in the worktree. Use `pytest -q` and `scripts/e2e_smoke_flow.py` unless the smoke script is restored.
- The repository contains both vendored `freqtrade/` and likely environment-installed `freqtrade`; choose one execution path consistently when debugging backtests.
- Many configs referenced in older docs are deleted in the current worktree; validate paths before running historical commands.
- Rank portfolio defaults in code have evolved from the original aggressive plan: current `RiskConfig.from_profile("aggressive")` defaults to a short-biased, rolling-IC filtered profile with lower gross/leverage caps unless CLI/env overrides are passed.
- LLM/cli-proxy paths depend on external services, local auth and proxy ports. `.env`/`.env.cliproxy` may contain provider-specific settings and should not be assumed portable.
- Full backtests and mining loops require local OHLCV coverage matching timeranges; missing feather files fail late if not preflighted.
