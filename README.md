# Agent Market

Agent Market is an experimentation platform for quantitative crypto research that blends
LLM-assisted feature discovery, freqtrade-based backtesting, and REST/React operational tooling.
It provides end-to-end automation for collecting market data, generating alpha expressions,
training ML/RL models, and evaluating strategies, while exposing job orchestration services and
UI controls.

> **Highlights**
> - Python runtime for feature engineering, model training, and agent orchestration.
> - FastAPI backend with job/connector/secret management and SSE log streaming.
> - React dashboard for triggering flows, inspecting backtests, and managing infrastructure.
> - Tight integration with freqtrade, including Conda environment helpers and reusable artifacts.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running the Services](#running-the-services)
- [Core Workflows](#core-workflows)
- [Agent Flow Automation](#agent-flow-automation)
- [Connectors, Triggers, and Secrets](#connectors-triggers-and-secrets)
- [Testing and Quality](#testing-and-quality)
- [Troubleshooting](#troubleshooting)
- [Further Reading](#further-reading)
- [Chinese Guide](#chinese-guide)

## Architecture Overview

The platform is composed of four major surfaces:

- **Data & Modeling** – Located under `src/agent_market/freqai/`, these modules assemble
  OHLCV datasets, transform features, and train ML or RL models via adapters (LightGBM,
  CatBoost, PyTorch).
- **Runtime Orchestration** – `src/agent_market/agent_flow.py` coordinates multi-step jobs
  (download → features → expressions → training → backtest) with cancellation, retry, and
  summary feedback after each run.
- **API Service** – `server/main.py` exposes REST endpoints for job control, expressions,
  connectors, triggers, and secrets storage. It relies on SQLite (`resources/user_data/app.db`) and a
  shared `JobManager` to launch subprocesses and stream logs.
- **Web UI** – `web/` contains a Vite + React SPA that consumes the REST/SSE APIs for
  monitoring and control, including a Runs tab that surfaces manifests created under `resources/user_data/runs/<run_id>/`.

Mermaid overview (see `docs/architecture/current.md`) places the browser on the left, the FastAPI
service in the middle, and the CLI/AgentFlow workers on the right, with artifacts stored under
`resources/user_data/` and `resources/data/`.

## Repository Layout

```text
├─ conf/                  # YAML configs for symbols, feeds, keywords, etc.
├─ configs/               # JSON configs for AgentFlow, freqtrade, RL, ML runs
├─ data/                  # Raw and cleaned market data (parquet)
├─ docs/                  # Architecture notes, LLM pipeline guides, quick start docs
├─ freqtrade/             # Vendored freqtrade fork with custom scripts
│  └─ scripts/            # Feature/Expression agents that call into agent_market modules
├─ scripts/               # CLI utilities (data ingestion, diagnostics, reports, idea pipeline)
├─ server/                # FastAPI application, DB layer, connector & trigger services
├─ src/agent_market/      # Core Python package (agent_flow, freqai, runtime utilities)
├─ tests/                 # Pytest suite covering flow, runtime, connectors, DB, etc.
├─ resources/user_data/             # SQLite DB, freqai outputs, backtest results, temporary logs
├─ web/                   # React application (Vite, TypeScript)
└─ venv/ / freqtrade env  # Optional local environments (Python venv, Conda freqtrade)
```

Key entry points:

- `src/agent_market/agent_flow.py` – Task orchestrator for multi-step pipelines.
- `server/main.py` – FastAPI app with dependency overrides and router wiring.
- `scripts/agent_flow.py` – CLI wrapper allowing specific pipeline stages.
- `web/src/App.tsx` – Root React component, orchestrating page-level tabs.

## Prerequisites

- **Python** 3.11+ (for backend, CLI, and core package).
- **Node.js** 20+ (for building/running the React application).
- **Conda (optional but recommended)** for the `freqtrade` toolchain (GPU toolkits are not
  required).
- GCC/Build tools if you plan to recompile frequency libraries (for Windows, install Build Tools).
- Access to external APIs (LLM provider, CCXT exchanges, News/Twitter, etc.) with keys stored in
  `.env`.

## Environment Setup

### 1. Clone and base Python environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r server/requirements.txt
pip install -r requirements-dev.txt
```

### 2. freqtrade Conda environment (optional)

Many data/ML scripts expect a `freqtrade` Conda env:

```powershell
conda env create -f freqtrade/environment.yml
conda activate freqtrade
pip install -e ./freqtrade
```

If the environment already exists, you can run arbitrary commands through it with:

```powershell
conda run --no-capture-output -n freqtrade python -m pytest
```

### 3. Node dependencies

```powershell
npm install --prefix web
```

### 4. Environment variables

Copy `.env.example` to `.env` and fill in provider credentials:

- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Exchange, Telegram, Discord, or News API tokens (for connectors/scripts)
- Optional observability settings (OTLP exporters)

## Running the Services

### Backend (FastAPI)

```powershell
uvicorn server.main:app --host 127.0.0.1 --port 8032
```

Endpoints of interest:

- `POST /run/expression` – Launch expression generation (LLM pipeline).
- `POST /run/backtest` – Launch freqtrade backtests.
- `POST /flow/run` – Run an AgentFlow configuration.
- `GET /jobs`, `/jobs/{id}/logs`, `/jobs/{id}/stream` – Inspect job status/logs.
- `GET /agents`, `/orders`, `/connectors`, `/secrets` – CRUD resources stored in SQLite.

Smoke tests:

```powershell
python scripts/server_smoke.py
python scripts/server_quickcheck.py
```

### Front-end (React SPA)

```powershell
npm --prefix web run dev       # Vite dev server
npm --prefix web run build     # Production build
```

By default the SPA assumes the backend is reachable on `http://127.0.0.1:8032`.

## Chinese Guide

For a Chinese-language overview of the repository—including directory structure, module summaries, common commands, and storage locations—refer to `docs/overview.zh.md`.

## Core Workflows

### Data ingestion

```powershell
python scripts/fetch_ccxt_ohlcv.py --conf conf/symbols.yaml
python scripts/fetch_binance_bulk.py --conf conf/symbols.yaml --limit-months 4
python scripts/clean_ohlcv.py --conf conf/symbols.yaml
python scripts/dq_report.py --mode ohlcv --conf conf/symbols.yaml
```

Resulting parquet files are placed under `data/raw/` or `data/clean/`.

### Feature generation

`freqtrade/scripts/freqai_feature_agent.py` builds baseline features into
`resources/user_data/freqai_features.json` using configuration from `configs/config_freqai_*.json`. The
transform logic lives in `src/agent_market/freqai/features.py`, which gracefully degrades when
TA-Lib is unavailable.

### Expression generation (LLM-assisted)

```powershell
conda run -n freqtrade python freqtrade/scripts/freqai_expression_agent.py `
    --feature-file resources/user_data/freqai_features.json `
    --output resources/user_data/freqai_expressions.json `
    --timeframe 1h `
    --llm-count 10 `
    --llm-loops 5
```

The agent queries the configured LLM, filters candidates with information coefficient (IC)
metrics, and emits ranked expressions.

### Model training

`src/agent_market/freqai/training/pipeline.py` consumes features to train ML models.
Example manual invocation:

```powershell
python scripts/train_ml.py --config configs/ml_config_real.json
```

Results (model artifacts, JSON summaries) are written to `artifacts/models/` or `resources/user_data/models/`.

### Backtesting and evaluation

```powershell
conda run -n freqtrade freqtrade backtesting `
    --config configs/config_freqai_multi.json `
    --strategy ExpressionLongStrategy `
    --strategy-path app/freqtrade/user_data/strategies `
    --timerange 20210101-20211231 `
    --freqaimodel LightGBMRegressor `
    --export trades `
    --export-filename resources/user_data/backtest_results/latest_trades_multi
```

AgentFlow automatically collects `.zip` exports and writes condensed summaries to
`resources/user_data/llm_feedback/latest_backtest_summary.json`.

## Agent Flow Automation

Agent Flow (`src/agent_market/agent_flow.py`) bundles multiple tasks into a managed pipeline with
cancellation, retries, concurrency control, and rate limiting. Example CLI run:

```powershell
python scripts/agent_flow.py `
    --config configs/agent_flow_multi.json `
    --steps download feature expression ml backtest
```

Configuration blocks include:

- `download` – run freqtrade download scripts with optional erase/new pair flags.
- `feature` – compute features and write JSON descriptors.
- `expression` – call the LLM agent and optionally seed it with feedback.
- `ml_training` / `rl_training` – feed configs to `TrainingPipeline` or RL trainers.
- `backtest` – invoke freqtrade CLI and capture results.

Each step writes progress via `[STEP]` markers consumed by the job manager and surfaced in the UI.
Every execution also materialises an artifact bundle under `resources/user_data/runs/<run_id>/`, including:

- `config.snapshot.json` — configuration snapshot at dispatch time.
- `events.jsonl` — JSON Lines stream of emitted `progress`/`log` events.
- `manifest.json` — run metadata (timestamps, step results, git/python info, artifact references).

These manifests allow downstream tooling to audit or archive experiments without scraping console output.

## Connectors, Triggers, and Secrets

- **Connectors** (`src/agent_market/connectors/`) standardize external integrations (Binance,
  OKX, CoinGecko, Telegram, Discord, EVM RPC). They validate credentials and expose a
  `test_connection` method. A built-in `mcp_browser` connector can be used to manage Model
  Context Protocol browser agents; create it via the Connectors API/UI with a config such as:
  ```json
  {
    "connector_type": "mcp_browser",
    "name": "Playwright MCP",
    "config": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    },
    "credentials": {}
  }
  ```
- **ConnectorService** (`server/connector_service.py`) stores configurations in SQLite, uses
  `SecretsManager` for encrypted credential storage, and can trigger jobs or triggers based on
  metadata.
- **Triggers** (`server/triggers.py`) support webhook, cron, and EVM event triggers. They use
  a background polling thread to evaluate conditions and enqueue jobs.
- **Secrets** (`server/secrets.py`) manages scoped secrets persisted in `resources/user_data/app.db`.

The React UI exposes tabs for Connectors and Secrets to simplify management.

## Testing and Quality

- Pytest suite: `python -m pytest` (or `conda run -n freqtrade python -m pytest`).
- Linting/formatting: `python -m ruff check .`, `python -m black --check .`,
  `npm --prefix web run lint`.
- Pre-commit: install via `python -m pre_commit install`.

CI-style quick checks exist under `scripts/` (`server_quickcheck.py`, `server_smoke.py`,
`test_server_endpoints.py`, etc.).

## Troubleshooting

- **Large environments** – `venv/` and `freqtrade/` consume most disk space. Recreate them if
  corrupted.
- **Missing TA-Lib** – Feature generation will fallback to NumPy implementations; install TA-Lib
  for more accurate indicators.
- **LLM requests failing** – Verify `.env` contains correct `LLM_BASE_URL` and `LLM_API_KEY`.
- **freqtrade CLI errors** – Ensure Conda env is activated or use `conda run -n freqtrade ...`.
- **Job logs not streaming** – Confirm SSE client connection to `/jobs/{id}/stream`; older browsers
  fall back to `GET /jobs/{id}/logs`.

## Further Reading

- `docs/architecture/current.md` – In-depth architecture diagram and component notes.
- `docs/llm_pipeline.md` – Detailed guide for the LLM expression workflow.
- `docs/dev-testing.md` – Developer testing practices and smoke check instructions.
- `docs/ai_framework.md` – Overview of the ML/RL strategy framework and future roadmap.
- `scripts/` – Explore `--help` on scripts for task-specific options.

---

Agent Market is an evolving research environment. Contributions, bug reports, and feature ideas are
welcome—see `docs/CONTRIBUTING.md` for guidelines.
