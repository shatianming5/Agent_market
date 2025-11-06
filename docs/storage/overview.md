# Storage Overview

## Relational Data (SQLite)
| Table | Location | Columns | Purpose |
| --- | --- | --- | --- |
| `agents` | `resources/user_data/app.db` | `id` TEXT PK, `name` TEXT, `created_at` ISO8601 | Lightweight registry for trading agents visible in the UI. |
| `orders` | `resources/user_data/app.db` | `id` TEXT PK, `agent_id` FK → `agents.id`, `side` (`buy`/`sell`), `qty` REAL, `status` TEXT, `created_at` ISO8601 | Tracks synthetic orders generated from the UI or automation. |
| `job_steps` | `resources/user_data/app.db` | `job_id` TEXT, `idx` INT, `total` INT, `label` TEXT, `ts` ISO8601 | Milestone log for long running jobs, populated by `JobManager.on_step`. |

> Current schema is initial and carries no secondary indexes. Once runtime DAGs land (TODO-07/08) we should add composite indexes on (`job_id`, `idx`) and (`agent_id`, `created_at`) to keep list queries responsive.

## Filesystem Buckets
| Path | Contents | Producers | Retention Notes |
| --- | --- | --- | --- |
| `resources/user_data/` | Primary working set: `app.db`, `freqai_expressions.json`, templated configs, generated features, ML artefacts, backtest zips. | `server/main.py`, `src/agent_market/agent_flow.py`, `freqtrade/scripts/*`. | Versioned by timestamped directories. Cleanup scripts should purge stale runs periodically. |
| `data/` | Raw/clean OHLCV datasets grouped by exchange/symbol. | `scripts/fetch_ccxt_ohlcv.py`, `scripts/clean_ohlcv.py`. | Immutable snapshots. Consider S3 offload when datasets grow. |
| `artifacts/` | Checkpoints produced by sweeps/backtests. | Batch jobs under `scripts/`. | Keep latest N to control disk usage. |
| `codex_logs/` | Rotating run logs captured by `codex_scheduler.py`. | Scheduler / CI harness. | Bounded by external rotation policy. |
| `catboost_info/` | CatBoost training metadata (temporary). | freqtrade AutoML routines. | Can be safely purged after model promotion. |

## External Integrations
- **Conda env** (`freqtrade/`) holds vendored strategies and required binaries for the ML pipeline.
- **Secrets** currently rely on `.env`; hardened secret storage is scheduled under TODO-12/13.

## Backup Checklist
1. `resources/user_data/app.db` (transactional state).
2. `resources/user_data/` (latest artefacts + expression catalog).
3. `data/` (input datasets) – optional if reproducible from upstream exchanges.
4. `docs/` configuration references to keep documentation in sync with data dumps.

Automate regular snapshots before large training runs to allow quick rollback in the event of corrupt artefacts or experimental schema updates.
