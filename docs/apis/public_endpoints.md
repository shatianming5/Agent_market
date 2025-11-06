# Public API Endpoints

All endpoints are served by FastAPI on `http://{APP_HOST}:{APP_PORT}` (defaults `127.0.0.1:8032`). Unless noted otherwise requests are JSON encoded and do not require authentication yet. Response schemas provided below reflect current implementations and may evolve alongside TODO-07/08 runtime work.

## Health
| Method | Path | Description | Response |
| --- | --- | --- | --- |
| GET | `/health` | Lightweight readiness probe. | `{"status": "ok"}` (HTTP 200). |
| GET | `/healthz` | Extensive check that inspects filesystem and freqtrade prerequisites. | `{"status": "ok", "details": {...}}` on success. |

## Expression Catalog
| Method | Path | Description | Payload / Query | Response |
| --- | --- | --- | --- | --- |
| GET | `/expressions` | Retrieve current expression list and backing file path. | – | `{ "path": "resources/user_data/freqai_expressions.json", "expressions": [...] }`. |
| GET | `/expressions/allowed` | Static allow-list of helper functions and baseline columns. | – | `{ "functions": [...], "base_columns": [...] }`. |
| POST | `/expressions/validate` | Quick static validation for an expression string. | Body: raw string | `{ "illegal_chars": [...], "allowed_funcs": [...], "used_funcs": [...], "unknown_identifiers": [...] }`. |
| PUT | `/expressions` | Replace the entire expression list. | `{ "expressions": [...] }` | `{ "status": "ok", "count": N, "path": "..." }`. |
| POST | `/expressions` | Append a single expression entry. | `{ ... }` arbitrary schema | `{ "status": "ok", "index": <int> }`. |
| PATCH | `/expressions/{index}` | Update an expression at position `index`. | `{ ... }` | `{ "status": "ok" }`. |
| DELETE | `/expressions/{index}` | Remove an expression at position `index`. | – | `{ "status": "ok", "removed": {...} }`. |
| POST | `/expressions/preview` | Compute expression outputs against a feature dataset. | Body includes `pair`, `timeframe`, `config`, `expression`, etc. | `{ "head": [...], "tail": [...], "stats": {...} }` or error descriptors. |

## Feature & Config Helpers
| Method | Path | Description |
| --- | --- | --- |
| GET | `/features` – returns merged feature configuration detected by `_load_feature_cfg`. |
| POST | `/config/use-features-file` – switch active feature file based on client-provided path. |
| POST | `/config/use-expressions-file` – mirror of above for expressions catalog. |

## Dataset Inspection
| Method | Path | Description | Response |
| --- | --- | --- | --- |
| GET | `/data/summary` | High level statistics (candles, date range) for configured pair/timeframe. | `{ "pair": "...", "timeframe": "...", "summary": {...} }`. |
| GET | `/data/check-missing` | Lists missing candles per timeframe. | `{ "missing": [...] }`. |
| GET | `/data/columns` | Returns available columns for a dataset. | `{ "columns": [...] }`. |

## Strategy Configuration
| Method | Path | Description | Response |
| --- | --- | --- | --- |
| GET | `/strategy/params` | Read current strategy parameters (typically freqtrade JSON). | `{ "params": {...} }`. |
| PUT | `/strategy/params` | Persist new strategy parameter set. | Body: `{ "params": {...} }` | `{ "status": "ok" }`. |
| GET | `/backtest/summary/latest` | Return summary of the most recent backtest artefact if present. | `{ "summary": {...}, "path": "..." }`. |

## Job Execution APIs
| Method | Path | Description | Notes |
| --- | --- | --- | --- |
| POST | `/run/expression` | Launch expression agent (LLM assisted). | Returns `{ "job_id": "...", "status": "submitted" }`. |
| POST | `/run/backtest` | Trigger freqtrade backtest workflow. | 〃 |
| POST | `/run/download-data` | Kick off OHLCV downloader. | 〃 |
| POST | `/run/train-ml` | Run ML training pipeline based on config. | 〃 |
| POST | `/run/train-xgb` | Helper for XGBoost pathway. | 〃 |
| POST | `/run/train-cat` | Helper for CatBoost pathway. | 〃 |
| POST | `/flow/run` | Execute declarative multi-step agent flow defined in `src/agent_market/agent_flow.py`. | Accepts flow JSON; returns job metadata. |

### Job Introspection
| Method | Path | Description | Response |
| --- | --- | --- | --- |
| GET | `/jobs` | List recent jobs with status flags. | `[{"id": "...", "running": true/false, ...}]`. |
| GET | `/jobs/{job_id}/status` | Poll a single job status. | `{ "id": "...", "running": bool, "returncode": int|null, "started_at": "...", "finished_at": "..." }`. |
| GET | `/jobs/{job_id}/logs` | Fetch buffered stdout lines. Supports `offset`, `limit`, `structured=true`. | `{ "lines": [...], "next_offset": int }`. |
| GET | `/jobs/{job_id}/stream` | Server-Sent Events stream of live log lines. | `text/event-stream`. |
| GET | `/jobs/{job_id}/progress` | Aggregated progress metrics. | `{ "progress": {...} }`. |
| GET | `/jobs/{job_id}/steps` | Sequenced milestones recorded via `_on_job_step`. | `{ "steps": [...] }`. |
| POST | `/jobs/{job_id}/terminate` | Request termination of a running job. | `{ "status": "terminating" }`. |
| GET | `/steps/stats` | Aggregated stats across job steps (duration, counts). | `{ "step_stats": [...] }`. |
| POST | `/jobs/dev/sleep` | Development helper to spawn a dummy sleeping job; useful for UI testing. | Accepts `{ "seconds": int }`. |

## Agents & Orders
| Method | Path | Description | Response |
| --- | --- | --- | --- |
| GET | `/agents` | List stored agents ordered by `created_at`. | `[{ "id": "...", "name": "...", "created_at": "..." }, ...]`. |
| POST | `/agents` | Create a new agent (name length 1–64). | Newly created agent record. |
| GET | `/agents/{agent_id}` | Fetch a specific agent. | Agent record or HTTP 404. |
| GET | `/orders` | List orders; optional query `agent_id`. | `[{ "id": "...", "agent_id": "...", "status": "...", ... }]`. |
| POST | `/orders` | Create an order linked to an agent. | Newly created order record; 404 if agent missing. |

## Authentication & Rate Limits
- Authentication is not yet enforced; implementations should front these endpoints with ingress rules or add API keys before internet exposure (tracked by TODO-12/13).
- Rate limiting is not enabled server side. Deploy behind a gateway (e.g., Traefik, Nginx) if multi-tenant exposure is planned.
