# Strategy Factory Operator Runbook

## Purpose

This runbook covers the unified `factor flow -> strategy miner -> promotion artifacts`
execution path driven by [scripts/agent_flow.py](../scripts/agent_flow.py).

The enterprise contract is:

- one top-level config
- one `run_id`
- one experiment registry lineage
- one replay manifest
- one resource dashboard

## Required Inputs

1. A top-level AgentFlow config that includes:
   - `experiment`
   - `factor_compile` and/or `factor_eval`
   - `strategy_miner`
2. Valid feature / expression / factor inputs
3. A valid strategy miner config

Reference example:

- [configs/agent_flow_strategy_factory_example.json](../configs/agent_flow_strategy_factory_example.json)

## Run

```bash
python3 scripts/agent_flow.py \
  --config configs/agent_flow_strategy_factory_example.json \
  --steps factor_compile factor_eval strategy_miner report
```

## What Happens Automatically

1. `factor_eval` writes:
   - `factor_memory.json`
   - `factor_cards.json`
   - `factor_failure_cards.json`
   - `factor_lineage.json`
2. `strategy_miner` automatically receives `factor_memory_path` from the same flow run if you did not hardcode it.
3. At flow completion the control plane writes:
   - `experiment_registry.jsonl`
   - `budget_plan.json`
   - `replay_manifest.json`
   - `lineage_graph.json`
   - `promotion_chain.json`
   - `resource_dashboard.json`

## Resume

### Resume the whole factory run

Use the same top-level config, then point the strategy miner section at an existing checkpoint if needed.

### Resume only the strategy miner

Use:

```bash
python3 scripts/strategy_miner.py \
  --resume artifacts/runs/<run_id>/strategy_miner/checkpoint.json
```

## Audit

Primary audit files live under:

- `artifacts/runs/<run_id>/run_meta.json`
- `artifacts/runs/<run_id>/experiment_registry.jsonl` via repo-global registry pointer
- `artifacts/runs/<run_id>/replay_manifest.json`
- `artifacts/runs/<run_id>/resource_dashboard.json`
- `artifacts/runs/<run_id>/lineage_graph.json`

API helpers:

- `/flow/run-meta/<run_id>`
- `/flow/factor-memory/<run_id>`
- `/flow/replay-manifest/<run_id>`
- `/flow/lineage/<run_id>`
- `/flow/promotion-chain/<run_id>`
- `/flow/resource-dashboard/<run_id>`

## Rollback

Offline rollback means treating a promoted candidate as invalid and refusing further promotion.

Use the strategy miner artifacts:

- `promotion_log.jsonl`
- `holdout_gate.json`
- `benchmark_verdict.json`
- `portfolio_plan.json`

The factory-level `promotion_chain.json` keeps the same decision path plus factor references
that contributed to the promoted candidate.

## Operational Notes

- `report` should run after `strategy_miner` so the bundle includes strategy-side artifacts.
- The factor-memory-to-strategy wiring is automatic only inside the same `AgentFlow` run.
- `experiment_registry.jsonl` is append-only and intended for cross-run analytics.
