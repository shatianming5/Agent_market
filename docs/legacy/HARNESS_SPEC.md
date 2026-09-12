# Strategy Miner Harness Spec

## Objective

The acceptance target is the `harness + model` system, not a single high-score strategy. A run is successful only if the harness can:

- explore under budget,
- validate with frozen benchmarks,
- promote safely through explicit gates,
- recover from interruption,
- replay and audit decisions,
- transfer knowledge across sessions and agents.

## State Flow

Run-level loop:

`strategy_gen -> train_model/backtest -> evaluation -> analysis -> complete`

Candidate-level lifecycle:

`generated -> trained -> smoke_passed -> hyperopt_done -> backtested -> evaluated -> holdout_tested -> promoted`

Invalid transitions fail closed through `VALID_TRANSITIONS`, `VALID_CANDIDATE_TRANSITIONS`, `safe_transition()`, and `update_candidate_stage()`.

## Control Plane

`goal_contract.py` is the machine-readable contract for:

- hypothesis and expected improvement,
- parent lineage,
- budget caps,
- stop conditions,
- promotion conditions,
- evaluation protocol,
- benchmark suite path.

Every run snapshots `goal_contract.json`.

## Evaluation Stack

Selection:

- quick funnel and main backtest run on `selection_timerange`,
- evaluation uses realistic daily metrics and constraint gates,
- optional walk-forward folds add robustness pressure.

Final validation:

- sealed holdout runs once at end of run,
- frozen benchmark pack executes after holdout,
- promotion happens only after holdout and benchmark pass.

Artifacts:

- `holdout_gate.json`
- `benchmark_verdict.json`
- `promotion_log.jsonl`

## Portfolio Layer

Final miner output also constructs a candidate portfolio:

- starts from top eligible candidates,
- derives daily return series from backtest evidence,
- prunes highly correlated candidates,
- produces HRP or equal-weight fallback allocation,
- writes `portfolio_plan.json`.

## Memory and Replay

Knowledge is split across:

- `knowledge_base.json` elites/failures,
- strategy cards,
- failure cards,
- lineage edges,
- agent traces,
- run events and economics.

Replay inputs are preserved through checkpoint, proposal, run meta, events, goal contract, benchmark verdict, and promotion log artifacts.
