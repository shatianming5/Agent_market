# Harness Acceptance

## Gate Rule

The system passes only when the harness is controllable, recoverable, benchmarked, and promotable. Offline score alone does not pass acceptance.

## Dimensions

### D1 Goal Contract

- Required: hypothesis, expected improvement axis, budget, stop conditions, promotion conditions, lineage.
- Evidence: `goal_contract.json`.
- Pass: random sample of experiments can explain why they were started, what they try to improve, what budget they consume, when they stop, and how they promote.

### D2 State Machine

- Required: explicit run and candidate transition tables, atomic checkpointing, crash recovery.
- Evidence: `checkpoint.json`, stage history, `events.jsonl`.
- Pass: injected interruption resumes without duplicate side effects.

### D3 Strategy IR

- Required: structured strategy schema separated from rendered Python.
- Evidence: `src/agent_market/strategy_miner/strategy_ir.py`, `strategy_ir_schema.json`.
- Pass: promoted strategies can be decomposed into alpha, regime, exit, execution, risk, and search-space modules.

### D4 Budget Scheduling

- Required: budget accounting, adaptive family allocation, cheap gates first.
- Evidence: `economics.json`, `economics_per_candidate.jsonl`, scheduler state.
- Pass: same budget yields better full-eval hit rate than static baseline.

### D5 Evaluation Correctness

- Required: selection and sealed holdout separation, frozen benchmark pack, seeded challenge suite.
- Evidence: `benchmark_pack/`, `holdout_gate.json`, `benchmark_verdict.json`.
- Pass: promotion uses selection metrics only for search, then sealed holdout plus benchmark pack for final verdict.

### D6 Verification and Repair

- Required: bounded repair loop with typed failure ledger and post-fix verification.
- Evidence: candidate `repair_ledger`, failure cards, trace grading.
- Pass: seeded syntax/runtime/integration failures are repaired only inside allowed scope.

### D7 Memory and Knowledge Transfer

- Required: episodic, semantic, and skill-like persistence plus lineage edges.
- Evidence: `knowledge_base.json`, strategy cards, failure cards.
- Pass: a new agent can resume from artifacts without replaying the entire trace.

### D8 Context Engineering

- Required: prompt metadata, context budget awareness, retrieval discipline.
- Evidence: agent traces with prompt metadata and context slices.
- Pass: retrieval is targeted instead of dumping full history into every prompt.

### D9 Observability and Replay

- Required: run metadata, event stream, artifact hashes or stable snapshots, deterministic replay inputs.
- Evidence: `run_meta.json`, `events.jsonl`, proposal, checkpoint, goal contract.
- Pass: promoted candidates can be traced to exact artifacts and decisions.

### D10 Promotion Chain

- Required: `evaluated -> holdout_tested -> promoted` only.
- Evidence: candidate stage history, `promotion_log.jsonl`.
- Pass: there is no shortcut around holdout or benchmark gate.

### D11 Security and Isolation

- Required: sandbox execution, secret scrubbing, least privilege, auditability.
- Evidence: sandbox execution path, scrubbed outputs, approval boundaries.
- Pass: experiment code cannot directly escape or leak secrets through artifacts.

### D12 Portfolio Construction

- Required: candidate correlation view, diversification filter, allocation output.
- Evidence: `portfolio_plan.json`.
- Pass: final output can recommend a diversified allocation instead of only one champion.

### D13 Economics

- Required: token, wall-clock, backtest, and cost accounting.
- Evidence: `economics.json`, `economics_per_candidate.jsonl`.
- Pass: useful exploration per unit cost improves against baseline.

## Final Gates

### Gate A Frozen Benchmark

- Run baseline and new harness on the same frozen benchmark for three cycles.
- Pass: new harness wins at least two cycles without relaxing risk thresholds.

### Gate B Crash and Handoff

- Inject crash, timeout, API error, and worker loss.
- Pass: recovery succeeds and a fresh agent can continue from artifacts.

### Gate C Shadow Validation

- Promote only through sealed holdout and benchmark pack, then observe paper/shadow behavior.
- Pass: drift, degradation, and rollback conditions are observable.

### Gate D Governance

- Audit staging and production separation, permissions, approvals, and logs.
- Pass: high-impact actions remain gated and attributable.
