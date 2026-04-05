# Benchmark Pack

`benchmark_pack/default/manifest.json` is the frozen benchmark and challenge suite used by the upgraded strategy-miner harness.

It defines:

- The selection and sealed-holdout windows.
- Hard benchmark gates for promotion.
- Seeded challenge checks for common leakage patterns.

The runner loads this manifest through `benchmark_suite` and writes the final verdict to `benchmark_verdict.json`.
