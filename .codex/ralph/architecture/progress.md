# Progress

## Done

- Added the target architecture doc at `docs/opencode_strategy_factory_architecture.md`.
- Added direct OpenCode factor-mining config at `configs/agent_flow_kucoin_factor_opencode.json`.
- Added direct strategy-miner runtime renderer at `scripts/make_strategy_miner_opencode_config.py`.
- Added repo-local Ralph loop bootstrap at `scripts/bootstrap_strategy_factory_loops.py`.
- Extended `scripts/freqai_expression_agent.py` and `src/agent_market/freqai/llm.py` with explicit OpenCode provider support.
- Verified the bounded pass with targeted pytest and py_compile.
- Smoke-verified the new expression CLI flags and the rendered strategy-miner runtime config.

## In Progress

- Keep the runtime/control-plane boundary explicit in docs and loop scaffolding.

## Blocked

- None.

## Next

- Next bounded task: wire factor compile/eval artifacts into a dedicated factor-card memory contract so the factor loop has a persistent semantic memory surface.
