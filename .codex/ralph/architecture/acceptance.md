# Acceptance

- The architecture doc matches the actual runnable entrypoints in this repo.
- Runtime plane still uses direct OpenCode execution for factor mining and strategy evolution.
- Each iteration updates progress.md, handoff.md, and loop_state.json.
- Verification must include at least one narrow command or file check.
- Verification commands for the current bounded pass:
  - `python3 -m pytest -q tests/test_freqai_llm.py tests/test_make_strategy_miner_opencode_config.py tests/test_agent_executor.py`
  - `python3 -m py_compile src/agent_market/freqai/llm.py scripts/freqai_expression_agent.py scripts/make_strategy_miner_opencode_config.py scripts/bootstrap_strategy_factory_loops.py`
  - `python3 scripts/freqai_expression_agent.py --help | rg -n "llm-provider|llm-agent-url|llm-workspace|llm-max-turns|llm-stale-timeout"`
  - `python3 scripts/make_strategy_miner_opencode_config.py --base configs/strategy_miner_alpha_spot_core.json --output <tmp.json>`
