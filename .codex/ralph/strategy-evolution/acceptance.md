# Acceptance

- Strategy evolution remains direct OpenCode at runtime.
- Runtime config rendering does not mutate base configs.
- Existing grouped alpha configs stay reusable.
- Each Ralph iteration closes exactly one bounded gap with verification.
- Baseline verification commands:
  - `python3 -m pytest -q tests/test_make_strategy_miner_opencode_config.py tests/test_strategy_miner.py`
  - `python3 scripts/make_strategy_miner_opencode_config.py --base configs/strategy_miner_alpha_spot_core.json --output runtime_configs/strategy_miner_alpha_spot_core_opencode.json`
