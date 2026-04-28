# Acceptance

- Factor mining still runs through direct OpenCode agent execution.
- The expression CLI accepts provider/workspace/agent settings.
- Generated artifacts remain JSON-parseable and machine-checkable.
- Each iteration records the exact verification command and result.
- Baseline verification commands:
  - `python3 -m pytest -q tests/test_freqai_llm.py`
  - `python3 scripts/agent_flow.py --config configs/agent_flow_kucoin_factor_opencode.json --steps feature expression`
