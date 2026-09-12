.PHONY: install install-full run smoke test-smoke test check e2e flow flow-smoke clean clean-dry

install:
	pip install -c constraints.txt -r server/requirements.txt -r requirements-dev.txt

install-full:
	pip install -c constraints.txt -r requirements-full.txt

run:
	uvicorn server.main:app --host 0.0.0.0 --port 8000

smoke:
	python scripts/smoke_test.py

test-smoke:
	pytest -q \
		tests/test_api_smoke.py \
		tests/test_no_bom.py \
		tests/test_security_and_gates.py \
		tests/test_sandbox_exec.py \
		tests/test_workspace_core.py \
		tests/test_workspace_filesystem_safety.py \
		tests/test_wq_brain_state_integrity.py \
		tests/test_wq_brain_pool.py \
		tests/test_wq_brain_submit_worker.py \
		tests/test_strategy_miner_runner.py \
		tests/test_strategy_miner_artifacts.py \
		tests/test_strategy_miner_phases.py \
		tests/test_factor_strategy_loop.py \
		tests/test_strategy_loop_final_gates.py \
		tests/test_rank_portfolio.py \
		tests/test_factor_memory.py \
		tests/test_gp_factor_mine_wrapper.py \
		tests/test_gp_factor_mine_v2.py \
		tests/test_backtest_results.py \
		tests/test_pipeline_leakage.py \
		tests/test_walkforward.py

test:
	pytest -q

check: smoke test

e2e:
	python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json

flow:
	python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json --steps feature expression ml backtest

flow-smoke:
	python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm_smoke.json --steps feature expression ml backtest

clean:
	python scripts/clean_workspace.py

clean-dry:
	python scripts/clean_workspace.py --dry-run
