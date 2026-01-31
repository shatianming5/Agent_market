.PHONY: install install-full run smoke test e2e flow flow-smoke clean clean-dry

install:
	pip install -r server/requirements.txt -r requirements-dev.txt

install-full:
	pip install -r requirements-full.txt

run:
	uvicorn server.main:app --host 0.0.0.0 --port 8000

smoke:
	python scripts/smoke_test.py

test:
	pytest -q

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

