# Verify Log

## Missing-001
- 2026-01-31: `pytest -q`
  - Result: PASS
  - Notes: Runs API smoke + e2e flow (feature/expression/ml/backtest) using local KuCoin feather demo data.
  - Artifacts: `user_data/backtest_results/backtest-result-*.zip`, `artifacts/models/lightgbm_real/training_summary.json`, `user_data/llm_feedback/latest_backtest_summary.json`

## Missing-002
- 2026-02-01: `python scripts/smoke_test.py && pytest -q tests/test_api_smoke.py`
  - Result: PASS
  - Notes: 本地预检通过；GitHub Actions 实际执行需推送到远端仓库后在 Actions 页面确认。
  - Artifacts: `.github/workflows/ci.yml`
- 2026-02-01: `GitHub Actions: CI / smoke (push) run 21549750134`
  - Result: PASS
  - Notes: 远端 Actions 已触发并通过（push 到 main）。
  - Artifacts: https://github.com/shatianming5/Agent_market/actions/runs/21549750134
- 2026-02-01: `GitHub Actions: CI / smoke (push) run 21550032300`
  - Result: PASS
  - Notes: 远端 Actions 已触发并通过（包含 BOM 回归测试：`tests/test_no_bom.py`）。
  - Artifacts: https://github.com/shatianming5/Agent_market/actions/runs/21550032300
- 2026-02-01: `GitHub Actions: CI / smoke (push) run 21550338467`
  - Result: PASS
  - Notes: 远端 Actions 已触发并通过（包含 requirements-full/Makefile/缺依赖友好报错等更新）。
  - Artifacts: https://github.com/shatianming5/Agent_market/actions/runs/21550338467
- 2026-02-01: `GitHub Actions: CI / smoke (push) run 21562881185`
  - Result: PASS
  - Notes: 远端 Actions 已触发并通过（Portfolio HRP 集成后 smoke 仍通过）。
  - Artifacts: https://github.com/shatianming5/Agent_market/actions/runs/21562881185

## Missing-003
- 2026-01-31: `pytest -q tests/test_e2e_flow_smoke.py`
  - Result: PASS
  - Notes: Flow 运行结束后生成 `artifacts/run_meta.json` 并包含 `run_id`/版本信息/关键产物路径。
  - Artifacts: `artifacts/run_meta.json`, `artifacts/runs/<run_id>/run_meta.json`

## Missing-004
- 2026-01-31: `pytest -q tests/test_e2e_flow_smoke.py`
  - Result: PASS
  - Notes: 校验 `/web/index.html` 默认指向黄金 Flow 配置，并验证 `run_meta` 检查端点可用于 UI 展示产物检查与链接。
  - Artifacts: `artifacts/run_meta.json`, `artifacts/runs/<run_id>/run_meta.json`

## Missing-005
- 2026-02-01: `pytest -q tests/test_demo_data_bootstrap.py`
  - Result: PASS
  - Notes: 在临时目录生成 demo OHLCV feather（KuCoin/BTC+ETH/1h），用于 fresh checkout 的 smoke 数据种子。
- 2026-02-01: `bash -lc 'bak="user_data/data.bak.$(date +%s)"; mv user_data/data "$bak"; trap "rm -rf user_data/data; mv \"$bak\" user_data/data" EXIT; python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm_smoke.json | rg "Markets not loaded \\(optimize mode\\)\\. Using synthesized markets"'`
  - Result: PASS
  - Notes: 验证在无 `user_data/data/` 的情况下也可自动生成 demo 数据并完成回测；同时确认离线 markets patch 生效（日志包含 synthesized markets 提示）。
  - Artifacts: `user_data/data/kucoin/*-1h.feather`, `user_data/backtest_results/backtest-result-*.zip`

## Missing-006
- 2026-02-01: `pytest -q tests/test_e2e_flow_smoke.py`
  - Result: PASS
  - Notes: 验证 `/flow/runs/list` 返回的最近 runs 中包含最新 run_id（用于前端 Run History 导航）。
  - Artifacts: `artifacts/runs/<run_id>/run_meta.json`, `artifacts/run_meta.json`

## Amb-001
- 2026-02-01: `pytest -q tests/test_e2e_flow_smoke.py`
  - Result: PASS
  - Notes: 校验前端静态页面中文编码正常（无 mojibake）且不依赖外部 CDN；同时确保端到端 Flow 冒烟仍可跑通。
  - Artifacts: `artifacts/run_meta.json`, `artifacts/runs/<run_id>/run_meta.json`

## Missing-007
- 2026-02-01: `pytest -q tests/test_e2e_flow_smoke.py`
  - Result: PASS
  - Notes: 验证移除被误追踪的 `user_data/llm_feedback/latest_backtest_summary.json` 后，e2e 仍可生成反馈摘要文件且测试通过。
  - Artifacts: `user_data/llm_feedback/latest_backtest_summary.json`, `artifacts/run_meta.json`

## Missing-008
- 2026-02-01: `pytest -q`
  - Result: PASS
  - Notes: 验证 Portfolio（HRP）模块/Flow step/API 读取均可工作（含 `tests/test_portfolio_hrp.py` 与 `tests/test_flow_portfolio_step.py`）。
  - Artifacts: `configs/agent_flow_kucoin_cpu_nollm_portfolio.json`, `artifacts/runs/<run_id>/portfolio/report.json`, `artifacts/run_meta.json`
