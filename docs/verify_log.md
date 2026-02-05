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

## Missing-011
- 2026-02-05: `pytest -q tests/test_micro_capture.py`
  - Result: PASS
  - Notes: Fixture 模式回放生成 `match.ndjson.gz`/`level2.ndjson.gz` 与 `manifest.json`（count>0）；并用 TestClient 验证 `POST /run/capture` 返回 `status=started` 与 `job_id`。
  - Artifacts: `tests/fixtures/kucoin_ws_sample.jsonl`, `server/api/routes/run.py`

## Missing-012
- 2026-02-05: `pytest -q tests/test_lob_rebuild.py`
  - Result: PASS
  - Notes: 使用 fixture 生成 capture（level2）后执行 `lob_rebuild`，产出 `lob_state.parquet` 与 `rebuild_report.json`；并用 TestClient 验证 `POST /run/lob_rebuild` 返回 `status=started` 与 `job_id`。
  - Artifacts: `tests/fixtures/kucoin_lob_snapshot.json`, `src/agent_market/microstructure/lob/rebuild.py`, `scripts/lob_rebuild.py`

## Missing-013
- 2026-02-05: `pytest -q tests/test_microstructure_features.py`
  - Result: PASS
  - Notes: 使用 fixture 的 `lob_state.parquet` 与 `match.ndjson.gz` 生成 microstructure features（mid/spread/microprice/depth/imbalance + trade_sign/vwap/ofi/arrival_intensity），并在 manifest 标注输入与 data_sources。
  - Artifacts: `src/agent_market/microstructure/features/feature_registry.py`, `scripts/micro_features.py`, `tests/test_microstructure_features.py`

## Missing-014
- 2026-02-05: `pytest -q tests/test_tca_report.py`
  - Result: PASS
  - Notes: 生成 `tca_report.json`（plan.md 5.1 v1 顶层结构位置齐全；允许 null/[] 占位）与 `tca_report.html`。
  - Artifacts: `src/agent_market/tca/schema.py`, `src/agent_market/tca/report.py`, `scripts/tca_report.py`

## Missing-015
- 2026-02-05: `pytest -q tests/test_factor_spec.py`
  - Result: PASS
  - Notes: FactorSpec/ExprNode Pydantic v2 校验 + JSON schema 导出 + canonical JSON/sha256 稳定性测试通过。
  - Artifacts: `src/agent_market/factor_compiler/api_models.py`, `tests/test_factor_spec.py`

## Missing-016
- 2026-02-05: `pytest -q tests/test_factor_dsl.py`
  - Result: PASS
  - Notes: Formula 通过 Python AST 解析为 canonical `ExprNode`，并支持序列化回 Formula；覆盖 roundtrip（Formula → AST → JSON → AST → Formula）与非法语法拒绝。
  - Artifacts: `src/agent_market/factor_compiler/dsl/parser.py`, `src/agent_market/factor_compiler/dsl/serializer.py`, `tests/test_factor_dsl.py`

## Missing-017
- 2026-02-05: `pytest -q tests/test_expression_engine_ops.py`
  - Result: PASS
  - Notes: ExpressionEngine 补齐 `ifelse/log/exp/sqrt/rolling_sum` 并强制 `shift(x,n)` 的 `n>=0`（校验+运行时）；DSL 编译（rolling_mean/zscore alias）输出可被 `safe_eval_expression()` 执行。
  - Artifacts: `src/agent_market/freqai/expression_engine.py`, `src/agent_market/factor_compiler/dsl/operators.py`, `tests/test_expression_engine_ops.py`

## Missing-018
- 2026-02-05: `pytest -q tests/test_factor_checks.py`
  - Result: PASS
  - Notes: 实现 complexity_budget（nodes/depth）与 leakage tests（负 shift 结构检查 + permutation leakage sanity check），并覆盖 FAIL code 触发。
  - Artifacts: `src/agent_market/factor_compiler/checks/complexity.py`, `src/agent_market/factor_compiler/checks/leakage.py`, `tests/test_factor_checks.py`

## Missing-019
- 2026-02-05: `pytest -q tests/test_factor_scoring.py`
  - Result: PASS
  - Notes: 实现 ScoreReport 最小版（IC/RankIC + nan/turnover gates + Pareto front）并落盘 `factor_scores.json` 与 `pareto.csv`。
  - Artifacts: `src/agent_market/factor_compiler/scoring/aggregate.py`, `src/agent_market/factor_compiler/scoring/objectives.py`, `tests/test_factor_scoring.py`

## Missing-020
- 2026-02-05: `pytest -q tests/test_factor_flow_integration.py`
  - Result: PASS
  - Notes: 接入 `/run/factor_compile`、`/run/factor_eval` 与 Flow steps（factor_compile/factor_eval），并在 `/flow/run-meta/*` checks 中暴露 factor artifacts 的 exists 校验。
  - Artifacts: `server/api/routes/run.py`, `src/agent_market/agent_flow.py`, `src/agent_market/flow_steps.py`, `server/api/routes/flow.py`

## Missing-021
- 2026-02-05: `pytest -q tests/test_flow_capture_lob_steps.py`
  - Result: PASS
  - Notes: Flow 接入 `capture`/`lob_rebuild`（fixture 模式）并在 run_meta 与 `/flow/run-meta/*` checks 中记录/校验 `capture_manifest`、`lob_state_parquet`、`rebuild_report`。
  - Artifacts: `src/agent_market/agent_flow.py`, `src/agent_market/flow_steps.py`, `tests/test_flow_capture_lob_steps.py`

## Missing-022
- 2026-02-05: `pytest -q tests/test_report_bundle.py`
  - Result: PASS
  - Notes: 新增 Flow `report` step 生成 `bundle.zip`，并通过 `/results/bundles/*` 提供列表与下载。
  - Artifacts: `src/agent_market/flow_steps.py`, `src/agent_market/agent_flow.py`, `server/api/routes/results.py`, `tests/test_report_bundle.py`

## Missing-023
- 2026-02-05: `pytest -q tests/test_web_factor_ui_links.py`
  - Result: PASS
  - Notes: 前端“产物检查”面板新增 factor scores JSON 预览链接与 bundle.zip 下载链接；后端提供 `/flow/factor-scores/*`。
  - Artifacts: `web/app.js`, `server/api/routes/flow.py`, `tests/test_web_factor_ui_links.py`

## Missing-024
- 2026-02-05: `pytest -q tests/test_eval_protocol.py`
  - Result: PASS
  - Notes: 增加最小评测协议：walk-forward splitter（purge/embargo）+ 成本入账（profit_abs - fees_total）并生成可复现 `eval_report.json`。
  - Artifacts: `src/agent_market/freqai/training/eval_protocol.py`, `scripts/eval_protocol.py`, `tests/test_eval_protocol.py`

## Missing-025
- 2026-02-05: `pytest -q tests/test_flow_ext_modules.py`
  - Result: PASS
  - Notes: 新增 `flow_ext/` 薄封装层（steps/artifacts/validators）并确保可导入；同时保持现有 Flow 行为不变。
  - Artifacts: `src/agent_market/flow_ext/steps.py`, `src/agent_market/flow_ext/artifacts.py`, `src/agent_market/flow_ext/validators.py`, `tests/test_flow_ext_modules.py`

## Missing-026
- 2026-02-05: `pytest -q tests/test_factor_compiler_planmd_paths.py`
  - Result: PASS
  - Notes: 补齐 plan.md 提到的 Factor Compiler 文件级模块（ast/types/time_safety/data_schema/unit_test_gen/novelty/stability/prompts），并提供最小可用实现与导入路径。
  - Artifacts: `src/agent_market/factor_compiler/dsl/types.py`, `src/agent_market/factor_compiler/checks/data_schema.py`, `src/agent_market/factor_compiler/scoring/novelty.py`, `src/agent_market/factor_compiler/prompts/factor_spec.fewshot.json`

## Missing-028
- 2026-02-05: `pytest -q tests/test_microstructure_planmd_modules.py`
  - Result: PASS
  - Notes: 补齐 plan.md 文件级 Microstructure 模块（ws_capture/exchange_adapters/checksum/volatility_features/schemas），并提供离线 fixture 可验收的最小行为。
  - Artifacts: `src/agent_market/microstructure/capture/ws_capture.py`, `src/agent_market/microstructure/lob/checksum.py`, `src/agent_market/microstructure/schemas/lob_parquet.py`, `tests/test_microstructure_planmd_modules.py`

## Missing-027
- 2026-02-05: `pytest -q tests/test_expression_engine_planmd_ops.py`
  - Result: PASS
  - Notes: ExpressionEngine 补齐 plan.md 3.4 提到的时间序列/鲁棒算子（diff/decay_linear/winsorize/robust_z），并提供 alias（rolling_mean/rolling_std/zscore）。
  - Artifacts: `src/agent_market/freqai/expression_engine.py`, `tests/test_expression_engine_planmd_ops.py`

## Missing-029
- 2026-02-05: `pytest -q tests/test_tca_orders_fills.py`
  - Result: PASS
  - Notes: 从 freqtrade backtest trades 提取 orders/fills 并填充到 TCA v1 schema；最小实现 IS 以 fees_total 入账（quote_ccy + bps）。
  - Artifacts: `src/agent_market/tca/report.py`, `src/agent_market/tca/adapters/simulated_exec.py`, `tests/test_tca_orders_fills.py`

## Missing-030
- 2026-02-05: `pytest -q tests/test_llm_factor_spec.py`
  - Result: PASS
  - Notes: 增加 FactorSpec prompt assets 载入与离线 parse/validate 入口（无 API Key 也可运行验证链路）。
  - Artifacts: `src/agent_market/freqai/llm.py`, `src/agent_market/factor_compiler/prompts/factor_spec.system.md`, `tests/test_llm_factor_spec.py`

## Missing-031
- 2026-02-05: `pytest -q tests/test_api_error_codes_planmd.py`
  - Result: PASS
  - Notes: `/run/factor_*` 与 `/run/lob_rebuild` 增加 preflight 校验并返回 plan.md error codes（含 `DATA_NOT_FOUND/UNKNOWN_OPERATOR/TYPECHECK_FAILED/LOOKAHEAD_DETECTED/COMPLEXITY_BUDGET_EXCEEDED/LOB_SEQUENCE_GAP`）。
  - Artifacts: `server/api/routes/run.py`, `tests/test_api_error_codes_planmd.py`

## Missing-032
- 2026-02-05: `pytest -q tests/test_factor_microstructure_ops.py`
  - Result: PASS
  - Notes: Factor DSL 编译补齐 plan.md 3.4.4 微观结构算子，并将其映射为 microstructure features 的稳定列名（避免扩展 ExpressionEngine 函数白名单）。
  - Artifacts: `src/agent_market/factor_compiler/dsl/operators.py`, `tests/test_factor_microstructure_ops.py`

## Missing-033
- 2026-02-05: `pytest -q tests/test_factor_scoring_planmd_fields.py`
  - Result: PASS
  - Notes: ScoreReport 字段扩展 + weighted score 与 corr gate（best-effort）；factor_eval 透传表达式用于复杂度统计。
  - Artifacts: `src/agent_market/factor_compiler/scoring/aggregate.py`, `scripts/factor_eval.py`, `tests/test_factor_scoring_planmd_fields.py`

## Missing-034
- 2026-02-05: `pytest -q tests/test_microstructure_feature_library_planmd.py`
  - Result: PASS
  - Notes: Microstructure feature library 扩展 buy/sell volume 与 LOB shape slope（注册表可生成稳定列名）。
  - Artifacts: `src/agent_market/microstructure/features/ofi_features.py`, `src/agent_market/microstructure/features/core_features.py`, `src/agent_market/microstructure/features/feature_registry.py`, `tests/test_microstructure_feature_library_planmd.py`

## Missing-035
- 2026-02-05: `pytest -q tests/test_expression_engine_xs_ops.py`
  - Result: PASS
  - Notes: ExpressionEngine 新增截面算子（rank_xs/zscore_xs/corr_xs/neutralize），默认按 `date/ts` 分组并支持显式传入 group series。
  - Artifacts: `src/agent_market/freqai/expression_engine.py`, `tests/test_expression_engine_xs_ops.py`

## Missing-036
- 2026-02-05: `pytest -q tests/test_expression_engine_exec_ops.py`
  - Result: PASS
  - Notes: ExpressionEngine 新增执行/成本 proxy 算子（fill_prob/impact_proxy/queue_pos_proxy），并提供缺列时的 fallback。
  - Artifacts: `src/agent_market/freqai/expression_engine.py`, `tests/test_expression_engine_exec_ops.py`

## Missing-037
- 2026-02-05: `pytest -q tests/test_microstructure_convexity_planmd.py`
  - Result: PASS
  - Notes: microstructure features 补齐 `convexity_{L}`（二阶形态 proxy），并在 registry 默认注册。
  - Artifacts: `src/agent_market/microstructure/features/core_features.py`, `src/agent_market/microstructure/features/feature_registry.py`, `tests/test_microstructure_convexity_planmd.py`

## Missing-038
- 2026-02-05: `pytest -q tests/test_microstructure_execution_proxies_planmd.py`
  - Result: PASS
  - Notes: microstructure features 补齐执行/毒性 proxy（expected_slippage_proxy/fill_prob_proxy/toxicity_proxy）。
  - Artifacts: `src/agent_market/microstructure/features/feature_registry.py`, `tests/test_microstructure_execution_proxies_planmd.py`

## Missing-039
- 2026-02-05: `pytest -q tests/test_factor_leakage_planmd.py`
  - Result: PASS
  - Notes: Leakage checks 补齐 Shift test 与 0-lag spike signature（best-effort）。
  - Artifacts: `src/agent_market/factor_compiler/checks/leakage.py`, `tests/test_factor_leakage_planmd.py`

## Missing-040
- 2026-02-05: `pytest -q tests/test_factor_budget_planmd.py`
  - Result: PASS
  - Notes: Factor Compiler budgets/复杂度细化（max_expensive_ops + static compute/turnover budget estimators）。
  - Artifacts: `src/agent_market/factor_compiler/api_models.py`, `src/agent_market/factor_compiler/checks/complexity.py`, `tests/test_factor_budget_planmd.py`

## Missing-041
- 2026-02-05: `pytest -q tests/test_factor_scoring_planmd_more_fields.py`
  - Result: PASS
  - Notes: ScoreReport 补齐剩余字段（regime_consistency/train_test_gap/capacity_proxy/expensive_ops + 微结构占位）。
  - Artifacts: `src/agent_market/factor_compiler/scoring/aggregate.py`, `tests/test_factor_scoring_planmd_more_fields.py`

## Missing-042
- 2026-02-05: `pytest -q tests/test_future_return_label_planmd.py`
  - Result: PASS
  - Notes: 训练 label 显式化为 `future_return(close, h)`，避免散落的 `shift(-h)`。
  - Artifacts: `src/agent_market/freqai/training/labels.py`, `src/agent_market/freqai/training/pipeline.py`, `tests/test_future_return_label_planmd.py`

## Missing-043
- 2026-02-05: `pytest -q tests/test_tca_cost_breakdown_planmd.py`
  - Result: PASS
  - Notes: TCA report 补齐 IS 分解（spread/delay/market_impact）bps 最小 proxy，并输出基于 OHLCV volume 的 participation proxy。
  - Artifacts: `src/agent_market/tca/schema.py`, `src/agent_market/tca/report.py`, `tests/test_tca_cost_breakdown_planmd.py`

## Missing-044
- 2026-02-05: `pytest -q tests/test_factor_availability_delay_planmd.py`
  - Result: PASS
  - Notes: Factor Compiler time-safety 增加 availability_delay_ms gate（best-effort），并在 `/run/factor_compile` 预检透传 `constraints.min_delay_ms`。
  - Artifacts: `src/agent_market/factor_compiler/checks/time_safety.py`, `server/api/routes/run.py`, `tests/test_factor_availability_delay_planmd.py`
