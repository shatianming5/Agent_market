# Mohu

## Missing
- [x] Missing-001: 增加 pytest 测试套件并把冒烟脚本迁移为可持续的测试
  - Location: `scripts/smoke_test.py`, `scripts/e2e_smoke_flow.py`（候选），新增 `tests/`
  - Acceptance: `pytest -q` 通过；至少覆盖 `/health`、`/settings`、以及黄金路径产物检查（可用小数据/短 timerange）。
  - Evidence: 当前仓库无 `tests/`；冒烟以脚本形式存在，无法在 CI 中稳定管理与分层。
  - Notes: 允许保留脚本作为本地快捷入口，但验收以 pytest 为准。
  - Implementation:
    - `pytest.ini` 限定 pytest 仅运行 `tests/`，避免误跑 `freqtrade/` 上游测试
    - `tests/test_api_smoke.py` 覆盖 `/health` 与 `/settings`
    - `tests/test_e2e_flow_smoke.py` 调用 `scripts/e2e_smoke_flow.py` 做端到端产物检查
    - `configs/agent_flow_kucoin_cpu_nollm_smoke.json` 提供短 timerange / 快速训练的 e2e 配置
    - `requirements-dev.txt` 增加 pytest 的开发依赖入口
  - Verified: 2026-01-31 (`pytest -q`)

- [x] Missing-002: 增加 CI（GitHub Actions）跑基础与端到端冒烟
  - Location: `.github/workflows/ci.yml`（新增）
  - Acceptance: PR/Push 自动执行 `python scripts/smoke_test.py`；可选执行 `python scripts/e2e_smoke_flow.py`（或其轻量版）。
  - Evidence: 需要把本地 smoke / e2e 验收迁移到 GitHub Actions，避免 PR 引入回归。
  - Notes: 端到端可按“快/慢”分离（smoke vs nightly）。
  - Implementation:
    - `.github/workflows/ci.yml`：新增 `smoke` job（push/pull_request 自动触发），运行：
      - `python scripts/smoke_test.py`
      - `pytest -q tests/test_api_smoke.py`
    - `.github/workflows/ci.yml`：新增可选 `e2e-flow-smoke` job（`workflow_dispatch` + `run_e2e=true` 时触发），运行：
      - `pytest -q tests/test_e2e_flow_smoke.py`
  - Verified: 2026-02-01 (`python scripts/smoke_test.py && pytest -q tests/test_api_smoke.py`)
  - Verified (GitHub Actions): 2026-02-01 (`CI` push run 21549750134)

- [x] Missing-003: 增加 run_id 与运行元信息落盘，建立可追溯的复现证据链
  - Location: `src/agent_market/agent_flow.py`, `src/agent_market/flow_steps.py`（或新增模块）
  - Acceptance: 每次 Flow 运行生成唯一 `run_id`；写入 `artifacts/run_meta.json`（包含：config 快照 hash、python/freqtrade 版本、关键产物路径）。
  - Evidence: 当前仅有日志与产物文件，缺少统一的“运行元信息”汇总文件。
  - Notes: 后续可用于前端展示与结果对比。
  - Implementation:
    - `src/agent_market/agent_flow.py`：为每次 `AgentFlow.run()` 生成 `run_id`，并在流程结束（成功/失败）时写入 `artifacts/run_meta.json` 与 `artifacts/runs/<run_id>/run_meta.json`
    - `src/agent_market/flow_steps.py`：新增 `get_freqtrade_version()`，通过 `freqtrade --version` 获取版本信息（失败时降级为 best-effort）
    - `scripts/agent_flow.py`：将 `--config` 路径透传给 `AgentFlow`，用于计算 config 文件快照 hash
    - `scripts/e2e_smoke_flow.py`：增加对 `artifacts/run_meta.json` 的存在性与关键字段检查
  - Verified: 2026-01-31 (`pytest -q tests/test_e2e_flow_smoke.py`)

- [x] Missing-004: 前端默认指向黄金配置并提供“一键黄金路径 + 产物检查”的 UX
  - Location: `web/app.js`, `web/index.html`
  - Acceptance: UI 默认加载 `configs/agent_flow_kucoin_cpu_nollm.json`；运行完成后展示产物检查结果与链接。
  - Evidence: 当前 UI 默认配置为 multi/binanceus，且没有产物检查汇总。
  - Notes: 与后端无强耦合，可纯前端实现。
  - Implementation:
    - `web/index.html`：默认输入改为黄金路径（KuCoin/CPU/no-llm）配置与参数，并新增 `#flowArtifacts` 面板
    - `web/app.js`：绑定顶部 `#btnTrainBt` 为“一键黄金路径”；Flow 结束后从日志提取 `run_id`，拉取 `/flow/run-meta/{run_id}` 并渲染产物检查 + 链接（features/top、results/summary、results/latest-training 等）
    - `server/api/routes/flow.py`：新增 `/flow/run-meta/latest` 与 `/flow/run-meta/{run_id}`（返回 run_meta + exists 检查），用于前端可视化
  - Next:
    - 自动验收（含默认 UI 配置 + run_meta 检查）：`pytest -q tests/test_e2e_flow_smoke.py`
    - 手动验收：`uvicorn server.main:app --host 0.0.0.0 --port 8000`，打开 `http://127.0.0.1:8000/web/index.html` → 点击顶部“训练 + 回测”，确认 `#flowArtifacts` 出现检查结果与可点击链接
  - Verified: 2026-01-31 (`pytest -q tests/test_e2e_flow_smoke.py`)

- [x] Missing-005: 黄金路径在“无外部数据/无交易所网络”下也可复现（自动生成 demo OHLCV + 离线 markets）
  - Location: `scripts/e2e_smoke_flow.py`, `src/agent_market/flow_steps.py`, `server/api/routes/run.py`（以及新增 wrapper 脚本）
  - Acceptance:
    - 在一个干净工作区（没有 `user_data/data/`）里执行 `python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm_smoke.json` 仍能跑通并生成完整产物
    - 回测在 backtesting/hyperopt 等 optimize 模式下不依赖交易所 API（不需要联网也能运行）
  - Evidence:
    - 训练/回测依赖 `user_data/data/<exchange>/*-<tf>.feather`，当前需要手动准备
    - freqtrade 在 optimize 模式下通常会 `load_markets` 访问交易所 API，离线场景会导致不稳定/不可复现
  - Notes: demo 数据仅用于 smoke/开发；真实研究仍应使用真实历史数据。
  - Implementation:
    - `src/agent_market/demo_data.py`：提供 demo OHLCV（feather）生成器（缺失时自动生成）
    - `scripts/e2e_smoke_flow.py`：运行 Flow 前自动调用 demo 数据生成器（fresh checkout 可直接跑）
    - `scripts/freqtrade_cli.py`：为 freqtrade CLI 注入离线 markets monkeypatch（optimize 模式下从 config pairs 合成 markets，避免交易所 API）
    - `src/agent_market/flow_steps.py`：回测优先使用 `scripts/freqtrade_cli.py` 执行，避免依赖系统 `freqtrade` 可执行文件与联网 markets
    - `server/api/routes/run.py`：`/run/backtest` 同样优先走 wrapper，保证 UI 触发回测的离线一致性
  - Next:
    - 快速验证 demo 数据生成：`pytest -q tests/test_demo_data_bootstrap.py`
    - 端到端验证（含回测 + 产物）：`pytest -q tests/test_e2e_flow_smoke.py`
  - Verified: 2026-02-01（see `docs/verify_log.md`）

- [x] Missing-006: 运行历史（Run History）列表：可查看最近 runs 并一键打开产物/摘要链接
  - Location: `server/api/routes/flow.py`, `web/app.js`, `web/index.html`
  - Acceptance:
    - 后端提供 `GET /flow/runs/list?limit=...` 返回最近 N 次 `run_id` 与关键字段（status/started_at/ended_at/配置 hash/最新回测 zip）
    - 前端展示运行历史列表，点击某个 run 可打开对应 run_meta，并复用产物检查面板展示该 run 的检查结果与链接
  - Evidence: 当前只能看“最新一次”结果与摘要，缺少可追溯的历史导航入口。
  - Implementation:
    - `server/api/routes/flow.py`：新增 `GET /flow/runs/list?limit=...`（扫描 `artifacts/runs/**/run_meta.json`，按 ended_at/mtime 倒序返回）
    - `web/index.html`：新增 Run History 面板（`#runHistoryLimit`/`#btnLoadRunHistory`/`#runHistory`）
    - `web/app.js`：新增 `loadRunHistory()` 拉取并渲染运行列表；`renderFlowArtifactsCheck(runIdOverride)` 支持点击某个 run_id 后复用产物检查面板
    - `tests/test_e2e_flow_smoke.py`：增加 `/flow/runs/list` 基础断言（包含最新 run_id）
  - Verified: 2026-02-01 (`pytest -q tests/test_e2e_flow_smoke.py`)

- [x] Missing-007: 清理被误追踪的运行产物（llm_feedback），避免每次运行污染 git 工作区
  - Location: `user_data/llm_feedback/latest_backtest_summary.json`, `.gitignore`
  - Acceptance:
    - `git ls-files user_data/llm_feedback/latest_backtest_summary.json` 不应输出任何内容
    - `pytest -q tests/test_e2e_flow_smoke.py` 通过（仍能生成反馈摘要文件）
  - Evidence: `.gitignore` 已忽略 `user_data/llm_feedback/`，但该文件仍被 git 追踪，导致每次 e2e/run 产生无意义 diff。
  - Notes: 不改变产物路径；仅移除 git tracking。
  - Implementation:
    - 从 git 索引移除（保留工作区文件，仍由运行过程写入）：
      - `git rm --cached user_data/llm_feedback/latest_backtest_summary.json`（推荐）
      - 或 `git update-index --force-remove -- user_data/llm_feedback/latest_backtest_summary.json`（等价）
  - Verified: 2026-02-01 (`pytest -q tests/test_e2e_flow_smoke.py`)

- [x] Missing-008: 增加 Portfolio（HRP 风险平价）步骤，并纳入 Flow 产物/证据链（含 API 读取）
  - Location: `src/agent_market/portfolio_opt.py`, `src/agent_market/agent_flow.py`, `server/api/routes/flow.py`, `web/app.js`, `configs/agent_flow_kucoin_cpu_nollm_portfolio.json`
  - Acceptance:
    - `python scripts/agent_flow.py --config configs/agent_flow_kucoin_cpu_nollm_portfolio.json --steps portfolio` 成功
    - 生成：
      - `artifacts/runs/<run_id>/portfolio/weights.json`
      - `artifacts/runs/<run_id>/portfolio/report.json`
      - （可选）`artifacts/runs/<run_id>/portfolio/returns.parquet`
    - API：
      - `GET /flow/portfolio/latest` 返回 `method=hrp` 与 `weights/stats/inputs`
  - Implementation:
    - `requirements-full.txt`：加入 `PyPortfolioOpt`
    - 新增 `src/agent_market/portfolio_opt.py`：feather → returns → HRP（PyPortfolioOpt）
    - `src/agent_market/agent_flow.py`：加入 `portfolio` step，并将 `portfolio_*` 写入 `run_meta.json`
    - `server/api/routes/flow.py`：run_meta checks + 新增 `/flow/portfolio/latest` 与 `/flow/portfolio/{run_id}`
    - `web/app.js`：产物检查面板在存在 portfolio 产物时显示链接
    - 新增测试：`tests/test_portfolio_hrp.py`, `tests/test_flow_portfolio_step.py`
    - 新增示例配置：`configs/agent_flow_kucoin_cpu_nollm_portfolio.json`
  - Verified: 2026-02-01 (`pytest -q`)

## Ambiguous
- [x] Amb-001: “完美落地”在前端体验上的范围与验收标准
  - Location: `web/`
  - Resolution (MVP UX):
    - UI 文案为 UTF-8 正常中文（修复 mojibake / 去 BOM）
    - UI 默认不依赖外部 CDN 资源（离线也能打开页面并完成黄金路径）
    - 保持黄金路径入口：一键运行、产物检查、Run History
  - Acceptance: `pytest -q tests/test_e2e_flow_smoke.py` 通过（包含 `/web/index.html` 的中文/无 CDN 断言）
  - Implementation:
    - `web/index.html`：修复中文乱码；移除 remixicon CDN 引用
    - `web/app.js`：修复中文乱码
    - `tests/test_e2e_flow_smoke.py`：增加对 `cdn.jsdelivr.net` 不出现与“自动布局”出现的断言
  - Verified: 2026-02-01 (`pytest -q tests/test_e2e_flow_smoke.py`)

## Log
- 2026-01-31: 新增黄金路径配置与端到端冒烟脚本；修复回测在无交易所网络时失败（离线 markets/pairlist 推断）；补齐 repo_inventory/plan/mohu 文档，开始进入“按 Missing 清零”的迭代循环。
- 2026-02-01: 补齐 Run History（后端 runs/list + 前端列表 + e2e 断言）；修复前端中文乱码并移除外部 CDN 依赖；补齐 CI（smoke 自动跑 + e2e 手动触发）；清理误追踪的 llm_feedback 产物；全套 pytest 通过。
