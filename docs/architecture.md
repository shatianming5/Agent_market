# Architecture — Agent Market 系统地图

> 给人类和 AI agent 一份**稳定的"系统边界"心智模型**，无需重排目录就能理解每个文件归属。
>
> 与 [`AGENTS.md`](../AGENTS.md) / [`docs/repo_inventory.md`](repo_inventory.md) / [`docs/INDEX.md`](INDEX.md) / [`scripts/README.md`](../scripts/README.md) 联动维护。如有冲突以仓库当前代码为准。

## 1. 一张图概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                              │
│  scripts/agent_flow.py · scripts/factor_lab.py · scripts/wq_brain.py│
│  scripts/strategy_miner.py · scripts/smoke_test.py · uvicorn        │
│         (CLI 包装；只做 argparse + dispatch)                        │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼─── L1  ORCHESTRATION ─────────────────────────────────┐
│  src/agent_market/agent_flow.py        — flow 主流水线编排           │
│  src/agent_market/flow_steps.py        — flow step 实现              │
│  src/agent_market/flow_ext/            — step_dispatch + artifacts   │
│  server/                               — FastAPI app + jobs + routes │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼─── L2  DOMAIN (业务子系统) ────────────────────────────┐
│  wq_brain        WorldQuant BRAIN agentic alpha mining (主驱动)      │
│  factor_lab      因子挖掘 / 回测 / 部署                              │
│  factor_hub      因子注册表 / 评估存储 / 部署 API                    │
│  strategy_miner  LLM 驱动的策略级挖掘                                │
│  factor_compiler 因子 DSL → 可执行 + checks + scoring                │
│  freqai          FreqAI 训练 (gradient_boosting / RL / 外部特征)    │
│  microstructure  capture / LOB / 微观特征                            │
│  tca             Transaction Cost Analysis                          │
│  portfolio_opt   HRP / 组合优化                                      │
│  agents          provider-agnostic agent executors                  │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼─── L3  CORE / RUNTIME (基础设施) ──────────────────────┐
│  paths.py / config.py / utils.py       — 路径 / 配置 / 通用工具      │
│  runtime_bootstrap.py / runtime_preflight.py — 引导 + 预检           │
│  src/runner_fsm/                       — 通用 OpenCode-FSM runner   │
│  factor_memory.py / strategy_registry.py / run_artifacts.py          │
│       — 注册表 / 记忆 / artifacts 写入助手                           │
└─────────────┬───────────────────────────────────────────────────────┘
              │
┌─────────────▼─── L4  RUNTIME ARTIFACTS (运行时产出) ─────────────────┐
│  artifacts/        — 各 run 的产物 / 模型 / factor_lab 输出          │
│  runtime_configs/  — 服务运行时 config 快照                          │
│  runtime_logs/     — 服务运行时日志                                  │
│  runtime_manifests/— 服务运行时 job 清单                             │
│  logs/             — 杂项日志                                        │
│  user_data/        — Freqtrade 工作区（含真实 OHLCV / 回测产物）     │
└─────────────────────────────────────────────────────────────────────┘

附属：
- workspace/        — 模板源（被 create_workspace.py 拷贝到 ws_<id>/）
- ws_production/    — 独立 Python 包：production-workspace 实验区（不属主流）
- freqtrade/        — vendored snapshot（只读）
```

## 2. 各层职责契约

### L1 — Orchestration（编排）

负责把"用户/agent 意图"→"按顺序执行的 step 列表"。**不做业务逻辑**；只做 dispatch + 元数据写入。

- `agent_flow.py` 解析 JSON config，加载步骤，调 `flow_ext.step_dispatch`
- `step_dispatch.py` 把每个 step 名分派给对应 handler（多数 delegate 到 `flow_steps.py`）
- `server/` 把 flow 暴露成 HTTP 端点 + 异步 job

**不要在这里实现业务**：每个 step 的实际工作落到 L2 的某个 domain package。

### L2 — Domain（业务子系统）

每个 package 是一个相对**独立的研究 / 生产子系统**，对外只通过明确的 API（`__init__.py` 顶层导出 + `dtypes.py`）暴露契约。

层内不互调（除非通过明确的 helper），层外只允许从 L1 编排或从 CLI 直接进入。

最重的子系统：
- **`wq_brain`**：8 轮 review loop 已硬化到 8.4/10；487 测试覆盖。`pool submit-worker` 是生产路径；`scan --auto-submit` / `pool resubmit-all` 是已知缺口（`AUTO_REVIEW.md` 详）。
- **`factor_lab`**：研究主 CLI；子命令 data / features / mine / validate / backtest / rank-export / rl / combo / deploy / hub。
- **`strategy_miner`**：OpenCode-driven agentic LLM 策略生成 + 验证。

### L3 — Core / Runtime（基础设施）

跨子系统共用的工具层。**不依赖任何 L2 domain**（否则会产生跨域耦合）。

- `paths.py` 是所有 artifacts 路径的唯一来源；任何 domain 写盘都该走它，方便测试隔离（`AGENT_MARKET_ARTIFACTS_ROOT` 环境变量）
- `runner_fsm` 是给 OpenCode-FSM 风格 agent 用的通用 runner；`wq_brain.agent_runner` 不是它的实例（用了 hermes/opencode CLI 直接调）

### L4 — Runtime artifacts（运行时产出）

**只读 / 由代码生成 / 不属仓库内容**。`.gitignore` 是权威清单；如果你看到 tracked 文件落在 `.gitignore` 范围内，那是历史遗留。

- `artifacts/` 是主要落点；每 run 一个 dir，含 `run_meta.json`
- `user_data/` 是 Freqtrade 工作区；含真实数据，`user_data/strategies/` 是策略源码

## 3. 顶层 flat 模块的归属（避免读者混淆）

`src/agent_market/` 下有 14 个平铺模块 + 11 个子包。flat 模块按上面的分层归属如下：

| 模块 | 归属 |
|---|---|
| `agent_flow.py` / `flow_steps.py` | L1 Orchestration |
| `runtime_bootstrap.py` / `runtime_preflight.py` | L3 Core/Runtime |
| `paths.py` / `config.py` / `utils.py` | L3 Core/Runtime |
| `factor_memory.py` / `strategy_registry.py` / `run_artifacts.py` | L3 Core/Runtime |
| `portfolio_opt.py` | L2 Domain |
| `backtest_results.py` | L3 Core/Runtime（dataclass 容器） |
| `factor_multiagent.py` | L2 Domain |
| `strategy_factory.py` (1700+ 行；待拆) | L2 Domain |
| `demo_data.py` | L4 Runtime artifacts (fixtures) |

> 未来如要重排（把 flat 模块移到 `core/` `flow/` 子包），需要先扫所有 import 路径 — 风险高，目前用本文件做"逻辑层级"足够。

## 4. agent / workspace / runtime artifacts 的边界

最容易混淆的就是这三类相邻概念：

| 概念 | 是什么 | 谁写 / 谁读 | 何时存在 |
|---|---|---|---|
| **agent state** | LLM agent 的会话 / 提示 / 决策日志 | agent CLI（hermes / opencode）写；`agent_runner.py` 读取最终摘要 | 一次会话期间 |
| **runtime state** | 服务/作业运行时配置 + 日志 + manifest | `server/` + `scripts/` 写；运维查 | 服务运行期间 |
| **artifacts** | 业务产物：因子分数 / 模型 / 回测 / TCA | L2 domain 写；后续 step 或 UI 读 | 持久（直到 GC） |
| **workspace** | 旧研究模板（被 `create_workspace.py` 拷贝出去） | 设计上 read-only；只有 `create_workspace.py` 写 | 历史保留 |
| **ws_production** | 独立 Python 包（adaptive_params / auto_improver / ...） | 自包含；不归 L1-L3 | 长期 |

## 5. 入口 → domain 的最短路径

| 你想跑什么 | 入口 | dispatch 到 |
|---|---|---|
| WQ BRAIN alpha mining | `scripts/wq_brain.py simulate / pool submit-worker` | `wq_brain.scan_runner` / `wq_brain.agent_runner` / `wq_brain.submit_gates` |
| 因子挖掘 + 回测 | `scripts/factor_lab.py mine / backtest` | `factor_lab.mining` / `factor_lab.backtest` |
| 离线主流水线（feature → expression → ml → backtest） | `scripts/agent_flow.py` | `agent_flow.py` → `flow_ext.step_dispatch` → 各 domain |
| LLM 策略级挖掘 | `scripts/strategy_miner.py` | `strategy_miner.runner` |
| HTTP 服务 / UI | `uvicorn server.main:app` | `server/app.py` 路由 → 各 domain |
| 冒烟 | `scripts/smoke_test.py` / `pytest -q` | 全栈 |

## 6. 当前已知架构债

详见 [`AUTO_REVIEW.md`](../AUTO_REVIEW.md)（review loop 历史）。摘要：

- `wq_brain.scan_runner` 的 `--auto-submit` 路径绕过 quota / persistence 栈
- `wq_brain pool resubmit-all` 缺 quota 预留 + outcome 持久化
- `src/agent_market/strategy_factory.py` 1700+ 行待拆
- `scripts/wq_brain.py` 2200 行 CLI monolith 待拆
- 双轨 plan 文件已通过命名澄清（`docs/proposals/agent_market_proposal.md` Proposal + `docs/plan.md` MVP，根 `plan.md` 是兼容指针），但未来仍可能合并
- `src/agent_market/__init__.py` 顶层 14 个 flat 模块未分到 `core/` `flow/` 子包

这些是结构债，不是 bug。生产路径（`pool submit-worker` 单进程）已通过 review。

## 7. 改动前先读

任何对本架构的修改（移文件 / 改导入 / 拆模块），必须先：

1. 读 `git ls-files | grep import` 找所有 import 引用
2. 跑 `pytest tests/test_wq_brain_*.py -q` 确认 487 测试基线
3. 看 `AUTO_REVIEW.md` 最新一轮的 deferred items 是否与你要做的事重合（避免重复劳动 / 撞反向回退）
