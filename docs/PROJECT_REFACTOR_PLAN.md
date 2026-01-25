## 目标

- **可读性高**：任何人 2 分钟内能定位“入口/配置/产物/核心逻辑”。
- **接口简单**：对外 HTTP API 保持稳定；对内模块 API 尽量小（少函数、少参数、少隐式依赖）。
- **易于逐个调试**：每个阶段都有独立入口（CLI/脚本）+ 可复现的最小命令 + 明确产物路径。
- **可渐进重构**：每一步都能跑烟测/单测验证，不做“一次性大挪移”。

## 现状（结论）

- 后端曾经把路由/校验/命令拼装/设置读写/进度解析（SSE/WS）混在单文件里，**不利于定位与逐步调试**；现已拆分为 `server/app.py + server/api/routes/*`，并保留 `server/main.py` 作为兼容入口。
- `scripts/` 覆盖面很广（Flow/训练/数据/清理/烟测），但没有按域分组；同时有一些能力依赖 `freqtrade`（可选）导致“装不齐就跑不通/测不通”。
- `src/agent_market` 核心包相对清晰，但仍存在“脚本侧拼 config / 服务侧拼命令”的重复与分散。

> 建议先从 **后端模块化** 开始：收益最大、风险可控、对外接口保持不变。

## 建议的目标结构（渐进式）

### 后端（server）

把“巨型入口文件”拆成小模块，保持对外路由不变：

```
server/
  main.py                 # 保留为 uvicorn 入口：from server.app import app
  app.py                  # create_app / app 初始化（CORS、静态 web、include_router）
  runtime.py              # ROOT/SRC/SETTINGS_PATH/jobs 等全局运行时对象
  job_manager.py          # 保留 JobManager（子进程 + 日志）
  api/
    __init__.py
    models.py             # Pydantic 请求体（ExpressionReq/BacktestReq/...）
    errors.py             # _error + 统一错误结构
    validators.py         # timeframe/pairs/path 相关校验
    routes/
      root.py             # / /index /health
      settings.py         # /settings
      jobs.py             # /jobs/*
      results.py          # /results/*
      run.py              # /run/*
      flow.py             # /flow/*
```

拆分原则：

- 每个 `routes/*.py` 只处理一个域（results/jobs/run/flow/settings），文件行数和函数数量都控制在可读范围。
- 重复的校验/路径解析/错误格式统一放到 `api/*`。
- `server/main.py` 只作为兼容入口（保证 `uvicorn server.main:app` 不变）。

### 核心包（src/agent_market）

保持现状为主，只做“接口收敛”：

- 约束跨层依赖：`server` 只调用“脚本/外部命令”或 “src/agent_market` 的少量 façade API”，避免直接依赖内部细节。
- 把“命令拼装/路径解析”逐步抽成 2~3 个小函数模块（避免大量散落在 server/main.py）。

### 脚本（scripts）

不建议立刻移动文件（会影响 API 内引用路径），先做“可调试性增强”：

- 保持现有脚本路径不变；后续如需分组，采用“新路径 + 旧路径薄封装”的方式迁移。
- 每个核心阶段保留 1 个稳定入口：
  - Flow：`scripts/agent_flow.py`
  - ML：`scripts/train_pipeline.py`
  - RL：`scripts/train_rl.py`
  - API 烟测：`scripts/smoke_test.py`

## 接口简化策略（落地规则）

- **统一返回结构**（server 内部强制）：成功返回 `{"status":"ok"| "started", ...}`；失败统一 `_error(code, message, **extra)`。
- **统一路径解析**：所有相对路径都以 `ROOT` 为基准 resolve；并明确 fallback（比如 backtest_results）。
- **把“拼命令”集中到一个地方**：每个 `/run/*` endpoint 只做参数校验与调用 `build_*_command(req)`，不掺杂其它逻辑。
- **类少函数**：
  - 对“编排器”类（如 `AgentFlow`）尽量做成薄壳：只负责步骤顺序与日志标记；具体执行放到独立函数（如 `agent_market/flow_steps.py`），避免类方法数量膨胀。
  - 对“工具类”优先用纯函数 + dataclass 组合，避免“大而全的 Manager”。

## 逐步改造路线（建议按顺序做）

### Phase 0：建立稳定的验证入口（1 次性）

- 目标：任何改动后能快速知道“服务还能起来、关键接口没炸”。
- 命令：
  - `python3 scripts/smoke_test.py`
  - （可选）`python3 -m pytest -q`

### Phase 1：后端模块化（收益最大，风险可控）

- 目标：拆分 `server/main.py` → `server/app.py + server/api/routes/*`，并保持 URL/行为不变。
- 完成标准：
  - `uvicorn server.main:app` 正常启动
  - `python3 scripts/smoke_test.py` 全部通过

状态：已完成（当前仓库内已拆分并通过 smoke）。

补充：回测摘要与 Flow Step 已收敛到可复用函数（`agent_market/backtest_results.py`、`agent_market/flow_steps.py`），降低 server 与核心流程的耦合，便于逐步调试与复用。

### Phase 2：run/* 的命令拼装收敛（让接口“更简单”）

- 目标：把 `/run/expression|feature|backtest|train|rl_train|hyperopt` 的“构建 cmd + env”统一到小函数模块（如 `server/api/commands.py`）。
- 完成标准：run 相关路由中不再出现大段 cmd 拼接逻辑，便于逐个调试每种 job 的命令构建。

### Phase 3：可选依赖与测试分层（提升可调试性）

- 目标：`freqtrade` / `stable-baselines3` 缺失时，相关测试/能力可明确跳过或给出提示，而不是直接 ImportError。
- 完成标准：
  - 在“只装 requirements.txt + server/requirements.txt + pytest”的环境下，至少能跑过核心单测/烟测（跳过可选部分）。

### Phase 4：脚本分域整理（不破坏兼容）

- 目标：按领域整理 `scripts/`（data/flow/train/debug），但保留旧脚本为薄 wrapper，兼容外部引用。
- 完成标准：旧路径仍可用，新路径更清晰，README/文档同步更新。
