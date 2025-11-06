# Agent Market 项目导览（中文）

面向新同学的快速上手指引，涵盖目录结构、核心模块、常用命令与数据落盘位置，帮助你在清理后的仓库中迅速定位关键文件。

## 目录总览

```text
├─ src/agent_market/     # Python 主包：AgentFlow、runtime、FreqAI 管道
│  ├─ agent_flow.py      # 端到端流程编排（下载→特征→表达式→训练→回测）
│  ├─ config.py          # 配置读取与 FreqAI 设置封装
│  ├─ runtime/           # 并发调度、任务执行器、workflow schema
│  └─ freqai/            # 特征工程、模型注册、训练管线
├─ server/               # FastAPI 服务：API 入口、JobManager、DB 访问层
│  ├─ main.py            # 应用入口，汇总路由与依赖
│  ├─ job_manager.py     # 子进程任务与日志管理
│  └─ routes_*.py        # agents / connectors / triggers / secrets 等路由
├─ web/                  # Vite + React 前端，包含 UI 组件与状态管理
├─ scripts/              # 实用脚本：数据收集、诊断、测试、流执行等
├─ configs/              # JSON 配置（AgentFlow、freqtrade、训练、回测）
├─ conf/                 # YAML 配置（交易对、关键字、触发器等）
├─ docs/                 # 项目文档（架构、流程、诊断指南等）
├─ tests/                # Pytest 套件，覆盖核心运行时与 API
├─ resources/user_data/            # 运行期生成的数据库、表达式、模型、图表
├─ data/                 # 市场行情原始/清洗数据（Parquet/Feather）
└─ artifacts/            # 训练模型、报告等产物（按实验分类）
```

> **提示**：`node_modules/`、`resources/user_data/logs/` 等临时或第三方目录已从仓库中清理或忽略，需要时可通过安装脚本重新生成。

## 核心模块速览

- **AgentFlow (`src/agent_market/agent_flow.py`)**  
  负责串联多步骤工作流，支持并发、重试、速率限制与事件输出。执行记录会写入 `resources/user_data/runs/<run_id>/`。

- **Runtime 执行器 (`src/agent_market/runtime/`)**  
  提供任务调度器、工作流 JSON Schema、上下文解析工具，实现 `$ref`/`$expr` 动态注入以及循环、条件、子流程等高级控制流。

- **FreqAI 管道 (`src/agent_market/freqai/`)**  
  完成特征计算、数据集构建与模型训练；训练结果默认写入 `artifacts/models/` 并生成 `training_summary.json`。

- **FastAPI 服务 (`server/` 目录)**  
  `server/main.py` 启动应用并加载运行期依赖；`server/db.py` 提供 SQLite ORM 包装；`server/job_manager.py` 统一管理 CLI 子进程和日志。

- **前端 (`web/`)**  
  通过 REST/SSE 消费后端接口，模块化的 Tab 包括数据实验室、表达式工作台、回测控制台、作业监控、运行历史、连接器与密钥管理。

## 常用命令

| 任务                   | 命令 |
|------------------------|------|
| 安装 Python 依赖       | `python -m pip install -e . && python -m pip install -r requirements-dev.txt` |
| 安装前端依赖           | `npm --prefix web install` |
| 启动 FastAPI 服务      | `uvicorn server.main:app --host 127.0.0.1 --port 8032` |
| 启动 AgentFlow         | `am-flow --config configs/agent_flow_multi.json` |
| 运行测试               | `pytest -q` |
| 执行 linters/格式化     | `ruff check . && black --check . && npm --prefix web run lint` |

常用命令可参考 `docs/QUICK_START.md` 或在 README 中的运行指南执行。

## 数据与产物

- **运行数据库**：`resources/user_data/app.db`（SQLite），存放 agents、orders、job steps 等元数据。  
- **表达式 & 特征**：`resources/user_data/freqai_expressions*.json`、`resources/user_data/freqai_features*.json`。  
- **模型产物**：`artifacts/models/<experiment>/` 下的模型文件与摘要。  
- **市场数据**：`data/` 与 `resources/user_data/data/`（按交易所与时间框架划分）。  
- **日志**：运行过程中产生的日志建议写入 `resources/user_data/agent_logs/`；常用日志文件已在 `.gitignore` 中忽略，并在清理脚本中自动删除。

## 清理策略（当前已执行）

- 删除所有 `__pycache__/` 目录与 `.pytest_cache/`。  
- 清除顶层与 `resources/user_data/logs/` 下的 `.log` 文件。  
- 移除历史 UI 测试产物目录 `test-results/`。  

未来如需再次清理，可以运行 `python scripts/diagnostics/cleanup.py`（可自行编写或扩展），或参考 `.gitignore` 内的路径列表手动删除。

## 入门顺序建议

1. 阅读 `README.md` 与本文档了解总体布局。  
2. 结合 `docs/architecture/current.md` 中的 Mermaid 图掌握端到端流程。  
3. 逐步阅读 `src/agent_market/agent_flow.py` 与 `server/main.py`，理解后端调度与 API。  
4. 通过 `web/src/App.tsx` 掌握前端如何调用 API，并在本地运行前后端进行联调。  
5. 使用 `tests/` 中的单元测试作为具体用例，快速定位某一模块的行为与契约。

> 若你正在扩展功能，建议在 `docs/` 目录下同步维护相应的设计/操作手册，以保持仓库长期可读。

---

如发现文档缺漏或目录再次堆积，可在 `docs/` 新增说明或提交 Issue 保持清晰度。
