# Feature READMEs

本目录按当前主要功能分组，每个功能一个子 README。根 [`README.md`](../../README.md) 只保留入口、快速开始和跳转；细节放在这里维护。

| 功能 | 子 README | 入口 |
|---|---|---|
| WQ BRAIN agentic alpha mining | [`wq_brain/README.md`](wq_brain/README.md) | `python scripts/wq_brain.py ...` |
| Agent Flow 离线主流水线 | [`agent_flow/README.md`](agent_flow/README.md) | `python scripts/agent_flow.py ...` |
| Factor Lab 研究 CLI | [`factor_lab/README.md`](factor_lab/README.md) | `python scripts/factor_lab.py ...` |
| Strategy Miner 策略挖掘 | [`strategy_miner/README.md`](strategy_miner/README.md) | `python scripts/strategy_miner.py ...` |
| FastAPI 服务与 Web UI | [`api_web/README.md`](api_web/README.md) | `uvicorn server.main:app ...` |
| Factor Compiler / Hub / Memory | [`factor_infrastructure/README.md`](factor_infrastructure/README.md) | `factor_compile`、`factor_eval`、`factor_lab.py hub` |
| 微观结构 / TCA / Rank Portfolio | [`microstructure_tca_rank/README.md`](microstructure_tca_rank/README.md) | Flow steps、`factor_lab.py rank-*` |
| 安装、测试与运行维护 | [`ops_testing/README.md`](ops_testing/README.md) | `make ...`、`pytest ...`、GC scripts |

