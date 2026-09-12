# 安装、测试与运行维护

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

本页汇总本地安装、环境变量、测试、清理、GC 和运行产物边界。

## 安装

推荐 Python 3.11。

完整研究 / 回测 / 训练路径：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -r requirements-full.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -c constraints.txt -r requirements-full.txt
```

只跑后端和离线 CI 冒烟测试：

```bash
pip install -c constraints.txt -r server/requirements.txt -r requirements-dev.txt
make test-smoke
```

最小安装仅覆盖 `Makefile` 中 `test-smoke` 的测试清单和 API 冒烟脚本，
不承诺支持 `pytest -q` 全套研究测试；不安装 Torch、LightGBM 或 RL 引擎。

macOS 如果要跑 TA-Lib 相关策略：

```bash
brew install ta-lib
```

## LLM 环境变量

| 变量 | 说明 |
|---|---|
| `OPENAI_BASE_URL` | OpenAI 兼容接口 base URL |
| `OPENAI_API_KEY` | LLM API key |
| `OPENAI_MODEL` | 默认模型 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 兼容别名 |
| `OPENAI_API_BASE` | 部分旧脚本兼容变量 |

## 服务和路径变量

| 变量 | 说明 |
|---|---|
| `AGENT_MARKET_API_KEY` | 后端 API key |
| `AGENT_MARKET_CORS_ORIGINS` | CORS origin 列表 |
| `AGENT_MARKET_ARTIFACTS_ROOT` | 重定向 `artifacts/` |
| `AGENT_MARKET_USER_DATA_ROOT` | 重定向 `user_data/` |
| `AGENT_MARKET_RUNS_ROOT` | 重定向 run 目录 |
| `AGENT_MARKET_MODELS_ROOT` | 重定向模型目录 |
| `AGENT_MARKET_MAX_CONCURRENT_JOBS` | 后端最大并发 job，默认 `2` |
| `AGENT_MARKET_MAX_QUEUED_JOBS` | 后端最大排队 job，默认 `50` |

## Makefile

```bash
make install-full
make run
make flow
make flow-smoke
make smoke
make test-smoke
make test
make check
make e2e
make clean-dry
make clean
```

## 测试

最小安装与 CI 使用同一份离线测试清单：

```bash
make test-smoke
python scripts/smoke_test.py
```

完整安装后可运行更广的研究测试：

```bash
pytest tests/test_wq_brain_*.py -q
pytest tests/test_rank_portfolio.py -q
pytest -q
```

全套测试依赖 `requirements-full.txt` 及所选可选引擎；网络测试默认跳过，
用 `--run-network` 显式启用。涉及真实回测 / 训练的测试还需要对应数据、
配置与外部工具，不能把最小安装或跳过可选测试视为完整路径验收。
真实端到端验证另行运行：

```bash
python scripts/e2e_smoke_flow.py --config configs/agent_flow_kucoin_cpu_nollm.json
```

## 清理与 GC

```bash
python scripts/clean_workspace.py --dry-run
python scripts/clean_workspace.py
```

Flow runs：

```bash
python scripts/gc_runs.py --keep 50 --prune-backtests --dry-run
python scripts/gc_runs.py --keep 50 --prune-backtests
```

Job logs / registry：

```bash
python scripts/gc_jobs.py --keep 200 --keep-days 14 --dry-run
python scripts/gc_jobs.py --keep 200 --keep-days 14
```

## 不要手动维护的目录

- `artifacts/`
- `runtime_configs/`
- `runtime_logs/`
- `runtime_manifests/`
- `logs/`
- `.tmp/`
- `.pytest_cache/`
- `.venv*/`
- `.opencode/`
- `.playwright-mcp/`
- `freqtrade/`，vendored snapshot，只读
- `user_data/`，含真实 OHLCV、策略和回测产物
- `workspace/`，旧模板源，只读为主
- `ws_production/`，独立实验区

`.gitignore` 是运行产物边界的权威清单。历史上可能存在 tracked artifact，不要把它们当作当前推荐手动维护的文件。
