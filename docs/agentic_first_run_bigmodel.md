# Agentic Strategy Miner 首跑（BigModel GLM API）

本指南用于：
1) 使用 BigModel API 作为 OpenAI-compatible LLM
2) 跑通第一条 agentic 策略任务
3) 做一次“收尾清理”（保留代码，清掉旧运行产物）

## 1. 环境变量（推荐，不落盘密钥）

```bash
export OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
export OPENAI_MODEL="glm-4.7"
export OPENAI_API_KEY="<YOUR_KEY>"
```

兼容变量同样可用（优先级更高）：

```bash
export LLM_BASE_URL="$OPENAI_BASE_URL"
export LLM_MODEL="$OPENAI_MODEL"
export LLM_API_KEY="$OPENAI_API_KEY"
```

> 说明：代码已兼容 BigModel 的 `/api/paas/v4` 路径，不会再强制追加 `/v1`。

## 2. 启动服务

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 3. 启动第一条 agentic 策略任务（API）

```bash
curl -sS -X POST "http://127.0.0.1:8000/strategy-miner/start" \
  -H "Content-Type: application/json" \
  -d '{
    "config": "configs/strategy_miner_default.json",
    "max_iterations": 1,
    "model": "glm-4.7"
  }'
```

返回中会有：`job_id`、`run_id`。

### 看进度

```bash
curl -sS "http://127.0.0.1:8000/strategy-miner/status/<job_id>"
curl -sS "http://127.0.0.1:8000/strategy-miner/runs/<run_id>"
```

### 批准候选落地到 user_data/strategies

```bash
curl -sS -X POST "http://127.0.0.1:8000/strategy-miner/runs/<run_id>/approve" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### 触发候选回测汇总

```bash
curl -sS -X POST "http://127.0.0.1:8000/strategy-miner/runs/<run_id>/backtest" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## 4. 收尾：清理旧版本运行产物（不删源码）

先预览：

```bash
python scripts/clean_workspace.py --dry-run
```

确认后执行：

```bash
python scripts/clean_workspace.py --keep-dirs
```

会清理：`.pytest_cache`、`artifacts`、`user_data/backtest_results`、`user_data/agent_logs` 等运行产物目录。

## 5. 代码收尾检查

```bash
pytest -q tests/test_strategy_miner.py tests/test_strategy_miner_api.py tests/test_strategy_miner_runner.py tests/test_runner_fsm.py
```

如果需要全量：

```bash
pytest -q
```
