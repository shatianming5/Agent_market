# FastAPI 服务与 Web UI

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

FastAPI 服务把脚本和 flow 包装成 HTTP API，并挂载静态前端 `web/`。前端负责提交任务、查看进度、读取 artifacts 和展示结果。

## 代码入口

| 位置 | 用途 |
|---|---|
| `server/main.py` | uvicorn 入口 |
| `server/app.py` | app factory、CORS、鉴权、静态前端挂载、路由注册 |
| `server/job_manager.py` | 后台 job、日志、状态 registry |
| `server/api/routes/` | API 路由 |
| `web/index.html` / `web/app.js` | 静态前端 |

## 启动

```bash
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/web/index.html
```

启用 API key：

```bash
AGENT_MARKET_API_KEY=your-secret-key \
  uvicorn server.main:app --host 127.0.0.1 --port 8000
```

请求头：

```text
X-API-Key: your-secret-key
```

## API 分组

| 分组 | 端点 |
|---|---|
| 健康检查 | `GET /health`、`GET /`、`GET /docs` |
| 单任务运行 | `POST /run/feature`、`/run/expression`、`/run/factor_compile`、`/run/factor_eval`、`/run/train`、`/run/rl_train`、`/run/backtest`、`/run/hyperopt`、`/run/capture`、`/run/lob_rebuild`、`/run/micro_feature`、`/run/tca` |
| Flow | `POST /flow/run`、`GET /flow/progress/{job_id}`、`GET /flow/stream/{job_id}`、`WS /flow/ws/{job_id}` |
| Flow artifacts | `GET /flow/run-meta/latest`、`GET /flow/run-meta/{run_id}`、`GET /flow/runs/list`、`GET /flow/tca/{run_id}`、`GET /flow/factor-scores/{run_id}` |
| Jobs | `GET /jobs/{job_id}/status`、`GET /jobs/{job_id}/logs`、`POST /jobs/{job_id}/cancel` |
| Results | `GET /results/list`、`GET /results/latest-summary`、`GET /results/summary`、`GET /results/gallery`、`GET /results/aggregate`、`GET /results/bundles/download/{run_id}` |
| Features | `GET /features/top` |
| Settings | `GET /settings`、`POST /settings` |
| Strategy Miner | `/strategy-miner/*` |

## JobManager 行为

- 后台启动脚本进程
- 合并 stdout/stderr
- 内存保留日志 ring buffer
- 磁盘写 `user_data/job_logs/<job_id>.log`
- 状态写 `user_data/job_registry/<job_id>.json`
- Flow 进度通过日志标记和启发式解析生成

## 前端能力

- 服务设置读写
- 表达式生成、回测、摘要查看
- Agent Flow 一键运行与进度追踪
- 结果列表、图集、聚合与 bundle 下载
- Strategy Miner run / candidate / leaderboard 查看
- SSE 优先，必要时轮询日志

## 配置变量

| 变量 | 说明 |
|---|---|
| `AGENT_MARKET_API_KEY` | 开启后保护 `/run/*`、`/flow/run` 等 API |
| `AGENT_MARKET_CORS_ORIGINS` | CORS origin 列表 |
| `AGENT_MARKET_MAX_CONCURRENT_JOBS` | 最大并发 job，默认 `2` |
| `AGENT_MARKET_MAX_QUEUED_JOBS` | 最大排队 job，默认 `50` |

## 验证

```bash
curl http://127.0.0.1:8000/health
python scripts/smoke_test.py
```

