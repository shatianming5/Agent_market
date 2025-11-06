## 快速开始（后端 API）

1) 准备环境（推荐 Conda 环境 freqtrade）：

```powershell
"C:\\Users\\90646\\.conda\\envs\\freqtrade\\python.exe" -m pip install -r server/requirements.txt
"C:\\Users\\90646\\.conda\\envs\\freqtrade\\python.exe" -m pip install -r requirements.txt
"C:\\Users\\90646\\.conda\\envs\\freqtrade\\python.exe" -m pip install -r requirements-dev.txt
```

2) 启动服务：

```powershell
"C:\\Users\\90646\\.conda\\envs\\freqtrade\\python.exe" -m uvicorn server.main:app --host 127.0.0.1 --port 8032
```

3) 快速验证（不等待重任务完成）：

```powershell
"C:\\Users\\90646\\.conda\\envs\\freqtrade\\python.exe" scripts/server_quickcheck.py
```

4) 最小数据路由（MVP）：

- `GET /agents`、`POST /agents`、`GET /agents/{id}`
- `GET /orders`、`POST /orders`

数据库默认位于 `resources/user_data/app.db`（可用 `APP_DB_PATH` 配置）。

5) 环境变量模板：参见 `.env.example`（包含 LLM/DEX/X/News 可选变量）

6) 常用命令速查（无 Makefile）：

```powershell
# 安装依赖
python -m pip install -r requirements.txt
python -m pip install -r server/requirements.txt
python -m pip install -r requirements-dev.txt

# 启动服务
python -m uvicorn server.main:app --host 127.0.0.1 --port 8032

# 快速体检
python scripts/server_quickcheck.py
```
