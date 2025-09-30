# Changelog

## v0.2.2

- Backend
  - 清理并统一 GET /、/index、/health 的返回体为简洁 JSON（彻底去除乱码）。
  - /flow/progress 提供每步状态 + 阶段（prepare/execute/summarize）+ 百分比估算。
  - 新增 /flow/stream (SSE) 与 /flow/ws (WebSocket) 实时进度与日志推送。
  - 统一错误返回结构 `_error(code,message)`，JobManager 标准化 code/status。
  - 加强 /run/expression、/run/feature、/run/train、/run/backtest 的参数/路径校验。
  - 新增 /settings 读写 LLM 与默认 timeframe 设置。
- Frontend
  - 本地化 react/echarts/dagre/html2canvas 至 `web/vendor`，ReactFlow 仍走 CDN 并提供回退绑定。
  - 加入全局 fetch 包装与“重试上次操作”，失败态与 Loading 更清晰。
  - Flow 面板增加单步快捷按钮，SSE 优先展示细粒度进度。
  - 图集/聚合卡片样式焕新（迷你图、极简配色）。
  - 修复“前端无响应”的多处根因（依赖加载、事件绑定回退、日志轮询）。
- Tooling
  - `scripts/frontend_smoke.mjs`：校验关键 DOM ID；若检测到乱码字符（U+FFFD）则失败。
  - `scripts/clean_workspace.*`：一键清理临时与缓存文件，README 同步说明。

## v0.2.1

- Flow 进度与前后端联调增强；SSE/WS 初步接入；错误码标准化；UI 按钮与样式优化。

## v0.2.0

- 初始 MVP：FastAPI 后端 + Web 前端；/run/* 任务发起与 JobManager；基础 Flow 与结果展示。

