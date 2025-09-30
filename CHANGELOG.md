# Changelog

## v0.2.0 — 中文化与联调（Flow 进度）
- 清理 GET / 与 /index 文本乱码
- 统一错误返回（_error），统一 jobs 错误包装
- 新增 `/flow/progress/{job_id}` 细粒度进度查询（脚本打点 + 回退启发式）
- `scripts/agent_flow.py` / `src/agent_market/agent_flow.py` 增加步骤打点日志（STEP_START/STEP_OK/STEP_FAIL）
- 前端接入进度轮询与标签高亮，统一加载与重试体验
- 图集渲染优化（默认限制 24 项）
- README 全面中文重写，补充 Flow 进度接口

## v0.1.0 — 初始版本
- FastAPI + Web 基础框架与主要 /run/* 路由
- JobManager 与日志轮询
- 简易前端页面
