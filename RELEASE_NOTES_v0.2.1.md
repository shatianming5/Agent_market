# v0.2.1 — Flow 实时进度推送与前端展示

## 亮点
- 进度模型增强：按阶段（prepare/execute/summarize）估算百分比；execute 阶段根据日志增长动态提升进度。
- 推送通道：
  - 新增 SSE `GET /flow/stream/{job_id}`，事件类型 `progress`，data 为 JSON（含 steps[].status/phase/percent 与日志增量 delta）。
  - 新增 WebSocket `/flow/ws/{job_id}`，消息为 JSON，与 SSE 载荷一致。
- 前端：
  - Flow 面板优先使用 SSE 实时更新步骤标签，显示阶段与百分比；不可用时回退轮询。
  - SSE 消息中的日志增量实时追加到日志面板。

## 相关文件
- 后端：`server/main.py`（/flow/progress、/flow/stream、/flow/ws）
- 脚本：`src/agent_market/agent_flow.py`（STEP/PHASE 打点）
- 前端：`web/app.js`（SSE 优先、轮询回退、标签与日志更新）
- 校验：`scripts/smoke_test.py`、`scripts/frontend_smoke.mjs`

## 升级与兼容
- 无破坏性变更；/flow/progress 返回体新增 `phase/percent` 字段。
- SSE/WS 为新增接口，旧前端逻辑自动回退，不受影响。
