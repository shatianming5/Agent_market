# Release Notes v0.2.2

本次版本聚焦“可用性 + 可靠性 + 观感”三方面，一次性解决首页乱码、前端无响应、进度不可见、依赖加载不稳定等问题，并补齐离线依赖、本地化图集样式与服务设置面板。

主要更新
- 后端
  - 统一根路径 `GET /`、`/index` 与 `/health` 的 JSON 返回，清除历史乱码。
  - Flow 进度：`GET /flow/progress/{job_id}` 返回每步 `status/phase/percent`；新增 `GET /flow/stream/{job_id}` (SSE) 与 `WS /flow/ws/{job_id}` 实时推送进度与日志。
  - 错误结构：统一使用 `_error(code,message)`；`JobManager` 输出 `status/code/kind/meta`，前端更易识别脚本失败。
  - 参数校验：表达式/特征/训练/回测等接口在路径与字段上更严格，错误更可读。
  - 设置接口：`GET|POST /settings` 支持 `llm_base_url/llm_model/default_timeframe` 读写。
- 前端
  - UMD 本地化：react/react-dom/echarts/dagre/html2canvas 均放入 `web/vendor`；ReactFlow 仍走 CDN，但在失败时回退到直接 DOM 绑定与日志轮询，避免“无响应”。
  - 全局请求包装：记录上次 /run/* 操作，失败时状态栏提示并提供一键重试；任务状态条根据 jobs.status/logs 高亮“运行中/成功/失败”。
  - Flow 面板：新增单步快捷按钮（Feature/Expr/ML/RL/BT），显示阶段与百分比，日志实时追加。
  - 图集/聚合：卡片与配色焕新，增加迷你柱状图与加载态；汇总面板展示更关键的指标（收益、胜率、回撤等）。
- 工具链
  - `scripts/frontend_smoke.mjs`：关键 DOM ID 校验 + 页面文本含 U+FFFD 则失败；`scripts/clean_workspace.*` 提供清理临时/缓存的脚本。

升级与兼容
- 需要 Python 3.10+（测试环境 3.13）与 Node（仅前端烟测）。
- 回测/超参等重负载流程需安装 `freqtrade` 与所需数据；若环境中缺失，对应功能会返回清楚的错误提示。

已知问题
- ReactFlow 仍使用 CDN；如网络受限，仅画布相关功能降级，但关键工作流可用。后续版本将本地化其 UMD 与 CSS。
- 超大图集的渲染仍需优化（虚拟化列表 + 图表实例复用）。

