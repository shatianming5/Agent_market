# UI 修复记录

## 2025-09-29 23:13:17
- 问题：左侧按钮点击无响应
- 深度原因：
  1) 事件绑定使用 useMemo，在 React 中 useMemo 不是用于副作用，可能在某些时机下未正确绑定；
  2) 画布（React Flow）可能覆盖在左侧侧栏之上（z-index/层级），导致点击命中画布而非按钮。
- 改动：
  - 将 web/app.js 中对按钮的事件绑定由 useMemo(() => {...}, []) 改为 useEffect(() => {...}, [])，确保在 DOM 就绪后绑定；
  - 调整 web/styles.css，为 .sidebar 添加 position: relative; z-index: 3;，为 .canvas 设置 z-index: 1;，避免画布层覆盖侧栏；
- 验证：本地启动后端与静态站点，侧栏按钮可正常点击、触发接口与实时日志；
- 提交：ix(ui): 确保左侧按钮可点击 - 将事件绑定从 useMemo 改为 useEffect；提高 sidebar z-index 防止画布覆盖


## 2025-09-29 23:32:24
- 问题：个别环境中仍出现侧栏元素无法响应点击
- 深度原因：
  1) React 事件绑定更适合通过 useEffect 保证 DOM 就绪；
  2) 需要一个内置“点击模拟器”，帮助快速验证是否已正确绑定；
- 改动：
  - 将 web/app.js 中的绑定确保使用 useEffect（再次校验）；
  - 新增 window._simulateClicks（支持 expr/backtest/summary），便于在浏览器控制台执行 _simulateClicks('expr') 快速验证；
  - web/styles.css 为 .sidebar 提升 z-index，避免画布层覆盖；
- 验证：在浏览器控制台执行 _simulateClicks('expr') 能触发“运行 表达式(LLM)”，日志实时输出；


## 2025-09-29 23:36:47
- 问题：访问 http://127.0.0.1:8000/ 返回 {detail: Not Found}
- 深度原因：FastAPI 未定义根路由，默认仅暴露我们注册的具体路由（如 /health、/run/*），根路径会 404。
- 改动：在 server/main.py 新增 GET / 根路由，返回接口索引、注意事项与示例，便于从根路径快速上手。
- 验证：重启 uvicorn 后访问 / 返回 endpoints/examples；访问 /docs 有交互式文档；/health 返回 ok。



## 2025-09-29 23:42:37
- UI 优化：新增节点面板（n8n 风格），支持拖入画布创建节点；新增删除选中功能；翻译侧栏按钮/标题为中文；修复部分乱码显示。
- 画布交互：为 .react-flow 绑定 dragover/drop 事件，支持从面板拖拽节点进入画布；暴露 window.__setNodes 方便调试添加；保留运行/保存/加载按钮。
- 兼容性：继续使用 fastapi 后端现有路由，无需变动；静态站点 http://127.0.0.1:8080/ 可直接打开。

