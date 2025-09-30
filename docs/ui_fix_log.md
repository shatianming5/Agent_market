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

