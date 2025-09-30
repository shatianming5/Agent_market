# Changelog

## v0.2.2

- Backend
  - æ¸…ç†å¹¶ç»Ÿä¸€ GET /ã€/indexã€/health çš„è¿”å›žä½“ä¸ºç®€æ´ JSONï¼ˆå½»åº•åŽ»é™¤ä¹±ç ï¼‰ã€‚
  - /flow/progress æä¾›æ¯æ­¥çŠ¶æ€ + é˜¶æ®µï¼ˆprepare/execute/summarizeï¼‰+ ç™¾åˆ†æ¯”ä¼°ç®—ã€‚
  - æ–°å¢ž /flow/stream (SSE) ä¸Ž /flow/ws (WebSocket) å®žæ—¶è¿›åº¦ä¸Žæ—¥å¿—æŽ¨é€ã€‚
  - ç»Ÿä¸€é”™è¯¯è¿”å›žç»“æž„ `_error(code,message)`ï¼ŒJobManager æ ‡å‡†åŒ– code/statusã€‚
  - åŠ å¼º /run/expressionã€/run/featureã€/run/trainã€/run/backtest çš„å‚æ•°/è·¯å¾„æ ¡éªŒã€‚
  - æ–°å¢ž /settings è¯»å†™ LLM ä¸Žé»˜è®¤ timeframe è®¾ç½®ã€‚
- Frontend
  - æœ¬åœ°åŒ– react/echarts/dagre/html2canvas è‡³ `web/vendor`ï¼ŒReactFlow ä»èµ° CDN å¹¶æä¾›å›žé€€ç»‘å®šã€‚
  - åŠ å…¥å…¨å±€ fetch åŒ…è£…ä¸Žâ€œé‡è¯•ä¸Šæ¬¡æ“ä½œâ€ï¼Œå¤±è´¥æ€ä¸Ž Loading æ›´æ¸…æ™°ã€‚
  - Flow é¢æ¿å¢žåŠ å•æ­¥å¿«æ·æŒ‰é’®ï¼ŒSSE ä¼˜å…ˆå±•ç¤ºç»†ç²’åº¦è¿›åº¦ã€‚
  - å›¾é›†/èšåˆå¡ç‰‡æ ·å¼ç„•æ–°ï¼ˆè¿·ä½ å›¾ã€æžç®€é…è‰²ï¼‰ã€‚
  - ä¿®å¤â€œå‰ç«¯æ— å“åº”â€çš„å¤šå¤„æ ¹å› ï¼ˆä¾èµ–åŠ è½½ã€äº‹ä»¶ç»‘å®šå›žé€€ã€æ—¥å¿—è½®è¯¢ï¼‰ã€‚
- Tooling
  - `scripts/frontend_smoke.mjs`ï¼šæ ¡éªŒå…³é”® DOM IDï¼›è‹¥æ£€æµ‹åˆ°ä¹±ç å­—ç¬¦ï¼ˆU+FFFDï¼‰åˆ™å¤±è´¥ã€‚
  - `scripts/clean_workspace.*`ï¼šä¸€é”®æ¸…ç†ä¸´æ—¶ä¸Žç¼“å­˜æ–‡ä»¶ï¼ŒREADME åŒæ­¥è¯´æ˜Žã€‚

## v0.2.1

- Flow è¿›åº¦ä¸Žå‰åŽç«¯è”è°ƒå¢žå¼ºï¼›SSE/WS åˆæ­¥æŽ¥å…¥ï¼›é”™è¯¯ç æ ‡å‡†åŒ–ï¼›UI æŒ‰é’®ä¸Žæ ·å¼ä¼˜åŒ–ã€‚

## v0.2.0

- åˆå§‹ MVPï¼šFastAPI åŽç«¯ + Web å‰ç«¯ï¼›/run/* ä»»åŠ¡å‘èµ·ä¸Ž JobManagerï¼›åŸºç¡€ Flow ä¸Žç»“æžœå±•ç¤ºã€‚

## v0.2.3

- 本地化 ReactFlow（通过 esbuild 打包为 IIFE，全局 `window.ReactFlow`），彻底移除最后的 CDN 依赖。
- 更新 `web/index.html` 使用 `vendor/reactflow.min.js`。
- 新增 `npm run build:vendor`，可复现离线打包流程。

## v0.2.3

- 本地化 ReactFlow（通过 esbuild 打包为 IIFE，全局 `window.ReactFlow`），彻底移除最后的 CDN 依赖。
- 更新 `web/index.html` 使用 `vendor/reactflow.min.js`。
- 新增 `npm run build:vendor`，可复现离线打包流程。
