const { createElement: h, useState, useEffect, useCallback } = window.React
const { createRoot } = window.ReactDOM

// ReactFlow UMD safety
const RFLib = window.ReactFlow || {}
const RF = RFLib.ReactFlow || RFLib.default || (typeof RFLib === 'function' ? RFLib : RFLib)
const Background = RFLib.Background || (RFLib.default && RFLib.default.Background) || (() => null)
const Controls = RFLib.Controls || (RFLib.default && RFLib.default.Controls) || (() => null)
const MiniMap = RFLib.MiniMap || (RFLib.default && RFLib.default.MiniMap) || (() => null)
const Handle = RFLib.Handle || (() => null)
const Position = RFLib.Position || { Top: 'top', Bottom: 'bottom', Left: 'left', Right: 'right' }
const MarkerType = RFLib.MarkerType || {}
const applyNodeChanges = RFLib.applyNodeChanges || ((chs, nds) => nds)
const applyEdgeChanges = RFLib.applyEdgeChanges || ((chs, eds) => eds)
const addEdgeLib = RFLib.addEdge || ((params, eds) => eds.concat({ id: (params.id || ('e' + Math.random().toString(16).slice(2,8))), ...params }))

// -------- Fallback (no React/ReactFlow) helpers --------
async function fallbackPollLogs(jobId) {
  const logsEl = document.getElementById('logs')
  let offset = 0
  while (true) {
    try {
      const r = await fetch(`${API}/jobs/${jobId}/logs?offset=${offset}`)
      const j = await r.json()
      const chunk = (j.logs || []).join('\n')
      if (chunk) { logsEl.textContent += chunk + '\n'; try { logsEl.scrollTop = logsEl.scrollHeight } catch {} }
      offset = j.next || offset
      if (!j.running) break
      await new Promise(r => setTimeout(r, 800))
    } catch { break }
  }
}

function wireFallback() {
  // API apply
  try {
    const apiEl = document.getElementById('apiUrl'); if (apiEl) apiEl.value = API
    const applyBtn = document.getElementById('applyApi'); if (applyBtn) applyBtn.onclick = async () => {
      try { const val = (document.getElementById('apiUrl')?.value || '').trim(); if (!val) return; setApiUrl(val); const r = await fetch(`${API}/health`); const j = await r.json(); alert(`API 探测成功: ${API} /health: ${JSON.stringify(j)}`) } catch (e) { alert(`API 探测失败: ${API}, 调用 /health 出错: ${e}`) }
    }
  } catch {}
  // Expr
  const be = document.getElementById('btnExpr'); if (be) be.onclick = async () => {
    try {
      const body = { config: document.getElementById('cfg').value, feature_file: document.getElementById('featureFile').value, output: 'user_data/freqai_expressions.json', timeframe: document.getElementById('timeframe').value, llm_model: document.getElementById('llmModel').value, llm_count: parseInt(document.getElementById('llmCount').value||'3',10), llm_loops:1, llm_timeout:60, feedback_top:0 }
      const r = await fetch(`${API}/run/expression`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })
      const j = await r.json(); if (j.job_id) { try { setStatus && setStatus('表达式', j.job_id, true) } catch {} ; await fallbackPollLogs(j.job_id); try { setStatus && setStatus('表达式', j.job_id, false) } catch {} } else { alert(JSON.stringify(j)) }
    } catch(e) { alert('表达式启动失败: '+e) }
  }
  // Backtest
  const bb = document.getElementById('btnBacktest'); if (bb) bb.onclick = async () => {
    try {
      const body = { config: document.getElementById('cfg').value, strategy: 'ExpressionLongStrategy', strategy_path: 'user_data/strategies', timerange: document.getElementById('timerange').value, freqaimodel: 'LightGBMRegressor', export: true, export_filename: 'user_data/backtest_results/latest_trades_multi' }
      const r = await fetch(`${API}/run/backtest`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) })
      const j = await r.json(); if (j.job_id) { try { setStatus && setStatus('回测', j.job_id, true) } catch {} ; await fallbackPollLogs(j.job_id); try { setStatus && setStatus('回测', j.job_id, false) } catch {} } else { alert(JSON.stringify(j)) }
    } catch(e) { alert('回测启动失败: '+e) }
  }
  // Summary
  const bs = document.getElementById('btnSummary'); if (bs) bs.onclick = async () => {
    try { const r = await fetch(`${API}/results/latest-summary`); const j = await r.json(); const summaryEl=document.getElementById('summary'); if (summaryEl) summaryEl.textContent = JSON.stringify(j,null,2) } catch(e) { alert('加载摘要失败: '+e) }
  }
}

// API base: default same-origin, controlled by input框
let API = (typeof location !== 'undefined' && /^https?:/i.test(location.origin || '')) ? location.origin.replace(/\/$/, '') : 'http://127.0.0.1:8000'
const GOLDEN = {
  flow_cfg: 'configs/agent_flow_kucoin_cpu_nollm.json',
  flow_steps: 'feature expression ml backtest',
  freqtrade_cfg: 'user_data/config_freqai_kucoin.json',
  feature_file: 'user_data/freqai_features_real.json',
  timeframe: '1h',
  timerange: '20251126-20260125',
}
function setApiUrl(url) {
  try {
    API = (url || API).replace(/\/$/, '')
    const el = document.getElementById('apiUrl'); if (el) el.value = API
  } catch {}
}

// Helpers
function setLoading(btn, on) {
  try {
    if (!btn) return
    if (on) { btn.classList.add('is-loading'); btn.disabled = true } else { btn.classList.remove('is-loading'); btn.disabled = false }
  } catch {}
}
async function runWithLoading(btn, fn) {
  setLoading(btn, true)
  try { await fn() } finally { setLoading(btn, false) }
}
function setStatus(phase, jobId, running) {
  const el = document.getElementById('statusBar')
  if (!el) return
  if (running) {
    el.className = 'status running'
    el.innerHTML = `<i class="ri-loader-4-line spin"></i> ${phase} 运行中 · job ${jobId}`
  } else {
    el.className = 'status'
    el.innerHTML = `<i class="ri-check-line"></i> ${phase} 完成 · job ${jobId}`
  }
}

async function pollLogs(jobId) {
  const logsEl = document.getElementById('logs')
  let offset = 0
  let last = null
  while (true) {
    const res = await fetch(`${API}/jobs/${jobId}/logs?offset=${offset}`)
    const data = await res.json()
    last = data
    const chunk = (data.logs || []).join('\n')
    if (chunk) {
      logsEl.textContent += chunk + '\n'
      try { logsEl.scrollTop = logsEl.scrollHeight } catch {}
    }
    offset = data.next || offset
    if (!data.running) break
    await new Promise(r => setTimeout(r, 800))
  }
  try {
    const bar = document.getElementById('statusBar')
    if (last && last.code && last.code !== 'OK') {
      bar.className = 'status failed'
      bar.innerHTML = `<i class="ri-error-warning-line"></i> 脚本失败 (code=${last.code})`
      const retryBtn = document.createElement('button')
      retryBtn.textContent = '重试上次操作'
      retryBtn.onclick = () => { try { window.__retryLast && window.__retryLast() } catch(e) { alert('重试失败: '+e) } }
      const cards = document.getElementById('cards') || bar.parentElement
      cards && cards.appendChild(retryBtn)
    } else if (last && last.code === 'OK') {
      bar.className = 'status ok'
    }
  } catch {}
}

function _escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch] || ch))
}

function _extractLastRunIdFromLogs(text) {
  const re = /\[FLOW\]\s+RUN_ID\s+([0-9a-f]{8,64})/ig
  let m = null
  let last = null
  while ((m = re.exec(String(text || '')))) last = m[1]
  return last
}

function _renderArtifactItem({ label, ok, href, detail }) {
  const cls = ok ? 'tag tag-ok' : 'tag tag-failed'
  const icon = ok ? 'ri-check-line' : 'ri-error-warning-line'
  const inner = `<span class="${cls}"><i class="${icon}"></i> ${_escapeHtml(label)}</span>`
  const link = href ? `<a href="${_escapeHtml(href)}" target="_blank" rel="noreferrer" style="text-decoration:none">${inner}</a>` : inner
  const sub = detail ? `<div style="font-size:12px; color:var(--muted); margin:4px 0 0 0;">${_escapeHtml(detail)}</div>` : ''
  return `<div style="margin:6px 0;">${link}${sub}</div>`
}

async function renderFlowArtifactsCheck(runIdOverride) {
  const el = document.getElementById('flowArtifacts')
  if (!el) return
  const logsText = document.getElementById('logs')?.textContent || ''
  const runId = (runIdOverride || '').trim() || _extractLastRunIdFromLogs(logsText)
  if (!runId) {
    el.innerHTML = `<div style="font-weight:600; margin-bottom:6px;">产物检查</div><div style="color:var(--muted); font-size:12px;">未在日志中发现 run_id（[FLOW] RUN_ID ...），跳过检查。</div>`
    return
  }

  el.innerHTML = `<div style="font-weight:600; margin-bottom:6px;">产物检查 (run_id=${_escapeHtml(runId)})</div><div style="color:var(--muted); font-size:12px;">加载中…</div>`
  let meta = null
  try {
    const r = await fetch(`${API}/flow/run-meta/${encodeURIComponent(runId)}`)
    meta = await r.json()
  } catch (e) {
    el.innerHTML = `<div style="font-weight:600; margin-bottom:6px;">产物检查 (run_id=${_escapeHtml(runId)})</div><div style="color:var(--muted); font-size:12px;">加载 run_meta 失败: ${_escapeHtml(e)}</div>`
    return
  }
  if (meta && meta.status === 'error' && meta.code) {
    el.innerHTML = `<div style="font-weight:600; margin-bottom:6px;">产物检查 (run_id=${_escapeHtml(runId)})</div><pre class="panel" style="white-space:pre-wrap;">${_escapeHtml(JSON.stringify(meta, null, 2))}</pre>`
    return
  }

  const checks = (meta && meta.checks) || {}
  const artifacts = (meta && meta.artifacts) || {}
  const btZips = (checks.backtest_zips && Array.isArray(checks.backtest_zips)) ? checks.backtest_zips : []
  const trSums = (checks.training_summaries && Array.isArray(checks.training_summaries)) ? checks.training_summaries : []
  const latestZipName = (() => {
    const raw = (artifacts.backtest_zips && Array.isArray(artifacts.backtest_zips)) ? artifacts.backtest_zips : []
    if (!raw.length) return null
    const lastPath = raw[raw.length - 1]
    try { return String(lastPath).split('/').pop() } catch { return null }
  })()

  const rows = []
  rows.push(_renderArtifactItem({
    label: `run_meta.json (${meta.status || 'unknown'})`,
    ok: true,
    href: `${API}/flow/run-meta/${encodeURIComponent(runId)}`,
    detail: meta.run_meta_path || '',
  }))
  rows.push(_renderArtifactItem({
    label: 'config snapshot',
    ok: !!(checks.config && checks.config.exists),
    href: null,
    detail: (checks.config && checks.config.path) ? checks.config.path : '',
  }))
  if (artifacts.portfolio_report || artifacts.portfolio_weights) {
    rows.push(_renderArtifactItem({
      label: 'portfolio (HRP)',
      ok: !!(checks.portfolio_report && checks.portfolio_report.exists),
      href: `${API}/flow/portfolio/${encodeURIComponent(runId)}`,
      detail: (checks.portfolio_report && checks.portfolio_report.path) ? checks.portfolio_report.path : (artifacts.portfolio_report || ''),
    }))
  }
  rows.push(_renderArtifactItem({
    label: 'features (TopN)',
    ok: !!(checks.feature_output && checks.feature_output.exists),
    href: (artifacts.feature_output ? `${API}/features/top?file=${encodeURIComponent(artifacts.feature_output)}&limit=20` : null),
    detail: (checks.feature_output && checks.feature_output.path) ? checks.feature_output.path : (artifacts.feature_output || ''),
  }))
  rows.push(_renderArtifactItem({
    label: 'expressions (file)',
    ok: !!(checks.expression_output && checks.expression_output.exists),
    href: null,
    detail: (checks.expression_output && checks.expression_output.path) ? checks.expression_output.path : (artifacts.expression_output || ''),
  }))
  rows.push(_renderArtifactItem({
    label: 'feedback summary (file)',
    ok: !!(checks.feedback_summary && checks.feedback_summary.exists),
    href: null,
    detail: (checks.feedback_summary && checks.feedback_summary.path) ? checks.feedback_summary.path : (artifacts.feedback_summary || ''),
  }))
  rows.push(_renderArtifactItem({
    label: 'training summary',
    ok: trSums.some(x => x && x.exists),
    href: `${API}/results/latest-training`,
    detail: trSums.length ? trSums.map(x => x.path).filter(Boolean).slice(0, 3).join('\n') : '',
  }))
  rows.push(_renderArtifactItem({
    label: 'backtest summary',
    ok: btZips.some(x => x && x.exists),
    href: latestZipName ? `${API}/results/summary?name=${encodeURIComponent(latestZipName)}` : `${API}/results/latest-summary`,
    detail: btZips.length ? btZips.map(x => x.path).filter(Boolean).slice(0, 3).join('\n') : '',
  }))
  rows.push(_renderArtifactItem({
    label: 'results list',
    ok: true,
    href: `${API}/results/list`,
    detail: '',
  }))

  const ver = (meta && meta.freqtrade && meta.freqtrade.version) ? String(meta.freqtrade.version) : ''
  const py = (meta && meta.python && meta.python.version) ? String(meta.python.version).split('\n')[0] : ''
  const cfgSha = (meta && meta.config && meta.config.sha256) ? String(meta.config.sha256).slice(0, 12) : ''

  el.innerHTML = `
    <div style="font-weight:600; margin-bottom:6px;">产物检查 (run_id=${_escapeHtml(runId)})</div>
    <div style="font-size:12px; color:var(--muted); margin-bottom:6px;">
      <span style="margin-right:10px;">cfg_sha=${_escapeHtml(cfgSha || '--')}</span>
      <span style="margin-right:10px;">py=${_escapeHtml(py || '--')}</span>
      <span>freqtrade=${_escapeHtml(ver || '--')}</span>
    </div>
    <div>${rows.join('')}</div>
  `
}

function _renderRunStatusTag(status) {
  const st = String(status || '').toLowerCase()
  const cls = st === 'success' ? 'tag tag-ok' : (st === 'failed' ? 'tag tag-failed' : 'tag tag-running')
  const icon = st === 'success' ? 'ri-check-line' : (st === 'failed' ? 'ri-error-warning-line' : 'ri-time-line')
  const label = status || 'unknown'
  return `<span class="${cls}" style="margin:0;"><i class="${icon}"></i> ${_escapeHtml(label)}</span>`
}

async function loadRunHistory() {
  const el = document.getElementById('runHistory')
  if (!el) return
  let limit = 10
  try { limit = parseInt(document.getElementById('runHistoryLimit')?.value || '10', 10) } catch {}
  if (!isFinite(limit) || limit <= 0) limit = 10
  limit = Math.max(1, Math.min(200, limit))

  el.innerHTML = `<div style="font-size:12px; color:var(--muted);">加载中…</div>`

  let data = null
  try {
    const r = await fetch(`${API}/flow/runs/list?limit=${limit}`)
    data = await r.json()
  } catch (e) {
    el.innerHTML = `<div style="font-size:12px; color:var(--muted);">加载运行历史失败: ${_escapeHtml(e)}</div>`
    return
  }
  if (data && data.status === 'error' && data.code) {
    el.innerHTML = `<pre class="panel" style="white-space:pre-wrap;">${_escapeHtml(JSON.stringify(data, null, 2))}</pre>`
    return
  }
  const items = (data && Array.isArray(data.items)) ? data.items : []
  if (!items.length) {
    el.innerHTML = `<div style="font-size:12px; color:var(--muted);">未找到运行记录。</div>`
    return
  }

  const rows = items.map(it => {
    const rid = String(it.run_id || '')
    const when = (it.ended_at || it.started_at || '')
    const whenShort = when ? String(when).replace('T', ' ').replace('Z', '').slice(0, 19) : '--'
    const cfgSha = it.config_sha256 ? String(it.config_sha256).slice(0, 12) : ''
    const zipName = it.latest_backtest_zip_name || ''
    return `
      <tr>
        <td style="padding:4px 6px; border-top:1px solid var(--border);">
          <a href="#" data-runid="${_escapeHtml(rid)}" style="text-decoration:none; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
            ${_escapeHtml(rid.slice(0, 12))}
          </a>
        </td>
        <td style="padding:4px 6px; border-top:1px solid var(--border);">${_renderRunStatusTag(it.status)}</td>
        <td style="padding:4px 6px; border-top:1px solid var(--border); white-space:nowrap;">${_escapeHtml(whenShort)}</td>
        <td style="padding:4px 6px; border-top:1px solid var(--border); font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">${_escapeHtml(cfgSha)}</td>
        <td style="padding:4px 6px; border-top:1px solid var(--border); white-space:nowrap;">${_escapeHtml(zipName)}</td>
      </tr>
    `
  })

  el.innerHTML = `
    <div style="font-weight:600; margin-bottom:6px;">最近运行</div>
    <table style="width:100%; font-size:12px; border-collapse:collapse;">
      <tr style="text-align:left; color:var(--muted); font-size:12px;">
        <th style="padding:4px 6px;">run_id</th>
        <th style="padding:4px 6px;">状态</th>
        <th style="padding:4px 6px;">结束时间</th>
        <th style="padding:4px 6px;">cfg_sha</th>
        <th style="padding:4px 6px;">zip</th>
      </tr>
      ${rows.join('')}
    </table>
    <div style="font-size:12px; color:var(--muted); margin-top:6px;">点击 run_id 查看产物检查。</div>
  `

  try {
    el.querySelectorAll('a[data-runid]').forEach(a => {
      a.onclick = (ev) => {
        try { ev.preventDefault() } catch {}
        const runId = a.getAttribute('data-runid') || ''
        if (!runId) return
        renderFlowArtifactsCheck(runId)
        try { document.getElementById('flowArtifacts')?.scrollIntoView({ behavior: 'smooth', block: 'start' }) } catch {}
      }
    })
  } catch {}
}

async function showSummary() {
  const summaryEl = document.getElementById('summary')
  const res = await fetch(`${API}/results/latest-summary`)
  const data = await res.json()
  summaryEl.textContent = JSON.stringify(data, null, 2)
  try {
    const toMetrics = (s) => {
      const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}
      return {
        profit_pct: c.profit_total_pct ?? s.profit_total_pct,
        trades: c.trades ?? s.trades,
        winrate: c.winrate ?? s.winrate,
        max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs,
      }
    }
    const m = toMetrics(data)
    const cards = document.getElementById('cards')
    if (cards) {
      let train = null
      try {
        const tr = await fetch(`${API}/results/latest-training`)
        const tj = await tr.json()
        if (!tj.status || tj.status !== 'error') train = tj
      } catch {}
      const trainRMSE = train && train.metrics ? (train.metrics.rmse_valid ?? train.metrics.rmse_train) : null
      const trainModel = train ? train.model : null
      const pf = Number(m.profit_pct || 0)
      const pfColor = (isFinite(pf) && pf >= 0) ? '#166534' : '#b91c1c'
      cards.innerHTML = `
        <div class="card"><div class="k">收益%</div><div class="v" style="color:${pfColor}">${m.profit_pct ?? '--'}</div></div>
        <div class="card"><div class="k">交易数</div><div class="v">${m.trades ?? '--'}</div></div>
        <div class="card"><div class="k">胜率</div><div class="v">${m.winrate ?? '--'}</div></div>
        <div class="card"><div class="k">最大回撤</div><div class="v">${m.max_dd ?? '--'}</div></div>
        <div class="card"><div class="k">最近训练</div><div class="v">${trainModel ?? '--'}</div></div>
        <div class="card"><div class="k">验证RMSE</div><div class="v">${trainRMSE ?? '--'}</div></div>
      `
    }
  } catch {}
  try {
    if (Array.isArray(data.trades)) {
      const sorted = data.trades.slice().sort((a,b) => (a.open_timestamp||0) - (b.open_timestamp||0))
      let cum = 0
      const ys = []
      const xs = []
      for (const t of sorted) {
        cum += Number(t.profit_abs || 0)
        ys.push(cum)
        xs.push(new Date(t.open_timestamp||0).toISOString().slice(0,10))
      }
      if (window.echarts) {
        const _el = document.getElementById('chart')
        const chart = (window.echarts.getInstanceByDom && _el) ? (echarts.getInstanceByDom(_el) || echarts.init(_el)) : echarts.init(_el)
        chart.setOption({ grid:{left:40,right:16,top:10,bottom:30}, xAxis:{ type:'category', data:xs, axisLabel:{ rotate:45 } }, yAxis:{ type:'value', scale:true }, tooltip:{ trigger:'axis' }, series:[{ name:'累计收益(USDT)', type:'line', data:ys }] })
      }
    }
  } catch (e) { console.warn('chart error', e) }
}

function Icon({ type }) {
  const map = { data: 'ri-database-2-line', expr: 'ri-function-line', bt: 'ri-line-chart-line', fb: 'ri-feedback-line', ho: 'ri-sliders-2-line', mv: 'ri-shuffle-line' }
  const cls = map[type] || 'ri-shape-2-line'
  return h('i', { className: cls })
}

function CustomNode({ id, data }) {
  const typeKey = data?.typeKey || 'node'
  const info = data?.cfg || {}
  const rows = []
  if (typeKey === 'data') rows.push(['tf', info.timeframe||'--'], ['pairs', (info.pairs||'--').split(' ').length])
  if (typeKey === 'expr') rows.push(['llm', info.llm_model||'--'], ['count', info.llm_count||'--'])
  if (typeKey === 'bt') rows.push(['range', info.timerange||'--'])
  const locked = !!data?.locked
  return h('div', { className: 'am-node' + (locked ? ' locked' : '') }, [
    h('div', { className: 'header' }, [ h(Icon, { type: typeKey }), h('div', { className: 'title' }, [ data?.label||id, locked ? h('i', { className: 'ri-lock-2-line', title: '已锁定' }) : null ]), h('div', { className: 'badge' }, typeKey) ]),
    h('div', { className: 'body' }, rows.map(([k,v]) => h('div', { className: 'kv' }, [ h('span', null, k), h('b', null, String(v)) ]))),
    h('div', { className: 'footer' }, [
      h('div', null, (info.output||info.results_dir||'')),
      h('div', { className: 'actions' }, [
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { window.__runNode && window.__runNode(id) } catch(e){} } }, '运行'),
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { window.__configureNode && window.__configureNode(id) } catch(e){} } }, '配置'),
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { showSummary() } catch(e){} } }, '摘要'),
      ])
    ]),
    h(Handle, { type: 'target', position: Position.Left }),
    h(Handle, { type: 'source', position: Position.Right }),
  ])
}

function App() {
  const [nodes, setNodes] = useState([
    { id: 'n1', type: 'amNode', position: { x: 50, y: 120 }, data: { label: 'Data', typeKey: 'data', cfg: { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' } } },
    { id: 'n2', type: 'amNode', position: { x: 320, y: 120 }, data: { label: 'Expression(LLM)', typeKey: 'expr', cfg: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' } } },
    { id: 'n3', type: 'amNode', position: { x: 610, y: 120 }, data: { label: 'Backtest', typeKey: 'bt', cfg: { timerange: '20210101-20211231' } } },
  ])
  const [edges, setEdges] = useState([
    { id: 'e1', source: 'n1', target: 'n2', type: 'smoothstep', animated: true },
    { id: 'e2', source: 'n2', target: 'n3', type: 'smoothstep', animated: true },
  ])
  const [snap, setSnap] = useState(true)
  const [grid, setGrid] = useState([16,16])
  const nodeTypes = (window.React && React.useMemo) ? React.useMemo(() => ({ amNode: CustomNode }), []) : { amNode: CustomNode }
  const defaultEdgeOptions = { animated: true, type: 'smoothstep', style: { stroke: '#8694ff', strokeWidth: 1.6 }, markerEnd: MarkerType.ArrowClosed ? { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#8694ff' } : undefined }

  // Connect toolbar
  useEffect(() => {
    const apiEl = document.getElementById('apiUrl'); if (apiEl) apiEl.value = API
    const applyBtn = document.getElementById('applyApi'); if (applyBtn) applyBtn.onclick = async () => {
      const val = (document.getElementById('apiUrl')?.value || '').trim()
      if (!val) return
      setApiUrl(val)
      try {
        const r = await fetch(`${API}/health`); const j = await r.json(); alert(`API 探测成功: ${API} /health: ${JSON.stringify(j)}`)
      } catch (e) { alert(`API 探测失败: ${API}, 调用 /health 出错: ${e}`) }
    }
    const themeBtn = document.getElementById('btnTheme'); if (themeBtn) themeBtn.onclick = () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'light'
      const next = cur === 'dark' ? 'light' : 'dark'
      document.documentElement.setAttribute('data-theme', next)
    }
    const brandSel = document.getElementById('brandSelect'); if (brandSel) brandSel.onchange = () => { document.documentElement.setAttribute('data-brand', brandSel.value || 'ocean') }
    const btnFit = document.getElementById('btnFit'); if (btnFit) btnFit.onclick = () => { try { window.__rf && window.__rf.fitView() } catch {} }
    const btnClear = document.getElementById('btnClear'); if (btnClear) btnClear.onclick = () => { setNodes([]); setEdges([]) }
    const layoutBtn = document.getElementById('btnLayout'); if (layoutBtn) layoutBtn.onclick = () => {
      // 简易水平布局
      const sorted = (nodes||[]).slice().sort((a,b)=> (a.position?.x||0)-(b.position?.x||0))
      const y = 120
      setNodes(sorted.map((n,i) => ({ ...n, position: { x: 80 + i*260, y } })))
    }
    const snapBtn = document.getElementById('btnSnap'); if (snapBtn) snapBtn.onclick = () => setSnap(s => !s)
    const gridInput = document.getElementById('gridSize'); if (gridInput) gridInput.onchange = () => { const v = Math.max(2, parseInt(gridInput.value||'16',10)||16); setGrid([v,v]) }
    const snapStrength = document.getElementById('snapStrength'); if (snapStrength) snapStrength.onchange = () => {
      const mode = snapStrength.value || 'medium'
      const base = Math.max(2, parseInt((document.getElementById('gridSize')?.value)||'16',10)||16)
      const v = mode === 'strong' ? Math.max(2, Math.round(base/2)) : (mode==='weak'? base*2 : base)
      setGrid([v,v])
    }
    const ax = document.getElementById('btnAlignX'); if (ax) ax.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 2) return alert('至少选中2个节点')
      const refY = sels[0].position?.y || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: n.position.x, y: refY } }) : n))
    }
    const ay = document.getElementById('btnAlignY'); if (ay) ay.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 2) return alert('至少选中2个节点')
      const refX = sels[0].position?.x || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: refX, y: n.position.y } }) : n))
    }
    const distX = document.getElementById('btnDistX'); if (distX) distX.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 3) { alert('至少选中3个节点'); return }
      const sorted = sels.slice().sort((a,b)=> (a.position?.x||0)-(b.position?.x||0))
      const minx = sorted[0].position.x, maxx = sorted[sorted.length-1].position.x
      const step = (maxx - minx) / (sorted.length - 1)
      const map = new Map(sorted.map((n,i)=> [n.id, minx + i*step]))
      setNodes(nds => nds.map(n => map.has(n.id) ? ({ ...n, position: { x: map.get(n.id), y: n.position.y } }) : n))
    }
    const distY = document.getElementById('btnDistY'); if (distY) distY.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 3) { alert('至少选中3个节点'); return }
      const sorted = sels.slice().sort((a,b)=> (a.position?.y||0)-(b.position?.y||0))
      const miny = sorted[0].position.y, maxy = sorted[sorted.length-1].position.y
      const step = (maxy - miny) / (sorted.length - 1)
      const map = new Map(sorted.map((n,i)=> [n.id, miny + i*step]))
      setNodes(nds => nds.map(n => map.has(n.id) ? ({ ...n, position: { x: n.position.x, y: map.get(n.id) } }) : n))
    }
    const trainBt = document.getElementById('btnTrainBt'); if (trainBt) trainBt.onclick = () => {
      try {
        const flowCfg = document.getElementById('flowCfg'); if (flowCfg) flowCfg.value = GOLDEN.flow_cfg
        const flowSteps = document.getElementById('flowSteps'); if (flowSteps) flowSteps.value = GOLDEN.flow_steps
        const cfg = document.getElementById('cfg'); if (cfg) cfg.value = GOLDEN.freqtrade_cfg
        const ff = document.getElementById('featureFile'); if (ff) ff.value = GOLDEN.feature_file
        const tf = document.getElementById('timeframe'); if (tf) tf.value = GOLDEN.timeframe
        const tr = document.getElementById('timerange'); if (tr) tr.value = GOLDEN.timerange
        const btn = document.getElementById('btnRunAgentFlow'); if (btn) btn.click()
      } catch (e) { alert('启动黄金路径失败: ' + e) }
    }
  }, [nodes])

  // Node editing stubs
  useEffect(() => { window.__runNode = async (id) => { /* could implement run by nodeId */ } }, [])

  // Wire core buttons
  useEffect(() => {
    const logsEl = document.getElementById('logs')
    const be = document.getElementById('btnExpr')
    const bb = document.getElementById('btnBacktest')
    const bs = document.getElementById('btnSummary')
    if (be) be.onclick = async () => {
      logsEl.textContent = ''
      const body = {
        config: document.getElementById('cfg').value,
        feature_file: document.getElementById('featureFile').value,
        output: 'user_data/freqai_expressions.json',
        timeframe: document.getElementById('timeframe').value,
        llm_model: document.getElementById('llmModel').value,
        llm_count: parseInt(document.getElementById('llmCount').value || '3', 10),
        llm_loops: 1,
        llm_timeout: 60,
        feedback_top: 0,
      }
      window.__retryLast = async () => {
        const r = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const j = await r.json(); if (j.job_id) { setStatus('表达式', j.job_id, true); await pollLogs(j.job_id); setStatus('表达式', j.job_id, false) } else { alert(JSON.stringify(j)) }
      }
      await window.__retryLast()
    }
    if (bb) bb.onclick = async () => {
      logsEl.textContent = ''
      const body = {
        config: document.getElementById('cfg').value,
        strategy: 'ExpressionLongStrategy',
        strategy_path: 'user_data/strategies',
        timerange: document.getElementById('timerange').value,
        freqaimodel: 'LightGBMRegressor',
        export: true,
        export_filename: 'user_data/backtest_results/latest_trades_multi',
      }
      window.__retryLast = async () => {
        const r = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const j = await r.json(); if (j.job_id) { setStatus('回测', j.job_id, true); await pollLogs(j.job_id); setStatus('回测', j.job_id, false) } else { alert(JSON.stringify(j)) }
      }
      await window.__retryLast()
    }
    if (bs) bs.onclick = () => showSummary()
    const cancel = document.getElementById('btnCancelJob'); if (cancel) cancel.onclick = async () => { try { /* requires job id track; skipped */ alert('请在任务执行时通过接口取消') } catch {} }
  }, [])

  // Settings load/save/apply
  useEffect(() => {
    const loadBtn = document.getElementById('btnLoadSettings')
    const saveBtn = document.getElementById('btnSaveSettings')
    const applyBtn = document.getElementById('btnApplySettings')
    if (loadBtn) loadBtn.onclick = async () => {
      try {
        const r = await fetch(`${API}/settings`)
        const s = await r.json()
        document.getElementById('llmBaseUrl').value = s.llm_base_url || ''
        document.getElementById('llmModelSet').value = s.llm_model || ''
        document.getElementById('defaultTf').value = s.default_timeframe || ''
      } catch(e) { alert('加载失败: '+e) }
    }
    if (saveBtn) saveBtn.onclick = async () => {
      const base = (document.getElementById('llmBaseUrl')?.value || '').trim()
      const model = (document.getElementById('llmModelSet')?.value || '').trim()
      const tf = (document.getElementById('defaultTf')?.value || '').trim()
      const body = {}; if (base) body.llm_base_url = base; if (model) body.llm_model = model; if (tf) body.default_timeframe = tf
      const r = await fetch(`${API}/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json(); if (j.status === 'ok') alert('已保存设置') else alert('保存失败: '+JSON.stringify(j))
    }
    if (applyBtn) applyBtn.onclick = () => {
      const model = (document.getElementById('llmModelSet')?.value || '').trim()
      const tf = (document.getElementById('defaultTf')?.value || '').trim()
      const mainLlm = document.getElementById('llmModel'); if (mainLlm && model) mainLlm.value = model
      const mainTf = document.getElementById('timeframe'); if (mainTf && tf) mainTf.value = tf
      alert('已应用到表单')
    }
  }, [])

  // Feature TopN
  useEffect(() => {
    const btn = document.getElementById('btnFeatTop'); if (btn) btn.onclick = async () => {
      const file = document.getElementById('featureFile').value || 'user_data/freqai_features.json'
      const res = await fetch(`${API}/features/top?file=${encodeURIComponent(file)}&limit=10`)
      const data = await res.json()
      const el = document.getElementById('featTop'); if (el) el.textContent = JSON.stringify(data, null, 2)
    }
  }, [])

  // Agent Flow
  useEffect(() => {
    const btn = document.getElementById('btnRunAgentFlow')
    if (!btn) return
    btn.onclick = async () => {
      const logsEl = document.getElementById('logs'); if (logsEl) logsEl.textContent = ''
      const fa = document.getElementById('flowArtifacts'); if (fa) fa.innerHTML = ''
      const cfg = (document.getElementById('flowCfg')?.value || 'configs/agent_flow_multi.json')
      const stepsRaw = (document.getElementById('flowSteps')?.value || '').trim()
      const body = { config: cfg }
      if (stepsRaw) body.steps = stepsRaw.split(/\s+/)
      const fs = document.getElementById('flowStatus')
      if (fs) {
        const steps = (body.steps && Array.isArray(body.steps) && body.steps.length) ? body.steps : ['feature','expression','ml','rl','backtest']
        fs.innerHTML = '<div style="margin-bottom:6px">步骤:</div>' + steps.map(s => `<span class="tag tag-running" data-step="${s}"><i class="ri-time-line"></i> ${s}</span>`).join(' ')
        const controls = document.createElement('div'); controls.className = 'buttons'
        controls.innerHTML = `
          <button class="btn" id="qFeature"><i class="ri-database-2-line"></i> Feature</button>
          <button class="btn" id="qExpr"><i class="ri-function-line"></i> Expr</button>
          <button class="btn" id="qML"><i class="ri-cpu-line"></i> ML</button>
          <button class="btn" id="qRL"><i class="ri-brain-line"></i> RL</button>
          <button class="btn" id="qBT"><i class="ri-line-chart-line"></i> BT</button>
        `
        fs.appendChild(controls)
        const bindQuick = (id, fn) => { const b = document.getElementById(id); if (b) b.onclick = () => runWithLoading(b, fn) }
        bindQuick('qFeature', async () => {
          const body = { config: document.getElementById('cfg').value, output: document.getElementById('featureFile').value, timeframe: document.getElementById('timeframe').value, pairs: 'BTC/USDT ETH/USDT' }
          const r = await fetch(`${API}/run/feature`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); const j = await r.json(); if (j.job_id) await pollLogs(j.job_id)
        })
        bindQuick('qExpr', async () => {
          const body = { config: document.getElementById('cfg').value, feature_file: document.getElementById('featureFile').value, output:'user_data/freqai_expressions.json', timeframe: document.getElementById('timeframe').value, llm_model: document.getElementById('llmModel').value, llm_count: parseInt(document.getElementById('llmCount').value||'3',10), llm_loops:1, llm_timeout:60, feedback_top:0 }
          const r = await fetch(`${API}/run/expression`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); const j = await r.json(); if (j.job_id) await pollLogs(j.job_id)
        })
        bindQuick('qML', async () => {
          const ff = document.getElementById('featureFile').value; const tf = document.getElementById('timeframe').value
          const body = { config_obj: { data: { feature_file: ff, data_dir: 'user_data/data', exchange: 'binanceus', timeframe: tf, pairs: ['BTC/USDT'] }, model: { name: 'lightgbm', params: { objective: 'regression', metric: 'rmse', num_boost_round: 120 } }, training: { validation_ratio: 0.2 }, output: { model_dir: 'artifacts/models/flow_ml' } } }
          const r = await fetch(`${API}/run/train`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); const j = await r.json(); if (j.job_id) await pollLogs(j.job_id)
        })
        bindQuick('qRL', async () => {
          const body = { config: 'configs/train_ppo.json' }
          const r = await fetch(`${API}/run/rl_train`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); const j = await r.json(); if (j.job_id) await pollLogs(j.job_id)
        })
        bindQuick('qBT', async () => {
          const body = { config: document.getElementById('cfg').value, strategy:'ExpressionLongStrategy', strategy_path:'user_data/strategies', timerange: document.getElementById('timerange').value, freqaimodel:'LightGBMRegressor', export:true, export_filename:'user_data/backtest_results/latest_trades_multi' }
          const r = await fetch(`${API}/run/backtest`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); const j = await r.json(); if (j.job_id) await pollLogs(j.job_id)
        })
      }
      const r = await fetch(`${API}/flow/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json(); if (j.job_id) {
        setStatus('Flow', j.job_id, true)
        const stepsCsv = (body.steps && Array.isArray(body.steps) && body.steps.length) ? body.steps.join(',') : 'feature,expression,ml,rl,backtest'
        const applyProgress = (pj) => {
          const items = (pj && pj.steps) || []
          for (const it of items) {
            const el = document.querySelector(`.tag[data-step="${it.name}"]`)
            if (!el) continue
            el.classList.remove('tag-ok','tag-failed','tag-running')
            if (it.status === 'ok') el.classList.add('tag-ok')
            else if (it.status === 'failed') el.classList.add('tag-failed')
            else el.classList.add('tag-running')
            // show phase + percent
            const pct = (typeof it.percent === 'number') ? ` ${it.percent}%` : ''
            const phase = it.phase ? ` (${it.phase}${pct})` : (pct ? ` (${pct})` : '')
            el.innerHTML = `<i class="ri-time-line"></i> ${it.name}${phase}`
          }
          // append log delta if present
          try {
            const delta = pj && pj.delta
            if (delta && Array.isArray(delta) && delta.length) {
              const logsEl = document.getElementById('logs')
              logsEl.textContent += delta.join('\n') + '\n'
              try { logsEl.scrollTop = logsEl.scrollHeight } catch {}
            }
          } catch {}
        }
        let es = null, timer = null
        try {
          es = new EventSource(`${API}/flow/stream/${j.job_id}?steps=${encodeURIComponent(stepsCsv)}`)
          es.onmessage = (ev) => {
            try { const pj = JSON.parse(ev.data); applyProgress(pj); if (pj && pj.running === false) { try { es.close() } catch {} } } catch {}
          }
          es.onerror = () => { try { es.close() } catch {} }
        } catch (e) {
          es = null
        }
        if (!es) {
          timer = setInterval(async () => {
            try { const pr = await fetch(`${API}/flow/progress/${j.job_id}?steps=${encodeURIComponent(stepsCsv)}`); const pj = await pr.json(); applyProgress(pj); if (!pj.running) { clearInterval(timer); timer = null } } catch {}
          }, 1000)
        }
        await pollLogs(j.job_id)
        if (es) { try { es.close() } catch {} }
        if (timer) { clearInterval(timer); timer = null }
        setStatus('Flow', j.job_id, false)
        try { await renderFlowArtifactsCheck() } catch (e) { console.warn('flowArtifacts error', e) }
        try { await showSummary() } catch {}
        try { await loadRunHistory() } catch {}
      } else { alert(JSON.stringify(j)) }
    }
  }, [])

  // Run History (Flow)
  useEffect(() => {
    const btn = document.getElementById('btnLoadRunHistory')
    if (btn) btn.onclick = () => runWithLoading(btn, loadRunHistory)
    try { loadRunHistory() } catch {}
  }, [])

  // Results list / compare / gallery / aggregate
  useEffect(() => {
    const listBtn = document.getElementById('btnList'); if (listBtn) listBtn.onclick = async () => {
      try { const res = await fetch(`${API}/results/list`); const data = await res.json(); const el = document.getElementById('comparePanel'); if (el) el.textContent = JSON.stringify(data, null, 2) } catch(e) { alert('加载失败:'+e) }
    }
    const cmpBtn = document.getElementById('btnCompare'); if (cmpBtn) cmpBtn.onclick = async () => {
      const a = (document.getElementById('resA').value||'').trim(); const b = (document.getElementById('resB').value||'').trim(); if (!a || !b) return alert('请先输入结果 A 与 B')
      const rd = 'user_data/backtest_results'
      const [ra, rb] = await Promise.all([
        fetch(`${API}/results/summary?name=${encodeURIComponent(a)}`),
        fetch(`${API}/results/summary?name=${encodeURIComponent(b)}`),
      ])
      const [sa, sb] = [await ra.json(), await rb.json()]
      const toM = (s) => { const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}; return { profit_pct: c.profit_total_pct ?? s.profit_total_pct, profit_abs: c.profit_total_abs ?? s.profit_total_abs, trades: c.trades ?? s.trades, winrate: c.winrate ?? s.winrate, max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs } }
      const ma = toM(sa), mb = toM(sb)
      const el = document.getElementById('comparePanel')
      if (el) el.innerHTML = `
        <table style="width:100%; font-size:12px; border-collapse:collapse">
          <tr><th>指标</th><th>A (${a})</th><th>B (${b})</th></tr>
          <tr><td>收益%</td><td>${ma.profit_pct ?? '--'}</td><td>${mb.profit_pct ?? '--'}</td></tr>
          <tr><td>收益(USDT)</td><td>${ma.profit_abs ?? '--'}</td><td>${mb.profit_abs ?? '--'}</td></tr>
          <tr><td>交易数</td><td>${ma.trades ?? '--'}</td><td>${mb.trades ?? '--'}</td></tr>
          <tr><td>胜率</td><td>${ma.winrate ?? '--'}</td><td>${mb.winrate ?? '--'}</td></tr>
          <tr><td>最大回撤(USDT)</td><td>${ma.max_dd ?? '--'}</td><td>${mb.max_dd ?? '--'}</td></tr>
        </table>`
    }
    const gbtn = document.getElementById('btnGallery'); if (gbtn) gbtn.onclick = async () => {
      await runWithLoading(gbtn, async () => {
        const res = await fetch(`${API}/results/gallery?limit=24`)
        const data = await res.json()
        const gp = document.getElementById('galleryPanel')
        if (gp) {
          const items = (data.items||[])
          const html = items.map(x => {
            const pct = Number(x.profit_total_pct||0)
            const width = Math.max(0, Math.min(100, Math.round(Math.abs(pct))))
            const color = pct >= 0 ? 'linear-gradient(90deg, #86efac, #22c55e)' : 'linear-gradient(90deg, #fecaca, #ef4444)'
            return `
              <div class="mini-card">
                <div class="title">${x.name}</div>
                <div class="row"><span>收益%</span><b>${x.profit_total_pct ?? '--'}</b></div>
                <div class="row"><span>交易数</span><b>${x.trades ?? '--'}</b></div>
                <div class="row"><span>最大回撤</span><b>${x.max_drawdown_abs ?? '--'}</b></div>
                <div class="mini-bar"><i style="width:${width}%; background:${color}"></i></div>
              </div>`
          }).join('') || '<em>暂无</em>'
          gp.innerHTML = `<div class="card-grid">${html}</div>`
        }
      })
    }
    const abtn = document.getElementById('btnAgg'); if (abtn) abtn.onclick = async () => {
      await runWithLoading(abtn, async () => {
        const names = (document.getElementById('aggNames').value||'').trim()
        if (!names) return alert('请输入若干结果名称, 例如 a.zip,b.zip')
        const res = await fetch(`${API}/results/aggregate?names=${encodeURIComponent(names)}`)
        const data = await res.json()
        const ap = document.getElementById('aggPanel')
        if (ap) {
          ap.innerHTML = '<div id="aggChart" style="height:220px"></div><div id="aggInfo" class="panel" style="margin-top:6px"></div>'
          if (window.echarts) {
            const chart = echarts.init(document.getElementById('aggChart'))
            const items = data.items || []
            const x = items.map(i => i.name)
            const y = items.map(i => i.profit_total_pct || 0)
            chart.setOption({ grid:{left:40,right:16,top:10,bottom:60}, xAxis:{ type:'category', data:x, axisLabel:{ rotate:60 } }, yAxis:{ type:'value', scale:true }, tooltip:{ trigger:'axis' }, series:[{ type:'bar', data:y }] })
          }
          const info = document.getElementById('aggInfo')
          if (info) info.innerHTML = `均值收益: ${data.mean_profit ?? '--'} | 收益波动: ${data.std_profit ?? '--'} | 鲁棒分: ${data.robust_score ?? '--'}`
        }
      })
    }
  }, [])

  const onConnect = useCallback((params) => setEdges((eds) => addEdgeLib(params, eds)), [])
  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), [])
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), [])
  const onNodeClick = useCallback((_, n) => { try { const el = document.getElementById('nodeForm'); if (el) el.textContent = JSON.stringify(n.data?.cfg||{}, null, 2) } catch {} }, [])
  const onDrop = useCallback((ev) => {
    ev.preventDefault()
    const typeKey = ev.dataTransfer.getData('application/node-type')
    if (!typeKey) return
    const id = 'n' + Math.random().toString(16).slice(2,8)
    const cfg = typeKey === 'expr' ? { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' }
      : (typeKey === 'bt' ? { timerange: '20210101-20211231' }
      : (typeKey === 'data' ? { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' } : {}))
    const position = { x: ev.clientX - 400, y: ev.clientY - 120 }
    const node = { id, type: 'amNode', position, data: { label: typeKey.toUpperCase(), typeKey, cfg } }
    setNodes(nds => nds.concat(node))
  }, [])
  const onDragOver = useCallback((ev) => { ev.preventDefault(); ev.dataTransfer.dropEffect = 'move' }, [])

  return h(RF, { nodes, edges, nodeTypes, defaultEdgeOptions, fitView: true, onConnect, onNodesChange, onEdgesChange, onNodeClick, onDrop, onDragOver, onInit: (inst) => { window.__rf = inst }, snapToGrid: !!snap, snapGrid: grid }, [
    h(Background, { variant: 'dots', gap: 16, size: 1, key: 'bg' }),
    h(Controls, { key: 'ctl' }),
    h(MiniMap, { key: 'mm' }),
  ])
}

function boot() {
  const hasReact = !!(window.React && window.ReactDOM)
  const hasRF = !!window.ReactFlow
  if (!hasReact || !hasRF) {
    console.warn('[fallback] 缺少依赖: React=', hasReact, ' ReactFlow=', hasRF, '，启用降级绑定')
    try { wireFallback() } catch (e2) { console.error('fallback 失败', e2) }
    return
  }
  try { createRoot(document.getElementById('root')).render(h(App)) } catch (e) {
    console.warn('[fallback] 渲染失败，启用降级绑定', e)
    try { wireFallback() } catch (e2) { console.error('fallback 失败', e2) }
  }
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  setTimeout(boot, 0)
} else {
  document.addEventListener('DOMContentLoaded', boot)
}

// Drag helpers for palette
document.addEventListener('DOMContentLoaded', () => {
  const paletteItems = Array.from(document.querySelectorAll('.palette-item'))
  const onDragStart = (e) => { const typeKey = e.target?.getAttribute?.('data-nodetype'); if (!typeKey) return; e.dataTransfer.setData('application/node-type', typeKey); e.dataTransfer.effectAllowed = 'move' }
  paletteItems.forEach(el => el.addEventListener('dragstart', onDragStart))
})
