/* ============================================
   Agent Market Pro — Main Application
   ============================================ */

const { createElement: h, useState, useEffect, useCallback } = window.React || {}
const { createRoot } = window.ReactDOM || {}

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
const addEdgeLib = RFLib.addEdge || ((params, eds) => eds.concat({ id: (params.id || ('e' + Math.random().toString(16).slice(2, 8))), ...params }))

// ============================================
// Global State
// ============================================
let API = (typeof location !== 'undefined' && /^https?:/i.test(location.origin || ''))
  ? location.origin.replace(/\/$/, '')
  : 'http://127.0.0.1:8000'

const GOLDEN = {
  flow_cfg: 'configs/agent_flow_kucoin_cpu_nollm.json',
  flow_steps: 'feature expression ml backtest',
  freqtrade_cfg: 'user_data/config_freqai_kucoin.json',
  feature_file: 'user_data/freqai_features_real.json',
  timeframe: '1h',
  timerange: '20251126-20260125',
}

function setApiUrl(url) {
  API = (url || API).replace(/\/$/, '')
  const el = document.getElementById('apiUrl')
  if (el) el.value = API
}

// ============================================
// Utilities
// ============================================
function _escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch] || ch))
}

function toast(msg, type = 'info') {
  const container = document.getElementById('toastContainer')
  if (!container) return
  const el = document.createElement('div')
  el.className = `toast ${type}`
  const icon = type === 'success' ? '&#10003;' : type === 'error' ? '&#10007;' : '&#8505;'
  el.innerHTML = `<span>${icon}</span><span>${_escapeHtml(msg)}</span>`
  container.appendChild(el)
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300) }, 4000)
}

function setGlobalStatus(text, state = 'ready') {
  const dot = document.getElementById('statusDot')
  const textEl = document.getElementById('statusText')
  if (!dot || !textEl) return
  dot.className = 'dot'
  if (state === 'running') dot.classList.add('loading')
  else if (state === 'error') dot.classList.add('offline')
  textEl.textContent = text
}

async function pollLogs(jobId, logElId = 'logs') {
  const logsEl = document.getElementById(logElId)
  if (!logsEl) return
  let offset = 0
  let last = null
  while (true) {
    try {
      const res = await fetch(`${API}/jobs/${jobId}/logs?offset=${offset}`)
      const data = await res.json()
      last = data
      const chunk = (data.logs || []).join('\n')
      if (chunk) {
        logsEl.textContent += chunk + '\n'
        try { logsEl.scrollTop = logsEl.scrollHeight } catch { }
      }
      offset = data.next || offset
      if (!data.running) break
      await new Promise(r => setTimeout(r, 800))
    } catch { break }
  }
  if (last && last.code && last.code !== 'OK') {
    setGlobalStatus(`任务失败 (code=${last.code})`, 'error')
  } else if (last && last.code === 'OK') {
    setGlobalStatus('就绪')
  }
  return last
}

function setLoading(btn, on) {
  if (!btn) return
  if (on) { btn.classList.add('is-loading'); btn.disabled = true }
  else { btn.classList.remove('is-loading'); btn.disabled = false }
}

async function runWithLoading(btn, fn) {
  setLoading(btn, true)
  try { await fn() } finally { setLoading(btn, false) }
}

// ============================================
// Tab Navigation
// ============================================
function initTabNav() {
  const tabs = document.querySelectorAll('.nav-tab')
  const panes = document.querySelectorAll('.tab-pane')

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab
      tabs.forEach(t => t.classList.remove('active'))
      panes.forEach(p => p.classList.remove('active'))
      tab.classList.add('active')
      const pane = document.getElementById('tab-' + target)
      if (pane) pane.classList.add('active')
      // Resize charts when switching to results tab
      if (target === 'results' && window.echarts) {
        setTimeout(() => {
          document.querySelectorAll('#tab-results [_echarts_instance_]').forEach(el => {
            const inst = echarts.getInstanceByDom(el)
            if (inst) inst.resize()
          })
        }, 50)
      }
    })
  })
}

// ============================================
// Settings Navigation
// ============================================
function initSettingsNav() {
  const items = document.querySelectorAll('.settings-nav-item')
  const sections = document.querySelectorAll('.settings-section')

  items.forEach(item => {
    item.addEventListener('click', () => {
      const target = item.dataset.section
      items.forEach(i => i.classList.remove('active'))
      sections.forEach(s => s.classList.remove('active'))
      item.classList.add('active')
      const section = document.getElementById('section-' + target)
      if (section) section.classList.add('active')
    })
  })
}

// ============================================
// Connection Test
// ============================================
function initConnection() {
  const applyBtn = document.getElementById('applyApi')
  if (applyBtn) applyBtn.addEventListener('click', async () => {
    const val = (document.getElementById('apiUrl')?.value || '').trim()
    if (!val) return
    setApiUrl(val)
    const result = document.getElementById('connectionResult')
    try {
      const r = await fetch(`${API}/health`)
      const j = await r.json()
      if (result) result.innerHTML = `<span class="text-green">&#10003; 连接成功: ${JSON.stringify(j)}</span>`
      setGlobalStatus('就绪')
      toast('API 连接成功', 'success')
    } catch (e) {
      if (result) result.innerHTML = `<span class="text-red">&#10007; 连接失败: ${e}</span>`
      setGlobalStatus('连接失败', 'error')
      toast('API 连接失败', 'error')
    }
  })
}

// ============================================
// Settings (LLM, load/save)
// ============================================
function initSettings() {
  const loadBtn = document.getElementById('btnLoadSettings')
  const saveBtn = document.getElementById('btnSaveSettings')
  const applyBtn = document.getElementById('btnApplySettings')

  if (loadBtn) loadBtn.addEventListener('click', async () => {
    try {
      const r = await fetch(`${API}/settings`)
      const s = await r.json()
      document.getElementById('llmBaseUrl').value = s.llm_base_url || ''
      document.getElementById('llmModelSet').value = s.llm_model || ''
      document.getElementById('defaultTf').value = s.default_timeframe || ''
      toast('设置已加载', 'success')
    } catch (e) { toast('加载失败: ' + e, 'error') }
  })

  if (saveBtn) saveBtn.addEventListener('click', async () => {
    const base = (document.getElementById('llmBaseUrl')?.value || '').trim()
    const model = (document.getElementById('llmModelSet')?.value || '').trim()
    const tf = (document.getElementById('defaultTf')?.value || '').trim()
    const body = {}
    if (base) body.llm_base_url = base
    if (model) body.llm_model = model
    if (tf) body.default_timeframe = tf
    try {
      const r = await fetch(`${API}/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json()
      if (j.status === 'ok') toast('设置已保存', 'success')
      else toast('保存失败', 'error')
    } catch (e) { toast('保存失败: ' + e, 'error') }
  })

  if (applyBtn) applyBtn.addEventListener('click', () => {
    const model = (document.getElementById('llmModelSet')?.value || '').trim()
    const tf = (document.getElementById('defaultTf')?.value || '').trim()
    const mainLlm = document.getElementById('llmModel')
    const mainTf = document.getElementById('timeframe')
    if (mainLlm && model) mainLlm.value = model
    if (mainTf && tf) mainTf.value = tf
    toast('已应用到参数表单', 'success')
  })
}

// ============================================
// Expression & Backtest
// ============================================
function initExprBacktest() {
  const be = document.getElementById('btnExpr')
  const bb = document.getElementById('btnBacktest')

  if (be) be.addEventListener('click', async () => {
    const logsEl = document.getElementById('logs')
    if (logsEl) logsEl.textContent = ''
    const body = {
      config: document.getElementById('cfg').value,
      feature_file: document.getElementById('featureFile').value,
      output: 'user_data/freqai_expressions.json',
      timeframe: document.getElementById('timeframe').value,
      llm_model: document.getElementById('llmModel').value,
      llm_count: parseInt(document.getElementById('llmCount').value || '3', 10),
      llm_loops: 1, llm_timeout: 60, feedback_top: 0,
    }
    try {
      setGlobalStatus('表达式生成中...', 'running')
      const r = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json()
      if (j.job_id) {
        toast('表达式任务已启动', 'success')
        await pollLogs(j.job_id)
        setGlobalStatus('就绪')
      } else toast(JSON.stringify(j), 'error')
    } catch (e) { toast('启动失败: ' + e, 'error'); setGlobalStatus('就绪') }
  })

  if (bb) bb.addEventListener('click', async () => {
    const logsEl = document.getElementById('logs')
    if (logsEl) logsEl.textContent = ''
    const body = {
      config: document.getElementById('cfg').value,
      strategy: 'ExpressionLongStrategy',
      strategy_path: 'user_data/strategies',
      timerange: document.getElementById('timerange').value,
      freqaimodel: 'LightGBMRegressor',
      export: true,
      export_filename: 'user_data/backtest_results/latest_trades_multi',
    }
    try {
      setGlobalStatus('回测运行中...', 'running')
      const r = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json()
      if (j.job_id) {
        toast('回测任务已启动', 'success')
        await pollLogs(j.job_id)
        setGlobalStatus('就绪')
      } else toast(JSON.stringify(j), 'error')
    } catch (e) { toast('启动失败: ' + e, 'error'); setGlobalStatus('就绪') }
  })
}

// ============================================
// Results Tab
// ============================================
function _getOrInitChart(elId) {
  if (!window.echarts) return null
  const el = document.getElementById(elId)
  if (!el) return null
  return echarts.getInstanceByDom(el) || echarts.init(el, 'dark')
}

async function showSummary() {
  try {
    const res = await fetch(`${API}/results/latest-summary`)
    const data = await res.json()
    const summaryEl = document.getElementById('summary')
    if (summaryEl) summaryEl.textContent = JSON.stringify(data, null, 2)

    const toMetrics = (s) => {
      const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}
      return {
        profit_pct: c.profit_total_pct ?? s.profit_total_pct,
        profit_abs: c.profit_total_abs ?? s.profit_total_abs,
        trades: c.trades ?? s.trades,
        winrate: c.winrate ?? s.winrate,
        max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs,
        sharpe: c.sharpe ?? s.sharpe,
      }
    }
    const m = toMetrics(data)

    // Update metric cards
    const setMetric = (id, val, cls) => {
      const el = document.getElementById(id)
      if (!el) return
      el.textContent = val ?? '--'
      el.className = 'value' + (cls ? ' ' + cls : '')
    }

    const pf = Number(m.profit_pct || 0)
    setMetric('metricSharpe', m.sharpe ?? '--')
    setMetric('metricProfit', m.profit_pct != null ? m.profit_pct + '%' : '--', pf >= 0 ? 'positive' : 'negative')
    setMetric('metricDrawdown', m.max_dd ?? '--', 'negative')
    setMetric('metricWinrate', m.winrate ?? '--')
    setMetric('metricTrades', m.trades ?? '--')

    // Try to get training info
    try {
      const tr = await fetch(`${API}/results/latest-training`)
      const tj = await tr.json()
      if (tj && !tj.status) {
        setMetric('metricModel', tj.model ?? '--')
      }
    } catch { }

    // Chart
    if (Array.isArray(data.trades) && window.echarts) {
      const sorted = data.trades.slice().sort((a, b) => (a.open_timestamp || 0) - (b.open_timestamp || 0))
      let cum = 0
      const ys = [], xs = []
      for (const t of sorted) {
        cum += Number(t.profit_abs || 0)
        ys.push(cum)
        xs.push(new Date(t.open_timestamp || 0).toISOString().slice(0, 10))
      }
      const chart = _getOrInitChart('chart')
      if (chart) {
        chart.setOption({
          backgroundColor: 'transparent',
          grid: { left: 50, right: 20, top: 20, bottom: 40 },
          xAxis: { type: 'category', data: xs, axisLabel: { rotate: 45, color: '#8b949e', fontSize: 11 }, axisLine: { lineStyle: { color: '#30363d' } } },
          yAxis: { type: 'value', scale: true, axisLabel: { color: '#8b949e', fontSize: 11 }, splitLine: { lineStyle: { color: '#21262d' } }, axisLine: { lineStyle: { color: '#30363d' } } },
          tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#e6edf3', fontSize: 12 } },
          series: [{
            name: '累计收益(USDT)',
            type: 'line',
            data: ys,
            smooth: true,
            lineStyle: { color: '#58a6ff', width: 2 },
            areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(88,166,255,0.3)' }, { offset: 1, color: 'rgba(88,166,255,0.02)' }] } },
            itemStyle: { color: '#58a6ff' },
          }]
        })
      }
    }
    toast('摘要已加载', 'success')
  } catch (e) { toast('加载摘要失败: ' + e, 'error') }
}

function initResults() {
  const bs = document.getElementById('btnSummary')
  if (bs) bs.addEventListener('click', () => showSummary())

  // Feature TopN
  const btnFeatTop = document.getElementById('btnFeatTop')
  if (btnFeatTop) btnFeatTop.addEventListener('click', async () => {
    try {
      const file = document.getElementById('featureFile')?.value || 'user_data/freqai_features.json'
      const res = await fetch(`${API}/features/top?file=${encodeURIComponent(file)}&limit=10`)
      const data = await res.json()
      const el = document.getElementById('featTop')
      if (el) el.textContent = JSON.stringify(data, null, 2)
      toast('TopN 已加载', 'success')
    } catch (e) { toast('加载失败: ' + e, 'error') }
  })

  // Feature Plot
  const btnFeatPlot = document.getElementById('btnFeatPlot')
  if (btnFeatPlot) btnFeatPlot.addEventListener('click', () => {
    try {
      const raw = document.getElementById('featTop')?.textContent || ''
      if (!raw || !window.echarts) return
      const data = JSON.parse(raw)
      const items = Array.isArray(data) ? data : (data.items || data.features || [])
      const chart = _getOrInitChart('featChart')
      if (chart && items.length) {
        chart.setOption({
          backgroundColor: 'transparent',
          grid: { left: 120, right: 20, top: 10, bottom: 20 },
          xAxis: { type: 'value', axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
          yAxis: { type: 'category', data: items.map(i => i.name || i.feature || '').reverse(), axisLabel: { color: '#8b949e', fontSize: 11 } },
          tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#e6edf3' } },
          series: [{ type: 'bar', data: items.map(i => i.importance || i.score || 0).reverse(), itemStyle: { color: '#58a6ff', borderRadius: [0, 4, 4, 0] } }]
        })
      }
    } catch (e) { toast('绘制失败: ' + e, 'error') }
  })

  // Results List
  const btnList = document.getElementById('btnList')
  if (btnList) btnList.addEventListener('click', async () => {
    try {
      const res = await fetch(`${API}/results/list`)
      const data = await res.json()
      const el = document.getElementById('comparePanel')
      if (el) {
        const items = Array.isArray(data) ? data : (data.items || [])
        if (items.length) {
          el.innerHTML = `<div style="max-height:200px;overflow-y:auto;" class="text-sm text-mono">${items.map(i => `<div style="padding:4px 0;border-bottom:1px solid var(--border-muted);">${_escapeHtml(typeof i === 'string' ? i : i.name || JSON.stringify(i))}</div>`).join('')}</div>`
        } else {
          el.textContent = JSON.stringify(data, null, 2)
        }
      }
    } catch (e) { toast('加载失败: ' + e, 'error') }
  })

  // Compare
  const btnCompare = document.getElementById('btnCompare')
  if (btnCompare) btnCompare.addEventListener('click', async () => {
    const a = (document.getElementById('resA')?.value || '').trim()
    const b = (document.getElementById('resB')?.value || '').trim()
    if (!a || !b) { toast('请输入结果 A 与 B', 'error'); return }
    try {
      const [ra, rb] = await Promise.all([
        fetch(`${API}/results/summary?name=${encodeURIComponent(a)}`),
        fetch(`${API}/results/summary?name=${encodeURIComponent(b)}`),
      ])
      const [sa, sb] = [await ra.json(), await rb.json()]
      const toM = (s) => {
        const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}
        return { profit_pct: c.profit_total_pct ?? s.profit_total_pct, profit_abs: c.profit_total_abs ?? s.profit_total_abs, trades: c.trades ?? s.trades, winrate: c.winrate ?? s.winrate, max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs }
      }
      const ma = toM(sa), mb = toM(sb)
      const el = document.getElementById('comparePanel')
      if (el) el.innerHTML = `
        <table class="compare-table mt-2">
          <tr><th>指标</th><th>A (${_escapeHtml(a)})</th><th>B (${_escapeHtml(b)})</th></tr>
          <tr><td>收益%</td><td>${ma.profit_pct ?? '--'}</td><td>${mb.profit_pct ?? '--'}</td></tr>
          <tr><td>收益(USDT)</td><td>${ma.profit_abs ?? '--'}</td><td>${mb.profit_abs ?? '--'}</td></tr>
          <tr><td>交易数</td><td>${ma.trades ?? '--'}</td><td>${mb.trades ?? '--'}</td></tr>
          <tr><td>胜率</td><td>${ma.winrate ?? '--'}</td><td>${mb.winrate ?? '--'}</td></tr>
          <tr><td>最大回撤(USDT)</td><td>${ma.max_dd ?? '--'}</td><td>${mb.max_dd ?? '--'}</td></tr>
        </table>`
    } catch (e) { toast('对比失败: ' + e, 'error') }
  })

  // Gallery
  const btnGallery = document.getElementById('btnGallery')
  if (btnGallery) btnGallery.addEventListener('click', async () => {
    await runWithLoading(btnGallery, async () => {
      const res = await fetch(`${API}/results/gallery?limit=100`)
      const data = await res.json()
      const gp = document.getElementById('galleryPanel')
      if (!gp) return
      const items = data.items || []
      if (!items.length) { gp.innerHTML = '<div class="empty-state text-sm">暂无数据</div>'; return }
      gp.innerHTML = `<div class="gallery-grid mt-2">${items.map(x => {
        const pct = Number(x.profit_total_pct || 0)
        const width = Math.max(0, Math.min(100, Math.round(Math.abs(pct))))
        const cls = pct >= 0 ? 'positive' : 'negative'
        return `<div class="gallery-card">
          <div class="name">${_escapeHtml(x.name)}</div>
          <div class="row"><span>收益%</span><b>${x.profit_total_pct ?? '--'}</b></div>
          <div class="row"><span>交易数</span><b>${x.trades ?? '--'}</b></div>
          <div class="row"><span>最大回撤</span><b>${x.max_drawdown_abs ?? '--'}</b></div>
          <div class="bar"><div class="fill ${cls}" style="width:${width}%"></div></div>
        </div>`
      }).join('')}</div>`
    })
  })

  // Aggregate
  const btnAgg = document.getElementById('btnAgg')
  if (btnAgg) btnAgg.addEventListener('click', async () => {
    await runWithLoading(btnAgg, async () => {
      const names = (document.getElementById('aggNames')?.value || '').trim()
      if (!names) { toast('请输入结果名称', 'error'); return }
      const res = await fetch(`${API}/results/aggregate?names=${encodeURIComponent(names)}`)
      const data = await res.json()
      const ap = document.getElementById('aggPanel')
      if (!ap) return
      ap.innerHTML = '<div id="aggChart" style="height:220px"></div><div class="text-sm text-muted mt-2" id="aggInfo"></div>'
      if (window.echarts) {
        const chart = _getOrInitChart('aggChart')
        if (chart) {
          const items = data.items || []
          chart.setOption({
            backgroundColor: 'transparent',
            grid: { left: 50, right: 20, top: 10, bottom: 60 },
            xAxis: { type: 'category', data: items.map(i => i.name), axisLabel: { rotate: 60, color: '#8b949e', fontSize: 10 } },
            yAxis: { type: 'value', scale: true, axisLabel: { color: '#8b949e' }, splitLine: { lineStyle: { color: '#21262d' } } },
            tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#30363d', textStyle: { color: '#e6edf3' } },
            series: [{ type: 'bar', data: items.map(i => i.profit_total_pct || 0), itemStyle: { color: '#58a6ff', borderRadius: [4, 4, 0, 0] } }]
          })
        }
      }
      const info = document.getElementById('aggInfo')
      if (info) info.innerHTML = `均值收益: ${data.mean_profit ?? '--'} | 收益波动: ${data.std_profit ?? '--'} | 鲁棒分: ${data.robust_score ?? '--'}`
    })
  })
}

// ============================================
// Run History
// ============================================
function _renderRunStatusTag(status) {
  const st = String(status || '').toLowerCase()
  const cls = st === 'success' ? 'tag tag-ok' : (st === 'failed' ? 'tag tag-failed' : 'tag tag-running')
  return `<span class="${cls}">${_escapeHtml(status || 'unknown')}</span>`
}

function _extractLastRunIdFromLogs(text) {
  const re = /\[FLOW\]\s+RUN_ID\s+([0-9a-f]{8,64})/ig
  let m = null, last = null
  while ((m = re.exec(String(text || '')))) last = m[1]
  return last
}

function _renderArtifactItem({ label, ok, href, detail }) {
  const cls = ok ? 'tag tag-ok' : 'tag tag-failed'
  const inner = `<span class="${cls}">${_escapeHtml(label)}</span>`
  const link = href ? `<a href="${_escapeHtml(href)}" target="_blank" rel="noreferrer" style="text-decoration:none">${inner}</a>` : inner
  const sub = detail ? `<div class="text-xs text-muted" style="margin:2px 0 0 0;">${_escapeHtml(detail)}</div>` : ''
  return `<div style="margin:4px 0;">${link}${sub}</div>`
}

async function renderFlowArtifactsCheck(runIdOverride) {
  const el = document.getElementById('flowArtifacts')
  if (!el) return
  const logsText = document.getElementById('logs')?.textContent || ''
  const runId = (runIdOverride || '').trim() || _extractLastRunIdFromLogs(logsText)
  if (!runId) {
    el.innerHTML = `<div class="text-sm text-muted mt-2">未在日志中发现 run_id</div>`
    return
  }
  el.innerHTML = `<div class="text-sm text-muted">加载产物检查…</div>`
  let meta = null
  try {
    const r = await fetch(`${API}/flow/run-meta/${encodeURIComponent(runId)}`)
    meta = await r.json()
  } catch (e) {
    el.innerHTML = `<div class="text-sm text-red">加载失败: ${_escapeHtml(e)}</div>`
    return
  }

  const checks = (meta && meta.checks) || {}
  const artifacts = (meta && meta.artifacts) || {}
  const btZips = (checks.backtest_zips && Array.isArray(checks.backtest_zips)) ? checks.backtest_zips : []
  const trSums = (checks.training_summaries && Array.isArray(checks.training_summaries)) ? checks.training_summaries : []
  const latestZipName = (() => {
    const raw = (artifacts.backtest_zips && Array.isArray(artifacts.backtest_zips)) ? artifacts.backtest_zips : []
    if (!raw.length) return null
    try { return String(raw[raw.length - 1]).split('/').pop() } catch { return null }
  })()

  const rows = []
  rows.push(_renderArtifactItem({ label: `run_meta (${meta.status || 'unknown'})`, ok: true, href: `${API}/flow/run-meta/${encodeURIComponent(runId)}` }))
  rows.push(_renderArtifactItem({ label: 'features', ok: !!(checks.feature_output && checks.feature_output.exists) }))
  rows.push(_renderArtifactItem({ label: 'expressions', ok: !!(checks.expression_output && checks.expression_output.exists) }))
  rows.push(_renderArtifactItem({
    label: 'factor scores',
    ok: !!(checks.factor_scores_json && checks.factor_scores_json.exists),
    href: `${API}/flow/factor-scores/${encodeURIComponent(runId)}`,
  }))
  rows.push(_renderArtifactItem({ label: 'training', ok: trSums.some(x => x && x.exists), href: `${API}/results/latest-training` }))
  rows.push(_renderArtifactItem({ label: 'backtest', ok: btZips.some(x => x && x.exists), href: latestZipName ? `${API}/results/summary?name=${encodeURIComponent(latestZipName)}` : `${API}/results/latest-summary` }))
  rows.push(_renderArtifactItem({
    label: 'bundle.zip',
    ok: !!(checks.bundle_zip && checks.bundle_zip.exists),
    href: `${API}/results/bundles/download/${encodeURIComponent(runId)}`,
  }))

  el.innerHTML = `<div class="section-title text-sm mt-2">产物检查 (${_escapeHtml(runId.slice(0, 12))})</div>${rows.join('')}`
}

async function loadRunHistory() {
  const el = document.getElementById('runHistory')
  if (!el) return
  let limit = 10
  try { limit = parseInt(document.getElementById('runHistoryLimit')?.value || '10', 10) } catch { }
  if (!isFinite(limit) || limit <= 0) limit = 10
  limit = Math.max(1, Math.min(200, limit))

  el.innerHTML = `<div class="text-sm text-muted">加载中…</div>`
  let data = null
  try {
    const r = await fetch(`${API}/flow/runs/list?limit=${limit}`)
    data = await r.json()
  } catch (e) {
    el.innerHTML = `<div class="text-sm text-red">加载失败: ${_escapeHtml(e)}</div>`
    return
  }
  const items = (data && Array.isArray(data.items)) ? data.items : []
  if (!items.length) {
    el.innerHTML = `<div class="text-sm text-muted">未找到运行记录</div>`
    return
  }

  el.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Run ID</th><th>状态</th><th>结束时间</th><th>cfg_sha</th></tr></thead>
      <tbody>${items.map(it => {
    const rid = String(it.run_id || '')
    const when = (it.ended_at || it.started_at || '')
    const whenShort = when ? String(when).replace('T', ' ').replace('Z', '').slice(0, 19) : '--'
    const cfgSha = it.config_sha256 ? String(it.config_sha256).slice(0, 8) : ''
    return `<tr>
          <td><a href="#" class="text-blue text-mono" data-runid="${_escapeHtml(rid)}">${_escapeHtml(rid.slice(0, 12))}</a></td>
          <td>${_renderRunStatusTag(it.status)}</td>
          <td class="text-sm">${_escapeHtml(whenShort)}</td>
          <td class="text-mono text-sm">${_escapeHtml(cfgSha)}</td>
        </tr>`
  }).join('')}</tbody>
    </table>`

  el.querySelectorAll('a[data-runid]').forEach(a => {
    a.onclick = (ev) => {
      ev.preventDefault()
      const runId = a.getAttribute('data-runid') || ''
      if (runId) renderFlowArtifactsCheck(runId)
    }
  })
}

function initRunHistory() {
  const btn = document.getElementById('btnLoadRunHistory')
  if (btn) btn.addEventListener('click', () => runWithLoading(btn, loadRunHistory))
}

// ============================================
// Agent Flow
// ============================================
function initAgentFlow() {
  const bind = (btnId, cfgId, stepsId) => {
    const btn = document.getElementById(btnId)
    if (!btn) return
    btn.addEventListener('click', async () => {
      const logsEl = document.getElementById('logs')
      if (logsEl) logsEl.textContent = ''
      const fa = document.getElementById('flowArtifacts')
      if (fa) fa.innerHTML = ''
      const cfg = document.getElementById(cfgId)?.value || 'configs/agent_flow_multi.json'
      const stepsRaw = (document.getElementById(stepsId)?.value || '').trim()
      const body = { config: cfg }
      if (stepsRaw) body.steps = stepsRaw.split(/\s+/)

      const fs = document.getElementById('flowStatus')
      if (fs) {
        const steps = body.steps || ['feature', 'expression', 'ml', 'rl', 'backtest']
        fs.innerHTML = '<div class="flex gap-2 mt-2" style="flex-wrap:wrap">' + steps.map(s => `<span class="tag tag-running" data-step="${s}">${s}</span>`).join('') + '</div>'
      }

      try {
        setGlobalStatus('Flow 运行中...', 'running')
        const r = await fetch(`${API}/flow/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const j = await r.json()
        if (j.job_id) {
          toast('Flow 已启动', 'success')
          const stepsCsv = body.steps ? body.steps.join(',') : 'feature,expression,ml,rl,backtest'

          const applyProgress = (pj) => {
            const items = (pj && pj.steps) || []
            for (const it of items) {
              const el = document.querySelector(`.tag[data-step="${it.name}"]`)
              if (!el) continue
              el.classList.remove('tag-ok', 'tag-failed', 'tag-running')
              if (it.status === 'ok') el.classList.add('tag-ok')
              else if (it.status === 'failed') el.classList.add('tag-failed')
              else el.classList.add('tag-running')
              const pct = (typeof it.percent === 'number') ? ` ${it.percent}%` : ''
              const phase = it.phase ? ` (${it.phase}${pct})` : (pct || '')
              el.textContent = `${it.name}${phase}`
            }
            // Append log delta
            try {
              const delta = pj && pj.delta
              if (delta && Array.isArray(delta) && delta.length) {
                const logsEl = document.getElementById('logs')
                if (logsEl) {
                  logsEl.textContent += delta.join('\n') + '\n'
                  logsEl.scrollTop = logsEl.scrollHeight
                }
              }
            } catch { }
          }

          // SSE
          let es = null, timer = null
          try {
            es = new EventSource(`${API}/flow/stream/${j.job_id}?steps=${encodeURIComponent(stepsCsv)}`)
            es.onmessage = (ev) => {
              try { const pj = JSON.parse(ev.data); applyProgress(pj); if (pj && pj.running === false) { try { es.close() } catch { } } } catch { }
            }
            es.onerror = () => { try { es.close() } catch { } }
          } catch { es = null }

          if (!es) {
            timer = setInterval(async () => {
              try {
                const pr = await fetch(`${API}/flow/progress/${j.job_id}?steps=${encodeURIComponent(stepsCsv)}`)
                const pj = await pr.json()
                applyProgress(pj)
                if (!pj.running) { clearInterval(timer); timer = null }
              } catch { }
            }, 1000)
          }

          await pollLogs(j.job_id)
          if (es) try { es.close() } catch { }
          if (timer) { clearInterval(timer); timer = null }
          setGlobalStatus('就绪')
          try { await renderFlowArtifactsCheck() } catch { }
          try { await showSummary() } catch { }
          try { await loadRunHistory() } catch { }
        } else {
          toast(JSON.stringify(j), 'error')
          setGlobalStatus('就绪')
        }
      } catch (e) { toast('Flow 启动失败: ' + e, 'error'); setGlobalStatus('就绪') }
    })
  }

  bind('btnRunAgentFlow', 'flowCfg', 'flowSteps')
  bind('btnRunAgentFlow2', 'flowCfg2', 'flowSteps2')

  // Train+BT shortcut
  const trainBt = document.getElementById('btnTrainBt')
  if (trainBt) trainBt.addEventListener('click', () => {
    try {
      const ids = { flowCfg: GOLDEN.flow_cfg, flowSteps: GOLDEN.flow_steps, cfg: GOLDEN.freqtrade_cfg, featureFile: GOLDEN.feature_file, timeframe: GOLDEN.timeframe, timerange: GOLDEN.timerange }
      for (const [k, v] of Object.entries(ids)) {
        const el = document.getElementById(k)
        if (el) el.value = v
      }
      document.getElementById('btnRunAgentFlow')?.click()
    } catch (e) { toast('启动失败: ' + e, 'error') }
  })
}

// ============================================
// Strategy Miner
// ============================================
function initStrategyMiner() {
  let currentMinerJobId = null

  // Start
  const btnStart = document.getElementById('btnStartMiner')
  if (btnStart) btnStart.addEventListener('click', async () => {
    const body = {
      config: document.getElementById('minerConfig')?.value || 'configs/strategy_miner_default.json',
    }
    const iters = document.getElementById('minerIters')?.value
    if (iters) body.max_iterations = parseInt(iters, 10)
    const model = (document.getElementById('minerModel')?.value || '').trim()
    if (model) body.model = model
    const resume = (document.getElementById('minerResume')?.value || '').trim()
    if (resume) body.resume = resume

    try {
      setGlobalStatus('策略挖掘运行中...', 'running')
      const r = await fetch(`${API}/strategy-miner/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const j = await r.json()
      if (j.job_id) {
        currentMinerJobId = j.job_id
        toast('策略挖掘已启动', 'success')
        const statusTag = document.getElementById('minerStatusTag')
        if (statusTag) { statusTag.className = 'tag tag-running'; statusTag.textContent = '运行中' }

        // Poll status
        const pollStatus = setInterval(async () => {
          try {
            const sr = await fetch(`${API}/strategy-miner/status/${currentMinerJobId}`)
            const sd = await sr.json()
            const iterEl = document.getElementById('minerIterCurrent')
            const barEl = document.getElementById('minerProgressBar')
            const rewardEl = document.getElementById('minerBestReward')
            const maxIters = parseInt(iters || '5', 10)
            const cur = sd.iteration || 0
            if (iterEl) iterEl.textContent = `${cur} / ${maxIters}`
            if (barEl) barEl.style.width = `${Math.min(100, Math.round((cur / maxIters) * 100))}%`
            if (rewardEl && sd.best_reward != null) rewardEl.textContent = sd.best_reward.toFixed(4)
            if (sd.last_lines) {
              const logsEl = document.getElementById('minerLogs')
              if (logsEl) {
                logsEl.textContent = sd.last_lines.join('\n')
                logsEl.scrollTop = logsEl.scrollHeight
              }
            }
            if (!sd.running) {
              clearInterval(pollStatus)
              if (statusTag) { statusTag.className = 'tag tag-ok'; statusTag.textContent = '已完成' }
              setGlobalStatus('就绪')
              loadMinerRuns()
              // Load leaderboard for latest run
              if (j.run_id) loadMinerLeaderboard(j.run_id)
            }
          } catch { }
        }, 2000)
      } else {
        toast(JSON.stringify(j), 'error')
        setGlobalStatus('就绪')
      }
    } catch (e) { toast('启动失败: ' + e, 'error'); setGlobalStatus('就绪') }
  })

  // Load miner runs
  const btnRuns = document.getElementById('btnLoadMinerRuns')
  if (btnRuns) btnRuns.addEventListener('click', () => loadMinerRuns())

  async function loadMinerRuns() {
    const el = document.getElementById('minerRunsList')
    if (!el) return
    try {
      const r = await fetch(`${API}/strategy-miner/runs?limit=10`)
      const data = await r.json()
      const items = data.items || []
      if (!items.length) { el.innerHTML = '<div class="text-sm text-muted">暂无运行记录</div>'; return }
      el.innerHTML = items.map(it => `
        <div class="leaderboard-row" style="cursor:pointer" data-runid="${_escapeHtml(it.run_id)}">
          <div class="flex-1">
            <div class="text-sm text-mono text-blue">${_escapeHtml(String(it.run_id).slice(0, 12))}</div>
            <div class="text-xs text-muted">Iter ${it.iteration || 0} | ${it.candidates_count || 0} candidates</div>
          </div>
          <div class="text-sm text-mono text-green">${it.best_reward != null ? Number(it.best_reward).toFixed(3) : '--'}</div>
        </div>
      `).join('')
      el.querySelectorAll('[data-runid]').forEach(row => {
        row.onclick = () => loadMinerLeaderboard(row.dataset.runid)
      })
    } catch (e) { el.innerHTML = `<div class="text-sm text-red">加载失败</div>` }
  }

  async function loadMinerLeaderboard(runId) {
    const el = document.getElementById('minerLeaderboard')
    if (!el) return
    try {
      const r = await fetch(`${API}/strategy-miner/runs/${encodeURIComponent(runId)}/candidates`)
      const data = await r.json()
      const items = data.items || []
      if (!items.length) { el.innerHTML = '<div class="empty-state text-sm">暂无候选</div>'; return }

      const sorted = items.slice().sort((a, b) => (b.reward || 0) - (a.reward || 0))
      el.innerHTML = sorted.map((c, i) => {
        const rankCls = i === 0 ? 'gold' : (i === 1 ? 'silver' : (i === 2 ? 'bronze' : ''))
        const passed = c.validation_passed
        return `
        <div class="leaderboard-row" style="cursor:pointer" data-candidate="${_escapeHtml(c.name || '')}">
          <div class="leaderboard-rank ${rankCls}">${i + 1}</div>
          <div class="flex-1">
            <div class="text-sm" style="font-weight:600;color:var(--text-primary)">${_escapeHtml(c.name || '未命名')}</div>
            <div class="text-xs text-muted">Iter ${c.iteration || 0} | ${passed ? '&#10003; 通过' : '&#10007; 未通过'}</div>
          </div>
          <div class="text-mono" style="font-size:16px;font-weight:700;color:${(c.reward || 0) >= 0 ? 'var(--success)' : 'var(--danger)'}">${c.reward != null ? Number(c.reward).toFixed(4) : '--'}</div>
        </div>`
      }).join('')

      el.querySelectorAll('[data-candidate]').forEach(row => {
        row.onclick = () => {
          const name = row.dataset.candidate
          const found = sorted.find(c => c.name === name)
          const detail = document.getElementById('minerCandidateDetail')
          if (detail && found) {
            detail.innerHTML = `
              <div class="flex justify-between items-center mb-3">
                <div class="text-sm" style="font-weight:700;color:var(--text-primary)">${_escapeHtml(found.name || '')}</div>
                <div class="btn-group">
                  <button class="btn btn-sm btn-primary" onclick="approveCandidate('${_escapeHtml(runId)}','${_escapeHtml(found.name || '')}')">采纳</button>
                  <button class="btn btn-sm" onclick="backtestCandidate('${_escapeHtml(runId)}','${_escapeHtml(found.name || '')}')">回测</button>
                </div>
              </div>
              <div class="text-sm"><span class="text-muted">Reward:</span> <span class="text-mono text-green">${found.reward ?? '--'}</span></div>
              <div class="text-sm"><span class="text-muted">Iteration:</span> <span class="text-mono">${found.iteration ?? '--'}</span></div>
              <div class="text-sm"><span class="text-muted">Validation:</span> ${found.validation_passed ? '<span class="text-green">Passed</span>' : '<span class="text-red">Failed</span>'}</div>
              ${found.diagnosis ? `<div class="text-sm mt-2"><span class="text-muted">Diagnosis:</span><div class="log-panel mt-2" style="max-height:120px;font-size:11px;">${_escapeHtml(found.diagnosis)}</div></div>` : ''}
              ${found.strategy_path ? `<div class="text-xs text-muted mt-2">Path: ${_escapeHtml(found.strategy_path)}</div>` : ''}
            `
          }
        }
      })
    } catch (e) {
      el.innerHTML = `<div class="text-sm text-red">加载失败: ${_escapeHtml(String(e))}</div>`
    }
  }

  // Expose for inline onclick
  window.approveCandidate = async (runId, name) => {
    try {
      const r = await fetch(`${API}/strategy-miner/runs/${encodeURIComponent(runId)}/approve`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate: name }) })
      const j = await r.json()
      if (j.status === 'ok') toast(`已采纳: ${j.dest}`, 'success')
      else toast(JSON.stringify(j), 'error')
    } catch (e) { toast('采纳失败: ' + e, 'error') }
  }

  window.backtestCandidate = async (runId, name) => {
    try {
      setGlobalStatus('回测候选中...', 'running')
      const r = await fetch(`${API}/strategy-miner/runs/${encodeURIComponent(runId)}/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ candidate: name }) })
      const j = await r.json()
      if (j.job_id) { toast('回测已启动', 'success'); await pollLogs(j.job_id, 'minerLogs'); setGlobalStatus('就绪') }
      else { toast(JSON.stringify(j), 'error'); setGlobalStatus('就绪') }
    } catch (e) { toast('回测失败: ' + e, 'error'); setGlobalStatus('就绪') }
  }
}

// ============================================
// ReactFlow App (Flow Tab)
// ============================================
function CustomNode({ id, data }) {
  const typeKey = data?.typeKey || 'node'
  const info = data?.cfg || {}
  const rows = []
  if (typeKey === 'data') rows.push(['tf', info.timeframe || '--'], ['pairs', String(info.pairs || '--').split(' ').length])
  if (typeKey === 'expr') rows.push(['llm', info.llm_model || '--'], ['count', info.llm_count || '--'])
  if (typeKey === 'bt') rows.push(['range', info.timerange || '--'])
  if (typeKey === 'ml') rows.push(['model', info.model_name || 'lightgbm'])
  if (typeKey === 'rl') rows.push(['algo', info.algorithm || 'PPO'])

  return h('div', { className: 'am-node' }, [
    h('div', { className: 'header' }, [
      h('div', { className: 'title' }, data?.label || id),
      h('div', { className: 'badge' }, typeKey),
    ]),
    h('div', { className: 'body' }, rows.map(([k, v]) =>
      h('div', { className: 'kv', key: k }, [h('span', null, k), h('b', null, String(v))])
    )),
    h('div', { className: 'footer' }, [
      h('div', null, info.output || info.results_dir || ''),
      h('div', { className: 'actions' }, [
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); showSummary() } }, '摘要'),
      ])
    ]),
    h(Handle, { type: 'target', position: Position.Left }),
    h(Handle, { type: 'source', position: Position.Right }),
  ])
}

function FlowApp() {
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
  const [grid, setGrid] = useState([16, 16])
  const nodeTypes = React.useMemo ? React.useMemo(() => ({ amNode: CustomNode }), []) : { amNode: CustomNode }
  const defaultEdgeOptions = {
    animated: true,
    type: 'smoothstep',
    style: { stroke: '#58a6ff', strokeWidth: 1.6 },
    markerEnd: MarkerType.ArrowClosed ? { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#58a6ff' } : undefined
  }

  useEffect(() => {
    const layoutBtn = document.getElementById('btnLayout')
    if (layoutBtn) layoutBtn.onclick = () => {
      const sorted = (nodes || []).slice().sort((a, b) => (a.position?.x || 0) - (b.position?.x || 0))
      setNodes(sorted.map((n, i) => ({ ...n, position: { x: 80 + i * 260, y: 120 } })))
    }
    const fitBtn = document.getElementById('btnFit')
    if (fitBtn) fitBtn.onclick = () => { try { window.__rf && window.__rf.fitView() } catch { } }
    const clearBtn = document.getElementById('btnClear')
    if (clearBtn) clearBtn.onclick = () => { setNodes([]); setEdges([]) }
    const snapBtn = document.getElementById('btnSnap')
    if (snapBtn) snapBtn.onclick = () => setSnap(s => !s)
    const ax = document.getElementById('btnAlignX')
    if (ax) ax.onclick = () => {
      const sels = (nodes || []).filter(n => n.selected)
      if (sels.length < 2) { toast('至少选中2个节点', 'error'); return }
      const refY = sels[0].position?.y || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: n.position.x, y: refY } }) : n))
    }
    const ay = document.getElementById('btnAlignY')
    if (ay) ay.onclick = () => {
      const sels = (nodes || []).filter(n => n.selected)
      if (sels.length < 2) { toast('至少选中2个节点', 'error'); return }
      const refX = sels[0].position?.x || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: refX, y: n.position.y } }) : n))
    }
  }, [nodes])

  const onConnect = useCallback((params) => setEdges((eds) => addEdgeLib(params, eds)), [])
  const onNodesChange = useCallback((changes) => setNodes((nds) => applyNodeChanges(changes, nds)), [])
  const onEdgesChange = useCallback((changes) => setEdges((eds) => applyEdgeChanges(changes, eds)), [])
  const onNodeClick = useCallback((_, n) => {
    const el = document.getElementById('nodeForm')
    if (el) el.innerHTML = `<div class="text-sm text-mono">${_escapeHtml(JSON.stringify(n.data?.cfg || {}, null, 2))}</div>`
  }, [])
  const onDrop = useCallback((ev) => {
    ev.preventDefault()
    const typeKey = ev.dataTransfer.getData('application/node-type')
    if (!typeKey) return
    const id = 'n' + Math.random().toString(16).slice(2, 8)
    const cfg = typeKey === 'expr' ? { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' }
      : (typeKey === 'bt' ? { timerange: '20210101-20211231' }
        : (typeKey === 'data' ? { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h' } : {}))
    const position = { x: ev.clientX - 300, y: ev.clientY - 120 }
    setNodes(nds => nds.concat({ id, type: 'amNode', position, data: { label: typeKey.toUpperCase(), typeKey, cfg } }))
  }, [])
  const onDragOver = useCallback((ev) => { ev.preventDefault(); ev.dataTransfer.dropEffect = 'move' }, [])

  return h(RF, {
    nodes, edges, nodeTypes, defaultEdgeOptions, fitView: true,
    onConnect, onNodesChange, onEdgesChange, onNodeClick, onDrop, onDragOver,
    onInit: (inst) => { window.__rf = inst },
    snapToGrid: !!snap, snapGrid: grid,
    style: { background: 'transparent' },
  }, [
    h(Background, { variant: 'dots', gap: 20, size: 1, color: '#21262d', key: 'bg' }),
    h(Controls, { key: 'ctl', style: { bottom: 10, left: 10 } }),
    h(MiniMap, { key: 'mm', style: { bottom: 10, right: 10 }, nodeColor: () => '#58a6ff', maskColor: 'rgba(13,17,23,0.8)' }),
  ])
}

// ============================================
// Boot
// ============================================
function boot() {
  // Tab navigation
  initTabNav()
  initSettingsNav()

  // Connection
  initConnection()

  // Settings
  initSettings()

  // Expression & Backtest
  initExprBacktest()

  // Results
  initResults()
  initRunHistory()

  // Agent Flow
  initAgentFlow()

  // Strategy Miner
  initStrategyMiner()

  // Palette drag
  document.querySelectorAll('.palette-item').forEach(el => {
    el.addEventListener('dragstart', (e) => {
      const typeKey = e.target?.closest?.('.palette-item')?.getAttribute('data-nodetype')
      if (!typeKey) return
      e.dataTransfer.setData('application/node-type', typeKey)
      e.dataTransfer.effectAllowed = 'move'
    })
  })

  // ReactFlow
  const hasReact = !!(window.React && window.ReactDOM)
  const hasRF = !!window.ReactFlow
  if (hasReact && hasRF) {
    try {
      createRoot(document.getElementById('root')).render(h(FlowApp))
    } catch (e) {
      console.warn('[fallback] ReactFlow render failed', e)
    }
  }

  // Auto-load run history on boot
  try { loadRunHistory() } catch { }
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  setTimeout(boot, 0)
} else {
  document.addEventListener('DOMContentLoaded', boot)
}
