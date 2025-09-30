const { createElement: h, useState, useEffect, useCallback } = window.React
const { createRoot } = window.ReactDOM
// 兼容 UMD 版本的 ReactFlow 全局导出
const RFLib = window.ReactFlow || {}
const RF = RFLib.ReactFlow || RFLib.default || (typeof RFLib === 'function' ? RFLib : RFLib)
const Background = RFLib.Background || (RFLib.default && RFLib.default.Background) || (() => null)
const Controls = RFLib.Controls || (RFLib.default && RFLib.default.Controls) || (() => null)
const MiniMap = RFLib.MiniMap || (RFLib.default && RFLib.default.MiniMap) || (() => null)
const Handle = RFLib.Handle || (() => null)
const Position = RFLib.Position || { Top: 'top', Bottom: 'bottom', Left: 'left', Right: 'right' }
const MarkerType = RFLib.MarkerType || {}
try { console.log('[rf]', Object.keys(RFLib)) } catch {}
const applyNodeChanges = RFLib.applyNodeChanges || ((chs, nds) => nds)
const applyEdgeChanges = RFLib.applyEdgeChanges || ((chs, eds) => eds)
const addEdgeLib = RFLib.addEdge || ((params, eds) => eds.concat({ id: (params.id || ('e' + Math.random().toString(16).slice(2,8))), ...params }))

const API = 'http://127.0.0.1:8000'

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
  if (typeKey === 'ho') rows.push(['epochs', info.epochs||'--'])
  const locked = !!data?.locked
  return h('div', { className: 'am-node' + (locked ? ' locked' : '') }, [
    h('div', { className: 'header' }, [ h(Icon, { type: typeKey }), h('div', { className: 'title' }, [ data?.label||id, locked ? h('i', { className: 'ri-lock-2-line', title: '已锁定' }) : null ]), h('div', { className: 'badge' }, typeKey) ]),
    h('div', { className: 'body' }, rows.map(([k,v]) => h('div', { className: 'kv' }, [ h('span', null, k), h('b', null, String(v)) ]))),
    h('div', { className: 'footer' }, [
      h('div', null, (info.output||info.results_dir||'')),
      h('div', { className: 'actions' }, [
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { window.__runNode && window.__runNode(id) } catch(e){} } }, '运行'),
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { window.__configureNode && window.__configureNode(id) } catch(e){} } }, '配置'),
        h('button', { className: 'mini', onClick: (e) => { e.stopPropagation(); e.preventDefault(); try { window._simulateClicks && window._simulateClicks('summary') } catch(e){} } }, '摘要'),
      ])
    ]),
    h(Handle, { type: 'target', position: Position.Left }),
    h(Handle, { type: 'source', position: Position.Right }),
  ])
}

function App() {
  const [nodes, setNodes] = useState([
    { id: 'n1', type: 'amNode', position: { x: 50, y: 120 }, data: { label: 'Data', typeKey: 'data', cfg: { pairs: 'BTC/USDT ETH/USDT SOL/USDT ADA/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' } } },
    { id: 'n2', type: 'amNode', position: { x: 320, y: 120 }, data: { label: 'Expression(LLM)', typeKey: 'expr', cfg: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' } } },
    { id: 'n3', type: 'amNode', position: { x: 610, y: 120 }, data: { label: 'Backtest', typeKey: 'bt', cfg: { timerange: '20210101-20211231' } } },
    { id: 'n4', type: 'amNode', position: { x: 900, y: 120 }, data: { label: 'Feedback', typeKey: 'fb', cfg: { results_dir: 'user_data/backtest_results' } } },
    { id: 'n5', type: 'amNode', position: { x: 1180, y: 120 }, data: { label: 'Hyperopt', typeKey: 'ho', cfg: { timerange: '20210101-20210430', spaces: 'buy sell protection', epochs: 20, loss: 'SharpeHyperOptLoss' } } },
  ])
  const [edges, setEdges] = useState([
    { id: 'e1', source: 'n1', target: 'n2', type: 'smoothstep', animated: true },
    { id: 'e2', source: 'n2', target: 'n3', type: 'smoothstep', animated: true },
  ])
  const [selected, setSelected] = useState(null)
  const [snap, setSnap] = useState(true)
  const [grid, setGrid] = useState([16,16])
  const nodeTypes = React.useMemo ? React.useMemo(() => ({ amNode: CustomNode }), []) : { amNode: CustomNode }
  const defaultEdgeOptions = { animated: true, type: 'smoothstep', style: { stroke: '#8694ff', strokeWidth: 1.6 }, markerEnd: MarkerType.ArrowClosed ? { type: MarkerType.ArrowClosed, width: 16, height: 16, color: '#8694ff' } : undefined }

  const logsEl = document.getElementById('logs')
  const summaryEl = document.getElementById('summary')
  const featTopEl = document.getElementById('featTop')
  const compareEl = document.getElementById('comparePanel')

  async function pollLogs(jobId) {
    let offset = 0
    while (true) {
      const res = await fetch(`${API}/jobs/${jobId}/logs?offset=${offset}`)
      const data = await res.json()
      const chunk = (data.logs || []).join('\n')
      if (chunk) {
        logsEl.textContent += chunk + '\n'
        try { logsEl.scrollTop = logsEl.scrollHeight } catch {}
      }
      offset = data.next || offset
      if (!data.running) break
      await new Promise(r => setTimeout(r, 800))
    }
  }

  async function runExpr() {
    logsEl.textContent = ''
    const body = {
      config: document.getElementById('cfg').value,
      feature_file: document.getElementById('featureFile').value,
      output: 'user_data/freqai_expressions.json',
      timeframe: document.getElementById('timeframe').value,
      llm_model: document.getElementById('llmModel').value,
      llm_count: parseInt(document.getElementById('llmCount').value || '3'),
      llm_loops: 1,
      llm_timeout: 60,
      feedback_top: 0
    }
    const res = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const data = await res.json()
    pollLogs(data.job_id)
  }

  async function runBacktest() {
    const body = {
      config: document.getElementById('cfg').value,
      strategy: 'ExpressionLongStrategy',
      strategy_path: 'freqtrade/user_data/strategies',
      timerange: document.getElementById('timerange').value,
      freqaimodel: 'LightGBMRegressor',
      export: true,
      export_filename: 'user_data/backtest_results/latest_trades_multi',
    }
    const res = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    const data = await res.json()
    pollLogs(data.job_id)
  }

  async function showSummary() {
    const res = await fetch(`${API}/results/latest-summary`)
    const data = await res.json()
    summaryEl.textContent = JSON.stringify(data, null, 2)
    try {
      // 指标卡片
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
        // 同时拉取最新训练摘要（如存在）
        let train = null
        try {
          const tr = await fetch(`${API}/results/latest-training`)
          const tj = await tr.json()
          if (!tj.error) train = tj
        } catch {}
        const trainRMSE = train && train.metrics ? train.metrics.rmse_valid ?? train.metrics.rmse_train : null
        const trainModel = train ? train.model : null
        cards.innerHTML = `
          <div class="card"><div class="k">总收益%</div><div class="v">${m.profit_pct ?? '--'}</div></div>
          <div class="card"><div class="k">交易数</div><div class="v">${m.trades ?? '--'}</div></div>
          <div class="card"><div class="k">胜率</div><div class="v">${m.winrate ?? '--'}</div></div>
          <div class="card"><div class="k">最大回撤(USDT)</div><div class="v">${m.max_dd ?? '--'}</div></div>
          <div class="card"><div class="k">训练模型</div><div class="v">${trainModel ?? '--'}</div></div>
          <div class="card"><div class="k">训练RMSE</div><div class="v">${trainRMSE ?? '--'}</div></div>
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
        const chart = echarts.init(document.getElementById('chart'))
        chart.setOption({
          grid: { left: 40, right: 16, top: 10, bottom: 30 },
          xAxis: { type: 'category', data: xs, axisLabel: { rotate: 45 } },
          yAxis: { type: 'value', scale: true },
          tooltip: { trigger: 'axis' },
          series: [{ name: '累计收益(USDT)', type: 'line', data: ys }],
        })
      }
    } catch (e) { console.warn('chart error', e) }
  }

  useEffect(() => {
    const be = document.getElementById('btnExpr')
    const bb = document.getElementById('btnBacktest')
    const bs = document.getElementById('btnSummary')
    if (be) be.onclick = runExpr
    if (bb) bb.onclick = runBacktest
    if (bs) bs.onclick = showSummary
    const fit = document.getElementById('btnFit'); if (fit) fit.onclick = () => { try { window.__rf && window.__rf.fitView && window.__rf.fitView({ padding: 0.2 }) } catch {} }
    const clr = document.getElementById('btnClear'); if (clr) clr.onclick = () => { setNodes([]); setEdges([]) }
    const theme = document.getElementById('btnTheme'); if (theme) theme.onclick = () => {
      const root = document.documentElement
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
      root.setAttribute('data-theme', next)
      try { localStorage.setItem('am_theme', next) } catch {}
    }
    const layout = document.getElementById('btnLayout'); if (layout) layout.onclick = () => {
      try {
        if (!window.dagre) { alert('布局库未加载'); return }
        const g = new window.dagre.graphlib.Graph(); g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 80 }); g.setDefaultEdgeLabel(() => ({}))
        const byId = Object.fromEntries(nodes.map(n => [n.id, n]))
        nodes.forEach(n => g.setNode(n.id, { width: 220, height: 120 }))
        edges.forEach(e => { if (byId[e.source] && byId[e.target]) g.setEdge(e.source, e.target) })
        window.dagre.layout(g)
        const laid = nodes.map(n => { const p = g.node(n.id) || { x: n.position.x, y: n.position.y }; return { ...n, position: { x: p.x - 110, y: p.y - 60 } } })
        setNodes(laid)
      } catch (e) { console.warn('layout error', e); alert('布局失败: '+e) }
    }
    const exportBtn = document.getElementById('btnExport'); if (exportBtn) exportBtn.onclick = async () => {
      try {
        const el = document.querySelector('.canvas')
        if (!el || !window.html2canvas) return alert('找不到画布或导出库')
        const canvas = await window.html2canvas(el, { backgroundColor: null, scale: 2 })
        const url = canvas.toDataURL('image/png')
        const a = document.createElement('a'); a.href = url; a.download = 'flow.png'; a.click()
      } catch (e) { console.warn('export error', e); alert('导出失败: '+e) }
    }
    try { const saved = localStorage.getItem('am_theme'); if (saved) document.documentElement.setAttribute('data-theme', saved) } catch {}
    // brand select
    const brandSel = document.getElementById('brandSelect'); if (brandSel) {
      try {
        const savedBrand = localStorage.getItem('am_brand') || 'ocean'
        brandSel.value = savedBrand; document.documentElement.setAttribute('data-brand', savedBrand)
      } catch {}
      brandSel.onchange = () => {
        const v = brandSel.value || 'ocean'
        document.documentElement.setAttribute('data-brand', v)
        try { localStorage.setItem('am_brand', v) } catch {}
      }
    }
    // snap toggle
    const snapBtn = document.getElementById('btnSnap'); if (snapBtn) snapBtn.onclick = () => setSnap(s => !s)
    const gridInput = document.getElementById('gridSize'); if (gridInput) {
      gridInput.onchange = () => {
        const v = Math.max(2, parseInt(gridInput.value||'16',10)||16)
        setGrid([v,v])
      }
    }
    const snapStrength = document.getElementById('snapStrength'); if (snapStrength) {
      snapStrength.onchange = () => {
        const mode = snapStrength.value || 'medium'
        const base = Math.max(2, parseInt((document.getElementById('gridSize')?.value)||'16',10)||16)
        const v = mode === 'strong' ? Math.max(2, Math.round(base/2)) : (mode==='weak'? base*2 : base)
        setGrid([v,v])
      }
    }
    const distX = document.getElementById('btnDistX'); if (distX) distX.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 3) { alert('请选择3个以上节点'); return }
      const sorted = sels.slice().sort((a,b)=> (a.position?.x||0)-(b.position?.x||0))
      const minx = sorted[0].position.x, maxx = sorted[sorted.length-1].position.x
      const step = (maxx - minx) / (sorted.length - 1)
      const map = new Map(sorted.map((n,i)=> [n.id, minx + i*step]))
      setNodes(nds => nds.map(n => map.has(n.id) ? ({ ...n, position: { x: map.get(n.id), y: n.position.y } }) : n))
    }
    const distY = document.getElementById('btnDistY'); if (distY) distY.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 3) { alert('请选择3个以上节点'); return }
      const sorted = sels.slice().sort((a,b)=> (a.position?.y||0)-(b.position?.y||0))
      const miny = sorted[0].position.y, maxy = sorted[sorted.length-1].position.y
      const step = (maxy - miny) / (sorted.length - 1)
      const map = new Map(sorted.map((n,i)=> [n.id, miny + i*step]))
      setNodes(nds => nds.map(n => map.has(n.id) ? ({ ...n, position: { x: n.position.x, y: map.get(n.id) } }) : n))
    }
    // align buttons
    const ax = document.getElementById('btnAlignX'); if (ax) ax.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 2) return alert('请选择2个以上节点')
      const refY = sels[0].position?.y || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: n.position.x, y: refY } }) : n))
    }
    const ay = document.getElementById('btnAlignY'); if (ay) ay.onclick = () => {
      const sels = (nodes||[]).filter(n => n.selected); if (sels.length < 2) return alert('请选择2个以上节点')
      const refX = sels[0].position?.x || 0
      setNodes(nds => nds.map(n => n.selected ? ({ ...n, position: { x: refX, y: n.position.y } }) : n))
    }
    const btnFeat = document.getElementById('btnFeatTop')
    if (btnFeat) btnFeat.onclick = async () => {
      const file = document.getElementById('featureFile').value || 'user_data/freqai_features.json'
      const res = await fetch(`${API}/features/top?file=${encodeURIComponent(file)}&limit=10`)
      const data = await res.json()
      if (featTopEl) featTopEl.textContent = JSON.stringify(data, null, 2)
    }
    // 暴露给全局：用于拖拽添加节点
    window.__setNodes = (node) => setNodes(nds => nds.concat(node))
    // 绑定调色板的拖拽开始，写入 dataTransfer
    const paletteItems = Array.from(document.querySelectorAll('.palette-item'))
    const onDragStart = (e) => {
      const typeKey = e.target?.getAttribute?.('data-nodetype')
      if (!typeKey) return
      e.dataTransfer.setData('application/node-type', typeKey)
      e.dataTransfer.effectAllowed = 'move'
    }
    paletteItems.forEach(el => el.addEventListener('dragstart', onDragStart))
    const btnRunAgentFlow = document.getElementById('btnRunAgentFlow')
    if (btnRunAgentFlow) btnRunAgentFlow.onclick = async () => {
      const cfg = (document.getElementById('flowCfg')?.value || 'configs/agent_flow_multi.json')
      const stepsRaw = (document.getElementById('flowSteps')?.value || '').trim()
      const body = { config: cfg }
      if (stepsRaw) { body.steps = stepsRaw.split(/\s+/) }
      logsEl.textContent = ''
      const res = await fetch(`${API}/flow/run`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json()
      await pollLogs(data.job_id)
      await showSummary()
    }
    // 清理函数，防止重复绑定
    return () => {
      paletteItems.forEach(el => el.removeEventListener('dragstart', onDragStart))
    }
    const btnFeatPlot = document.getElementById('btnFeatPlot')
    if (btnFeatPlot) btnFeatPlot.onclick = async () => {
      const file = document.getElementById('featureFile').value || 'user_data/freqai_features.json'
      const res = await fetch(`${API}/features/top?file=${encodeURIComponent(file)}&limit=10`)
      const data = await res.json()
      const items = data.items || []
      const names = items.map(x => x.name)
      const vals = items.map(x => x.score)
      const chart = echarts.init(document.getElementById('featChart'))
      chart.setOption({
        grid: { left: 40, right: 16, top: 10, bottom: 60 },
        xAxis: { type: 'category', data: names, axisLabel: { rotate: 60 } },
        yAxis: { type: 'value', scale: true },
        tooltip: { trigger: 'axis' },
        series: [{ name: 'score', type: 'bar', data: vals }],
      })
    }
    const btnList = document.getElementById('btnList')
    if (btnList) btnList.onclick = async () => {
      const res = await fetch(`${API}/results/list`)
      const data = await res.json()
      if (data.items && data.items.length) {
        const [a,b] = data.items
        if (a) document.getElementById('resA').value = a.name
        if (b) document.getElementById('resB').value = b.name
      }
    }
    const btnCmp = document.getElementById('btnCompare')
    if (btnCmp) btnCmp.onclick = async () => {
      const a = (document.getElementById('resA').value||'').trim()
      const b = (document.getElementById('resB').value||'').trim()
      if (!a || !b) return alert('请填写结果文件名 A / B')
      const [sa, sb] = await Promise.all([
        fetch(`${API}/results/summary?name=${encodeURIComponent(a)}`).then(r => r.json()),
        fetch(`${API}/results/summary?name=${encodeURIComponent(b)}`).then(r => r.json()),
      ])
      function toSeries(summary, label) {
        const trades = Array.isArray(summary.trades) ? summary.trades.slice().sort((x,y)=> (x.open_timestamp||0)-(y.open_timestamp||0)) : []
        let cum = 0
        const xs = []
        const ys = []
        const dd = []
        let peak = 0
        for (const t of trades) { cum += Number(t.profit_abs||0); xs.push(new Date(t.open_timestamp||0).toISOString().slice(0,10)); ys.push(cum); peak = Math.max(peak, cum); dd.push(peak ? (cum - peak) : 0) }
        return { xs, ys, dd, label }
      }
      const A = toSeries(sa, 'A:'+a)
      const B = toSeries(sb, 'B:'+b)
      if (compareEl) {
        compareEl.innerHTML = '<div id="cmpEquity" style="height:220px;margin:6px 0;"></div><div id="cmpDD" style="height:160px;margin:6px 0 10px 0;"></div>'
        const eq = echarts.init(document.getElementById('cmpEquity'))
        const dd = echarts.init(document.getElementById('cmpDD'))
        const x = A.xs.length >= B.xs.length ? A.xs : B.xs
        eq.setOption({
          grid: { left: 40, right: 16, top: 10, bottom: 20 },
          xAxis: { type: 'category', data: x, boundaryGap: false },
          yAxis: { type: 'value', scale: true },
          tooltip: { trigger: 'axis' }, legend: {}, dataZoom: [{ type:'inside' }, { type:'slider' }],
          series: [ { name: A.label, type: 'line', data: A.ys, smooth: true }, { name: B.label, type: 'line', data: B.ys, smooth: true } ]
        })
        dd.setOption({
          grid: { left: 40, right: 16, top: 10, bottom: 16 },
          xAxis: { type: 'category', data: x, boundaryGap: false, axisLabel: { show: false } },
          yAxis: { type: 'value', scale: true },
          tooltip: { trigger: 'axis' }, legend: {}, dataZoom: [{ type:'inside' }, { type:'slider' }],
          series: [ { name: A.label, type: 'line', data: A.dd, smooth: true, areaStyle: {} }, { name: B.label, type: 'line', data: B.dd, smooth: true, areaStyle: {} } ]
        })
        try { echarts.connect([eq, dd]) } catch {}
        function metrics(s) {
          const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}
          return {
            profit_pct: c.profit_total_pct ?? s.profit_total_pct,
            profit_abs: c.profit_total_abs ?? s.profit_total_abs,
            trades: c.trades ?? s.trades,
            winrate: c.winrate ?? s.winrate,
            max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs,
          }
        }
        const ma = metrics(sa), mb = metrics(sb)
        const tbl = document.createElement('div')
        tbl.innerHTML = `<table border="1" cellspacing="0" cellpadding="4" style="width:100%; font-size:12px; margin-top:6px">
            <tr><th>指标</th><th>A (${a})</th><th>B (${b})</th></tr>
            <tr><td>总收益%</td><td>${ma.profit_pct ?? '--'}</td><td>${mb.profit_pct ?? '--'}</td></tr>
            <tr><td>总收益(USDT)</td><td>${ma.profit_abs ?? '--'}</td><td>${mb.profit_abs ?? '--'}</td></tr>
            <tr><td>交易数</td><td>${ma.trades ?? '--'}</td><td>${mb.trades ?? '--'}</td></tr>
            <tr><td>胜率</td><td>${ma.winrate ?? '--'}</td><td>${mb.winrate ?? '--'}</td></tr>
            <tr><td>最大回撤(USDT)</td><td>${ma.max_dd ?? '--'}</td><td>${mb.max_dd ?? '--'}</td></tr>
          </table>`
        compareEl.appendChild(tbl)
      }
      summaryEl.textContent = JSON.stringify({ A: sa, B: sb }, null, 2)
    }
    // 提供简单的点击模拟器，便于快速自测（在浏览器控制台运行：_simulateClicks('expr')）
    window._simulateClicks = async (what = 'expr') => {
      const map = {
        expr: 'btnExpr',
        backtest: 'btnBacktest',
        summary: 'btnSummary',
      }
      const id = map[what] || what
      const el = document.getElementById(id)
      if (!el) { console.warn('simulate: not found', id); return false }
      el.click()
      return true
    }
  }, [])

  // ReactFlow 画布的拖拽回调，确保可以把调色板节点丢到画布
  const onDragOver = useCallback((e) => {
    e.preventDefault()
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'move'
  }, [])
  const onDrop = useCallback((e) => {
    e.preventDefault()
    const pane = document.querySelector('.react-flow')
    const bounds = pane ? pane.getBoundingClientRect() : { left: 0, top: 0 }
    const typeKey = e.dataTransfer.getData('application/node-type') || ''
    if (!typeKey) return
    let x = e.clientX - bounds.left
    let y = e.clientY - bounds.top
    try {
      const vp = document.querySelector('.react-flow__viewport')
      const m = vp ? window.getComputedStyle(vp).transform : 'none'
      if (m && m !== 'none' && m.startsWith('matrix(')) {
        const nums = m.replace('matrix(', '').replace(')', '').split(',').map(parseFloat)
        const scale = nums[0] || 1
        const tx = nums[4] || 0
        const ty = nums[5] || 0
        x = (x - tx) / scale
        y = (y - ty) / scale
      }
    } catch {}
    const position = { x, y }
    const id = 'n' + Math.random().toString(16).slice(2,8)
    const labelMap = { data: 'Data', expr: 'Expression(LLM)', bt: 'Backtest', ml: 'Train(ML)', rl: 'Train(RL)', fb: 'Feedback', ho: 'Hyperopt', mv: 'MultiValidate' }
    const cfgMap = {
      data: { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' },
      expr: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' },
      ml: { config: 'configs/train_pytorch_mlp.json', inline: false, model: 'lightgbm', params: '{}', feature_file: 'user_data/freqai_features_multi.json', timeframe: '4h', pairs: 'BTC/USDT ETH/USDT', autobt: false },
      rl: { config: 'configs/train_ppo.json' },
      bt: { timerange: '20210101-20211231' },
      fb: { results_dir: 'user_data/backtest_results' },
      ho: { timerange: '20210101-20210430', spaces: 'buy sell protection', epochs: 20, loss: 'SharpeHyperOptLoss' },
      mv: { timeranges: '20210101-20210331,20210401-20210630' },
    }
    const node = { id, type: 'amNode', position, data: { label: labelMap[typeKey] || typeKey, typeKey, cfg: cfgMap[typeKey] || {} } }
    setNodes(nds => nds.concat(node))
  }, [])

  // Node helpers
  function genId(prefix='n') { return prefix + Math.random().toString(16).slice(2,8) }
  const onConnect = useCallback((params) => {
    setEdges((eds) => addEdgeLib({ id: genId('e'), ...params }, eds))
  }, [])
  const onNodesChange = useCallback((chs) => { setNodes((nds) => applyNodeChanges(chs, nds)) }, [])
  const onEdgesChange = useCallback((chs) => { setEdges((eds) => applyEdgeChanges(chs, eds)) }, [])
  const onNodeClick = useCallback((_, n) => { setSelected(n) }, [])

  function nodeCfgSchema(typeKey) {
    if (typeKey === 'data') return [
      { key: 'pairs', label: 'Pairs', def: 'BTC/USDT ETH/USDT' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
      { key: 'output', label: 'Output', def: 'user_data/freqai_features_multi.json' },
    ]
    if (typeKey === 'expr') return [
      { key: 'feature_file', label: 'Feature File', def: (document.getElementById('featureFile')?.value || 'user_data/freqai_features.json') },
      { key: 'llm_model', label: 'LLM Model', def: 'gpt-3.5-turbo' },
      { key: 'llm_count', label: 'LLM Count', def: 12, type: 'number' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
    ]
    if (typeKey === 'ml') return [
      { key: 'config', label: 'Config', def: 'configs/train_pytorch_mlp.json' },
      { key: 'inline', label: 'Inline(true/false)', def: false },
      { key: 'model', label: 'Model', def: 'lightgbm' },
      { key: 'params', label: 'Params(JSON)', def: '{}' },
      { key: 'feature_file', label: 'Feature File', def: 'user_data/freqai_features_multi.json' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
      { key: 'pairs', label: 'Pairs', def: 'BTC/USDT ETH/USDT' },
      { key: 'autobt', label: 'AutoBacktest(true/false)', def: false },
    ]
    if (typeKey === 'rl') return [
      { key: 'config', label: 'Config', def: 'configs/train_ppo.json' },
    ]
    if (typeKey === 'bt') return [
      { key: 'timerange', label: 'Timerange', def: '20210101-20211231' },
    ]
    if (typeKey === 'fb') return [
      { key: 'results_dir', label: 'Results Dir', def: 'user_data/backtest_results' },
    ]
    if (typeKey === 'ho') return [
      { key: 'timerange', label: 'Timerange', def: '20210101-20210430' },
      { key: 'spaces', label: 'Spaces', def: 'buy sell protection' },
      { key: 'epochs', label: 'Epochs', def: 20, type: 'number' },
      { key: 'loss', label: 'Loss', def: 'SharpeHyperOptLoss' },
    ]
    return []
  }

  function renderNodeForm() {
    const el = document.getElementById('nodeForm')
    if (!selected) { el.innerHTML = '<em>未选中节点</em>'; return }
    const typeKey = selected?.data?.typeKey
    if (!typeKey) { el.innerHTML = '<em>未知节点</em>'; return }
    const cfg = selected.data.cfg || {}
    const schema = nodeCfgSchema(typeKey)
    let html = `<div><b>${selected.data.label}</b> (${typeKey})</div>`
    schema.forEach(s => {
      const val = cfg[s.key] ?? s.def
      const t = s.type === 'number' ? 'number' : 'text'
      html += `<label>${s.label}</label><input data-k="${s.key}" type="${t}" value="${val}" />`
    })
    el.innerHTML = html
    Array.from(el.querySelectorAll('input[data-k]')).forEach(inp => {
      inp.addEventListener('change', (e) => {
        const k = e.target.getAttribute('data-k')
        const v = (e.target.type === 'number') ? Number(e.target.value) : e.target.value
        setNodes(nds => nds.map(n => n.id === selected.id ? ({...n, data: {...n.data, cfg: {...(n.data.cfg||{}), [k]: v }}}) : n))
        selected.data.cfg = {...selected.data.cfg, [k]: v}
      })
    })
  }

  useEffect(() => { renderNodeForm() }, [selected, nodes])

  function addNode(typeKey) {
    const id = genId()
    const label = typeKey === 'data' ? 'Data' : (typeKey === 'expr' ? 'Expression(LLM)' : (typeKey === 'bt' ? 'Backtest' : (typeKey === 'fb' ? 'Feedback' : (typeKey === 'ho' ? 'Hyperopt' : 'MultiValidate'))))
    const cfg = typeKey === 'expr' ? { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' }
      : (typeKey === 'bt' ? { timerange: '20210101-20211231' }
      : (typeKey === 'data' ? { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' }
      : (typeKey === 'fb' ? { results_dir: 'user_data/backtest_results' } : { timerange: '20210101-20210430', spaces: 'buy sell protection', epochs: 20, loss: 'SharpeHyperOptLoss' })))
    const node = { id, position: { x: 120 + Math.random()*320, y: 120 + Math.random()*180 }, data: { label, typeKey, cfg } }
    setNodes(nds => nds.concat(node))
  }

  async function runFlow() {
    logsEl.textContent = ''
    let feedbackPath = null
    // topo order (simple forward traversal)
    const id2node = Object.fromEntries(nodes.map(n => [n.id, n]))
    const incoming = {}
    nodes.forEach(n => incoming[n.id] = 0)
    edges.forEach(e => incoming[e.target] = (incoming[e.target]||0)+1)
    const queue = nodes.filter(n => (incoming[n.id]||0) === 0).map(n => n.id)
    const order = []
    const outs = edges.reduce((m,e) => { (m[e.source] ||= []).push(e.target); return m }, {})
    while (queue.length) {
      const id = queue.shift()
      order.push(id)
      for (const nb of (outs[id]||[])) {
        incoming[nb] -= 1
        if (incoming[nb] === 0) queue.push(nb)
      }
    }
    // execute in order: data -> expr -> bt -> fb
    for (const id of order) {
      const n = id2node[id]
      if (!n?.data?.typeKey) continue
      if (n.data.typeKey === 'data') {
        const body = {
          config: document.getElementById('cfg').value,
          output: n.data.cfg?.output || 'user_data/freqai_features_multi.json',
          timeframe: n.data.cfg?.timeframe || document.getElementById('timeframe').value,
          pairs: n.data.cfg?.pairs || 'BTC/USDT ETH/USDT',
        }
        const res = await fetch(`${API}/run/feature`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
      }
      if (n.data.typeKey === 'expr') {
        const body = {
          config: document.getElementById('cfg').value,
          feature_file: document.getElementById('featureFile').value,
          output: 'user_data/freqai_expressions.json',
          timeframe: n.data.cfg?.timeframe || '4h',
          llm_model: n.data.cfg?.llm_model || 'gpt-3.5-turbo',
          llm_count: Number(n.data.cfg?.llm_count || 12),
          llm_loops: 1,
          llm_timeout: 60,
          feedback_top: 0
          feedback: feedbackPath || undefined,
        }
        const res = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
      }
      if (n.data.typeKey === 'ml') {
        const cfg = n.data.cfg || {}
        let body
        const inline = String(cfg.inline).toLowerCase() === 'true' || cfg.inline === true
        if (inline) {
          let paramsObj = {}
          try { paramsObj = JSON.parse(cfg.params || '{}') } catch {}
          body = {
            config_obj: {
              data: { feature_file: cfg.feature_file || 'user_data/freqai_features_multi.json', data_dir: 'freqtrade/user_data/data', exchange: 'binanceus', timeframe: cfg.timeframe || '4h', pairs: String(cfg.pairs||'BTC/USDT').split(/\s+/) },
              model: { name: cfg.model || 'lightgbm', params: paramsObj },
              training: { validation_ratio: 0.2 },
              output: { model_dir: 'artifacts/models/inline_ml' },
            }
          }
        } else {
          body = { config: cfg.config || 'configs/train_pytorch_mlp.json' }
        }
        const res = await fetch(`${API}/run/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
        const autobt = String(cfg.autobt).toLowerCase() === 'true' || cfg.autobt === true
        if (autobt) {
          const btReq = { config: document.getElementById('cfg').value, strategy: 'ExpressionLongStrategy', strategy_path: 'freqtrade/user_data/strategies', timerange: document.getElementById('timerange').value, freqaimodel: 'LightGBMRegressor', export: true, export_filename: 'user_data/backtest_results/latest_trades_multi' }
          const rb = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(btReq) })
          const jb = await rb.json(); await pollLogs(jb.job_id); await showSummary()
        }
      }
      if (n.data.typeKey === 'rl') {
        const body = { config: n.data.cfg?.config || 'configs/train_ppo.json' }
        const res = await fetch(`${API}/run/rl_train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
      }
      if (n.data.typeKey === 'bt') {
        const body = {
          config: document.getElementById('cfg').value,
          strategy: 'ExpressionLongStrategy',
          strategy_path: 'freqtrade/user_data/strategies',
          timerange: n.data.cfg?.timerange || document.getElementById('timerange').value,
          freqaimodel: 'LightGBMRegressor',
          export: true,
          export_filename: 'user_data/backtest_results/latest_trades_multi',
        }
        const res = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
        await showSummary()
      }
      if (n.data.typeKey === 'fb') {
        const body = { results_dir: n.data.cfg?.results_dir || 'user_data/backtest_results', out: 'user_data/llm_feedback/latest_backtest_summary.json' }
        const res = await fetch(`${API}/results/prepare-feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const fb = await res.json()
        feedbackPath = fb.feedback_path || null
      }
      if (n.data.typeKey === 'ho') {
        const body = {
          config: document.getElementById('cfg').value,
          strategy: 'ExpressionLongStrategy',
          strategy_path: 'freqtrade/user_data/strategies',
          timerange: n.data.cfg?.timerange || '20210101-20210430',
          spaces: n.data.cfg?.spaces || 'buy sell protection',
          hyperopt_loss: n.data.cfg?.loss || 'SharpeHyperOptLoss',
          epochs: Number(n.data.cfg?.epochs || 20),
          freqaimodel: 'LightGBMRegressor',
          job_workers: -1,
        }
        const res = await fetch(`${API}/run/hyperopt`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        const data = await res.json()
        await pollLogs(data.job_id)
        // auto backtest after hyperopt
        const btReq = {
          config: document.getElementById('cfg').value,
          strategy: 'ExpressionLongStrategy',
          strategy_path: 'freqtrade/user_data/strategies',
          timerange: n.data.cfg?.timerange || '20210101-20210430',
          freqaimodel: 'LightGBMRegressor',
          export: true,
          export_filename: 'user_data/backtest_results/latest_trades_multi',
        }
        const btRes = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(btReq) })
        const btJob = await btRes.json()
        await pollLogs(btJob.job_id)
        await showSummary()
      }
    }
  }

  // 单节点执行：供自定义节点“运行”按钮调用
  async function runNodeById(nodeId) {
    const n = (nodes || []).find(x => x.id === nodeId)
    if (!n || !n.data || !n.data.typeKey) return
    let feedbackPath = null
    if (n.data.typeKey === 'data') {
      const body = {
        config: document.getElementById('cfg').value,
        output: n.data.cfg?.output || 'user_data/freqai_features_multi.json',
        timeframe: n.data.cfg?.timeframe || document.getElementById('timeframe').value,
        pairs: n.data.cfg?.pairs || 'BTC/USDT ETH/USDT',
      }
      const res = await fetch(`${API}/run/feature`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id)
    }
    if (n.data.typeKey === 'expr') {
      const body = {
        config: document.getElementById('cfg').value,
        feature_file: n.data.cfg?.feature_file || document.getElementById('featureFile').value,
        output: 'user_data/freqai_expressions.json',
        timeframe: n.data.cfg?.timeframe || '4h',
        llm_model: n.data.cfg?.llm_model || 'gpt-3.5-turbo',
        llm_count: Number(n.data.cfg?.llm_count || 12),
        llm_loops: 1,
        llm_timeout: 60,
        feedback_top: 0
        feedback: undefined,
      }
      const res = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id)
    }
    if (n.data.typeKey === 'bt') {
      const body = {
        config: document.getElementById('cfg').value,
        strategy: 'ExpressionLongStrategy',
        strategy_path: 'freqtrade/user_data/strategies',
        timerange: n.data.cfg?.timerange || document.getElementById('timerange').value,
        freqaimodel: 'LightGBMRegressor',
        export: true,
        export_filename: 'user_data/backtest_results/latest_trades_multi',
      }
      const res = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id); await showSummary()
    }
    if (n.data.typeKey === 'ml') {
      const cfg = n.data.cfg || {}
      let body
      const inline = String(cfg.inline).toLowerCase() === 'true' || cfg.inline === true
      if (inline) {
        let paramsObj = {}
        try { paramsObj = JSON.parse(cfg.params || '{}') } catch {}
        body = {
          config_obj: {
            data: { feature_file: cfg.feature_file || 'user_data/freqai_features_multi.json', data_dir: 'freqtrade/user_data/data', exchange: 'binanceus', timeframe: cfg.timeframe || '4h', pairs: String(cfg.pairs||'BTC/USDT').split(/\s+/) },
            model: { name: cfg.model || 'lightgbm', params: paramsObj },
            training: { validation_ratio: 0.2 },
            output: { model_dir: 'artifacts/models/inline_ml' },
          }
        }
      } else {
        body = { config: cfg.config || 'configs/train_pytorch_mlp.json' }
      }
      const res = await fetch(`${API}/run/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id)
      const autobt = String(cfg.autobt).toLowerCase() === 'true' || cfg.autobt === true
      if (autobt) {
        const btReq = { config: document.getElementById('cfg').value, strategy: 'ExpressionLongStrategy', strategy_path: 'freqtrade/user_data/strategies', timerange: document.getElementById('timerange').value, freqaimodel: 'LightGBMRegressor', export: true, export_filename: 'user_data/backtest_results/latest_trades_multi' }
        const rb = await fetch(`${API}/run/backtest`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(btReq) })
        const jb = await rb.json(); await pollLogs(jb.job_id); await showSummary()
      }
    }
    if (n.data.typeKey === 'rl') {
      const body = { config: n.data.cfg?.config || 'configs/train_ppo.json' }
      const res = await fetch(`${API}/run/rl_train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id)
    }
    if (n.data.typeKey === 'fb') {
      const body = { results_dir: n.data.cfg?.results_dir || 'user_data/backtest_results', out: 'user_data/llm_feedback/latest_backtest_summary.json' }
      const res = await fetch(`${API}/results/prepare-feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const fb = await res.json(); feedbackPath = fb.feedback_path || null
    }
    if (n.data.typeKey === 'ho') {
      const body = {
        config: document.getElementById('cfg').value,
        strategy: 'ExpressionLongStrategy',
        strategy_path: 'freqtrade/user_data/strategies',
        timerange: n.data.cfg?.timerange || '20210101-20210430',
        spaces: n.data.cfg?.spaces || 'buy sell protection',
        hyperopt_loss: n.data.cfg?.loss || 'SharpeHyperOptLoss',
        epochs: Number(n.data.cfg?.epochs || 20),
        freqaimodel: 'LightGBMRegressor',
        job_workers: -1,
      }
      const res = await fetch(`${API}/run/hyperopt`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const data = await res.json(); await pollLogs(data.job_id)
    }
  }

  useEffect(() => { window.__runNode = (id) => runNodeById(id) }, [nodes])
  useEffect(() => { window.__configureNode = (id) => { try { const n = (nodes||[]).find(x=>x.id===id); if (!n) return; setSelected(n); document.getElementById('nodeForm')?.scrollIntoView({ behavior: 'smooth', block: 'center' }) } catch {} } }, [nodes])

  function showCtxMenu(x, y, target) {
    const el = document.getElementById('ctxMenu'); if (!el) return
    const items = []
    if (target.type === 'node') {
      items.push({ k: 'copyNode', label: '复制节点' })
      items.push({ k: 'toggleLock', label: '锁定/解锁' })
      items.push({ k: 'deleteNode', label: '删除节点' })
    } else if (target.type === 'edge') {
      items.push({ k: 'deleteEdge', label: '删除连线' })
    }
    el.innerHTML = items.map(it => `<div class="item" data-k="${it.k}">${it.label}</div>`).join('')
    el.style.left = `${x}px`; el.style.top = `${y}px`; el.style.display = 'block'
    Array.from(el.querySelectorAll('.item')).forEach(item => {
      item.addEventListener('click', () => {
        const k = item.getAttribute('data-k')
        if (k === 'deleteNode') {
          setNodes(nds => nds.filter(n => n.id !== target.id))
          setEdges(eds => eds.filter(e => e.source !== target.id && e.target !== target.id))
        }
        if (k === 'copyNode') {
          const src = (nodes||[]).find(n => n.id === target.id); if (!src) return
          const id = genId(); const pos = { x: (src.position?.x||0) + 40, y: (src.position?.y||0) + 40 }
          const clone = { ...src, id, position: pos }
          setNodes(nds => nds.concat(clone))
        }
        if (k === 'toggleLock') {
          setNodes(nds => nds.map(n => n.id === target.id ? ({ ...n, draggable: !(n.draggable === false), data: { ...(n.data||{}), locked: !(n.draggable === false) ? true : false } }) : n))
        }
        if (k === 'deleteEdge') {
          setEdges(eds => eds.filter(e => e.id !== target.id))
        }
        el.style.display = 'none'
      })
    })
  }

  function saveFlow() {
    const payload = { nodes, edges }
    localStorage.setItem('agent_market_flow', JSON.stringify(payload))
    alert('已保存到 localStorage')
  }
  function loadFlow() {
    const raw = localStorage.getItem('agent_market_flow')
    if (!raw) return alert('没有保存的 flow')
    try {
      const obj = JSON.parse(raw)
      setNodes(obj.nodes||[])
      setEdges(obj.edges||[])
    } catch(e) { alert('解析失败:'+e) }
  }

  useEffect(() => {
    const run = document.getElementById('runFlow'); if (run) run.onclick = runFlow
    const save = document.getElementById('saveFlow'); if (save) save.onclick = saveFlow
    const load = document.getElementById('loadFlow'); if (load) load.onclick = loadFlow
    const del = document.getElementById('delNode'); if (del) del.onclick = () => {
      if (!selected) return;
      setNodes(nds => nds.filter(n => n.id !== selected.id))
      setEdges(eds => eds.filter(e => e.source !== selected.id && e.target !== selected.id))
      setSelected(null)
    }
  }, [])

  const onNodeContextMenu = useCallback((e, n) => {
    e.preventDefault(); showCtxMenu(e.clientX, e.clientY, { type: 'node', id: n.id })
  }, [])
  const onEdgeContextMenu = useCallback((e, eobj) => {
    e.preventDefault(); showCtxMenu(e.clientX, e.clientY, { type: 'edge', id: eobj.id })
  }, [])

  return h(RF, { nodes, edges, nodeTypes, defaultEdgeOptions, fitView: true, onConnect, onNodesChange, onEdgesChange, onNodeClick, onDrop, onDragOver,
    onNodeContextMenu, onEdgeContextMenu,
    panOnDrag: [0,1,2], selectionOnDrag: false, panOnScroll: false, zoomOnScroll: true, zoomOnPinch: true,
    snapToGrid: !!snap, snapGrid: grid,
    nodesDraggable: true, nodesConnectable: true, elementsSelectable: true, onInit: (inst) => { window.__rf = inst } }, [
    h(Background, { variant: 'dots', gap: 16, size: 1, key: 'bg' }),
    h(Controls, { key: 'ctl' }),
    h(MiniMap, { key: 'mm' }),
  ])
}

createRoot(document.getElementById('root')).render(h(App))










function addNodeAt(typeKey, position){ const id='n'+Math.random().toString(16).slice(2,8); const labelMap={data:'Data',expr:'Expression(LLM)',bt:'Backtest',fb:'Feedback',ho:'Hyperopt',mv:'MultiValidate'}; const cfgMap={data:{pairs:'BTC/USDT ETH/USDT',timeframe:'4h',output:'user_data/freqai_features_multi.json'},expr:{llm_model:'gpt-3.5-turbo',llm_count:12,timeframe:'4h'},bt:{timerange:'20210101-20211231'},fb:{results_dir:'user_data/backtest_results'},ho:{timerange:'20210101-20210430',spaces:'buy sell protection',epochs:20,loss:'SharpeHyperOptLoss'},mv:{timeranges:'20210101-20210331,20210401-20210630'}}; const node={id,type:'amNode',position,data:{label:labelMap[typeKey]||typeKey,typeKey,cfg:cfgMap[typeKey]||{}}}; window.__setNodes && window.__setNodes(node); }








