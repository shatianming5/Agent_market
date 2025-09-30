import { createElement as h, useState, useEffect, useCallback } from 'https://unpkg.com/react@18/umd/react.development.js'
import { createRoot } from 'https://unpkg.com/react-dom@18/umd/react-dom.development.js'
import ReactFlow, { Background, Controls, MiniMap } from 'https://unpkg.com/reactflow@11.10.2/dist/standalone.js'

const API = 'http://127.0.0.1:8000'

function App() {
  const [nodes, setNodes] = useState([
    { id: 'n1', position: { x: 50, y: 100 }, data: { label: 'Data', typeKey: 'data', cfg: { pairs: 'BTC/USDT ETH/USDT SOL/USDT ADA/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' } }, type: 'input' },
    { id: 'n2', position: { x: 300, y: 100 }, data: { label: 'Expression(LLM)', typeKey: 'expr', cfg: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' } } },
    { id: 'n3', position: { x: 560, y: 100 }, data: { label: 'Backtest', typeKey: 'bt', cfg: { timerange: '20210101-20211231' } }, type: 'output' },
    { id: 'n4', position: { x: 820, y: 100 }, data: { label: 'Feedback', typeKey: 'fb', cfg: { results_dir: 'user_data/backtest_results' } } },
    { id: 'n5', position: { x: 1060, y: 100 }, data: { label: 'Hyperopt', typeKey: 'ho', cfg: { timerange: '20210101-20210430', spaces: 'buy sell protection', epochs: 20, loss: 'SharpeHyperOptLoss' } } },
  ])
  const [edges, setEdges] = useState([
    { id: 'e1', source: 'n1', target: 'n2' },
    { id: 'e2', source: 'n2', target: 'n3' },
  ])
  const [selected, setSelected] = useState(null)

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
      feedback_top: 0,
      llm_api_key: document.getElementById('apiKey').value || undefined,
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
          series: [{ name: 'ç´¯ç§¯æ”¶ç›Š(USDT)', type: 'line', data: ys }],
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
    const btnFeat = document.getElementById('btnFeatTop')
    if (btnFeat) btnFeat.onclick = async () => {
      const file = document.getElementById('featureFile').value || 'user_data/freqai_features.json'
      const res = await fetch(`${API}/features/top?file=${encodeURIComponent(file)}&limit=10`)
      const data = await res.json()
      if (featTopEl) featTopEl.textContent = JSON.stringify(data, null, 2)
    }
    // 暴露给全局：用于拖拽添加节点
    window.__setNodes = (node) => {
      setNodes(nds => nds.concat(node))
    }
    // 为画布绑定拖拽事件
    const rf = document.querySelector('.react-flow')
    if (rf) {
      const onOver = (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move' }
      const onDrop = (e) => {
        e.preventDefault()
        const typeKey = e.dataTransfer.getData('application/node-type')
        if (!typeKey) return
        const bounds = rf.getBoundingClientRect()
        const position = { x: e.clientX - bounds.left, y: e.clientY - bounds.top }
        const id = 'n' + Math.random().toString(16).slice(2,8)
        const labelMap = { data: 'Data', expr: 'Expression(LLM)', bt: 'Backtest', fb: 'Feedback', ho: 'Hyperopt', mv: 'MultiValidate' }
        const cfgMap = {
          data: { pairs: 'BTC/USDT ETH/USDT', timeframe: '4h', output: 'user_data/freqai_features_multi.json' },
          expr: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' },
          bt: { timerange: '20210101-20211231' },
          fb: { results_dir: 'user_data/backtest_results' },
          ho: { timerange: '20210101-20210430', spaces: 'buy sell protection', epochs: 20, loss: 'SharpeHyperOptLoss' },
          mv: { timeranges: '20210101-20210331,20210401-20210630' },
        }
        const node = { id, position, data: { label: labelMap[typeKey] || typeKey, typeKey, cfg: cfgMap[typeKey] || {} } }
        setNodes(nds => nds.concat(node))
      }
      rf.addEventListener('dragover', onOver)
      rf.addEventListener('drop', onDrop)
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
      const chart = echarts.init(document.getElementById('chart'))
      function toSeries(summary, label) {
        const trades = Array.isArray(summary.trades) ? summary.trades.slice().sort((x,y)=> (x.open_timestamp||0)-(y.open_timestamp||0)) : []
        let cum = 0
        const xs = []
        const ys = []
        for (const t of trades) { cum += Number(t.profit_abs||0); xs.push(new Date(t.open_timestamp||0).toISOString().slice(0,10)); ys.push(cum) }
        return { xs, ys, label }
      }
      const A = toSeries(sa, 'A:'+a)
      const B = toSeries(sb, 'B:'+b)
      const x = A.xs.length >= B.xs.length ? A.xs : B.xs
      chart.setOption({
        grid: { left: 40, right: 16, top: 10, bottom: 30 },
        xAxis: { type: 'category', data: x, axisLabel: { rotate: 45 } },
        yAxis: { type: 'value', scale: true },
        tooltip: { trigger: 'axis' },
        legend: {},
        series: [
          { name: A.label, type: 'line', data: A.ys },
          { name: B.label, type: 'line', data: B.ys },
        ],
      })
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
      const ma = metrics(sa)
      const mb = metrics(sb)
      if (compareEl) {
        compareEl.innerHTML = `
          <table border="1" cellspacing="0" cellpadding="4" style="width:100%; font-size:12px">
            <tr><th>指标</th><th>A (${a})</th><th>B (${b})</th></tr>
            <tr><td>总收益%</td><td>${ma.profit_pct ?? '--'}</td><td>${mb.profit_pct ?? '--'}</td></tr>
            <tr><td>总收益(USDT)</td><td>${ma.profit_abs ?? '--'}</td><td>${mb.profit_abs ?? '--'}</td></tr>
            <tr><td>交易数</td><td>${ma.trades ?? '--'}</td><td>${mb.trades ?? '--'}</td></tr>
            <tr><td>胜率</td><td>${ma.winrate ?? '--'}</td><td>${mb.winrate ?? '--'}</td></tr>
            <tr><td>最大回撤(USDT)</td><td>${ma.max_dd ?? '--'}</td><td>${mb.max_dd ?? '--'}</td></tr>
          </table>`
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

  // Node helpers
  function genId(prefix='n') { return prefix + Math.random().toString(16).slice(2,8) }
  const onConnect = useCallback((params) => {
    setEdges((eds) => eds.concat({ id: genId('e'), ...params }))
  }, [])
  const onNodesChange = useCallback((chs) => {}, [])
  const onEdgesChange = useCallback((chs) => {}, [])
  const onNodeClick = useCallback((_, n) => { setSelected(n) }, [])

  function nodeCfgSchema(typeKey) {
    if (typeKey === 'data') return [
      { key: 'pairs', label: 'Pairs', def: 'BTC/USDT ETH/USDT' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
      { key: 'output', label: 'Output', def: 'user_data/freqai_features_multi.json' },
    ]
    if (typeKey === 'expr') return [
      { key: 'llm_model', label: 'LLM Model', def: 'gpt-3.5-turbo' },
      { key: 'llm_count', label: 'LLM Count', def: 12, type: 'number' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
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
    if (!selected) { el.innerHTML = '<em>æœªé€‰ä¸­èŠ‚ç‚¹</em>'; return }
    const typeKey = selected?.data?.typeKey
    if (!typeKey) { el.innerHTML = '<em>æœªçŸ¥èŠ‚ç‚¹</em>'; return }
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
          feedback_top: 0,
          llm_api_key: document.getElementById('apiKey').value || undefined,
          feedback: feedbackPath || undefined,
        }
        const res = await fetch(`${API}/run/expression`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
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

  function saveFlow() {
    const payload = { nodes, edges }
    localStorage.setItem('agent_market_flow', JSON.stringify(payload))
    alert('å·²ä¿å­˜åˆ° localStorage')
  }
  function loadFlow() {
    const raw = localStorage.getItem('agent_market_flow')
    if (!raw) return alert('æ²¡æœ‰ä¿å­˜çš„ flow')
    try {
      const obj = JSON.parse(raw)
      setNodes(obj.nodes||[])
      setEdges(obj.edges||[])
    } catch(e) { alert('è§£æžå¤±è´¥:'+e) }
  }

  useEffect(() => {
    document.getElementById('addData').onclick = () => addNode('data')
    document.getElementById('addExpr').onclick = () => addNode('expr')
    document.getElementById('addBt').onclick = () => addNode('bt')
    document.getElementById('addHo').onclick = () => addNode('ho')
    document.getElementById('addFb').onclick = () => addNode('fb')
    document.getElementById('runFlow').onclick = runFlow
    document.getElementById('saveFlow').onclick = saveFlow
    document.getElementById('loadFlow').onclick = loadFlow
  }, [])

  return h(ReactFlow, { nodes, edges, fitView: true, onConnect, onNodesChange, onEdgesChange, onNodeClick, onDrop, onDragOver }, [
    h(Background, { variant: 'dots', gap: 16, size: 1, key: 'bg' }),
    h(Controls, { key: 'ctl' }),
    h(MiniMap, { key: 'mm' }),
  ])
}

createRoot(document.getElementById('root')).render(h(App))










function addNodeAt(typeKey, position){ const id='n'+Math.random().toString(16).slice(2,8); const labelMap={data:'Data',expr:'Expression(LLM)',bt:'Backtest',fb:'Feedback',ho:'Hyperopt',mv:'MultiValidate'}; const cfgMap={data:{pairs:'BTC/USDT ETH/USDT',timeframe:'4h',output:'user_data/freqai_features_multi.json'},expr:{llm_model:'gpt-3.5-turbo',llm_count:12,timeframe:'4h'},bt:{timerange:'20210101-20211231'},fb:{results_dir:'user_data/backtest_results'},ho:{timerange:'20210101-20210430',spaces:'buy sell protection',epochs:20,loss:'SharpeHyperOptLoss'},mv:{timeranges:'20210101-20210331,20210401-20210630'}}; const node={id,position,data:{label:labelMap[typeKey]||typeKey,typeKey,cfg:cfgMap[typeKey]||{}}}; window.__setNodes && window.__setNodes(node); }
