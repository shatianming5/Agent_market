import { createElement as h, useState, useMemo, useEffect, useCallback } from 'https://unpkg.com/react@18/umd/react.development.js'
import { createRoot } from 'https://unpkg.com/react-dom@18/umd/react-dom.development.js'
import ReactFlow, { Background, Controls, MiniMap } from 'https://unpkg.com/reactflow@11.10.2/dist/standalone.js'

const API = 'http://127.0.0.1:8000'

function App() {
  const [nodes, setNodes] = useState([
    { id: 'n1', position: { x: 50, y: 100 }, data: { label: 'Data', typeKey: 'data', cfg: { } }, type: 'input' },
    { id: 'n2', position: { x: 300, y: 100 }, data: { label: 'Expression(LLM)', typeKey: 'expr', cfg: { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' } } },
    { id: 'n3', position: { x: 560, y: 100 }, data: { label: 'Backtest', typeKey: 'bt', cfg: { timerange: '20210101-20211231' } }, type: 'output' },
  ])
  const [edges, setEdges] = useState([
    { id: 'e1', source: 'n1', target: 'n2' },
    { id: 'e2', source: 'n2', target: 'n3' },
  ])
  const [selected, setSelected] = useState(null)

  const logsEl = document.getElementById('logs')
  const summaryEl = document.getElementById('summary')

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
  }

  useMemo(() => {
    document.getElementById('btnExpr').onclick = runExpr
    document.getElementById('btnBacktest').onclick = runBacktest
    document.getElementById('btnSummary').onclick = showSummary
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
    if (typeKey === 'data') return []
    if (typeKey === 'expr') return [
      { key: 'llm_model', label: 'LLM Model', def: 'gpt-3.5-turbo' },
      { key: 'llm_count', label: 'LLM Count', def: 12, type: 'number' },
      { key: 'timeframe', label: 'Timeframe', def: '4h' },
    ]
    if (typeKey === 'bt') return [
      { key: 'timerange', label: 'Timerange', def: '20210101-20211231' },
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
    const label = typeKey === 'data' ? 'Data' : (typeKey === 'expr' ? 'Expression(LLM)' : 'Backtest')
    const cfg = typeKey === 'expr' ? { llm_model: 'gpt-3.5-turbo', llm_count: 12, timeframe: '4h' } : (typeKey === 'bt' ? { timerange: '20210101-20211231' } : {})
    const node = { id, position: { x: 120 + Math.random()*320, y: 120 + Math.random()*180 }, data: { label, typeKey, cfg } }
    setNodes(nds => nds.concat(node))
  }

  async function runFlow() {
    logsEl.textContent = ''
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
    // execute expr -> backtest in order
    for (const id of order) {
      const n = id2node[id]
      if (!n?.data?.typeKey) continue
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
    }
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
    document.getElementById('addData').onclick = () => addNode('data')
    document.getElementById('addExpr').onclick = () => addNode('expr')
    document.getElementById('addBt').onclick = () => addNode('bt')
    document.getElementById('runFlow').onclick = runFlow
    document.getElementById('saveFlow').onclick = saveFlow
    document.getElementById('loadFlow').onclick = loadFlow
  }, [])

  return h(ReactFlow, { nodes, edges, fitView: true, onConnect, onNodesChange, onEdgesChange, onNodeClick }, [
    h(Background, { variant: 'dots', gap: 16, size: 1, key: 'bg' }),
    h(Controls, { key: 'ctl' }),
    h(MiniMap, { key: 'mm' }),
  ])
}

createRoot(document.getElementById('root')).render(h(App))
