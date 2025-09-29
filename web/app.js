import { createElement as h, useState, useMemo } from 'https://unpkg.com/react@18/umd/react.development.js'
import { createRoot } from 'https://unpkg.com/react-dom@18/umd/react-dom.development.js'
import ReactFlow, { Background, Controls, MiniMap } from 'https://unpkg.com/reactflow@11.10.2/dist/standalone.js'

const API = 'http://127.0.0.1:8000'

function App() {
  const [nodes, setNodes] = useState([
    { id: 'data', position: { x: 50, y: 80 }, data: { label: 'Data' }, type: 'input' },
    { id: 'expr', position: { x: 260, y: 80 }, data: { label: 'Expression(LLM)' } },
    { id: 'bt', position: { x: 470, y: 80 }, data: { label: 'Backtest' }, type: 'output' },
  ])
  const [edges, setEdges] = useState([
    { id: 'e1', source: 'data', target: 'expr' },
    { id: 'e2', source: 'expr', target: 'bt' },
  ])

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

  return h(ReactFlow, { nodes, edges, fitView: true }, [
    h(Background, { variant: 'dots', gap: 16, size: 1, key: 'bg' }),
    h(Controls, { key: 'ctl' }),
    h(MiniMap, { key: 'mm' }),
  ])
}

createRoot(document.getElementById('root')).render(h(App))

