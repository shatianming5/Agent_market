import React, { useEffect, useState } from 'react'
import { API_BASE, Agent, Order, getJSON, postJSON, postEmpty } from './api'

export default function App() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [name, setName] = useState('demo-agent')
  const [selected, setSelected] = useState<Agent | null>(null)
  const [orders, setOrders] = useState<Order[]>([])
  const [qty, setQty] = useState(1)
  const [side, setSide] = useState<'buy' | 'sell'>('buy')
  const [dlTimeframe, setDlTimeframe] = useState('4h')
  const [dlTimerange, setDlTimerange] = useState('20200701-20210131')
  const [dlPairs, setDlPairs] = useState('BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT')
  const [trainPairs, setTrainPairs] = useState('BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT')

  type JobStatus = { id: string; running: boolean; returncode: number | null; lines: number; started_at?: string; finished_at?: string }
  const [jobs, setJobs] = useState<JobStatus[]>([])
  const [selectedJob, setSelectedJob] = useState<JobStatus | null>(null)
  const [jobLogs, setJobLogs] = useState<{ line: number; text: string }[]>([])
  const [es, setEs] = useState<EventSource | null>(null)

  function parseStep(logs: string[]) {
    const re = /^\[STEP\]\s+\d{2}:\d{2}:\d{2}\s+(\d+)\/(\d+)\s+(.+)$/
    let current = 0, total = 0, label = ''
    for (const line of logs) {
      const m = line.match(re)
      if (m) {
        current = parseInt(m[1], 10)
        total = parseInt(m[2], 10)
        label = m[3]
      }
    }
    return { current, total, label }
  }

  async function refreshAgents() {
    const data = await getJSON<Agent[]>('/agents/')
    setAgents(data)
  }

  async function createAgent() {
    const agent = await postJSON<Agent>('/agents/', { name })
    setName('')
    setSelected(agent)
    await refreshAgents()
  }

  async function loadAgent(agent: Agent) {
    const detail = await getJSON<Agent>(`/agents/${agent.id}`)
    setSelected(detail)
    const list = await getJSON<Order[]>(`/orders/?agent_id=${agent.id}`)
    setOrders(list)
  }

  async function createOrder() {
    if (!selected) return
    await postJSON<Order>('/orders/', { agent_id: selected.id, side, qty })
    const list = await getJSON<Order[]>(`/orders/?agent_id=${selected.id}`)
    setOrders(list)
  }

  async function refreshJobs() {
    const list = await getJSON<JobStatus[]>('/jobs')
    setJobs(list.sort((a,b) => (b.started_at || '').localeCompare(a.started_at || '')))
  }

  async function openJob(job: JobStatus) {
    setSelectedJob(job)
    // close previous stream if any
    if (es) { try { es.close() } catch {} }
    setJobLogs([])
    // try SSE first
    try {
      const stream = new EventSource(`${API_BASE}/jobs/${job.id}/stream?offset=0`)
      stream.onmessage = (ev) => {
        setJobLogs(prev => [...prev, { line: prev.length, text: ev.data }])
      }
      stream.addEventListener('end', () => {
        stream.close()
      })
      stream.onerror = () => {
        stream.close()
      }
      setEs(stream)
    } catch {
      // fallback to one-shot fetch
      const res = await getJSON<{ entries: { line: number; text: string }[] }>(`/jobs/${job.id}/logs?offset=0&limit=500&structured=true`)
      setJobLogs(res.entries || [])
    }
  }

  async function runExpression() {
    await postJSON('/run/expression', {
      config: 'configs/config_freqai_multi.json',
      feature_file: 'user_data/freqai_features.json',
      output: 'user_data/freqai_expressions.json',
      timeframe: '1h',
      llm_model: 'gpt-3.5-turbo',
      llm_count: 1,
      llm_loops: 1,
      llm_timeout: 5,
      feedback_top: 0,
      fast: true,
    })
    await refreshJobs()
  }

  async function runBacktest() {
    await postJSON('/run/backtest', {
      config: 'configs/config_freqai_multi.json',
      strategy: 'ExpressionLongStrategy',
      strategy_path: 'freqtrade/user_data/strategies',
      timerange: '20200701-20210131',
      freqaimodel: 'LightGBMRegressor',
      export: false,
      fast: true,
    })
    await refreshJobs()
  }

  async function runDownload() {
    const pairs = dlPairs.split(',').map(s => s.trim()).filter(Boolean)
    await postJSON('/run/download-data', {
      config: 'configs/config_freqai_multi.json',
      timeframes: [dlTimeframe],
      timerange: dlTimerange,
      pairs,
    })
    await refreshJobs()
  }

  async function runTrainML() {
    const pairs = trainPairs.split(',').map(s => s.trim()).filter(Boolean)
    const cfg = {
      model: { name: 'lightgbm', params: {} },
      data: {
        feature_file: 'user_data/freqai_features_multi.json',
        data_dir: 'freqtrade/user_data/data',
        exchange: 'binanceus',
        timeframe: dlTimeframe,
        pairs,
        label_period: 18,
      },
      training: { validation_ratio: 0.2 },
      output: { model_dir: 'artifacts/models/lightgbm_multi' },
    }
    await postJSON('/run/train-ml', { config: cfg })
    await refreshJobs()
  }

  async function terminateJob() {
    if (!selectedJob) return
    await postEmpty(`/jobs/${selectedJob.id}/terminate`)
    await refreshJobs()
  }

  useEffect(() => {
    refreshAgents()
  }, [])

  useEffect(() => {
    refreshJobs().catch(() => {})
    const t = setInterval(() => refreshJobs().catch(() => {}), 4000)
    return () => clearInterval(t)
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <h1>Agent Market Web</h1>
      <p>API: {API_BASE}</p>

      <section style={{ display: 'flex', gap: 24 }}>
        <div style={{ flex: 1 }}>
          <h2>Agents</h2>
          <div style={{ marginBottom: 8 }}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="agent name" />
            <button onClick={createAgent} disabled={!name.trim()}>Create</button>
          </div>
          <ul>
            {agents.map(a => (
              <li key={a.id}>
                <button onClick={() => loadAgent(a)}>{a.name}</button>
                <small style={{ marginLeft: 8, color: '#555' }}>{a.id}</small>
              </li>
            ))}
          </ul>
        </div>

        <div style={{ flex: 2 }}>
          <h2>Agent Detail</h2>
          {selected ? (
            <div>
              <div><b>{selected.name}</b> (id: {selected.id})</div>
              <div style={{ marginTop: 8 }}>
                <select value={side} onChange={(e) => setSide(e.target.value as any)}>
                  <option value="buy">buy</option>
                  <option value="sell">sell</option>
                </select>
                <input type="number" value={qty} onChange={(e) => setQty(parseFloat(e.target.value))} min={0} step={0.1} style={{ width: 100, marginLeft: 8 }} />
                <button onClick={createOrder} style={{ marginLeft: 8 }}>Place Order</button>
              </div>
              <h3 style={{ marginTop: 16 }}>Orders</h3>
              <table>
                <thead>
                  <tr><th>ID</th><th>Side</th><th>Qty</th><th>Status</th><th>Created</th></tr>
                </thead>
                <tbody>
                  {orders.map(o => (
                    <tr key={o.id}>
                      <td>{o.id}</td>
                      <td>{o.side}</td>
                      <td>{o.qty}</td>
                      <td>{o.status}</td>
                      <td>{o.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div>Select an agent to view details</div>
          )}
        </div>
        <div style={{ flex: 2 }}>
          <h2>Tasks</h2>
          <div style={{ marginBottom: 8 }}>
            <button onClick={runExpression}>Run Expression</button>
            <button onClick={runBacktest} style={{ marginLeft: 8 }}>Run Backtest</button>
            <button onClick={runDownload} style={{ marginLeft: 8 }}>Download Data</button>
            <button onClick={runTrainML} style={{ marginLeft: 8 }}>Train ML</button>
            <button onClick={refreshJobs} style={{ marginLeft: 8 }}>Refresh</button>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
            <label>Timeframe:</label>
            <input style={{ width: 80 }} value={dlTimeframe} onChange={e => setDlTimeframe(e.target.value)} />
            <label>Timerange:</label>
            <input style={{ width: 180 }} value={dlTimerange} onChange={e => setDlTimerange(e.target.value)} />
            <label>Pairs (comma):</label>
            <input style={{ width: 320 }} value={dlPairs} onChange={e => setDlPairs(e.target.value)} />
            <label>Train Pairs:</label>
            <input style={{ width: 320 }} value={trainPairs} onChange={e => setTrainPairs(e.target.value)} />
          </div>
          <div style={{ display: 'flex', gap: 16 }}>
            <div style={{ flex: 1 }}>
              <h3>Jobs</h3>
              <ul>
                {jobs.map(j => (
                  <li key={j.id}>
                    <button onClick={() => openJob(j)}>
                      {j.id} {j.running ? '(running)' : `(rc=${j.returncode})`}
                    </button>
                    <small style={{ marginLeft: 8, color: '#777' }}>lines={j.lines}</small>
                  </li>
                ))}
              </ul>
            </div>
            <div style={{ flex: 1 }}>
          <h3>Logs {selectedJob ? `(${selectedJob.id})` : ''}</h3>
          {selectedJob ? (() => {
            const logs = jobLogs.map(j => j.text)
            const step = parseStep(logs)
            const pct = step.total > 0 ? Math.min(100, Math.floor(step.current / step.total * 100)) : 0
            // elapsed from job started_at
            let elapsed = ''
            try {
              const started = (selectedJob as any).started_at ? new Date((selectedJob as any).started_at).getTime() : null
              if (started) {
                const sec = Math.floor((Date.now() - started) / 1000)
                elapsed = `${sec}s`
              }
            } catch {}
            return (
              <div style={{ margin: '8px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <div>Step: {step.current}/{step.total} {step.label}</div>
                  <div>Elapsed: {elapsed}</div>
                </div>
                <div style={{ height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden', marginTop: 4 }}>
                  <div style={{ width: pct + '%', height: '100%', background: '#4caf50' }} />
                </div>
              </div>
            )
          })() : null}
          {selectedJob ? (
            <div style={{ marginBottom: 8 }}>
              <button onClick={terminateJob}>Terminate</button>
            </div>
          ) : null}
              <div style={{ height: 240, overflow: 'auto', border: '1px solid #ddd', padding: 8 }}>
                <pre style={{ margin: 0 }}>
                  {jobLogs.map(e => `${e.line}: ${e.text}`).join('\n')}
                </pre>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
