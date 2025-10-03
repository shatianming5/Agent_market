import React, { useEffect, useState } from 'react'
import { API_BASE, getJSON, postJSON, postEmpty } from './api'
import DataManager from './components/DataManager'
import ExpressionsManager from './components/ExpressionsManager'
import JobsPanel from './components/JobsPanel'
import StrategyParams from './components/StrategyParams'

export default function App() {
  const [activeTab, setActiveTab] = useState<'data' | 'expressions' | 'backtest' | 'jobs'>('data')
  const [dlTimeframe, setDlTimeframe] = useState('4h')
  const [dlTimerange, setDlTimerange] = useState('20200701-20210131')
  const [dlPairs, setDlPairs] = useState('BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT')

  type JobStatus = { id: string; running: boolean; returncode: number | null; lines: number; started_at?: string; finished_at?: string }
  const [jobs, setJobs] = useState<JobStatus[]>([])
  const [selectedJob, setSelectedJob] = useState<JobStatus | null>(null)
  const [jobLogs, setJobLogs] = useState<{ line: number; text: string }[]>([])
  const [es, setEs] = useState<EventSource | null>(null)
  const [btJobId, setBtJobId] = useState<string | null>(null)
  const [btPct, setBtPct] = useState(0)
  const [btLabel, setBtLabel] = useState('')
  const [btRunning, setBtRunning] = useState(false)
  const [btElapsed, setBtElapsed] = useState<number | undefined>(undefined)
  const [flowJobId, setFlowJobId] = useState<string | null>(null)
  const [flowPct, setFlowPct] = useState(0)
  const [flowLabel, setFlowLabel] = useState('')
  const [flowRunning, setFlowRunning] = useState(false)
  const [flowElapsed, setFlowElapsed] = useState<number | undefined>(undefined)

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

  async function runBacktest() {
    const res = await postJSON<{ status:string; job_id:string }>(''/run/backtest'.replace("''",""), {
      config: 'configs/config_freqai_multi.json',
      strategy: 'ExpressionLongStrategy',
      strategy_path: 'freqtrade/user_data/strategies',
      timerange: '20200701-20210131',
      freqaimodel: 'LightGBMRegressor',
      export: false,
      fast: true,
    })
    startTrack(res.job_id, 'bt')
  }

  async function terminateJob() {
    if (!selectedJob) return
    await postEmpty(`/jobs/${selectedJob.id}/terminate`)
    await refreshJobs()
  }

  useEffect(() => {
    refreshJobs().catch(() => {})
    const t = setInterval(() => refreshJobs().catch(() => {}), 4000)
    return () => clearInterval(t)
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <h1>Agent Market</h1>
      <p>API: {API_BASE}</p>
      <nav style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <button onClick={() => setActiveTab('data')} disabled={activeTab==='data'}>Data</button>
        <button onClick={() => setActiveTab('expressions')} disabled={activeTab==='expressions'}>Expressions</button>
        <button onClick={() => setActiveTab('backtest')} disabled={activeTab==='backtest'}>Backtest</button>
        <button onClick={() => setActiveTab('jobs')} disabled={activeTab==='jobs'}>Jobs</button>
        <button onClick={refreshJobs} style={{ marginLeft: 'auto' }}>Refresh Jobs</button>
      </nav>

      {activeTab === 'data' ? (
        <DataManager defaultTimeframe={dlTimeframe} defaultTimerange={dlTimerange} defaultPairs={dlPairs} onJob={refreshJobs} />
      ) : null}
      {activeTab === 'expressions' ? (
        <ExpressionsManager onJob={refreshJobs} />
      ) : null}
      {activeTab === 'backtest' ? (
        <div>
          <div style={{ marginBottom: 8 }}>
            <button onClick={runBacktest}>Run Backtest</button>
            <button onClick={runFlow} style={{ marginLeft: 8 }}>Run Full Flow</button>
          </div>
          <p>Backtest runs with current server config. Full Flow will execute: download → feature → expression → ml → backtest.</p>
          {btJobId ? (
            <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
              <b>Backtest Job:</b> {btJobId}
              <div style={{ height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden', marginTop: 4 }}>
                <div style={{ width: btPct + '%', height: '100%', background: '#3b82f6' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span>{btLabel || 'running...'}</span>
                <span>{btPct}% {btElapsed ? `(elapsed ${btElapsed}s)` : ''}</span>
              </div>
            </div>
          ) : null}
          {flowJobId ? (
            <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
              <b>Flow Job:</b> {flowJobId}
              <div style={{ height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden', marginTop: 4 }}>
                <div style={{ width: flowPct + '%', height: '100%', background: '#16a34a' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span>{flowLabel || 'running...'}</span>
                <span>{flowPct}% {flowElapsed ? `(elapsed ${flowElapsed}s)` : ''}</span>
              </div>
            </div>
          ) : null}
          <div style={{ marginTop: 12 }}>
            <StrategyParams />
          </div>
        </div>
      ) : null}
      {activeTab === 'jobs' ? (
        <JobsPanel jobs={jobs} selectedJob={selectedJob} onOpen={openJob} onTerminate={terminateJob} jobLogs={jobLogs} />
      ) : null}
    </div>
  )
}
  async function runFlow() {
    const res = await postJSON<{ job_id: string }>('/flow/run', { config: 'configs/agent_flow_multi.json' })
    startTrack(res.job_id, 'flow')
  }

  async function pollProgress(id: string, target: 'bt'|'flow') {
    try {
      const p = await getJSON<any>(`/jobs/${id}/progress`)
      if (target === 'bt') {
        setBtPct(p.percent || 0); setBtLabel(p.label || ''); setBtRunning(!!p.running); setBtElapsed(p.elapsed)
      } else {
        setFlowPct(p.percent || 0); setFlowLabel(p.label || ''); setFlowRunning(!!p.running); setFlowElapsed(p.elapsed)
      }
    } catch {}
  }

  function startTrack(id: string, target: 'bt'|'flow') {
    if (target === 'bt') { setBtJobId(id); setBtRunning(true); setBtPct(0); setBtLabel('') } else { setFlowJobId(id); setFlowRunning(true); setFlowPct(0); setFlowLabel('') }
    // light polling for simplicity
    const tick = async () => {
      await pollProgress(id, target)
      const running = target === 'bt' ? btRunning : flowRunning
      if (!(target === 'bt' ? btRunning : flowRunning)) {
        await refreshJobs()
        return
      }
      setTimeout(tick, 1200)
    }
    setTimeout(tick, 1000)
  }
