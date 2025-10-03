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
  const [btSteps, setBtSteps] = useState<any[]>([])
  const [flowJobId, setFlowJobId] = useState<string | null>(null)
  const [flowPct, setFlowPct] = useState(0)
  const [flowLabel, setFlowLabel] = useState('')
  const [flowRunning, setFlowRunning] = useState(false)
  const [flowElapsed, setFlowElapsed] = useState<number | undefined>(undefined)
  const [flowSteps, setFlowSteps] = useState<any[]>([])
  const [stepStats, setStepStats] = useState<any[]>([])
  // Backtest controls
  const [btModel, setBtModel] = useState('LightGBMRegressor')
  const [btExpr, setBtExpr] = useState<any[]>([])
  const [btExprLoaded, setBtExprLoaded] = useState(false)
  const [btSummary, setBtSummary] = useState<any | null>(null)
  const [btFeaturePath, setBtFeaturePath] = useState('user_data/freqai_features_multi.json')
  const [btExprPath, setBtExprPath] = useState('user_data/freqai_expressions.json')
  const [btExprFilter, setBtExprFilter] = useState('')
  const [btExprMode, setBtExprMode] = useState<'all'|'enabled'|'disabled'>('all')
  const [btFeaturesActivePath, setBtFeaturesActivePath] = useState<string>('')
  const [btShowAllFactors, setBtShowAllFactors] = useState(false)

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

  useEffect(() => {
    // Load active features path once
    getJSON<any>('/features').then((res) => {
      if (res && res.path) setBtFeaturesActivePath(res.path)
    }).catch(()=>{})
  }, [])

  async function runBacktest() {
    // Save enabled flags if expressions loaded
    try {
      if (btExprLoaded) {
        await postJSON('/expressions', { expressions: btExpr })
      }
    } catch {}
    const res = await postJSON<{ status:string; job_id:string }>(`/run/backtest`, {
      config: 'configs/config_freqai_multi.json',
      strategy: 'ExpressionLongStrategy',
      strategy_path: 'freqtrade/user_data/strategies',
      timerange: '20200701-20210131',
      freqaimodel: btModel,
      export: false,
      fast: true,
    })
    startTrack(res.job_id, 'bt')
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
    try {
      const steps = await getJSON<any>(`/jobs/${id}/steps`)
      const stats = await getJSON<any>('/steps/stats')
      if (target === 'bt') setBtSteps(steps.steps || []); else setFlowSteps(steps.steps || [])
      setStepStats(stats.stats || [])
    } catch {}
  }

  function startTrack(id: string, target: 'bt'|'flow') {
    if (target === 'bt') { setBtJobId(id); setBtRunning(true); setBtPct(0); setBtLabel('') } else { setFlowJobId(id); setFlowRunning(true); setFlowPct(0); setFlowLabel('') }
    const tick = async () => {
      await pollProgress(id, target)
      const running = target === 'bt' ? btRunning : flowRunning
      if (!(target === 'bt' ? btRunning : flowRunning)) {
        await refreshJobs()
        // Auto load summary after backtest
        if (target === 'bt') {
          try {
            const sum = await getJSON<any>('/backtest/summary/latest')
            setBtSummary(sum)
          } catch {}
        }
        return
      }
      setTimeout(tick, 1200)
    }
    setTimeout(tick, 1000)
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
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', margin: '8px 0' }}>
            <label>ML Model:</label>
            <select value={btModel} onChange={e => setBtModel(e.target.value)}>
              <option>LightGBMRegressor</option>
              <option>XGBoostRegressor</option>
              <option>RandomForestRegressor</option>
              <option>CatBoostRegressor</option>
            </select>
            <label>Features file:</label>
            <input value={btFeaturePath} onChange={e => setBtFeaturePath(e.target.value)} style={{ width: 260 }} />
            <button onClick={async ()=>{ try { await postJSON('/config/use-features-file', { path: btFeaturePath }); alert('Features file applied.'); } catch { alert('Apply features failed') } }}>Use</button>
            <label>Expressions file:</label>
            <input value={btExprPath} onChange={e => setBtExprPath(e.target.value)} style={{ width: 260 }} />
            <button onClick={async ()=>{ try { await postJSON('/config/use-expressions-file', { path: btExprPath }); const data = await getJSON<any>('/expressions'); setBtExpr(data.expressions || []); setBtExprLoaded(true); alert('Expressions file applied & loaded.'); } catch { alert('Apply expressions failed') } }}>Use & Load</button>
            <button onClick={async () => {
              try {
                const data = await getJSON<any>('/expressions')
                setBtExpr(data.expressions || [])
                setBtExprLoaded(true)
              } catch {}
            }}>Load Factors</button>
            {btExprLoaded ? <button onClick={async ()=>{ try { await postJSON('/expressions', { expressions: btExpr }); alert('Factors saved'); } catch (e) { alert('Save failed') } }}>Save Factors</button> : null}
          </div>
          {btExprLoaded ? (
            <div style={{ border: '1px solid #eee', padding: 8, borderRadius: 4, maxHeight: 220, overflow: 'auto', marginBottom: 8 }}>
              <div style={{ display:'flex', gap:8, alignItems:'center', marginBottom: 6 }}>
                <div style={{ fontWeight: 600 }}>Select Factors</div>
                <input placeholder='search name...' value={btExprFilter} onChange={e=>setBtExprFilter(e.target.value)} />
                <select value={btExprMode} onChange={e=>setBtExprMode(e.target.value as any)}>
                  <option value='all'>all</option>
                  <option value='enabled'>enabled</option>
                  <option value='disabled'>disabled</option>
                </select>
              </div>
              {(btExpr || []).filter((x:any)=>{
                const name = (x.name || '').toLowerCase();
                const okName = !btExprFilter || name.includes(btExprFilter.toLowerCase());
                const okMode = btExprMode==='all' || (btExprMode==='enabled' ? !!x.enabled : !x.enabled);
                return okName && okMode;
              }).map((x:any, i:number) => (
                <label key={i} style={{ display: 'inline-block', marginRight: 12 }}>
                  <input type="checkbox" checked={!!x.enabled} onChange={e => setBtExpr(prev => prev.map((y:any, j:number) => j===i ? ({ ...y, enabled: e.target.checked }) : y))} />
                  <span style={{ marginLeft: 4 }}>{x.name || `expr_${i}`}</span>
                </label>
              ))}
            </div>
          ) : null}
          <p>Backtest runs with current server config. Full Flow will execute: download → feature → expression → ml → backtest.</p>
          {btJobId ? (
            <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
              <b>Backtest Job:</b> {btJobId}
              <div style={{ marginTop: 4, color: '#444' }}>
                <div>Features: {btFeaturesActivePath ? btFeaturesActivePath.replace(/^.*[\\\\\/]/,'') : 'unknown'}</div>
                {btExprLoaded ? (()=>{ 
                  const enabled = (btExpr || []).filter((x:any)=>!!x.enabled);
                  const names = enabled.map((x:any)=>x.name||'').filter(Boolean);
                  const head = btShowAllFactors ? names : names.slice(0,8);
                  const more = Math.max(0, names.length - head.length);
                  return (
                    <div>
                      <div>
                        Enabled factors: {enabled.length}{names.length? ' → ' + head.join(', ') + (more? ` ... (+${more})`:'') : ''}
                        {names.length>8 ? (
                          <button onClick={()=>setBtShowAllFactors(v=>!v)} style={{ marginLeft: 8 }}>
                            {btShowAllFactors ? 'Collapse' : 'Show all'}
                          </button>
                        ) : null}
                      </div>
                    </div>
                  )
                })() : null}
              </div>
              <div style={{ height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden', marginTop: 4 }}>
                <div style={{ width: btPct + '%', height: '100%', background: '#3b82f6' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <span>{btLabel || 'running...'}</span>
                <span>{btPct}% {btElapsed ? `(elapsed ${btElapsed}s)` : ''}</span>
              </div>
              {btSummary ? (
                <div style={{ marginTop: 8, borderTop: '1px solid #eee', paddingTop: 6 }}>
                  <div><b>Summary:</b> {btSummary.strategy} ({btSummary.source})</div>
                  {Array.isArray(btSummary.pairs) && btSummary.pairs.length ? (
                    <table style={{ marginTop: 4 }}>
                      <thead><tr><th>Pair</th><th>Trades</th><th>Profit %</th></tr></thead>
                      <tbody>
                        {btSummary.pairs.map((p:any, idx:number) => (
                          <tr key={idx}><td>{p.pair}</td><td style={{ textAlign:'right' }}>{p.trades}</td><td style={{ textAlign:'right' }}>{p.profit_total_pct}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
          {btSteps.length ? (
            <div style={{ marginTop: 8 }}>
              <h4>Steps (Backtest)</h4>
              <table>
                <thead><tr><th>#</th><th>Label</th><th>Duration(s)</th><th>Avg(s)</th><th>Delta</th></tr></thead>
                <tbody>
                  {btSteps.map((s:any, i:number) => {
                    const stat = stepStats.find((x:any) => x.label === s.label)
                    const avg = stat ? Math.round(stat.avg_seconds) : null
                    const dur = typeof s.duration === 'number' ? s.duration : null
                    const delta = (avg != null && dur != null) ? (dur - avg) : null
                    return <tr key={i}><td>{s.idx}/{s.total}</td><td>{s.label}</td><td style={{ textAlign:'right' }}>{dur ?? '-'}</td><td style={{ textAlign:'right' }}>{avg ?? '-'}</td><td style={{ textAlign:'right', color: (delta!=null && delta>5)?'crimson':undefined }}>{delta ?? '-'}</td></tr>
                  })}
                </tbody>
              </table>
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
          {flowSteps.length ? (
            <div style={{ marginTop: 8 }}>
              <h4>Steps (Flow)</h4>
              <table>
                <thead><tr><th>#</th><th>Label</th><th>Duration(s)</th><th>Avg(s)</th><th>Delta</th></tr></thead>
                <tbody>
                  {flowSteps.map((s:any, i:number) => {
                    const stat = stepStats.find((x:any) => x.label === s.label)
                    const avg = stat ? Math.round(stat.avg_seconds) : null
                    const dur = typeof s.duration === 'number' ? s.duration : null
                    const delta = (avg != null && dur != null) ? (dur - avg) : null
                    return <tr key={i}><td>{s.idx}/{s.total}</td><td>{s.label}</td><td style={{ textAlign:'right' }}>{dur ?? '-'}</td><td style={{ textAlign:'right' }}>{avg ?? '-'}</td><td style={{ textAlign:'right', color: (delta!=null && delta>5)?'crimson':undefined }}>{delta ?? '-'}</td></tr>
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {btSummary ? (
            <div style={{ marginTop: 12, border: '1px solid #ddd', padding: 8 }}>
              <b>Last Summary</b>
              <div>Strategy: {btSummary.strategy} Source: {btSummary.source}</div>
              {btSummary.metrics ? (
                <div>
                  <div>trades: {btSummary.metrics.trades} profit_total_pct: {btSummary.metrics.profit_total_pct} winrate: {btSummary.metrics.winrate}</div>
                </div>
              ) : null}
              {Array.isArray(btSummary.pairs) && btSummary.pairs.length ? (
                <table style={{ marginTop: 6 }}>
                  <thead><tr><th>Pair</th><th>Trades</th><th>Profit %</th></tr></thead>
                  <tbody>
                    {btSummary.pairs.map((p:any, idx:number) => (
                      <tr key={idx}><td>{p.pair}</td><td style={{ textAlign:'right' }}>{p.trades}</td><td style={{ textAlign:'right' }}>{p.profit_total_pct}</td></tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
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
