import React, { useEffect, useState } from 'react'
import { getJSON, postEmpty } from '../api'

type JobStatus = { id: string; running: boolean; returncode: number | null; lines: number; started_at?: string; finished_at?: string }

export default function JobsPanel({ jobs, selectedJob, onOpen, onTerminate, jobLogs }: {
  jobs: JobStatus[]
  selectedJob: JobStatus | null
  onOpen: (j: JobStatus) => void
  onTerminate: () => void
  jobLogs: { line: number; text: string }[]
}) {
  const [steps, setSteps] = useState<any[]>([])
  const [stats, setStats] = useState<any[]>([])
  const [summary, setSummary] = useState<any | null>(null)
  const [isBacktest, setIsBacktest] = useState<boolean>(false)

  useEffect(() => {
    let timer: any
    async function poll() {
      if (!selectedJob) return
      try {
        const st = await getJSON<any>(`/jobs/${selectedJob.id}/status`)
        try {
          const cmd = (st && st.cmd) || []
          const s = (Array.isArray(cmd) ? cmd.join(' ') : String(cmd || '')).toLowerCase()
          setIsBacktest(s.includes('backtesting'))
        } catch {}
        const s = await getJSON<any>(`/jobs/${selectedJob.id}/steps`)
        const st = await getJSON<any>('/steps/stats')
        setSteps(s.steps || [])
        setStats(st.stats || [])
      } catch {}
      timer = setTimeout(poll, 1500)
    }
    poll()
    return () => { if (timer) clearTimeout(timer) }
  }, [selectedJob?.id])

  async function loadSummary() {
    try { const s = await getJSON<any>('/backtest/summary/latest'); setSummary(s) } catch {}
  }
  function parseStep(logs: string[]) {
    const re = /^\[STEP\]\s+\d{2}:\d{2}:\d{2}\s+(\d+)\/(\d+)\s+(.+)$/
    let current = 0, total = 0, label = ''
    for (const line of logs) {
      const m = line.match(re)
      if (m) { current = parseInt(m[1], 10); total = parseInt(m[2], 10); label = m[3] }
    }
    return { current, total, label }
  }
  return (
    <div>
      <h2>Jobs</h2>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <ul>
            {jobs.map(j => (
              <li key={j.id}>
                <button onClick={() => onOpen(j)}>
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
            let elapsed = ''
            try {
              const started = (selectedJob as any).started_at ? new Date((selectedJob as any).started_at).getTime() : null
              if (started) { const sec = Math.floor((Date.now() - started) / 1000); elapsed = `${sec}s` }
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
              <button onClick={onTerminate}>Terminate</button>
            </div>
          ) : null}
          <div style={{ height: 240, overflow: 'auto', border: '1px solid #ddd', padding: 8 }}>
            <pre style={{ margin: 0 }}>
              {jobLogs.map(e => `${e.line}: ${e.text}`).join('\n')}
            </pre>
          </div>
          {steps.length ? (
            <div style={{ marginTop: 8 }}>
              <h4>Steps</h4>
              <table>
                <thead><tr><th>#</th><th>Label</th><th>Duration(s)</th><th>Avg(s)</th><th>Delta</th></tr></thead>
                <tbody>
                  {steps.map((s:any, i:number) => {
                    const stat = stats.find((x:any) => x.label === s.label)
                    const avg = stat ? Math.round(stat.avg_seconds) : null
                    const dur = typeof s.duration === 'number' ? s.duration : null
                    const delta = (avg != null && dur != null) ? (dur - avg) : null
                    return <tr key={i}><td>{s.idx}/{s.total}</td><td>{s.label}</td><td style={{ textAlign:'right' }}>{dur ?? '-'}</td><td style={{ textAlign:'right' }}>{avg ?? '-'}</td><td style={{ textAlign:'right', color: (delta!=null && delta>5)?'crimson':undefined }}>{delta ?? '-'}</td></tr>
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {isBacktest ? (
            <div style={{ marginTop: 8 }}>
              <button onClick={loadSummary}>Show Last Backtest Summary</button>
            </div>
          ) : null}
          {summary ? (
            <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
              <b>Last Summary</b>
              <div>Strategy: {summary.strategy} Source: {summary.source}</div>
              {Array.isArray(summary.pairs) && summary.pairs.length ? (
                <table style={{ marginTop: 4 }}>
                  <thead><tr><th>Pair</th><th>Trades</th><th>Profit %</th></tr></thead>
                  <tbody>
                    {summary.pairs.map((p:any, idx:number) => (
                      <tr key={idx}><td>{p.pair}</td><td style={{ textAlign:'right' }}>{p.trades}</td><td style={{ textAlign:'right' }}>{p.profit_total_pct}</td></tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
