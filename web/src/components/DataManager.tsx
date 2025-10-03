import React, { useEffect, useMemo, useState } from 'react'
import { getJSON, postJSON } from '../api'

type SummaryRow = { pair: string; count: number; start?: string; end?: string }

export default function DataManager({ defaultTimeframe, defaultTimerange, defaultPairs, onJob }: { defaultTimeframe: string; defaultTimerange: string; defaultPairs: string; onJob?: () => void }) {
  const [exchange, setExchange] = useState('binanceus')
  const [timeframes, setTimeframes] = useState(defaultTimeframe)
  const [pairs, setPairs] = useState(defaultPairs)
  const [timerange, setTimerange] = useState(defaultTimerange)
  const [summary, setSummary] = useState<Record<string, SummaryRow[]>>({})
  const [loading, setLoading] = useState(false)
  const [missing, setMissing] = useState<any>({ missing: [], insufficient: [] })

  async function refresh() {
    setLoading(true)
    try {
      const res = await getJSON<{ timeframes: Record<string, SummaryRow[]> }>(`/data/summary?exchange=${encodeURIComponent(exchange)}&timeframes=${encodeURIComponent(timeframes)}`)
      setSummary(res.timeframes || {})
    } finally {
      setLoading(false)
    }
  }

  async function checkMissing() {
    setLoading(true)
    try {
      const qs = new URLSearchParams({ exchange, timeframes, pairs, timerange })
      const res = await getJSON<{ missing: any[]; insufficient: any[] }>(`/data/check-missing?${qs.toString()}`)
      setMissing(res)
    } finally {
      setLoading(false)
    }
  }

  async function download() {
    const tfs = timeframes.split(',').map(s => s.trim()).filter(Boolean)
    const plist = pairs.split(',').map(s => s.trim()).filter(Boolean)
    await postJSON('/run/download-data', { config: 'configs/config_freqai_multi.json', timeframes: tfs, pairs: plist, timerange })
    if (onJob) onJob()
  }

  useEffect(() => { refresh().catch(()=>{}) }, [])

  const tfList = useMemo(() => Object.keys(summary), [summary])

  return (
    <div>
      <h2>Data Manager</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <label>Exchange:</label>
        <input value={exchange} onChange={e => setExchange(e.target.value)} style={{ width: 140 }} />
        <label>Timeframes (comma):</label>
        <input value={timeframes} onChange={e => setTimeframes(e.target.value)} style={{ width: 180 }} />
        <label>Timerange:</label>
        <input value={timerange} onChange={e => setTimerange(e.target.value)} style={{ width: 180 }} />
        <label>Pairs (comma):</label>
        <input value={pairs} onChange={e => setPairs(e.target.value)} style={{ width: 320 }} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <button onClick={refresh} disabled={loading}>Refresh Summary</button>
        <button onClick={checkMissing} style={{ marginLeft: 8 }} disabled={loading}>Check Missing</button>
        <button onClick={download} style={{ marginLeft: 8 }} disabled={loading}>Download</button>
      </div>
      {tfList.map(tf => (
        <div key={tf} style={{ marginTop: 8 }}>
          <h3>{tf}</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr><th style={{ textAlign: 'left' }}>Pair</th><th>Count</th><th>Start</th><th>End</th></tr>
            </thead>
            <tbody>
              {(summary[tf] || []).map((r, idx) => (
                <tr key={idx}><td>{r.pair}</td><td style={{ textAlign: 'right' }}>{r.count}</td><td>{r.start || '-'}</td><td>{r.end || '-'}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <div style={{ marginTop: 12 }}>
        <h3>Missing / Insufficient</h3>
        <div>
          <b>Missing:</b> {missing.missing.length} items
          <ul>
            {missing.missing.map((m: any, i: number) => <li key={i}>{m.pair} {m.timeframe} - {m.reason}</li>)}
          </ul>
        </div>
        <div>
          <b>Insufficient:</b> {missing.insufficient.length} items
          <ul>
            {missing.insufficient.map((m: any, i: number) => <li key={i}>{m.pair} {m.timeframe} - have [{m.file_start} .. {m.file_end}] want [{m.want_start} .. {m.want_end}]</li>)}
          </ul>
        </div>
      </div>
    </div>
  )
}

