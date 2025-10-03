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
  const [selMissing, setSelMissing] = useState<Record<string, boolean>>({})
  const [selInsuf, setSelInsuf] = useState<Record<string, boolean>>({})
  const [error, setError] = useState<string>('')
  const [erase, setErase] = useState(false)
  const [newPairs, setNewPairs] = useState(false)
  const [prepend, setPrepend] = useState(false)

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      const res = await getJSON<{ timeframes: Record<string, SummaryRow[]> }>(`/data/summary?exchange=${encodeURIComponent(exchange)}&timeframes=${encodeURIComponent(timeframes)}`)
      setSummary(res.timeframes || {})
    } catch (e:any) {
      setError(e?.message || 'Refresh summary failed')
    } finally {
      setLoading(false)
    }
  }

  async function checkMissing() {
    setLoading(true)
    setError('')
    try {
      const qs = new URLSearchParams({ exchange, timeframes, pairs, timerange })
      const res = await getJSON<{ missing: any[]; insufficient: any[] }>(`/data/check-missing?${qs.toString()}`)
      setMissing(res)
      const nm: Record<string, boolean> = {}
      const ni: Record<string, boolean> = {}
      for (const m of res.missing || []) nm[`${m.pair}|${m.timeframe}`] = true
      for (const m of res.insufficient || []) ni[`${m.pair}|${m.timeframe}`] = true
      setSelMissing(nm); setSelInsuf(ni)
    } catch (e:any) {
      setError(e?.message || 'Check missing failed')
    } finally {
      setLoading(false)
    }
  }

  async function download() {
    const tfs = timeframes.split(',').map(s => s.trim()).filter(Boolean)
    const plist = pairs.split(',').map(s => s.trim()).filter(Boolean)
    await postJSON('/run/download-data', { config: 'configs/config_freqai_multi.json', timeframes: tfs, pairs: plist, timerange, erase, new_pairs: newPairs, prepend })
    if (onJob) onJob()
  }

  async function downloadMissing() {
    const tset = new Set<string>()
    const pset = new Set<string>()
    for (const m of missing.missing || []) { if (selMissing[`${m.pair}|${m.timeframe}`]) { tset.add(m.timeframe); pset.add(m.pair) } }
    for (const m of missing.insufficient || []) { if (selInsuf[`${m.pair}|${m.timeframe}`]) { tset.add(m.timeframe); pset.add(m.pair) } }
    const tfs = Array.from(tset)
    const plist = Array.from(pset)
    if (tfs.length === 0 || plist.length === 0) return
    await postJSON('/run/download-data', { config: 'configs/config_freqai_multi.json', timeframes: tfs, pairs: plist, timerange, erase, new_pairs: newPairs, prepend })
    if (onJob) onJob()
  }

  useEffect(() => { refresh().catch(()=>{}) }, [])

  const tfList = useMemo(() => Object.keys(summary), [summary])

  return (
    <div>
      <h2>Data Manager</h2>
      {error ? <div style={{ color: 'crimson', marginBottom: 8 }}>{error}</div> : null}
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
        <button onClick={downloadMissing} style={{ marginLeft: 8 }} disabled={loading}>Download Missing</button>
        <label style={{ marginLeft: 8 }}><input type="checkbox" checked={erase} onChange={e => setErase(e.target.checked)} /> erase</label>
        <label style={{ marginLeft: 8 }}><input type="checkbox" checked={newPairs} onChange={e => setNewPairs(e.target.checked)} /> new-pairs</label>
        <label style={{ marginLeft: 8 }}><input type="checkbox" checked={prepend} onChange={e => setPrepend(e.target.checked)} /> prepend</label>
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
          <table>
            <thead><tr><th></th><th>Pair</th><th>TF</th><th>Reason</th></tr></thead>
            <tbody>
              {missing.missing.map((m: any, i: number) => (
                <tr key={`m-${i}`}>
                  <td><input type="checkbox" checked={!!selMissing[`${m.pair}|${m.timeframe}`]} onChange={e => setSelMissing(prev => ({ ...prev, [`${m.pair}|${m.timeframe}`]: e.target.checked }))} /></td>
                  <td>{m.pair}</td><td>{m.timeframe}</td><td>{m.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8 }}>
          <b>Insufficient:</b> {missing.insufficient.length} items
          <table>
            <thead><tr><th></th><th>Pair</th><th>TF</th><th>Have</th><th>Want</th></tr></thead>
            <tbody>
              {missing.insufficient.map((m: any, i: number) => (
                <tr key={`i-${i}`}>
                  <td><input type="checkbox" checked={!!selInsuf[`${m.pair}|${m.timeframe}`]} onChange={e => setSelInsuf(prev => ({ ...prev, [`${m.pair}|${m.timeframe}`]: e.target.checked }))} /></td>
                  <td>{m.pair}</td><td>{m.timeframe}</td><td>[{m.file_start} .. {m.file_end}]</td><td>[{m.want_start} .. {m.want_end}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
