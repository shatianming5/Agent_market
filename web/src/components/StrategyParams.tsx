import React, { useEffect, useState } from 'react'
import { getJSON, postJSON } from '../api'

export default function StrategyParams() {
  const [data, setData] = useState<any>(null)
  const [path, setPath] = useState('')
  const [saving, setSaving] = useState(false)
  const [summary, setSummary] = useState<any>(null)

  async function load() {
    const res = await getJSON<{ path: string; data: any }>(`/strategy/params`)
    setData(res.data)
    setPath(res.path)
  }

  async function save() {
    setSaving(true)
    try {
      await putJSON('/strategy/params', data)
    } finally {
      setSaving(false)
    }
  }

  async function loadSummary() {
    const res = await getJSON<any>('/backtest/summary/latest')
    setSummary(res)
  }

  useEffect(() => { load().catch(()=>{}) }, [])

  function set(pathStr: string, val: any) {
    const keys = pathStr.split('.')
    setData((prev: any) => {
      const cp = JSON.parse(JSON.stringify(prev || {}))
      let cur = cp
      for (let i = 0; i < keys.length - 1; i++) { cur[keys[i]] ||= {}; cur = cur[keys[i]] }
      cur[keys[keys.length-1]] = val
      return cp
    })
  }

  const buy = (data?.params?.buy) || {}
  const sell = (data?.params?.sell) || {}

  return (
    <div>
      <h2>Strategy Params</h2>
      <div style={{ color: '#777', marginBottom: 8 }}>File: {path}</div>
      <div style={{ display: 'flex', gap: 16 }}>
        <div>
          <h3>Buy</h3>
          <div>dynamic_entry_q: <input type="number" step="0.01" value={buy.dynamic_entry_q ?? ''} onChange={e => set('params.buy.dynamic_entry_q', parseFloat(e.target.value))} /></div>
          <div>signal_entry_min: <input type="number" step="0.01" value={buy.signal_entry_min ?? ''} onChange={e => set('params.buy.signal_entry_min', parseFloat(e.target.value))} /></div>
          <div>vote_entry_threshold_p: <input type="number" step="0.01" value={buy.vote_entry_threshold_p ?? ''} onChange={e => set('params.buy.vote_entry_threshold_p', parseFloat(e.target.value))} /></div>
        </div>
        <div>
          <h3>Sell</h3>
          <div>dynamic_exit_q: <input type="number" step="0.01" value={sell.dynamic_exit_q ?? ''} onChange={e => set('params.sell.dynamic_exit_q', parseFloat(e.target.value))} /></div>
          <div>signal_exit_max: <input type="number" step="0.01" value={sell.signal_exit_max ?? ''} onChange={e => set('params.sell.signal_exit_max', parseFloat(e.target.value))} /></div>
          <div>vote_exit_threshold_p: <input type="number" step="0.01" value={sell.vote_exit_threshold_p ?? ''} onChange={e => set('params.sell.vote_exit_threshold_p', parseFloat(e.target.value))} /></div>
        </div>
      </div>
      <div style={{ marginTop: 8 }}>
        <button onClick={save} disabled={saving}>Save</button>
        <button onClick={loadSummary} style={{ marginLeft: 8 }}>Show Last Summary</button>
      </div>
      {summary ? (
        <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
          <b>Latest Backtest</b>
          <div>strategy: {summary.strategy} source: {summary.source}</div>
          {summary.metrics ? (
            <div>
              <div>trades: {Array.isArray(summary?.metrics?.trades) ? summary.metrics.trades.length : summary?.metrics?.trades} profit_total_pct: {(typeof summary?.metrics?.profit_total_pct === 'object' ? '' : summary?.metrics?.profit_total_pct)} winrate: {summary.metrics.winrate}</div>
              <div>best_pair: {summary.metrics.best_pair?.key} worst_pair: {summary.metrics.worst_pair?.key}</div>
            </div>
          ) : null}
          {Array.isArray(summary.pairs) && summary.pairs.length ? (
            <div style={{ marginTop: 8 }}>
              <h4>Per-Pair</h4>
              <table>
                <thead><tr><th>Pair</th><th>Trades</th><th>Profit %</th></tr></thead>
                <tbody>
                  {summary.pairs.map((p:any, i:number) => (
                    <tr key={i}><td>{p.pair}</td><td style={{ textAlign:'right' }}>{p.trades}</td><td style={{ textAlign:'right' }}>{p.profit_total_pct}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}



