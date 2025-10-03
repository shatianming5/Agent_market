import React, { useEffect, useState } from 'react'
import { getJSON, postJSON } from '../api'

type ExpressionItem = { [k: string]: any }

export default function ExpressionsManager({ onJob }: { onJob?: () => void }) {
  const [expressions, setExpressions] = useState<ExpressionItem[]>([])
  const [path, setPath] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [allowed, setAllowed] = useState<{ functions: string[]; base_columns: string[] }>({ functions: [], base_columns: [] })
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewStats, setPreviewStats] = useState<any | null>(null)
  const [pair, setPair] = useState('ADA/USDT')
  const [timeframe, setTimeframe] = useState('4h')
  const [staticIssues, setStaticIssues] = useState<Record<number, any>>({})
  const [columns, setColumns] = useState<string[]>([])

  async function load() {
    setLoading(true)
    setError('')
    try {
      const res = await getJSON<{ path: string; expressions: ExpressionItem[] }>(`/expressions`)
      setPath(res.path)
      setExpressions(res.expressions || [])
    } catch (e: any) {
      setError(e?.message || 'Failed to load expressions')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])
  useEffect(() => { getJSON('/expressions/allowed').then(setAllowed).catch(()=>{}) }, [])

  function update(idx: number, key: string, val: any) {
    setExpressions(prev => prev.map((x, i) => i===idx ? { ...x, [key]: val } : x))
  }

  function addNew() {
    setExpressions(prev => [...prev, { enabled: true, name: '', expression: '', entry_threshold: '', exit_threshold: '', offline_profit: '', win_ratio: '', offline_trade_count: '', signal_sharpe: '', stability_mean: '', adjusted_score: '' }])
  }

  function remove(idx: number) {
    setExpressions(prev => prev.filter((_, i) => i !== idx))
  }

  async function saveAll() {
    setLoading(true)
    setError('')
    try {
      await postJSON(`/expressions`, { expressions })
    } catch (e: any) {
      setError(e?.message || 'Failed to save expressions')
    } finally {
      setLoading(false)
    }
  }

  async function validateExpression(idx: number) {
    const item = expressions[idx]
    if (!item?.expression) return
    setLoading(true)
    setError('')
    try {
      const stat = await postJSON('/expressions/validate', { expression: item.expression })
      setStaticIssues(prev => ({ ...prev, [idx]: stat }))
      const res = await postJSON('/expressions/preview', { pair, timeframe, expression: item.expression, config: 'configs/config_freqai_multi.json', apply_features: true })
      setPreviewIdx(idx); setPreviewStats(res)
    } catch (e: any) {
      setError(e?.message || 'Validate failed')
      setPreviewIdx(idx); setPreviewStats(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadColumns() {
    try {
      const qs = new URLSearchParams({ pair, timeframe, config: 'configs/config_freqai_multi.json', apply_features: 'true' })
      const res = await getJSON<{ columns: string[] }>(`/data/columns?${qs.toString()}`)
      setColumns(res.columns || [])
    } catch {}
  }

  return (
    <div>
      <h2>Expressions</h2>
      <div style={{ marginBottom: 8, color: '#777' }}>File: {path || '(auto)'}</div>
      <div style={{ marginBottom: 8 }}>
        <button onClick={addNew}>Add</button>
        <button onClick={saveAll} style={{ marginLeft: 8 }} disabled={loading}>Save All</button>
        <button onClick={load} style={{ marginLeft: 8 }} disabled={loading}>Reload</button>
        {error ? <span style={{ marginLeft: 8, color: 'crimson' }}>{error}</span> : null}
      </div>
      <datalist id="expr-suggestions">
        {allowed.functions.map((f, i) => <option key={`f-${i}`} value={`${f}(`} />)}
        {columns.map((c, i) => <option key={`c-${i}`} value={c} />)}
      </datalist>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>On</th>
            <th style={{ textAlign: 'left' }}>Name</th>
            <th style={{ textAlign: 'left' }}>Expression</th>
            <th>Entry Thr</th>
            <th>Exit Thr</th>
            <th>offline_profit</th>
            <th>win_ratio</th>
            <th>trade_cnt</th>
            <th>sharpe</th>
            <th>stability</th>
            <th>adjusted</th>
            <th style={{ width: 120 }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {expressions.map((item, idx) => (
            <tr key={idx}>
              <td><input type="checkbox" checked={!!item.enabled} onChange={e => update(idx, 'enabled', e.target.checked)} /></td>
              <td><input value={item.name || ''} onChange={e => update(idx, 'name', e.target.value)} style={{ width: 160 }} /></td>
              <td><input list="expr-suggestions" value={item.expression || ''} onChange={e => update(idx, 'expression', e.target.value)} onFocus={loadColumns} style={{ width: '100%' }} /></td>
              <td><input value={item.entry_threshold ?? ''} onChange={e => update(idx, 'entry_threshold', e.target.value)} style={{ width: 80 }} /></td>
              <td><input value={item.exit_threshold ?? ''} onChange={e => update(idx, 'exit_threshold', e.target.value)} style={{ width: 80 }} /></td>
              <td><input value={item.offline_profit ?? ''} onChange={e => update(idx, 'offline_profit', e.target.value)} style={{ width: 80 }} /></td>
              <td><input value={item.win_ratio ?? ''} onChange={e => update(idx, 'win_ratio', e.target.value)} style={{ width: 70 }} /></td>
              <td><input value={item.offline_trade_count ?? ''} onChange={e => update(idx, 'offline_trade_count', e.target.value)} style={{ width: 70 }} /></td>
              <td><input value={item.signal_sharpe ?? ''} onChange={e => update(idx, 'signal_sharpe', e.target.value)} style={{ width: 70 }} /></td>
              <td><input value={item.stability_mean ?? ''} onChange={e => update(idx, 'stability_mean', e.target.value)} style={{ width: 70 }} /></td>
              <td><input value={item.adjusted_score ?? ''} onChange={e => update(idx, 'adjusted_score', e.target.value)} style={{ width: 70 }} /></td>
              <td>
                <button onClick={() => validateExpression(idx)}>Preview</button>
                <button onClick={() => remove(idx)} style={{ marginLeft: 4 }}>Delete</button>
                {staticIssues[idx] ? (
                  <div style={{ color: 'crimson', marginTop: 4, fontSize: 12 }}>
                    {staticIssues[idx].illegal_chars?.length ? `Illegal: ${staticIssues[idx].illegal_chars.join('')}` : ''}
                    {staticIssues[idx].unknown_identifiers?.length ? ` Unknown: ${staticIssues[idx].unknown_identifiers.join(', ')}` : ''}
                  </div>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
        <b>Preview on:</b>
        <input value={pair} onChange={e => setPair(e.target.value)} style={{ width: 140 }} />
        <input value={timeframe} onChange={e => setTimeframe(e.target.value)} style={{ width: 80 }} />
        <span style={{ color: '#777' }}>
          Allowed funcs: {allowed.functions.join(', ')}
        </span>
        <button onClick={loadColumns}>Reload Columns</button>
      </div>
      {previewStats ? (
        <div style={{ marginTop: 8, border: '1px solid #ddd', padding: 8 }}>
          <b>Preview stats {previewIdx !== null ? `(row ${previewIdx})` : ''}</b>
          <div>count={previewStats.count} mean={Number.isFinite(previewStats.mean) ? previewStats.mean.toFixed(6) : previewStats.mean} std={Number.isFinite(previewStats.std) ? previewStats.std.toFixed(6) : previewStats.std} min={Number.isFinite(previewStats.min) ? previewStats.min.toFixed(6) : previewStats.min} max={Number.isFinite(previewStats.max) ? previewStats.max.toFixed(6) : previewStats.max}</div>
          <div>quantiles: {Object.entries(previewStats.quantiles || {}).map(([k,v]) => `${k}:${(Number(v)).toFixed(4)}`).join(' , ')}</div>
          <div>z-quantiles: {Object.entries(previewStats.z_quantiles || {}).map(([k,v]) => `${k}:${(Number(v)).toFixed(4)}`).join(' , ')}</div>
        </div>
      ) : null}
    </div>
  )
}
