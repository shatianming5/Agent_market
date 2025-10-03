import React, { useEffect, useState } from 'react'
import { getJSON, postJSON } from '../api'

type ExpressionItem = { [k: string]: any }

export default function ExpressionsManager({ onJob }: { onJob?: () => void }) {
  const [expressions, setExpressions] = useState<ExpressionItem[]>([])
  const [path, setPath] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

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

  function update(idx: number, key: string, val: any) {
    setExpressions(prev => prev.map((x, i) => i===idx ? { ...x, [key]: val } : x))
  }

  function addNew() {
    setExpressions(prev => [...prev, { name: '', expression: '', entry_threshold: '', exit_threshold: '' }])
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
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>Name</th>
            <th style={{ textAlign: 'left' }}>Expression</th>
            <th>Entry Thr</th>
            <th>Exit Thr</th>
            <th style={{ width: 80 }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {expressions.map((item, idx) => (
            <tr key={idx}>
              <td><input value={item.name || ''} onChange={e => update(idx, 'name', e.target.value)} style={{ width: 180 }} /></td>
              <td><input value={item.expression || ''} onChange={e => update(idx, 'expression', e.target.value)} style={{ width: '100%' }} /></td>
              <td><input value={item.entry_threshold ?? ''} onChange={e => update(idx, 'entry_threshold', e.target.value)} style={{ width: 80 }} /></td>
              <td><input value={item.exit_threshold ?? ''} onChange={e => update(idx, 'exit_threshold', e.target.value)} style={{ width: 80 }} /></td>
              <td><button onClick={() => remove(idx)}>Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: '#555', marginTop: 8 }}>Tip: Allowed columns include open/high/low/close/volume and rolling features; strategy will safe-eval using whitelisted functions.</p>
    </div>
  )
}

