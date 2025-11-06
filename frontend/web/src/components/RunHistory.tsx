import React, { useEffect, useMemo, useState } from 'react'
import { getRunManifest, listRuns, getRunFileUrl, getRunEvents } from '../api'
import type { RunManifest, RunSummary } from '../types'
import { Card, CardHeader, CardBody, StatusChip, SectionTitle, Stack } from './ui'

type RunFilter = 'all' | 'completed' | 'failed' | 'running'

const FILTER_OPTIONS: Array<{ value: RunFilter; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'running', label: 'Running' },
]

const formatDateTime = (value?: string) => {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const statusMeta = (status?: string) => {
  const value = (status || '').toLowerCase()
  if (value === 'completed' || value === 'success') return { label: 'Completed', tone: 'success' as const }
  if (value === 'failed') return { label: 'Failed', tone: 'danger' as const }
  if (value === 'running' || value === 'in_progress') return { label: 'Running', tone: 'running' as const }
  if (value === 'cancelled') return { label: 'Cancelled', tone: 'warning' as const }
  return { label: status || 'Unknown', tone: 'default' as const }
}

export default function RunHistory() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [filter, setFilter] = useState<RunFilter>('all')
  const [selected, setSelected] = useState<RunSummary | null>(null)
  const [manifest, setManifest] = useState<RunManifest | null>(null)
  const [events, setEvents] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function loadRuns() {
    try {
      const data = await listRuns()
      setRuns(data)
      if (!selected && data.length) {
        setSelected(data[0])
      }
    } catch (err) {
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    loadRuns()
  }, [])

  async function openRun(run: RunSummary) {
    setSelected(run)
    setLoading(true)
    setError(null)
    try {
      const data = await getRunManifest(run.run_id)
      setManifest(data)
      try {
        const ev = await getRunEvents(run.run_id, 200)
        setEvents(ev)
      } catch {
        setEvents([])
      }
    } catch (err) {
      setManifest(null)
      setEvents([])
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => {
      if (filter === 'all') return true
      if (filter === 'completed') return (run.status || '').toLowerCase() === 'completed'
      if (filter === 'failed') return (run.status || '').toLowerCase() === 'failed'
      if (filter === 'running') return (run.status || '').toLowerCase() === 'running'
      return true
    })
  }, [runs, filter])

  function downloadLink(path?: string | null): string | null {
    if (!manifest || !path) return null
    return getRunFileUrl(manifest.run_id, path)
  }

  const selectedMeta = manifest ? statusMeta(manifest.status) : undefined

  return (
    <div className="runs-layout">
      <div className="runs-column">
        <Card>
          <CardHeader
            title="Run history"
            subtitle="Browse agent flow executions and drill into their artefacts."
            actions={
              <select value={filter} onChange={(e) => setFilter(e.target.value as RunFilter)}>
                {FILTER_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            }
          />
          <CardBody>
            <table className="compact-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Status</th>
                  <th>Started</th>
                  <th>Finished</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredRuns.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="ui-table__empty">
                      No runs available.
                    </td>
                  </tr>
                ) : (
                  filteredRuns.map((run) => {
                    const meta = statusMeta(run.status)
                    const isActive = selected && selected.run_id === run.run_id
                    return (
                      <tr key={run.run_id} className={isActive ? 'runs-row--active' : undefined}>
                        <td>{run.run_id}</td>
                        <td>
                          <StatusChip label={meta.label} tone={meta.tone} />
                        </td>
                        <td>{formatDateTime(run.started_at)}</td>
                        <td>{formatDateTime(run.finished_at)}</td>
                        <td>
                          <button className="jobs-row__id" onClick={() => openRun(run)}>
                            View details
                          </button>
                        </td>
                      </tr>
                    )
                  })
                )}
              </tbody>
            </table>
          </CardBody>
        </Card>
      </div>

      <div className="runs-column runs-column--detail">
        <Card>
          <CardHeader
            title={selected ? `Run detail - ${selected.run_id}` : 'Run detail'}
            subtitle={selected ? `Started ${formatDateTime(selected.started_at)}` : 'Select a run to inspect its manifest.'}
            actions={selectedMeta ? <StatusChip label={selectedMeta.label} tone={selectedMeta.tone} /> : undefined}
          />
          <CardBody>
            {error ? <div className="data-lab__status danger">{error}</div> : null}
            {loading ? <div className="expr-empty">Loading run data...</div> : null}
            {!selected ? (
              <div className="expr-empty">Choose a run from the table to view artefacts and logs.</div>
            ) : manifest ? (
              <Stack gap={16}>
                <div className="runs-detail-grid">
                  <div>
                    <span className="jobs-detail-label">Run directory</span>
                    <span className="jobs-detail-value">{manifest.run_dir || '--'}</span>
                  </div>
                  <div>
                    <span className="jobs-detail-label">Finished</span>
                    <span className="jobs-detail-value">{formatDateTime(manifest.finished_at)}</span>
                  </div>
                  {manifest.error ? (
                    <div>
                      <span className="jobs-detail-label">Error</span>
                      <span className="jobs-detail-value jobs-detail-value--command">{manifest.error}</span>
                    </div>
                  ) : null}
                </div>
                <div className="runs-links">
                  {manifest.config_snapshot ? (
                    <a href={downloadLink(manifest.config_snapshot) || '#'} target="_blank" rel="noreferrer">
                      Download config snapshot
                    </a>
                  ) : null}
                  {manifest.events_log ? (
                    <a href={downloadLink(manifest.events_log) || '#'} target="_blank" rel="noreferrer">
                      Download events log
                    </a>
                  ) : null}
                </div>
                <div className="runs-steps">
                  <SectionTitle>Step timeline</SectionTitle>
                  <table className="compact-table">
                    <thead>
                      <tr>
                        <th>Step</th>
                        <th>Status</th>
                        <th>Started</th>
                        <th>Finished</th>
                      </tr>
                    </thead>
                    <tbody>
                      {manifest.steps?.length ? (
                        manifest.steps.map((step, idx) => (
                          <tr key={idx}>
                            <td>{step.name}</td>
                            <td>{step.status}</td>
                            <td>{formatDateTime(step.started_at)}</td>
                            <td>{formatDateTime(step.finished_at)}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} className="ui-table__empty">
                            No steps were recorded.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="runs-events">
                  <SectionTitle>Recent events (max 200)</SectionTitle>
                  <div className="jobs-logs__panel">
                    <pre>{events.map((evt) => JSON.stringify(evt)).join('\n') || 'No events captured.'}</pre>
                  </div>
                </div>
              </Stack>
            ) : (
              <div className="expr-empty">Unable to load run manifest.</div>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
