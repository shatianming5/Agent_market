import React, { useEffect, useMemo, useState } from 'react'
import { getJSON } from '../api'
import type { JobStatus, JobLogEntry } from '../types'
import { Card, CardHeader, CardBody, SectionTitle, StatusChip, Stack } from './ui'

type FilterMode = 'all' | 'running' | 'completed' | 'failed'

const FILTER_OPTIONS: Array<{ value: FilterMode; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
]

const formatDateTime = (value?: string) => {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const jobStatusMeta = (job: JobStatus) => {
  if (job.running) return { label: 'Running', tone: 'running' as const }
  if (job.returncode === 0) return { label: 'Completed', tone: 'success' as const }
  if (job.returncode === null) return { label: 'Queued', tone: 'default' as const }
  return { label: `Failed (${job.returncode})`, tone: 'danger' as const }
}

const parseProgress = (entries: JobLogEntry[]) => {
  let current = 0
  let total = 0
  let label = ''
  let status = ''
  let percent: number | null = null
  for (const entry of entries) {
    const evt = entry.event
    if (evt && typeof evt === 'object' && (evt as any).event === 'progress') {
      const progress = evt as any
      if (progress.current !== undefined) current = Number(progress.current)
      if (progress.total !== undefined) total = Number(progress.total)
      if (progress.label) label = String(progress.label)
      if (progress.status) status = String(progress.status)
      if (progress.percent !== undefined) percent = Number(progress.percent)
    }
  }
  return { current, total, label, status, percent }
}

const commandToString = (cmd?: string[] | null) => {
  if (!cmd || !Array.isArray(cmd) || !cmd.length) return '--'
  return cmd.join(' ')
}

const parseTrainingSummary = (entries: any[]): { model_path?: string; metrics?: Record<string, any> } | null => {
  for (let i = entries.length - 1; i >= 0; i--) {
    const text = entries[i].text
    try {
      const payload = JSON.parse(text)
      if (payload && payload.metrics) {
        return { model_path: payload.model_path, metrics: payload.metrics }
      }
    } catch {
      // ignore
    }
  }
  const metrics: Record<string, number> = {}
  let modelPath = ''
  for (let i = entries.length - 1; i >= 0; i--) {
    const raw = String(entries[i].text || '').trim()
    if (!modelPath && raw.startsWith('Model:')) modelPath = raw.replace(/^Model:\s*/, '')
    if (!Object.keys(metrics).length && raw.startsWith('Metrics:')) {
      const parts = raw.replace(/^Metrics:\s*/, '').split(',').map((item) => item.trim())
      parts.forEach((item) => {
        const [key, value] = item.split('=')
        if (key && value) metrics[key.trim()] = Number(value)
      })
    }
  }
  if (!modelPath && !Object.keys(metrics).length) return null
  return { model_path: modelPath, metrics }
}

export default function JobsPanel({
  jobs,
  selectedJob,
  onOpen,
  onTerminate,
  jobLogs,
}: {
  jobs: JobStatus[]
  selectedJob: JobStatus | null
  onOpen: (job: JobStatus) => void
  onTerminate: () => void
  jobLogs: JobLogEntry[]
}) {
  const [filter, setFilter] = useState<FilterMode>('all')
  const [steps, setSteps] = useState<any[]>([])
  const [stats, setStats] = useState<any[]>([])
  const [summary, setSummary] = useState<any | null>(null)
  const [trainResult, setTrainResult] = useState<any | null>(null)
  const [command, setCommand] = useState('--')
  const [isBacktest, setIsBacktest] = useState(false)
  const [isTraining, setIsTraining] = useState(false)

  useEffect(() => {
    setSummary(null)
    setTrainResult(null)
    setSteps([])
    setStats([])
    setCommand('--')
    setIsBacktest(false)
    setIsTraining(false)
  }, [selectedJob?.id])

  useEffect(() => {
    if (!selectedJob) return
    let timer: number | undefined

    const fetchDetails = async () => {
      try {
        const statusRes = await getJSON<any>(`/jobs/${selectedJob.id}/status`)
        const cmd = Array.isArray(statusRes?.cmd) ? statusRes.cmd : []
        const cmdString = commandToString(cmd)
        setCommand(cmdString)
        const lower = cmdString.toLowerCase()
        const backtest = lower.includes('backtest')
        const training =
          lower.includes('train_ml.py') || lower.includes('train-xgb') || lower.includes('train-cat') || lower.includes('train')
        setIsBacktest(backtest)
        setIsTraining(training)

        const stepsRes = await getJSON<{ steps?: any[] }>(`/jobs/${selectedJob.id}/steps`)
        setSteps(stepsRes.steps || [])
        const statsRes = await getJSON<{ stats?: any[] }>('/steps/stats')
        setStats(statsRes.stats || [])

        if (statusRes && statusRes.running === false) {
          if (backtest) {
            try {
              const s = await getJSON<any>('/backtest/summary/latest')
              setSummary(s)
            } catch {
              setSummary(null)
            }
          }
          if (training && !trainResult) {
            try {
              const logs = await getJSON<any>(`/jobs/${selectedJob.id}/logs?offset=0&limit=1000&structured=true`)
              const entries = logs.entries || []
              const parsed = parseTrainingSummary(entries)
              if (parsed) setTrainResult(parsed)
            } catch {
              // ignore
            }
          }
        }
      } catch (err) {
        console.warn('Failed to fetch job details', err)
      }
    }

    fetchDetails()
    if (selectedJob.running) {
      timer = window.setInterval(fetchDetails, 3000)
    }
    return () => {
      if (timer) window.clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJob, trainResult])

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (filter === 'all') return true
      if (filter === 'running') return job.running
      if (filter === 'completed') return !job.running && job.returncode === 0
      if (filter === 'failed') return !job.running && job.returncode !== 0 && job.returncode !== null
      return true
    })
  }, [jobs, filter])

  const activeJob = selectedJob
  const progress = useMemo(() => parseProgress(jobLogs), [jobLogs, activeJob?.id])

  return (
    <div className="jobs-layout">
      <div className="jobs-column jobs-column--table">
        <Card>
          <CardHeader
            title="Job monitor"
            subtitle="Review the latest executions and inspect their status."
            actions={<StatusChip label={`${jobs.length} jobs`} tone="default" />}
          />
          <CardBody>
            <div className="jobs-filter">
              <select value={filter} onChange={(e) => setFilter(e.target.value as FilterMode)}>
                {FILTER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="jobs-table">
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Started</th>
                    <th>Finished</th>
                    <th>Lines</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredJobs.length === 0 ? (
                    <tr>
                      <td colSpan={5}>
                        <div className="expr-empty">No jobs match the current filter.</div>
                      </td>
                    </tr>
                  ) : (
                    filteredJobs.map((job) => {
                      const meta = jobStatusMeta(job)
                      const isActive = activeJob && activeJob.id === job.id
                      return (
                        <tr
                          key={job.id}
                          className={isActive ? 'jobs-row--active' : ''}
                          onClick={() => onOpen(job)}
                          role="button"
                        >
                          <td>{job.id}</td>
                          <td>
                            <StatusChip label={meta.label} tone={meta.tone} />
                          </td>
                          <td>{formatDateTime(job.started_at)}</td>
                          <td>{formatDateTime(job.finished_at)}</td>
                          <td className="text-right">{job.lines}</td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      </div>

      <div className="jobs-column jobs-column--detail">
        <Card>
          <CardHeader
            title={activeJob ? `Job details - ${activeJob.id}` : 'Job details'}
            subtitle={
              activeJob ? `Started ${formatDateTime(activeJob.started_at)}` : 'Select a job to inspect runtime details.'
            }
            actions={
              activeJob ? (
                <StatusChip label={jobStatusMeta(activeJob).label} tone={jobStatusMeta(activeJob).tone} />
              ) : undefined
            }
          />
          <CardBody>
            {!activeJob ? (
              <div className="expr-empty">Select a job from the list to display its logs and metrics.</div>
            ) : (
              <Stack gap={16}>
                <div className="jobs-meta">
                  <div>
                    <SectionTitle>Command</SectionTitle>
                    <code className="jobs-command">{command}</code>
                  </div>
                  <div className="jobs-meta-grid">
                    <div>
                      <span className="jobs-detail-label">Started</span>
                      <span>{formatDateTime(activeJob.started_at)}</span>
                    </div>
                    <div>
                      <span className="jobs-detail-label">Finished</span>
                      <span>{formatDateTime(activeJob.finished_at)}</span>
                    </div>
                    <div>
                      <span className="jobs-detail-label">Lines</span>
                      <span>{activeJob.lines}</span>
                    </div>
                  </div>
                </div>
                <div className="status-card">
                  <div className="status-card-header">
                    <div>
                      <div className="status-title">{progress.label || 'Progress'}</div>
                      <div className="status-id">{progress.status || 'Idle'}</div>
                    </div>
                    <StatusChip
                      label={
                        progress.percent != null
                          ? `${Math.round(progress.percent)}%`
                          : progress.total
                            ? `${progress.current}/${progress.total}`
                            : 'Tracking'
                      }
                      tone="running"
                    />
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar-fill accent"
                      style={{ width: `${Math.min(progress.percent ?? (progress.total ? (progress.current / Math.max(progress.total, 1)) * 100 : 0), 100)}%` }}
                    />
                  </div>
                  <div className="status-meta">
                    <span>
                      Step {progress.current}/{progress.total || '?'}
                    </span>
                    <span>{progress.status || ''}</span>
                  </div>
                </div>
                <div className="jobs-detail-actions">
                  <button className="ghost-button" onClick={onTerminate} disabled={!activeJob.running}>
                    Terminate job
                  </button>
                  <button className="ghost-button" onClick={() => onOpen(activeJob)}>
                    Refresh details
                  </button>
                </div>
                <div className="jobs-logs">
                  <SectionTitle>Logs</SectionTitle>
                  <div className="jobs-logs__panel">
                    <pre>{jobLogs.map((entry) => `${entry.line}: ${entry.text}`).join('\n') || 'No logs available.'}</pre>
                  </div>
                </div>
                {steps.length ? (
                  <div className="jobs-steps">
                    <SectionTitle>Steps diagnostics</SectionTitle>
                    <table className="compact-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Step</th>
                          <th className="text-right">Duration (s)</th>
                          <th className="text-right">Average (s)</th>
                          <th className="text-right">Delta</th>
                        </tr>
                      </thead>
                      <tbody>
                        {steps.map((step: any, idx: number) => {
                          const stat = stats.find((item: any) => item.label === step.label)
                          const avg = stat ? Math.round(stat.avg_seconds) : null
                          const duration = typeof step.duration === 'number' ? step.duration : null
                          const delta = avg != null && duration != null ? duration - avg : null
                          return (
                            <tr key={idx}>
                              <td>
                                {step.idx}/{step.total}
                              </td>
                              <td>{step.label}</td>
                              <td className="text-right">{duration ?? '--'}</td>
                              <td className="text-right">{avg ?? '--'}</td>
                              <td className={`text-right ${delta != null && delta > 5 ? 'text-danger' : ''}`}>
                                {delta ?? '--'}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : null}
                {isBacktest && summary ? (
                  <div className="jobs-summary">
                    <SectionTitle>Backtest summary</SectionTitle>
                    <div className="status-card">
                      <div className="status-card-header">
                        <div>
                          <div className="status-title">{summary.strategy}</div>
                          <div className="status-id">{summary.source}</div>
                        </div>
                        <StatusChip label={`${summary.metrics?.profit_total_pct ?? '--'}%`} tone="success" />
                      </div>
                      <div className="status-details">
                        <div>Total trades: {summary.metrics?.trades ?? '--'}</div>
                        <div>Win rate: {summary.metrics?.winrate ?? '--'}</div>
                      </div>
                    </div>
                  </div>
                ) : null}
                {isTraining && trainResult ? (
                  <div className="jobs-summary">
                    <SectionTitle>Training summary</SectionTitle>
                    <div className="status-card">
                      <div className="status-details">
                        <div>
                          <strong>Model path</strong> {trainResult.model_path || '--'}
                        </div>
                        <div>
                          <strong>Metrics</strong>{' '}
                          {trainResult.metrics
                            ? Object.entries(trainResult.metrics)
                                .map(([key, value]) => `${key}: ${value}`)
                                .join(' | ')
                            : '--'}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}
              </Stack>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
