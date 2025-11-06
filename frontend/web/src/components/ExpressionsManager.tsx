import React, { useEffect, useMemo, useRef, useState } from 'react'
import { getJSON, postJSON, putJSON } from '../api'
import { Card, CardHeader, CardBody, SectionTitle, StatusChip, Stack } from './ui'

type ExpressionItem = {
  enabled?: boolean
  name?: string
  expression?: string
  entry_threshold?: string
  exit_threshold?: string
  offline_profit?: string
  win_ratio?: string
  offline_trade_count?: string
  signal_sharpe?: string
  stability_mean?: string
  adjusted_score?: string
  [key: string]: any
}

type FilterMode = 'all' | 'enabled' | 'disabled'

type AllowedCatalog = {
  functions: string[]
  base_columns: string[]
}

type StaticIssues = {
  illegal_chars?: string[]
  unknown_identifiers?: string[]
  used_funcs?: string[]
}

type PreviewStats = {
  count?: number
  mean?: number
  std?: number
  quantiles?: Record<string, number>
  z_quantiles?: Record<string, number>
  head?: unknown
  tail?: unknown
}

const FILTER_OPTIONS: Array<{ value: FilterMode; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'enabled', label: 'Enabled' },
  { value: 'disabled', label: 'Disabled' },
]

const DEFAULT_PAIR = 'ADA/USDT'
const DEFAULT_TIMEFRAME = '4h'
const DEFAULT_FEATURE_FILE = '../../resources/user_data/freqai_features_multi.json'
const DEFAULT_OUTPUT_PATH = '../../resources/user_data/freqai_expressions.json'
const DEFAULT_CONFIG_PATH = '../../config/configs/config_freqai_multi.json'
const DEFAULT_TIMERANGE = '20200701-20210131'
const DEFAULT_STRATEGY_PATH = '../freqtrade/user_data/strategies'

const createExpression = (): ExpressionItem => ({
  enabled: true,
  name: '',
  expression: '',
  entry_threshold: '',
  exit_threshold: '',
  offline_profit: '',
  win_ratio: '',
  offline_trade_count: '',
  signal_sharpe: '',
  stability_mean: '',
  adjusted_score: '',
})

export default function ExpressionsManager({ onJob }: { onJob?: () => void }) {
  const [expressions, setExpressions] = useState<ExpressionItem[]>([])
  const [activeIndex, setActiveIndex] = useState<number>(-1)
  const [path, setPath] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [filterText, setFilterText] = useState('')
  const [filterMode, setFilterMode] = useState<FilterMode>('all')
  const [allowed, setAllowed] = useState<AllowedCatalog>({ functions: [], base_columns: [] })
  const [previewIdx, setPreviewIdx] = useState<number | null>(null)
  const [previewStats, setPreviewStats] = useState<PreviewStats | null>(null)
  const [pair, setPair] = useState(DEFAULT_PAIR)
  const [timeframe, setTimeframe] = useState(DEFAULT_TIMEFRAME)
  const [staticIssues, setStaticIssues] = useState<Record<number, StaticIssues>>({})
  const [columns, setColumns] = useState<string[]>([])
  const [llmModel, setLlmModel] = useState('gpt-4o-mini')
  const [llmCount, setLlmCount] = useState(20)
  const [llmLoops, setLlmLoops] = useState(1)
  const [llmTimeout, setLlmTimeout] = useState(60)
  const [apiKey, setApiKey] = useState('')
  const [featureFile, setFeatureFile] = useState(DEFAULT_FEATURE_FILE)
  const [outputPath, setOutputPath] = useState(DEFAULT_OUTPUT_PATH)
  const [genJobId, setGenJobId] = useState<string | null>(null)
  const [genPct, setGenPct] = useState(0)
  const [genLabel, setGenLabel] = useState('')
  const [genRunning, setGenRunning] = useState(false)
  const [genElapsed, setGenElapsed] = useState<number | undefined>(undefined)
  const pollRef = useRef<number | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [evaluation, setEvaluation] = useState<any | null>(null)
  const [evaluationLoading, setEvaluationLoading] = useState(false)
  const [autoBacktest, setAutoBacktest] = useState(true)

  const enabledCount = useMemo(() => expressions.filter((expr) => !!expr.enabled).length, [expressions])

  const filteredExpressions = useMemo(() => {
    const search = filterText.trim().toLowerCase()
    return expressions
      .map((expr, idx) => ({ expr, idx }))
      .filter(({ expr }) => {
        if (filterMode === 'enabled' && !expr.enabled) return false
        if (filterMode === 'disabled' && expr.enabled) return false
        if (!search) return true
        const name = (expr.name || '').toLowerCase()
        const body = (expr.expression || '').toLowerCase()
        return name.includes(search) || body.includes(search)
      })
  }, [expressions, filterText, filterMode])

  const activeExpression = activeIndex >= 0 ? expressions[activeIndex] : null

  const clearPoll = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  async function load() {
    setLoading(true)
    setError('')
    try {
      const res = await getJSON<{ path: string; expressions: ExpressionItem[] }>('/expressions')
      setPath(res.path)
      const next = res.expressions || []
      setExpressions(next)
      setActiveIndex(next.length ? 0 : -1)
    } catch (err: any) {
      setError(err?.message || 'Failed to load expressions.')
    } finally {
      setLoading(false)
    }
  }

  async function refreshAllowed() {
    try {
      const res = await getJSON<AllowedCatalog>('/expressions/allowed')
      setAllowed({
        functions: Array.isArray(res.functions) ? res.functions : [],
        base_columns: Array.isArray(res.base_columns) ? res.base_columns : [],
      })
    } catch (err) {
      console.warn('Failed to load allowed functions', err)
    }
  }

  useEffect(() => {
    load()
    refreshAllowed().catch(() => undefined)
    return () => {
      clearPoll()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!expressions.length) {
      setActiveIndex(-1)
      return
    }
    setActiveIndex((prev) => {
      if (prev < 0) return 0
      if (prev >= expressions.length) return expressions.length - 1
      return prev
    })
  }, [expressions])

  const updateExpression = (idx: number, updates: Partial<ExpressionItem>) => {
    setExpressions((prev) => prev.map((item, i) => (i === idx ? { ...item, ...updates } : item)))
  }

  const addNew = () => {
    setExpressions((prev) => {
      const next = [...prev, createExpression()]
      setActiveIndex(next.length - 1)
      return next
    })
  }

  const remove = (idx: number) => {
    setExpressions((prev) => prev.filter((_, i) => i !== idx))
    setStaticIssues((prev) => {
      const next: Record<number, StaticIssues> = {}
      Object.entries(prev).forEach(([key, value]) => {
        const oldIdx = Number(key)
        if (oldIdx === idx) return
        next[oldIdx > idx ? oldIdx - 1 : oldIdx] = value
      })
      return next
    })
    setPreviewIdx((prev) => {
      if (prev === null) return prev
      if (prev === idx) return null
      return prev > idx ? prev - 1 : prev
    })
  }

  const waitForJob = async (jobId: string, label: string) => {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        const status = await getJSON<{ running?: boolean }>(`/jobs/${jobId}/status`)
        if (status && status.running === false) {
          return
        }
      } catch (err) {
        console.warn(`Failed to poll ${label} status`, err)
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1500))
    }
  }

  async function saveAll() {
    setLoading(true)
    setError('')
    try {
      await putJSON('/expressions', { expressions })
    } catch (err: any) {
      setError(err?.message || 'Failed to save expressions.')
    } finally {
      setLoading(false)
    }
  }

  async function validate(idx: number) {
    if (idx < 0 || idx >= expressions.length) return
    const item = expressions[idx]
    if (!item?.expression) return
    setLoading(true)
    setError('')
    try {
      const issues = await postJSON<StaticIssues>('/expressions/validate', item.expression)
      setStaticIssues((prev) => ({ ...prev, [idx]: issues }))
      const preview = await postJSON<PreviewStats>('/expressions/preview', {
        pair,
        timeframe,
        expression: item.expression,
        config: DEFAULT_CONFIG_PATH,
        apply_features: true,
      })
      setPreviewIdx(idx)
      setPreviewStats(preview)
    } catch (err: any) {
      setError(err?.message || 'Validation failed.')
      setPreviewIdx(idx)
      setPreviewStats(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadColumns() {
    setError('')
    try {
      const qs = new URLSearchParams({
        pair,
        timeframe,
        config: DEFAULT_CONFIG_PATH,
        apply_features: 'true',
      })
      const res = await getJSON<{ columns: string[] }>(`/data/columns?${qs.toString()}`)
      setColumns(res.columns || [])
    } catch (err: any) {
      setError(err?.message || 'Failed to load feature columns.')
    }
  }

  const startTrack = (id: string) => {
    setGenJobId(id)
    setGenRunning(true)
    setGenPct(0)
    setGenLabel('')
    setGenElapsed(undefined)
    clearPoll()
    pollRef.current = window.setInterval(async () => {
      try {
        const progress = await getJSON<{ percent?: number; label?: string; running?: boolean; elapsed?: number }>(
          `/jobs/${id}/progress`,
        )
        setGenPct(progress.percent ?? 0)
        setGenLabel(progress.label || '')
        setGenRunning(progress.running ?? false)
        setGenElapsed(progress.elapsed)
        if (!progress.running) {
          clearPoll()
          await load()
          if (onJob) onJob()
        }
      } catch (err) {
        console.warn('Failed to poll generation progress', err)
      }
    }, 1200)
  }

  async function generateExpressions() {
    setError('')
    try {
      const res = await postJSON<{ status: string; job_id: string }>('/run/expression', {
        config: DEFAULT_CONFIG_PATH,
        feature_file: featureFile,
        output: outputPath,
        timeframe,
        llm_model: llmModel,
        llm_count: llmCount,
        llm_loops: llmLoops,
        llm_timeout: llmTimeout,
        feedback_top: 0,
        fast: true,
        llm_api_key: apiKey || undefined,
      })
      startTrack(res.job_id)
      return res.job_id
    } catch (err: any) {
      setError(err?.message || 'Failed to generate expressions.')
      return null
    }
  }

  async function generateAndBacktest() {
    setEvaluation(null)
    setEvaluationLoading(true)
    const id = await generateExpressions()
    if (!id) {
      setEvaluationLoading(false)
      return
    }
    const waitForCompletion = async () => {
      try {
        await waitForJob(id, 'generation')
        const backtest = await postJSON<{ status: string; job_id: string }>('/run/backtest', {
          config: DEFAULT_CONFIG_PATH,
          strategy: 'ExpressionLongStrategy',
          strategy_path: DEFAULT_STRATEGY_PATH,
          timerange: DEFAULT_TIMERANGE,
          freqaimodel: 'LightGBMRegressor',
          export: false,
          fast: true,
        })
        if (backtest?.job_id) {
          await waitForJob(backtest.job_id, 'backtest')
          try {
            const latest = await getJSON<any>('/backtest/summary/latest')
            setEvaluation(latest)
          } catch (err) {
            console.warn('Failed to fetch backtest summary', err)
            setEvaluation(null)
            setError((err as Error)?.message || 'Backtest summary fetch failed.')
          }
          if (onJob) onJob()
        } else {
          setError('Backtest job failed to start.')
        }
      } catch (err) {
        console.warn('Waiting for generation/backtest job failed', err)
        setError((err as Error)?.message || 'Failed to complete generation pipeline.')
      } finally {
        setEvaluationLoading(false)
      }
    }
    waitForCompletion().catch(() => undefined)
  }

  async function handleGenerateClick() {
    if (autoBacktest) {
      await generateAndBacktest()
    } else {
      setEvaluation(null)
      setEvaluationLoading(false)
      await generateExpressions()
    }
  }

  const renderIssues = (issues: StaticIssues | undefined) => {
    if (!issues) return <div className="expr-empty">No validation results yet.</div>
    return (
      <Stack gap={8}>
        {issues.illegal_chars?.length ? (
          <div className="expr-issue">
            <strong>Illegal characters:</strong> {issues.illegal_chars.join(', ')}
          </div>
        ) : null}
        {issues.unknown_identifiers?.length ? (
          <div className="expr-issue">
            <strong>Unknown functions:</strong> {issues.unknown_identifiers.join(', ')}
          </div>
        ) : null}
        {issues.used_funcs?.length ? (
          <div className="expr-issue">
            <strong>Used functions:</strong> {issues.used_funcs.join(', ')}
          </div>
        ) : null}
      </Stack>
    )
  }

  const previewActive = previewIdx !== null && previewIdx === activeIndex && !!previewStats

  return (
    <div className="expr-layout">
      <div className="expr-column expr-column--list">
        <Card className="expr-card expr-card--list">
          <CardHeader
            title="Expression list"
            subtitle="Filter, enable or disable expressions"
            actions={<StatusChip label={`${enabledCount}/${expressions.length} enabled`} tone="success" />}
          />
          <CardBody>
            {path ? (
              <div className="expr-path">
                Current file:<code>{path}</code>
              </div>
            ) : null}
            {error ? <div className="data-lab__status danger">{error}</div> : null}
            <div className="expr-filter">
              <input
                placeholder="Search name or expression"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
              />
              <select value={filterMode} onChange={(e) => setFilterMode(e.target.value as FilterMode)}>
                {FILTER_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="expr-list">
              {filteredExpressions.length === 0 ? (
                <div className="expr-empty">No expressions match the current filters.</div>
              ) : (
                filteredExpressions.map(({ expr, idx }) => {
                  const title = expr.name || `Expression #${idx + 1}`
                  return (
                    <div key={idx} className={`expr-item ${idx === activeIndex ? 'expr-item--active' : ''}`}>
                      <button className="expr-item__main" onClick={() => setActiveIndex(idx)}>
                        <span className="expr-item__title">{title}</span>
                        <span className="expr-item__preview">{expr.expression || 'No expression defined'}</span>
                      </button>
                      <div className="expr-item__actions">
                        <label className="expr-switch">
                          <input
                            type="checkbox"
                            checked={!!expr.enabled}
                            onChange={(e) => updateExpression(idx, { enabled: e.target.checked })}
                          />
                          <span>Enabled</span>
                        </label>
                        <button className="expr-delete" onClick={() => remove(idx)} title="Delete expression">
                          Remove
                        </button>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
            <div className="expr-list__actions">
              <button onClick={addNew}>Add expression</button>
            </div>
          </CardBody>
        </Card>
      </div>

      <div className="expr-column expr-column--editor">
        <Card className="expr-card expr-card--editor">
          <CardHeader
            title={
              activeExpression?.name ||
              (activeIndex >= 0 ? `Expression #${activeIndex + 1}` : 'No expression selected')
            }
            subtitle="Edit expression content and metrics"
            actions={
              activeExpression ? (
                <StatusChip label={activeExpression.enabled ? 'Enabled' : 'Disabled'} tone={activeExpression.enabled ? 'success' : 'default'} />
              ) : undefined
            }
          />
          <CardBody>
            {!activeExpression ? (
              <div className="expr-empty">Select or add an expression to start editing.</div>
            ) : (
              <Stack gap={16}>
                <label className="form-control">
                  <span className="form-label">Name</span>
                  <input
                    value={activeExpression.name || ''}
                    onChange={(e) => updateExpression(activeIndex, { name: e.target.value })}
                    placeholder="Give the expression a name"
                  />
                </label>
                <label className="form-control">
                  <span className="form-label">Expression</span>
                  <textarea
                    value={activeExpression.expression || ''}
                    onChange={(e) => updateExpression(activeIndex, { expression: e.target.value })}
                    rows={6}
                    placeholder="Example: ema(close, 12) - ema(close, 26)"
                  />
                </label>
                <div className="expr-field-grid">
                  <label className="form-control">
                    <span className="form-label">Entry threshold</span>
                    <input
                      value={activeExpression.entry_threshold || ''}
                      onChange={(e) => updateExpression(activeIndex, { entry_threshold: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Exit threshold</span>
                    <input
                      value={activeExpression.exit_threshold || ''}
                      onChange={(e) => updateExpression(activeIndex, { exit_threshold: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Offline profit</span>
                    <input
                      value={activeExpression.offline_profit || ''}
                      onChange={(e) => updateExpression(activeIndex, { offline_profit: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Win ratio</span>
                    <input
                      value={activeExpression.win_ratio || ''}
                      onChange={(e) => updateExpression(activeIndex, { win_ratio: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Trade count</span>
                    <input
                      value={activeExpression.offline_trade_count || ''}
                      onChange={(e) => updateExpression(activeIndex, { offline_trade_count: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Sharpe</span>
                    <input
                      value={activeExpression.signal_sharpe || ''}
                      onChange={(e) => updateExpression(activeIndex, { signal_sharpe: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Stability</span>
                    <input
                      value={activeExpression.stability_mean || ''}
                      onChange={(e) => updateExpression(activeIndex, { stability_mean: e.target.value })}
                    />
                  </label>
                  <label className="form-control">
                    <span className="form-label">Overall score</span>
                    <input
                      value={activeExpression.adjusted_score || ''}
                      onChange={(e) => updateExpression(activeIndex, { adjusted_score: e.target.value })}
                    />
                  </label>
                </div>
                <div className="expr-preview-config">
                  <label>
                    Pair
                    <input value={pair} onChange={(e) => setPair(e.target.value)} />
                  </label>
                  <label>
                    Timeframe
                    <input value={timeframe} onChange={(e) => setTimeframe(e.target.value)} />
                  </label>
                  <button className="ghost-button" onClick={loadColumns}>
                    Load feature columns
                  </button>
                </div>
                <div className="button-group">
                  <button onClick={() => validate(activeIndex)} disabled={loading}>
                    Validate & preview
                  </button>
                  <button className="ghost-button" onClick={saveAll} disabled={loading}>
                    Save changes
                  </button>
                </div>
              </Stack>
            )}
          </CardBody>
        </Card>

        <Card className="expr-card expr-card--insight">
          <CardHeader
            title="LLM generation"
            subtitle="Configure parameters and trigger expression generation"
            actions={
              genJobId ? (
                <StatusChip label={genRunning ? 'Generating' : 'Completed'} tone={genRunning ? 'running' : 'success'} />
              ) : undefined
            }
          />
          <CardBody>
            <div className="expr-generator-grid">
              <label>
                Model
                <input value={llmModel} onChange={(e) => setLlmModel(e.target.value)} />
              </label>
              <label>
                Generation count
                <input
                  type="number"
                  min={1}
                  value={llmCount}
                  onChange={(e) => setLlmCount(Number(e.target.value) || 0)}
                />
              </label>
              <label className="expr-generator-wide">
                Feature file
                <input value={featureFile} onChange={(e) => setFeatureFile(e.target.value)} />
              </label>
              <label className="expr-generator-wide">
                Output path
                <input value={outputPath} onChange={(e) => setOutputPath(e.target.value)} />
              </label>
            </div>
            <div className="expr-generator-toolbar">
                <div className="button-group expr-generator-actions">
                  <button onClick={handleGenerateClick} disabled={genRunning}>
                    {genRunning ? 'Running...' : autoBacktest ? 'Generate & backtest' : 'Generate factors'}
                  </button>
                  <label className="expr-generator-toggle">
                    <input
                      type="checkbox"
                      checked={autoBacktest}
                      onChange={(e) => setAutoBacktest(e.target.checked)}
                    />
                    <span>Auto-run backtest</span>
                  </label>
                </div>
              <button
                type="button"
                className="link-button"
                onClick={() => setShowAdvanced((prev) => !prev)}
              >
                {showAdvanced ? 'Hide advanced options' : 'Show advanced options'}
              </button>
            </div>
            {showAdvanced ? (
              <div className="expr-generator-grid expr-generator-grid--advanced">
                <label>
                  Loop count
                  <input
                    type="number"
                    min={1}
                    value={llmLoops}
                    onChange={(e) => setLlmLoops(Number(e.target.value) || 0)}
                  />
                </label>
                <label>
                  Timeout (s)
                  <input
                    type="number"
                    min={10}
                    value={llmTimeout}
                    onChange={(e) => setLlmTimeout(Number(e.target.value) || 0)}
                  />
                </label>
                <label className="expr-generator-wide">
                  API key (optional)
                  <input
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Leave empty to use the server configuration"
                  />
                </label>
              </div>
            ) : null}
            {genJobId ? (
              <div className="status-card" style={{ marginTop: 16 }}>
                <div className="status-card-header">
                  <div>
                    <div className="status-title">Generation job</div>
                    <div className="status-id">{genJobId}</div>
                  </div>
                  <StatusChip label={genRunning ? 'In progress' : 'Completed'} tone={genRunning ? 'running' : 'success'} />
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill warning" style={{ width: `${genPct}%` }} />
                </div>
                <div className="status-meta">
                  <span>{genLabel || 'Waiting for progress updates...'}</span>
                  <span>
                    {genPct}%{genElapsed ? ` | ${genElapsed}s` : ''}
                  </span>
                </div>
              </div>
            ) : null}
          </CardBody>
        </Card>
      </div>

      <div className="expr-column expr-column--insight">
        <Card className="expr-card expr-card--insight">
          <CardHeader title="Static validation" subtitle="Quickly review syntax issues and function usage" />
          <CardBody>{renderIssues(staticIssues[activeIndex])}</CardBody>
        </Card>
        <Card className="expr-card expr-card--generator">
          <CardHeader title="Preview statistics" subtitle="Inspect sampled metrics for the selected expression" />
          <CardBody>
            {!previewActive ? (
              <div className="expr-empty">Run a validation to populate preview statistics.</div>
            ) : (
              <div className="expr-preview-stats">
                <div>
                  <span className="expr-preview-label">Sample count</span>
                  <span className="expr-preview-value">{previewStats?.count ?? '--'}</span>
                </div>
                <div>
                  <span className="expr-preview-label">Mean</span>
                  <span className="expr-preview-value">{(previewStats?.mean ?? 0).toFixed(4)}</span>
                </div>
                <div>
                  <span className="expr-preview-label">Std dev</span>
                  <span className="expr-preview-value">{(previewStats?.std ?? 0).toFixed(4)}</span>
                </div>
                <div className="expr-preview-quantiles">
                  <strong>Quantiles</strong>
                  <pre>{JSON.stringify(previewStats?.quantiles || {}, null, 2)}</pre>
                </div>
                <div className="expr-preview-quantiles">
                  <strong>Z-score quantiles</strong>
                  <pre>{JSON.stringify(previewStats?.z_quantiles || {}, null, 2)}</pre>
                </div>
                <div className="expr-preview-quantiles">
                  <strong>Sample head & tail</strong>
                  <pre>{JSON.stringify({ head: previewStats?.head, tail: previewStats?.tail }, null, 2)}</pre>
                </div>
              </div>
            )}
          </CardBody>
        </Card>
        <Card className="expr-card expr-card--insight">
          <CardHeader title="Latest evaluation" subtitle="Backtest summary for generated factors" />
          <CardBody>
            {evaluationLoading ? (
              <div className="expr-empty">Backtest running...</div>
            ) : evaluation ? (
              <div className="expr-eval-summary">
                <div className="expr-eval-row">
                  <span className="expr-preview-label">Strategy</span>
                  <span className="expr-preview-value">{evaluation?.strategy ?? '--'}</span>
                </div>
                <div className="expr-eval-row">
                  <span className="expr-preview-label">Total return</span>
                  <span className="expr-preview-value">
                    {evaluation?.metrics?.profit_total_pct != null
                      ? `${Number(evaluation.metrics.profit_total_pct).toFixed(2)}%`
                      : '--'}
                  </span>
                </div>
                <div className="expr-eval-row">
                  <span className="expr-preview-label">Trades</span>
                  <span className="expr-preview-value">
                    {Array.isArray(evaluation?.metrics?.trades)
                      ? evaluation.metrics.trades.length
                      : evaluation?.metrics?.total_trades ?? '--'}
                  </span>
                </div>
                <div className="expr-eval-row">
                  <span className="expr-preview-label">Winrate</span>
                  <span className="expr-preview-value">
                    {evaluation?.metrics?.winrate != null
                      ? `${(Number(evaluation.metrics.winrate) * 100).toFixed(2)}%`
                      : '--'}
                  </span>
                </div>
                <div className="expr-eval-row">
                  <span className="expr-preview-label">Sharpe</span>
                  <span className="expr-preview-value">
                    {evaluation?.metrics?.sharpe != null
                      ? Number(evaluation.metrics.sharpe).toFixed(2)
                      : '--'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="expr-empty">
                {autoBacktest
                  ? 'Generate factors to populate the evaluation stats.'
                  : 'Enable auto-run backtest to populate this evaluation panel automatically.'}
              </div>
            )}
          </CardBody>
        </Card>
        <Card className="expr-card expr-card--insight">
          <CardHeader title="Available functions & features" subtitle="Use these references when authoring expressions" />
          <CardBody>
            <SectionTitle>Available functions</SectionTitle>
            <div className="expr-chip-cloud">
              {(allowed.functions || []).map((fn) => (
                <span key={fn} className="expr-chip">
                  {fn}
                </span>
              ))}
            </div>
            <SectionTitle>Baseline columns</SectionTitle>
            <div className="expr-chip-cloud">
              {(allowed.base_columns || []).map((col) => (
                <span key={col} className="expr-chip">
                  {col}
                </span>
              ))}
            </div>
            <SectionTitle>Example feature columns</SectionTitle>
            <div className="expr-chip-cloud">
              {columns.length ? (
                columns.slice(0, 60).map((col) => (
                  <span key={col} className="expr-chip expr-chip--ghost">
                    {col}
                  </span>
                ))
              ) : (
                <div className="expr-empty">Click &quot;Load feature columns&quot; to fetch an example list.</div>
              )}
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
