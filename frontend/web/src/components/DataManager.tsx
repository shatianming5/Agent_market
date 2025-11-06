import React, { useEffect, useMemo, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api'
import { Card, CardHeader, CardBody, SectionTitle, SectionDescription, StatusChip, Stack } from './ui'

type SummaryRow = { pair: string; count: number; start?: string; end?: string }
type MissingRecord = Record<string, any>

const EXCHANGE_PRESETS = ['binanceus', 'binance', 'okx']
const TIMEFRAME_PRESETS = ['1h', '4h', '12h', '1d']
const TIMERANGE_PRESETS = ['20200701-20210131', '20200101-20241231']
const PAIR_PRESETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'ATOM/USDT', 'XRP/USDT']

const parseList = (value: string): string[] =>
  value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

const formatTimestamp = (value?: string | null) => {
  if (!value) return 'N/A'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const combineList = (list: string[]) => Array.from(new Set(list.filter(Boolean))).join(',')

export default function DataManager({
  defaultTimeframe,
  defaultTimerange,
  defaultPairs,
  onJob,
}: {
  defaultTimeframe: string
  defaultTimerange: string
  defaultPairs: string
  onJob?: () => void
}) {
  const [exchange, setExchange] = useState(EXCHANGE_PRESETS[0])
  const [timeframes, setTimeframes] = useState(defaultTimeframe)
  const [pairs, setPairs] = useState(defaultPairs)
  const [timerange, setTimerange] = useState(defaultTimerange)
  const [summary, setSummary] = useState<Record<string, SummaryRow[]>>({})
  const [summaryUpdatedAt, setSummaryUpdatedAt] = useState<string | null>(null)
  const [missing, setMissing] = useState<{ missing: MissingRecord[]; insufficient: MissingRecord[] }>({ missing: [], insufficient: [] })
  const [selectedMissing, setSelectedMissing] = useState<Record<string, boolean>>({})
  const [selectedInsufficient, setSelectedInsufficient] = useState<Record<string, boolean>>({})
  const [formErrors, setFormErrors] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [lastCheckAt, setLastCheckAt] = useState<string | null>(null)
  const [erase, setErase] = useState(false)
  const [newPairs, setNewPairs] = useState(false)
  const [prepend, setPrepend] = useState(false)

  const [jobId, setJobId] = useState<string | null>(null)
  const [jobPct, setJobPct] = useState(0)
  const [jobLabel, setJobLabel] = useState('')
  const [jobRunning, setJobRunning] = useState(false)
  const [jobElapsed, setJobElapsed] = useState<number | undefined>(undefined)
  const pollRef = useRef<number | null>(null)

  const timeframeList = useMemo(() => parseList(timeframes), [timeframes])
  const pairList = useMemo(() => parseList(pairs), [pairs])

  const missingCount = missing.missing.length
  const insufficientCount = missing.insufficient.length

  const selectedMissingCount = useMemo(() => {
    const miss = Object.values(selectedMissing).filter(Boolean).length
    const insuf = Object.values(selectedInsufficient).filter(Boolean).length
    return miss + insuf
  }, [selectedMissing, selectedInsufficient])

  const coverageTotal = useMemo(() => {
    const totals = Object.entries(summary).map(([tf, rows]) => ({
      timeframe: tf,
      pairs: rows.length,
      candles: rows.reduce((acc, row) => acc + (row.count || 0), 0),
    }))
    return totals
  }, [summary])

  const validateForm = () => {
    const errors: string[] = []
    if (!exchange.trim()) errors.push('Exchange is required.')
    if (!timeframeList.length) errors.push('Select at least one timeframe.')
    if (!pairList.length) errors.push('Provide at least one trading pair.')
    if (!timerange.trim()) {
      errors.push('Timerange is required.')
    } else if (!/^\d{8}-\d{8}$/.test(timerange.trim())) {
      errors.push('Timerange must follow the YYYYMMDD-YYYYMMDD pattern.')
    }
    return errors
  }

  const setPresetExchange = (value: string) => {
    setExchange(value)
  }

  const toggleTimeframe = (value: string) => {
    setTimeframes((prev) => {
      const list = parseList(prev)
      const next = list.includes(value) ? list.filter((item) => item !== value) : [...list, value]
      return combineList(next)
    })
  }

  const appendPair = (value: string) => {
    setPairs((prev) => {
      const list = parseList(prev)
      if (list.includes(value)) return prev
      return combineList([...list, value])
    })
  }

  const removePair = (value: string) => {
    setPairs((prev) => combineList(parseList(prev).filter((item) => item !== value)))
  }

  const clearJobPoll = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const refreshSummary = async () => {
    const errors = validateForm()
    setFormErrors(errors)
    if (errors.length) return
    setLoading(true)
    setStatusMessage('')
    setError('')
    try {
      const qs = new URLSearchParams({ exchange, timeframes: timeframeList.join(',') })
      const res = await getJSON<{ timeframes: Record<string, SummaryRow[]> }>(`/data/summary?${qs.toString()}`)
      setSummary(res.timeframes || {})
      setSummaryUpdatedAt(new Date().toISOString())
      setStatusMessage('Coverage summary updated.')
    } catch (err: any) {
      setError(err?.message || 'Failed to load coverage summary.')
    } finally {
      setLoading(false)
    }
  }

  const runCheckMissing = async () => {
    const errors = validateForm()
    setFormErrors(errors)
    if (errors.length) return
    setLoading(true)
    setStatusMessage('')
    setError('')
    try {
      const qs = new URLSearchParams({
        exchange,
        timeframes: timeframeList.join(','),
        pairs: pairList.join(','),
        timerange,
      })
      const res = await getJSON<{ missing: MissingRecord[]; insufficient: MissingRecord[] }>(`/data/check-missing?${qs.toString()}`)
      setMissing(res)
      const missingKeys: Record<string, boolean> = {}
      const insufKeys: Record<string, boolean> = {}
      for (const item of res.missing || []) missingKeys[`${item.pair}|${item.timeframe}`] = true
      for (const item of res.insufficient || []) insufKeys[`${item.pair}|${item.timeframe}`] = true
      setSelectedMissing(missingKeys)
      setSelectedInsufficient(insufKeys)
      setLastCheckAt(new Date().toISOString())
      setStatusMessage('Gap analysis refreshed.')
    } catch (err: any) {
      setError(err?.message || 'Failed to check missing data.')
    } finally {
      setLoading(false)
    }
  }

  const startTrack = (id: string) => {
    setJobId(id)
    setJobRunning(true)
    setJobPct(0)
    setJobLabel('')
    setJobElapsed(undefined)
    clearJobPoll()
    pollRef.current = window.setInterval(async () => {
      try {
        const progress = await getJSON<any>(`/jobs/${id}/progress`)
        setJobPct(progress.percent || 0)
        setJobLabel(progress.label || '')
        setJobRunning(!!progress.running)
        setJobElapsed(progress.elapsed)
        if (!progress.running) {
          clearJobPoll()
          setStatusMessage('Download job finished.')
          refreshSummary().catch(() => undefined)
          if (onJob) onJob()
        }
      } catch (err) {
        console.error(err)
      }
    }, 1200)
  }

  const runDownload = async (targetPairs?: string[], targetTimeframes?: string[]) => {
    const errors = validateForm()
    setFormErrors(errors)
    if (errors.length) return
    const tfs = targetTimeframes && targetTimeframes.length ? targetTimeframes : timeframeList
    const plist = targetPairs && targetPairs.length ? targetPairs : pairList
    if (!tfs.length || !plist.length) {
      setError('Select at least one timeframe and pair before downloading.')
      return
    }
    setError('')
    setStatusMessage('')
    try {
      const res = await postJSON<{ status: string; job_id: string }>('/run/download-data', {
        config: 'configs/config_freqai_multi.json',
        timeframes: tfs,
        pairs: plist,
        timerange,
        erase,
        new_pairs: newPairs,
        prepend,
      })
      startTrack(res.job_id)
    } catch (err: any) {
      setError(err?.message || 'Failed to start download job.')
    }
  }

  const downloadSelected = () => {
    const selectedPairs = new Set<string>()
    const selectedTimeframes = new Set<string>()
    for (const item of missing.missing || []) {
      if (selectedMissing[`${item.pair}|${item.timeframe}`]) {
        selectedPairs.add(item.pair)
        selectedTimeframes.add(item.timeframe)
      }
    }
    for (const item of missing.insufficient || []) {
      if (selectedInsufficient[`${item.pair}|${item.timeframe}`]) {
        selectedPairs.add(item.pair)
        selectedTimeframes.add(item.timeframe)
      }
    }
    if (!selectedPairs.size || !selectedTimeframes.size) {
      setError('Select at least one gap before downloading.')
      return
    }
    runDownload(Array.from(selectedPairs), Array.from(selectedTimeframes))
  }

  useEffect(() => {
    refreshSummary().catch(() => undefined)
    return () => clearJobPoll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const configChip = formErrors.length
    ? { label: 'Needs attention', tone: 'warning' as const }
    : pairList.length && timeframeList.length
      ? { label: 'Ready', tone: 'success' as const }
      : { label: 'Incomplete', tone: 'default' as const }

  const checkChip = lastCheckAt
    ? { label: `Last checked at ${formatTimestamp(lastCheckAt)}`, tone: 'success' as const }
    : { label: 'Not checked yet', tone: 'running' as const }

  const downloadChip = jobRunning
    ? { label: 'Download running', tone: 'running' as const }
    : jobId
      ? { label: 'Download finished', tone: 'success' as const }
      : { label: 'No downloads yet', tone: 'default' as const }

  return (
    <div className="data-lab-layout">
      <div className="data-lab-column">
        <Card>
          <CardHeader
            title="Step 1 - Configure Download Scope"
            subtitle="Set the exchange, timeframes, and pairs before auditing or downloading data."
            actions={<StatusChip label={configChip.label} tone={configChip.tone} />}
          />
          <CardBody>
            {statusMessage ? <div className="data-lab__status success">{statusMessage}</div> : null}
            {error ? <div className="data-lab__status danger">{error}</div> : null}
            {formErrors.length ? (
              <div className="data-lab__status warning">
                {formErrors.map((errMsg) => (
                  <div key={errMsg}>{errMsg}</div>
                ))}
              </div>
            ) : null}
            <SectionTitle>Download Parameters</SectionTitle>
            <SectionDescription>
              Use quick chips or type directly to describe the dataset you need.
            </SectionDescription>
            <div className="form-grid">
              <label className="form-control">
                <span className="form-label">Exchange</span>
                <input value={exchange} onChange={(e) => setExchange(e.target.value)} placeholder="binanceus" />
                <div className="pill-row">
                  {EXCHANGE_PRESETS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`pill ${exchange === item ? 'pill--active' : ''}`}
                      onClick={() => setPresetExchange(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </label>
              <label className="form-control">
                <span className="form-label">Timeframes</span>
                <input value={timeframes} onChange={(e) => setTimeframes(e.target.value)} placeholder="4h,1d" />
                <div className="pill-row">
                  {TIMEFRAME_PRESETS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`pill ${timeframeList.includes(item) ? 'pill--active' : ''}`}
                      onClick={() => toggleTimeframe(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </label>
              <label className="form-control">
                <span className="form-label">Timerange</span>
                <input value={timerange} onChange={(e) => setTimerange(e.target.value)} placeholder="YYYYMMDD-YYYYMMDD" />
                <div className="pill-row">
                  {TIMERANGE_PRESETS.map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`pill ${timerange === item ? 'pill--active' : ''}`}
                      onClick={() => setTimerange(item)}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </label>
              <label className="form-control form-control-wide">
                <span className="form-label">Trading pairs</span>
                <input value={pairs} onChange={(e) => setPairs(e.target.value)} placeholder="BTC/USDT,ETH/USDT" />
                <div className="pill-row">
                  {PAIR_PRESETS.map((item) => (
                    <button key={item} type="button" className="pill" onClick={() => appendPair(item)}>
                      {item}
                    </button>
                  ))}
                </div>
                <div className="token-row">
                  {pairList.map((item) => (
                    <button
                      key={item}
                      type="button"
                      className="token"
                      onClick={() => removePair(item)}
                      title="Remove pair"
                    >
                      {item}
                      <span className="token__close">x</span>
                    </button>
                  ))}
                </div>
              </label>
            </div>
            <div className="button-group controls-inline">
              <button onClick={refreshSummary} disabled={loading}>
                {loading ? 'Working...' : 'Refresh coverage'}
              </button>
              <button onClick={runCheckMissing} disabled={loading}>
                {loading ? 'Working...' : 'Check gaps'}
              </button>
            </div>
            <div className="control-switches">
              <label className="control-switch">
                <input type="checkbox" checked={erase} onChange={(e) => setErase(e.target.checked)} />
                <span>Erase existing data</span>
              </label>
              <label className="control-switch">
                <input type="checkbox" checked={newPairs} onChange={(e) => setNewPairs(e.target.checked)} />
                <span>Only append new pairs</span>
              </label>
              <label className="control-switch">
                <input type="checkbox" checked={prepend} onChange={(e) => setPrepend(e.target.checked)} />
                <span>Backfill history before timerange</span>
              </label>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader
            title="Step 2 - Review Gaps"
            subtitle="Select the pairs and timeframes you want to redownload."
            actions={<StatusChip label={checkChip.label} tone={checkChip.tone} />}
          />
          <CardBody>
            <div className="data-lab-stat-row">
              <div>
                <span className="data-lab-stat__value">{missingCount}</span>
                <span className="data-lab-stat__label">Missing combos</span>
              </div>
              <div>
                <span className="data-lab-stat__value">{insufficientCount}</span>
                <span className="data-lab-stat__label">Insufficient coverage</span>
              </div>
              <div>
                <span className="data-lab-stat__value">{selectedMissingCount}</span>
                <span className="data-lab-stat__label">Selected items</span>
              </div>
            </div>
            <Stack gap={14}>
              <div>
                <SectionTitle>Missing combinations</SectionTitle>
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Pair</th>
                      <th>Timeframe</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {missing.missing.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="ui-table__empty">
                          No missing combinations for the current configuration.
                        </td>
                      </tr>
                    ) : (
                      missing.missing.map((item, index) => {
                        const key = `${item.pair}|${item.timeframe}`
                        return (
                          <tr key={`missing-${index}`}>
                            <td>
                              <input
                                type="checkbox"
                                checked={!!selectedMissing[key]}
                                onChange={(e) =>
                                  setSelectedMissing((prev) => ({ ...prev, [key]: e.target.checked }))
                                }
                              />
                            </td>
                            <td>{item.pair}</td>
                            <td>{item.timeframe}</td>
                            <td>{item.reason || 'Unknown'}</td>
                          </tr>
                        )
                      })
                    )}
                  </tbody>
                </table>
              </div>

              <div>
                <SectionTitle>Insufficient coverage</SectionTitle>
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Pair</th>
                      <th>Timeframe</th>
                      <th>Existing range</th>
                      <th>Target range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {missing.insufficient.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="ui-table__empty">
                          No rows fall short of the requested timerange.
                        </td>
                      </tr>
                    ) : (
                      missing.insufficient.map((item, index) => {
                        const key = `${item.pair}|${item.timeframe}`
                        return (
                          <tr key={`insufficient-${index}`}>
                            <td>
                              <input
                                type="checkbox"
                                checked={!!selectedInsufficient[key]}
                                onChange={(e) =>
                                  setSelectedInsufficient((prev) => ({ ...prev, [key]: e.target.checked }))
                                }
                              />
                            </td>
                            <td>{item.pair}</td>
                            <td>{item.timeframe}</td>
                            <td>[{item.file_start || 'N/A'} - {item.file_end || 'N/A'}]</td>
                            <td>
                              [
                                {item.want_start || timerange.split('-')[0]} -{' '}
                                {item.want_end || timerange.split('-')[1]}
                              ]
                            </td>
                          </tr>
                        )
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </Stack>
            <div className="button-group" style={{ justifyContent: 'flex-end', marginTop: 16 }}>
              <button className="ghost-button" onClick={() => runDownload()} disabled={loading}>
                {loading ? 'Working...' : 'Download full scope'}
              </button>
              <button onClick={downloadSelected} disabled={loading || selectedMissingCount === 0}>
                {loading ? 'Working...' : 'Download selection'}
              </button>
            </div>
          </CardBody>
        </Card>
        <Card>
          <CardHeader
            title="Step 3 - Monitor Downloads"
            subtitle="Track live progress for your most recent download job."
            actions={<StatusChip label={downloadChip.label} tone={downloadChip.tone} />}
          />
          <CardBody>
            {jobId ? (
              <div className="status-card">
                <div className="status-card-header">
                  <div>
                    <div className="status-title">Job ID</div>
                    <div className="status-id">{jobId}</div>
                  </div>
                  <StatusChip
                    label={jobRunning ? 'Download running' : 'Download finished'}
                    tone={jobRunning ? 'running' : 'success'}
                  />
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill accent" style={{ width: `${jobPct}%` }} />
                </div>
                <div className="status-meta">
                  <span>{jobLabel || 'In progress'}</span>
                  <span>
                    {jobPct}%{jobElapsed ? ` | ${jobElapsed}s` : ''}
                  </span>
                </div>
              </div>
            ) : (
              <div className="data-lab__status info">No download job has been started yet.</div>
            )}
          </CardBody>
        </Card>
      </div>
      <div className="data-lab-column data-lab-column--insight">
        <Card>
          <CardHeader
            title="Coverage overview"
            subtitle="Latest coverage counts grouped by timeframe."
            actions={
              <StatusChip
                label={summaryUpdatedAt ? `Summary updated at ${formatTimestamp(summaryUpdatedAt)}` : 'No summary yet'}
                tone={summaryUpdatedAt ? 'success' : 'default'}
              />
            }
          />
          <CardBody>
            {Object.keys(summary).length === 0 ? (
              <div className="data-lab__status info">No coverage data yet. Refresh coverage to load a summary.</div>
            ) : (
              Object.entries(summary).map(([tf, rows]) => (
                <div key={tf} className="data-lab-summary">
                  <div className="data-lab-summary__header">
                    <h4>{tf}</h4>
                    <span>{rows.length} pairs</span>
                  </div>
                  <table className="compact-table">
                    <thead>
                      <tr>
                        <th>Pair</th>
                        <th className="text-right">Candles</th>
                        <th>Start</th>
                        <th>End</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, index) => (
                        <tr key={`${tf}-${index}`}>
                          <td>{row.pair}</td>
                          <td className="text-right">{row.count}</td>
                          <td>{row.start || 'N/A'}</td>
                          <td>{row.end || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Coverage totals" subtitle="Quick totals across the selected timeframes." />
          <CardBody>
            {coverageTotal.length === 0 ? (
              <div className="data-lab__status info">No totals yet. Refresh coverage to calculate them.</div>
            ) : (
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Timeframe</th>
                    <th className="text-right">Pairs</th>
                    <th className="text-right">Candles</th>
                  </tr>
                </thead>
                <tbody>
                  {coverageTotal.map((item) => (
                    <tr key={item.timeframe}>
                      <td>{item.timeframe}</td>
                      <td className="text-right">{item.pairs}</td>
                      <td className="text-right">{item.candles}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
