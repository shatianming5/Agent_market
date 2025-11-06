import React, { useEffect, useState, useMemo } from 'react'

import { API_BASE, getJSON, postJSON, postEmpty, putJSON } from './api'
import ConnectorsManager from './components/ConnectorsManager'
import DataManager from './components/DataManager'
import ExpressionsManager from './components/ExpressionsManager'
import JobsPanel from './components/JobsPanel'
import RunHistory from './components/RunHistory'
import SecretsViewer from './components/SecretsViewer'
import StrategyParams from './components/StrategyParams'
import { Card } from './components/ui'
import { SimpleTable } from './components/ui/Table'
import type { JobLogEntry, JobStatus } from './types'

type SeriesPoint = { label: string; value: number; timestamp?: number }

const PLACEHOLDER_TRADES = [
  {
    pair: 'BTC/USDT',
    stake_amount: 1000,
    profit_ratio: 0.032,
    profit_abs: 320,
    trade_duration: 720,
    exit_reason: 'roi',
    open_date: '2024-09-16 08:00:00+00:00',
    close_date: '2024-09-16 20:00:00+00:00',
    open_timestamp: 1726473600000,
    close_timestamp: 1726516800000,
  },
  {
    pair: 'ETH/USDT',
    stake_amount: 1000,
    profit_ratio: -0.018,
    profit_abs: -180,
    trade_duration: 360,
    exit_reason: 'stop_loss',
    open_date: '2024-09-17 04:00:00+00:00',
    close_date: '2024-09-17 10:00:00+00:00',
    open_timestamp: 1726545600000,
    close_timestamp: 1726567200000,
  },
  {
    pair: 'SOL/USDT',
    stake_amount: 1000,
    profit_ratio: 0.026,
    profit_abs: 260,
    trade_duration: 600,
    exit_reason: 'roi',
    open_date: '2024-09-18 02:00:00+00:00',
    close_date: '2024-09-18 12:00:00+00:00',
    open_timestamp: 1726615200000,
    close_timestamp: 1726660800000,
  },
  {
    pair: 'ADA/USDT',
    stake_amount: 1000,
    profit_ratio: 0.009,
    profit_abs: 90,
    trade_duration: 480,
    exit_reason: 'roi',
    open_date: '2024-09-19 01:00:00+00:00',
    close_date: '2024-09-19 09:00:00+00:00',
    open_timestamp: 1726698000000,
    close_timestamp: 1726736400000,
  },
  {
    pair: 'AVAX/USDT',
    stake_amount: 1000,
    profit_ratio: -0.012,
    profit_abs: -120,
    trade_duration: 720,
    exit_reason: 'stop_loss',
    open_date: '2024-09-20 03:00:00+00:00',
    close_date: '2024-09-20 15:00:00+00:00',
    open_timestamp: 1726772400000,
    close_timestamp: 1726844400000,
  },
  {
    pair: 'DOGE/USDT',
    stake_amount: 1000,
    profit_ratio: 0.015,
    profit_abs: 150,
    trade_duration: 960,
    exit_reason: 'roi',
    open_date: '2024-09-20 18:00:00+00:00',
    close_date: '2024-09-21 18:00:00+00:00',
    open_timestamp: 1726855200000,
    close_timestamp: 1726941600000,
  },
]

const PLACEHOLDER_DAILY_PROFIT = [
  ['2024-09-12', -85],
  ['2024-09-13', 120],
  ['2024-09-14', 90],
  ['2024-09-15', -45],
  ['2024-09-16', 320],
  ['2024-09-17', -180],
  ['2024-09-18', 260],
  ['2024-09-19', 90],
  ['2024-09-20', -40],
  ['2024-09-21', 150],
  ['2024-09-22', 60],
  ['2024-09-23', 110],
  ['2024-09-24', 45],
  ['2024-09-25', 80],
]

const PLACEHOLDER_EXIT_SUMMARY = [
  { key: 'roi', trades: 38, profit_total_pct: 26.4, winrate: 0.68 },
  { key: 'stop_loss', trades: 12, profit_total_pct: -8.6, winrate: 0.22 },
  { key: 'force_exit', trades: 4, profit_total_pct: -1.3, winrate: 0.25 },
]

const PLACEHOLDER_PAIR_RESULTS = [
  { pair: 'BTC/USDT', trades: 18, profit_total_pct: 12.4 },
  { pair: 'ETH/USDT', trades: 16, profit_total_pct: 8.1 },
  { pair: 'SOL/USDT', trades: 15, profit_total_pct: 10.7 },
  { pair: 'ADA/USDT', trades: 14, profit_total_pct: 6.2 },
  { pair: 'AVAX/USDT', trades: 10, profit_total_pct: 4.4 },
  { pair: 'DOGE/USDT', trades: 12, profit_total_pct: 5.8 },
]

const PLACEHOLDER_COMPARISON = [
  { key: 'ExpressionLongStrategy', trades: 120, profit_total_pct: 48.6, winrate: 0.62 },
  { key: 'BaselineEMA', trades: 120, profit_total_pct: 17.4, winrate: 0.44 },
  { key: 'BuyAndHold', trades: 1, profit_total_pct: 9.1, winrate: 1.0 },
]

const PLACEHOLDER_BACKTEST_SUMMARY = {
  source: 'demo-sample-run.zip',
  strategy: 'ExpressionLongStrategy',
  metrics: {
    stake_currency: 'USDT',
    starting_balance: 10000,
    dry_run_wallet: 10000,
    stake_amount: 1000,
    winrate: 0.62,
    expectancy: 2.45,
    sharpe: 1.84,
    trades_per_day: 3.1,
    sortino: 2.27,
    profit_factor: 1.96,
    profit_total_pct: 5.2,
    profit_total_abs: 520,
    profit_mean: 0.0086,
    max_drawdown_abs: 520.35,
    max_drawdown_account: 0.051,
    max_consecutive_wins: 7,
    max_consecutive_losses: 3,
    trades: PLACEHOLDER_TRADES,
    daily_profit: PLACEHOLDER_DAILY_PROFIT,
    exit_reason_summary: PLACEHOLDER_EXIT_SUMMARY,
  },
  pairs: PLACEHOLDER_PAIR_RESULTS,
  comparison: PLACEHOLDER_COMPARISON,
} satisfies Record<string, any>

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

function formatCurrency(raw: unknown, currency = 'USDT') {
  const num = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isFinite(num)) return '--'
  return `${currencyFormatter.format(num)} ${currency}`
}

function formatPercent(raw: unknown, fractionDigits = 2, isRatio = false) {
  const num = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isFinite(num)) return '--'
  const pct = isRatio ? num * 100 : num
  return `${pct.toFixed(fractionDigits)}%`
}

function parseDurationMinutes(raw: unknown) {
  const num = typeof raw === 'number' ? raw : Number(raw)
  if (!Number.isFinite(num) || num <= 0) return '0m'
  const hours = Math.floor(num / 60)
  const minutes = Math.round(num % 60)
  if (!hours) return `${minutes}m`
  if (!minutes) return `${hours}h`
  return `${hours}h ${minutes}m`
}

function normalizeExitSummary(summary: any): Array<{ label: string; trades: number; profitPct: number; winrate: string }> {
  if (!summary) return []
  const entries: Array<{ label: string; trades: number; profitPct: number; winrate: string }> = []
  if (Array.isArray(summary)) {
    for (const item of summary) {
      const key = item?.key || item?.name
      if (!key) continue
      const label = key.toString()
      const trades = typeof item?.trades === 'number' ? item.trades : Number(item?.count) || 0
      const profitPct = typeof item?.profit_total_pct === 'number'
        ? item.profit_total_pct
        : typeof item?.profit_total === 'number'
          ? item.profit_total * 100
          : 0
      const winrate = typeof item?.winrate === 'number'
        ? formatPercent(item.winrate, 1, true)
        : formatPercent(item?.win_rate, 1, true)
      entries.push({ label, trades, profitPct, winrate })
    }
  } else if (typeof summary === 'object') {
    for (const [key, value] of Object.entries(summary)) {
      if (!value || typeof value !== 'object') continue
      const trades = typeof (value as any).trades === 'number' ? (value as any).trades : Number((value as any).count) || 0
      const profitPct = typeof (value as any).profit_total_pct === 'number'
        ? (value as any).profit_total_pct
        : typeof (value as any).profit_total === 'number'
          ? (value as any).profit_total * 100
          : 0
      const winrate = typeof (value as any).winrate === 'number'
        ? formatPercent((value as any).winrate, 1, true)
        : '--'
      entries.push({ label: key, trades, profitPct, winrate })
    }
  }
  return entries
    .filter((entry) => entry.label && entry.label !== 'TOTAL')
    .sort((a, b) => Math.abs(b.profitPct) - Math.abs(a.profitPct))
}

function buildPairRows(pairs: any): Array<{ pair: string; trades: string; profit: string }> {
  if (!Array.isArray(pairs)) return []
  return pairs
    .filter((item) => item && (item.pair || item.key))
    .map((item) => ({
      pair: item.pair || item.key,
      trades: `${item.trades ?? 0}`,
      profit: formatPercent(item.profit_total_pct ?? 0),
    }))
}

function buildTopTrades(trades: any[], limit = 5) {
  if (!Array.isArray(trades)) return []
  const positives = trades
    .filter((trade) => Number.isFinite(Number(trade?.profit_ratio)) && Number(trade?.profit_ratio) > 0)
    .sort((a, b) => Number(b.profit_ratio) - Number(a.profit_ratio))
  if (!positives.length) {
    return trades
      .filter((trade) => Number.isFinite(Number(trade?.profit_ratio)))
      .sort((a, b) => Number(b.profit_ratio) - Number(a.profit_ratio))
      .slice(0, limit)
  }
  return positives.slice(0, limit)
}

function buildWorstTrades(trades: any[], limit = 5) {
  if (!Array.isArray(trades)) return []
  const negatives = trades
    .filter((trade) => Number.isFinite(Number(trade?.profit_ratio)) && Number(trade?.profit_ratio) < 0)
    .sort((a, b) => Number(a.profit_ratio) - Number(b.profit_ratio))
  if (!negatives.length) {
    return trades
      .filter((trade) => Number.isFinite(Number(trade?.profit_ratio)))
      .sort((a, b) => Number(a.profit_ratio) - Number(b.profit_ratio))
      .slice(0, limit)
  }
  return negatives.slice(0, limit)
}

function buildEquitySeries(trades: any[], startingBalance: number): SeriesPoint[] {
  if (!Array.isArray(trades) || !trades.length) return startingBalance ? [{ label: 'Start', value: startingBalance }] : []
  const points: SeriesPoint[] = []
  const startValue = Number.isFinite(startingBalance) ? startingBalance : 0
  if (Number.isFinite(startValue)) {
    points.push({ label: 'Start', value: Number(startValue.toFixed(2)), timestamp: trades[0]?.open_timestamp })
  }
  let balance = startValue
  trades.forEach((trade, idx) => {
    const gain = Number(trade?.profit_abs)
    if (Number.isFinite(gain)) {
      balance += gain
    }
    const ts = trade?.close_timestamp ?? trade?.open_timestamp ?? undefined
    const label = trade?.close_date || trade?.open_date || `Trade ${idx + 1}`
    points.push({ label, value: Number(balance.toFixed(2)), timestamp: ts })
  })
  return points
}

type SparklineProps = {
  values: number[]
  width?: number
  height?: number
  stroke?: string
  fill?: string
  showBaseline?: boolean
}

function Sparkline({
  values,
  width = 240,
  height = 72,
  stroke = '#7a8bbd',
  fill = 'rgba(122, 139, 189, 0.22)',
  showBaseline = true,
}: SparklineProps) {
  if (!values || !values.length) {
    return <div className="sparkline sparkline--empty">No data yet</div>
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const step = values.length > 1 ? width / (values.length - 1) : 0
  const coords = values.map((value, idx) => {
    const x = idx * step
    const norm = (value - min) / range
    const y = height - norm * height
    return [x, Number.isFinite(y) ? y : height / 2] as const
  })
  const linePath = coords.map(([x, y], idx) => `${idx === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')
  const areaPath = `M 0 ${height} ${coords.map(([x, y]) => `L ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ')} L ${width} ${height} Z`
  const baselineY = height - ((0 - min) / range) * height
  return (
    <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <path d={areaPath} fill={fill} stroke="none" />
      <path d={linePath} fill="none" stroke={stroke} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      {showBaseline && baselineY >= 0 && baselineY <= height ? (
        <line x1={0} x2={width} y1={baselineY} y2={baselineY} stroke="rgba(122,139,189,0.18)" strokeDasharray="4 4" />
      ) : null}
    </svg>
  )
}

type MiniBarDatum = { label: string; value: number }

function MiniBarChart({ data, unit = '%' }: { data: MiniBarDatum[]; unit?: string }) {
  if (!data.length) return null
  const max = Math.max(...data.map((d) => Math.abs(d.value))) || 1
  return (
    <div className="mini-bar-chart">
      {data.map((datum) => {
        const positive = datum.value >= 0
        const width = (Math.abs(datum.value) / max) * 100
        return (
          <div key={datum.label} className="mini-bar-row">
            <span className="mini-bar-label">{datum.label}</span>
            <div className="mini-bar-track">
              <div
                className={`mini-bar-fill ${positive ? 'positive' : 'negative'}`}
                style={{ width: `${width}%` }}
              />
            </div>
            <span className={`mini-bar-value ${positive ? 'positive' : 'negative'}`}>{datum.value.toFixed(2)}{unit}</span>
          </div>
        )
      })}
    </div>
  )
}

type TabId = 'data' | 'expressions' | 'backtest' | 'jobs' | 'runs' | 'connectors' | 'secrets'

const TAB_ITEMS: Array<{ id: TabId; label: string; description: string; icon: string }> = [
  { id: 'data', label: 'Data Lab', description: 'Ingest and audit market time series', icon: 'DT' },
  { id: 'expressions', label: 'Expression Studio', description: 'Curate and test LLM-generated alpha factors', icon: 'EX' },
  { id: 'backtest', label: 'Backtests & Flows', description: 'Trigger ML training, backtests, and the full agent flow', icon: 'BT' },
  { id: 'jobs', label: 'Job Monitor', description: 'Inspect running jobs and live log streams', icon: 'JB' },
  { id: 'runs', label: 'Run History', description: 'Browse manifests and artifacts from previous executions', icon: 'RN' },
  { id: 'connectors', label: 'Connectors', description: 'Manage integrations, MCP bridges, and external services', icon: 'CN' },
  { id: 'secrets', label: 'Secrets Vault', description: 'Store API keys and credentials securely', icon: 'SC' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('data')
  const dlTimeframe = '4h'
  const dlTimerange = '20200701-20210131'
  const dlPairs = 'BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT'

  const [jobs, setJobs] = useState<JobStatus[]>([])
  const [selectedJob, setSelectedJob] = useState<JobStatus | null>(null)
  const [jobLogs, setJobLogs] = useState<JobLogEntry[]>([])
  const [es, setEs] = useState<EventSource | null>(null)
  const [btJobId, setBtJobId] = useState<string | null>(null)
  const [btPct, setBtPct] = useState(0)
  const [btLabel, setBtLabel] = useState('')
  const [btRunning, setBtRunning] = useState(false)
  const [btElapsed, setBtElapsed] = useState<number | undefined>(undefined)
  const [btSteps, setBtSteps] = useState<any[]>([])
  const [flowJobId, setFlowJobId] = useState<string | null>(null)
  const [flowPct, setFlowPct] = useState(0)
  const [flowLabel, setFlowLabel] = useState('')
  const [flowRunning, setFlowRunning] = useState(false)
  const [flowElapsed, setFlowElapsed] = useState<number | undefined>(undefined)
  const [flowSteps, setFlowSteps] = useState<any[]>([])
  const [stepStats, setStepStats] = useState<any[]>([])
  // Backtest controls
  const [btModel, setBtModel] = useState('LightGBMRegressor')
  const [btExpr, setBtExpr] = useState<any[]>([])
  const [btExprLoaded, setBtExprLoaded] = useState(false)
  const [btSummary, setBtSummary] = useState<any | null>(null)
  const [btFeaturePath, setBtFeaturePath] = useState('../../resources/user_data/freqai_features_multi.json')
  const [btExprPath, setBtExprPath] = useState('../../resources/user_data/freqai_expressions.json')
  const [btExprFilter, setBtExprFilter] = useState('')
  const [btExprMode, setBtExprMode] = useState<'all'|'enabled'|'disabled'>('all')
  const [btFeaturesActivePath, setBtFeaturesActivePath] = useState<string>('')
  // Training jobs
  const [trainJobId, setTrainJobId] = useState<string | null>(null)
  const [trainPct, setTrainPct] = useState(0)
  const [trainLabel, setTrainLabel] = useState('')
  const [trainRunning, setTrainRunning] = useState(false)
  const [trainElapsed, setTrainElapsed] = useState<number | undefined>(undefined)
  const [trainMetrics, setTrainMetrics] = useState<any | null>(null)

  const activeBtSummary = useMemo(() => {
    if (btSummary && !(btSummary as any).status && (btSummary as any).metrics) {
      return btSummary
    }
    return PLACEHOLDER_BACKTEST_SUMMARY
  }, [btSummary])

  const usingPlaceholderSummary = activeBtSummary === PLACEHOLDER_BACKTEST_SUMMARY

  const formatDateTime = (value?: string) => {
    if (!value) return '--'
    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return value
    return date.toLocaleString()
  }

  const formatProfitMetric = (raw: any) => {
    if (raw === null || raw === undefined) return '--'
    if (typeof raw === 'number') return `${raw.toFixed(2)}%`
    if (typeof raw === 'string') {
      const num = Number(raw)
      return Number.isFinite(num) ? `${num.toFixed(2)}%` : raw
    }
    if (typeof raw === 'object') {
      if ('value' in raw && typeof (raw as any).value === 'number') {
        return `${((raw as any).value as number).toFixed(2)}%`
      }
      if ('total' in raw && typeof (raw as any).total === 'number') {
        return `${((raw as any).total as number).toFixed(2)}%`
      }
    }
    return '--'
  }

  const filenameFromPath = (value?: string) => {
    if (!value) return '--'
    const parts = value.split(/[\\/]/)
    return parts[parts.length - 1] || value
  }

  function formatLogText(payload: any, fallback = ''): string {
    try {
      if (payload && typeof payload === 'object') {
        if (payload.event === 'log') {
          const msg = payload.message ?? ''
          return msg || fallback
        }
        if (payload.event === 'progress') {
          const parts: string[] = []
          if (payload.current !== undefined && payload.total !== undefined) {
            parts.push(`[progress] ${payload.current}/${payload.total}`)
          }
          if (payload.label) parts.push(String(payload.label))
          if (payload.status) parts.push(`status=${payload.status}`)
          if (payload.percent !== undefined) parts.push(`${payload.percent}%`)
          return parts.join(' ') || fallback || '[progress]'
        }
      }
    } catch (error) {
      console.warn('Failed to format log payload', error)
    }
    return fallback || (typeof payload === 'string' ? payload : '')
  }

  async function refreshJobs() {
    const list = await getJSON<JobStatus[]>('/jobs')
    setJobs(list.sort((a,b) => (b.started_at || '').localeCompare(a.started_at || '')))
  }

  async function openJob(job: JobStatus) {
    setSelectedJob(job)
    // close previous stream if any
    if (es) {
      try {
        es.close()
      } catch (error) {
        console.warn('Failed to close previous EventSource', error)
      }
    }
    setJobLogs([])
    // try SSE first
    try {
      const stream = new EventSource(`${API_BASE}/jobs/${job.id}/stream?offset=0`)
      stream.onmessage = (ev) => {
        setJobLogs(prev => {
          try {
            const payload = JSON.parse(ev.data)
            return [...prev, { line: prev.length, text: formatLogText(payload, ev.data), event: payload }]
          } catch (error) {
            console.warn('Failed to parse SSE payload', error)
            return [...prev, { line: prev.length, text: ev.data }]
          }
        })
      }
      stream.addEventListener('end', () => {
        stream.close()
      })
      stream.onerror = () => {
        stream.close()
      }
      setEs(stream)
    } catch (error) {
      console.warn('Falling back to job log fetch', error)
      // fallback to one-shot fetch
      const res = await getJSON<any>(`/jobs/${job.id}/logs?offset=0&limit=500&structured=true`)
      const start = res.offset || 0
      if (Array.isArray(res.events) && res.events.length) {
        const mapped: JobLogEntry[] = res.events.map((evt: any, idx: number) => ({
          line: start + idx,
          text: formatLogText(evt),
          event: evt,
        }))
        setJobLogs(mapped)
      } else if (Array.isArray(res.entries)) {
        setJobLogs(res.entries.map((entry: any, idx: number) => ({
          line: entry?.line ?? (start + idx),
          text: entry?.text ?? '',
          event: entry?.event,
        })))
      } else {
        const logs: string[] = res.logs || []
        setJobLogs(logs.map((text: string, idx: number) => ({ line: start + idx, text })))
      }
    }
  }

  useEffect(() => {
    // Load active features path once
    getJSON<any>('/features')
      .then((res) => {
        if (res && res.path) setBtFeaturesActivePath(res.path)
      })
      .catch((error) => {
        console.warn('Failed to load active features path', error)
      })
  }, [])

  async function runBacktest() {
    // Save enabled flags if expressions loaded
    try {
      if (btExprLoaded) {
        await putJSON('/expressions', { expressions: btExpr })
      }
    } catch (error) {
      console.warn('Failed to persist expressions before backtest', error)
    }
      const res = await postJSON<{ status:string; job_id:string }>(`/run/backtest`, {
        config: '../../config/configs/config_freqai_multi.json',
        strategy: 'ExpressionLongStrategy',
        strategy_path: '../freqtrade/user_data/strategies',
        timerange: '20200701-20210131',
        freqaimodel: btModel,
        export: false,
        fast: true,
      })
    if (!res || !res.job_id) {
      alert('Backtest failed to start: no job_id returned')
      return
    }
    startTrack(res.job_id, 'bt')
  }

  async function runFlow() {
    const res = await postJSON<{ job_id: string }>('/flow/run', { config: '../../config/configs/agent_flow_multi.json' })
    if (!res || !res.job_id) { alert('Flow failed to start: no job_id'); return }
    startTrack(res.job_id, 'flow')
  }

  async function pollProgress(id: string, target: 'bt'|'flow'): Promise<boolean> {
    try {
      const p = await getJSON<any>(`/jobs/${id}/progress`)
      if (target === 'bt') {
        setBtPct(p.percent || 0); setBtLabel(p.label || ''); setBtRunning(!!p.running); setBtElapsed(p.elapsed)
      } else {
        setFlowPct(p.percent || 0); setFlowLabel(p.label || ''); setFlowRunning(!!p.running); setFlowElapsed(p.elapsed)
      }
      return !!p.running
    } catch (error) {
      console.warn('Failed to poll job progress', error)
    }
    try {
      const steps = await getJSON<any>(`/jobs/${id}/steps`)
      const stats = await getJSON<any>('/steps/stats')
      if (target === 'bt') setBtSteps(steps.steps || []); else setFlowSteps(steps.steps || [])
      setStepStats(stats.stats || [])
    } catch (error) {
      console.warn('Failed to fetch job steps summary', error)
    }
    return true
  }

  function startTrack(id: string, target: 'bt'|'flow') {
    if (target === 'bt') { setBtJobId(id); setBtRunning(true); setBtPct(0); setBtLabel('') } else { setFlowJobId(id); setFlowRunning(true); setFlowPct(0); setFlowLabel('') }
    const tick = async () => {
      const stillRunning = await pollProgress(id, target)
      if (!stillRunning) {
        await refreshJobs()
        // Auto load summary after backtest
        if (target === 'bt') {
          try {
            const sum = await getJSON<any>('/backtest/summary/latest')
            setBtSummary(sum)
          } catch (error) {
            console.warn('Failed to load latest backtest summary', error)
          }
        }
        return
      }
      setTimeout(tick, 1200)
    }
    setTimeout(tick, 1000)
  }

  async function trainXGB() {
    const res = await postJSON<{ status: string; job_id: string }>(`/run/train-xgb`, {})
    startTrainTrack(res.job_id)
  }

  async function trainCAT() {
    const res = await postJSON<{ status: string; job_id: string }>(`/run/train-cat`, {})
    startTrainTrack(res.job_id)
  }

  async function startTrainTrack(id: string) {
    setTrainJobId(id); setTrainRunning(true); setTrainPct(0); setTrainLabel(''); setTrainMetrics(null)
    const tick = async () => {
      try {
        const p = await getJSON<any>(`/jobs/${id}/progress`)
        setTrainPct(p.percent || 0)
        setTrainLabel(p.label || '')
        setTrainRunning(!!p.running)
        setTrainElapsed(p.elapsed)
        if (!p.running) {
          // fetch logs and parse last JSON line
          try {
            const logs = await getJSON<any>(`/jobs/${id}/logs?offset=0&limit=1000&structured=true`)
            const entries = logs.entries || []
            for (let i=entries.length-1;i>=0;i--) {
              const t = entries[i].text
              try {
                const obj = JSON.parse(t)
                if (obj && obj.metrics) {
                  setTrainMetrics(obj)
                  break
                }
              } catch (error) {
                console.warn('Failed to parse structured training log entry', error)
              }
            }
            // Fallback parsing for plain text lines ("Metrics:", "Model:")
            if (!trainMetrics) {
              const metrics: Record<string, number | string> = {}
              let modelPath = ''
              for (let i=entries.length-1;i>=0;i--) {
                const t = (entries[i].text || '').trim()
                if (!modelPath && t.startsWith('Model:')) {
                  modelPath = t.replace(/^Model:\s*/, '')
                }
                if (Object.keys(metrics).length === 0 && t.startsWith('Metrics:')) {
                  const rest = t.replace(/^Metrics:\s*/, '')
                  const parts = rest.split(/\s*,\s*/)
                  for (const part of parts) {
                    const kv = part.split(':')
                    if (kv.length >= 2) {
                      const k = kv[0].trim()
                      const vraw = kv.slice(1).join(':').trim()
                      const vnum = Number(vraw)
                      metrics[k] = Number.isNaN(vnum) ? vraw : vnum
                    }
                  }
                }
              }
              if (Object.keys(metrics).length || modelPath) {
                setTrainMetrics({ metrics, model_path: modelPath })
              }
            }
          } catch (error) {
            console.warn('Failed to inspect training job logs', error)
          }
          return
        }
      } catch (error) {
        console.warn('Failed to poll training progress', error)
      }
      setTimeout(tick, 1200)
    }
    setTimeout(tick, 1000)
  }

  async function terminateJob() {
    if (!selectedJob) return
    await postEmpty(`/jobs/${selectedJob.id}/terminate`)
    await refreshJobs()
  }

  useEffect(() => {
    refreshJobs().catch((error) => {
      console.warn('Initial job refresh failed', error)
    })
    const t = setInterval(() => {
      refreshJobs().catch((error) => console.warn('Periodic job refresh failed', error))
    }, 4000)
    return () => clearInterval(t)
  }, [])

  const tabMeta = TAB_ITEMS.find(tab => tab.id === activeTab) ?? TAB_ITEMS[0]
  const runningJobs = jobs.filter(job => job.running).length
  const successfulJobs = jobs.filter(job => !job.running && job.returncode === 0).length
  const failedJobs = jobs.filter(job => !job.running && job.returncode !== null && job.returncode !== 0).length
  const totalJobs = jobs.length
  const activeJob = selectedJob || jobs.find(job => job.running) || null
  const activeJobId = activeJob?.id ?? 'None'
  const activeJobStatus = activeJob ? `${activeJob.running ? 'Running' : activeJob.returncode === 0 ? 'Succeeded' : 'Stopped'} | ${formatDateTime(activeJob.started_at)}` : 'Idle'
  const summaryProfit = formatProfitMetric(activeBtSummary?.metrics?.profit_total_pct)
  const flowStatus = flowRunning
    ? `${flowPct}% | ${flowLabel || 'running'}`
    : flowJobId
      ? flowLabel
        ? `Last | ${flowLabel}`
        : 'Completed'
      : 'Idle'
  const flowElapsedText = flowElapsed
    ? `${flowElapsed}s elapsed`
    : flowJobId ? 'Previous flow completed' : 'Awaiting next dispatch'

  const hasBacktestSummary = !!btSummary?.metrics
  const resolvedBacktestStatus = hasBacktestSummary
    ? `Ready | ${formatProfitMetric(btSummary?.metrics?.profit_total_pct)}`
    : btRunning
      ? `${btPct}% | ${btLabel || 'running'}`
      : btJobId
        ? (btLabel ? `Last | ${btLabel}` : 'Completed')
        : 'Run pending'
  const backtestStatus = resolvedBacktestStatus
  const backtestElapsedText = btElapsed
    ? `${btElapsed}s elapsed`
    : hasBacktestSummary
      ? 'Latest summary cached'
      : btJobId
        ? 'Awaiting summary'
        : 'Run a backtest to populate'

  const featuresBadge = filenameFromPath(btFeaturesActivePath || 'resources/user_data/freqai_features_multi.json')
  const enabledFactorCount = btExprLoaded ? (btExpr || []).filter((expr: any) => !!expr.enabled).length : 0
  const totalFactors = btExpr?.length ?? 0
  const lastStartedAt = jobs[0]?.started_at
  const lastActivity = formatDateTime(lastStartedAt)

  const trades = useMemo(() => {
    if (!Array.isArray(activeBtSummary?.metrics?.trades)) return []
    return [...activeBtSummary.metrics.trades].sort((a: any, b: any) => {
      const aTs = a.close_timestamp ?? a.open_timestamp ?? 0
      const bTs = b.close_timestamp ?? b.open_timestamp ?? 0
      return aTs - bTs
    })
  }, [activeBtSummary])

  const startingBalance = useMemo(() => {
    const raw = activeBtSummary?.metrics?.starting_balance
      ?? activeBtSummary?.metrics?.dry_run_wallet
      ?? activeBtSummary?.metrics?.stake_amount
      ?? 0
    const num = Number(raw)
    return Number.isFinite(num) ? num : 0
  }, [activeBtSummary])

  const equitySeries = useMemo(() => buildEquitySeries(trades, startingBalance), [trades, startingBalance])
  const latestBalance = equitySeries.length ? equitySeries[equitySeries.length - 1].value : startingBalance
  const balanceDelta = latestBalance - (equitySeries[0]?.value ?? startingBalance)
  const balanceDeltaPct = startingBalance ? (balanceDelta / startingBalance) * 100 : 0

  const dailySeries = useMemo(() => {
    const src = activeBtSummary?.metrics?.daily_profit
    if (!Array.isArray(src) || !src.length) {
      return []
    }
    return src.map((entry: any) => ({
      label: entry[0],
      value: typeof entry[1] === 'number' ? entry[1] : Number(entry[1]) || 0,
    }))
  }, [activeBtSummary])

  const exitBreakdown = useMemo(
    () => normalizeExitSummary(activeBtSummary?.metrics?.exit_reason_summary),
    [activeBtSummary],
  )
  const pairPerformance = useMemo(() => buildPairRows(activeBtSummary?.pairs), [activeBtSummary])
  const topTrades = useMemo(() => buildTopTrades(trades, 5), [trades])
  const worstTrades = useMemo(() => buildWorstTrades(trades, 5), [trades])
  const drawdownStats = useMemo(() => {
    if (!activeBtSummary?.metrics) return []
    const m = activeBtSummary.metrics
    const safe = (value: any, formatter: (n: number) => string = (n) => n.toFixed(2)) => {
      const num = Number(value)
      return Number.isFinite(num) ? formatter(num) : '--'
    }
    return [
      { label: 'Max Drawdown (abs)', value: safe(m.max_drawdown_abs, formatCurrency) },
      { label: 'Max Drawdown (%)', value: safe(m.max_drawdown_account, (n) => `${(n * 100).toFixed(2)}%`) },
      { label: 'Max Consecutive Wins', value: m.max_consecutive_wins ?? '--' },
      { label: 'Max Consecutive Losses', value: m.max_consecutive_losses ?? '--' },
      { label: 'Winrate', value: typeof m.winrate === 'number' ? `${(m.winrate * 100).toFixed(2)}%` : m.winrate ?? '--' },
      { label: 'Sharpe', value: safe(m.sharpe) },
      { label: 'Sortino', value: safe(m.sortino) },
      { label: 'Profit Factor', value: safe(m.profit_factor) },
    ]
  }, [activeBtSummary])

  const renderBacktestTab = (): React.ReactNode => {
    const summary = activeBtSummary
    const currency = summary?.metrics?.stake_currency ?? 'USDT'
    const totalTradesCount = trades.length
    const winrateDisplay = typeof summary?.metrics?.winrate === 'number'
      ? formatPercent(summary.metrics.winrate, 1, true)
      : summary?.metrics?.winrate ?? '--'
    const expectancyDisplay = typeof summary?.metrics?.expectancy === 'number'
      ? summary.metrics.expectancy.toFixed(2)
      : summary?.metrics?.expectancy ?? '--'
    const sharpeDisplay = typeof summary?.metrics?.sharpe === 'number'
      ? summary.metrics.sharpe.toFixed(2)
      : summary?.metrics?.sharpe ?? '--'
    const tradesPerDayDisplay = typeof summary?.metrics?.trades_per_day === 'number'
      ? summary.metrics.trades_per_day.toFixed(1)
      : summary?.metrics?.trades_per_day ?? '--'
    const equityValues = equitySeries.map((point) => point.value)
    const recentDaily = dailySeries.slice(-14)
    const recentDailyValues = recentDaily.map((item) => item.value)
    const latestDaily = recentDaily.length ? recentDaily[recentDaily.length - 1].value : 0
    const avgDaily = recentDaily.length
      ? recentDaily.reduce((sum, item) => sum + item.value, 0) / recentDaily.length
      : 0
    const exitChartData = exitBreakdown.map((entry) => ({
      label: `${entry.label} (${entry.trades})`,
      value: entry.profitPct,
    }))
    const pairColumns = [
      { key: 'pair', title: 'Pair' },
      { key: 'trades', title: 'Trades', align: 'right' as const },
      { key: 'profit', title: 'Return %', align: 'right' as const },
    ]
    const pairTableRows = pairPerformance.slice(0, 8).map((row) => ({
      pair: row.pair,
      trades: row.trades,
      profit: row.profit,
    }))
    const topTradeColumns = [
      { key: 'pair', title: 'Pair' },
      { key: 'profit', title: 'Return %', align: 'right' as const },
      { key: 'duration', title: 'Holding', align: 'right' as const },
      { key: 'exit', title: 'Exit Reason' },
    ]
    const topTradeRows = topTrades.map((trade) => ({
      pair: trade.pair ?? '--',
      profit: formatPercent(trade.profit_ratio ?? 0, 2, true),
      duration: parseDurationMinutes(trade.trade_duration),
      exit: trade.exit_reason ?? '--',
    }))
    const worstTradeRows = worstTrades.map((trade) => ({
      pair: trade.pair ?? '--',
      profit: formatPercent(trade.profit_ratio ?? 0, 2, true),
      duration: parseDurationMinutes(trade.trade_duration),
      exit: trade.exit_reason ?? '--',
    }))
    const exitColumns = [
      { key: 'reason', title: 'Exit Strategy' },
      { key: 'trades', title: 'Trades', align: 'right' as const },
      { key: 'profit', title: 'Return %', align: 'right' as const },
      { key: 'winrate', title: 'Winrate', align: 'right' as const },
    ]
    const exitRows = exitBreakdown.map((entry) => ({
      reason: entry.label,
      trades: `${entry.trades}`,
      profit: formatPercent(entry.profitPct),
      winrate: entry.winrate,
    }))
    const drawdownColumns = [
      { key: 'metric', title: 'Metric' },
      { key: 'value', title: 'Value', align: 'right' as const },
    ]
    const drawdownRows = drawdownStats.map((item) => ({
      metric: item.label,
      value: item.value,
    }))
    const comparisonColumns = [
        { key: 'strategy', title: 'Strategy' },
        { key: 'trades', title: 'Trades', align: 'right' as const },
        { key: 'profit', title: 'Return %', align: 'right' as const },
        { key: 'winrate', title: 'Winrate', align: 'right' as const },
      ]
    const comparisonSource = Array.isArray(summary?.comparison) ? (summary?.comparison as any[]) : []
    const comparisonRows = comparisonSource.map((item: any) => ({
        strategy: item?.key || item?.strategy || '--',
        trades: item?.trades ?? '--',
        profit: typeof item?.profit_total_pct === 'number'
          ? formatPercent(item.profit_total_pct)
          : typeof item?.profit_total === 'number'
            ? formatPercent(item.profit_total, 2, true)
            : '--',
        winrate: typeof item?.winrate === 'number'
          ? formatPercent(item.winrate, 1, true)
          : item?.winrate ?? '--',
      }))
    const finalBalanceDisplay = formatCurrency(latestBalance, currency)
    const startingBalanceDisplay = formatCurrency(startingBalance, currency)
    const balanceDeltaDisplay = `${balanceDelta >= 0 ? '+' : '-'}${currencyFormatter.format(Math.abs(balanceDelta))} ${currency}`
    const balanceDeltaPctDisplay = `${balanceDeltaPct >= 0 ? '+' : '-'}${Math.abs(balanceDeltaPct).toFixed(2)}%`
    const summaryTitle = summary?.strategy || 'ExpressionLongStrategy'
    const summarySource = summary?.source ? filenameFromPath(summary.source) : '--'

    return (
      <div className="content-stack">
        <Card className="backtest-panel">
          <div className="backtest-column backtest-column--controls">
            <section className="control-section">
              <div className="card-header">
                <h2>Pipeline Control</h2>
                <p>Trigger backtests or the full Agent Flow whenever you need it.</p>
              </div>
              <div className="button-group">
                <button onClick={runBacktest}>Run Backtest</button>
                <button className="ghost-button" onClick={runFlow}>Run Full Flow</button>
              </div>
              <div className="inline-form">
                <div className="form-control">
                  <label>ML Model</label>
                  <select value={btModel} onChange={e => setBtModel(e.target.value)}>
                    <option>LightGBMRegressor</option>
                    <option>XGBoostRegressor</option>
                    <option>RandomForestRegressor</option>
                    <option>CatBoostRegressor</option>
                  </select>
                </div>
                <button onClick={trainXGB}>Train XGB</button>
                <button onClick={trainCAT}>Train CAT</button>
                <button onClick={async () => {
                  try {
                    const cfg = {
                      config: '../../config/configs/ml_config_lightgbm.json',
                      output: { model_dir: 'artifacts/models/lightgbm_multi' },
                    }
                    const res = await postJSON<{ status: string; job_id: string }>('/run/train-ml', { config: cfg })
                    startTrainTrack(res.job_id)
                  } catch (error) {
                    console.warn('Failed to trigger LightGBM training', error)
                    alert('LightGBM training launch failed')
                  }
                }}>Train LGBM</button>
              </div>
              <div className="inline-form">
                <div className="form-control grow">
                  <label>Features file</label>
                  <input value={btFeaturePath} onChange={e => setBtFeaturePath(e.target.value)} />
                </div>
                <button onClick={async () => {
                  try {
                    await postJSON('/config/use-features-file', { path: btFeaturePath })
                    alert('Features file applied.')
                  } catch (error) {
                    console.warn('Applying features file failed', error)
                    alert('Apply features failed')
                  }
                }}>Use</button>
              </div>
              <div className="inline-form">
                <div className="form-control grow">
                  <label>Expressions file</label>
                  <input value={btExprPath} onChange={e => setBtExprPath(e.target.value)} />
                </div>
                <button onClick={async () => {
                  try {
                    await postJSON('/config/use-expressions-file', { path: btExprPath })
                    const data = await getJSON<any>('/expressions')
                    setBtExpr(data.expressions || [])
                    setBtExprLoaded(true)
                    alert('Expressions file applied and loaded.')
                  } catch (error) {
                    console.warn('Applying expressions file failed', error)
                    alert('Apply expressions failed')
                  }
                }}>Use & Load</button>
                <button onClick={async () => {
                  try {
                    const data = await getJSON<any>('/expressions')
                    setBtExpr(data.expressions || [])
                    setBtExprLoaded(true)
                  } catch (error) {
                    console.warn('Loading expressions failed', error)
                    alert('Load factors failed')
                  }
                }}>Load Factors</button>
                {btExprLoaded ? (
                  <button onClick={async () => {
                    try {
                      await putJSON('/expressions', { expressions: btExpr })
                      alert('Factors saved')
                    } catch (error) {
                      console.warn('Saving expressions failed', error)
                      alert('Save failed')
                    }
                  }}>Save Factors</button>
                ) : null}
              </div>
              {btExprLoaded ? (
                <div className="scroll-panel factor-panel">
                  <div className="selection-toolbar">
                    <div className="selection-title">Enable Factors</div>
                    <input placeholder="Search name..." value={btExprFilter} onChange={e => setBtExprFilter(e.target.value)} />
                    <select value={btExprMode} onChange={e => setBtExprMode(e.target.value as any)}>
                      <option value="all">All</option>
                      <option value="enabled">Enabled</option>
                      <option value="disabled">Disabled</option>
                    </select>
                  </div>
                  <div className="selection-grid">
                    {(btExpr || []).map((expr: any, idx: number) => {
                      const name = expr.name || `expr_${idx}`
                      const normalized = name.toLowerCase()
                      const matchesFilter = !btExprFilter || normalized.includes(btExprFilter.toLowerCase())
                      const matchesMode = btExprMode === 'all' || (btExprMode === 'enabled' ? !!expr.enabled : !expr.enabled)
                      if (!matchesFilter || !matchesMode) return null
                      return (
                        <label key={idx} className="selection-item">
                          <input
                            type="checkbox"
                            checked={!!expr.enabled}
                            onChange={(e) => setBtExpr(prev => prev.map((item: any, itemIdx: number) => itemIdx === idx ? ({ ...item, enabled: e.target.checked }) : item))}
                          />
                          <span>{name}</span>
                        </label>
                      )
                    })}
                  </div>
                </div>
              ) : (
                <p className="footnote">Load an expressions file to toggle factors.</p>
              )}
              <p className="footnote">Tip: backtests run using the current server configuration; the Full Flow covers download -&gt; feature -&gt; expression -&gt; ML -&gt; backtest.</p>
            </section>
            <section className="status-stack">
              {trainJobId ? (
                <div className="status-card">
                  <div className="status-card-header">
                    <div>
                      <div className="status-title">Training Job</div>
                      <div className="status-id">{trainJobId}</div>
                    </div>
                    <span className={`status-chip ${trainRunning ? 'running' : 'done'}`}>{trainRunning ? 'Running' : 'Finished'}</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill warning" style={{ width: `${trainPct}%` }} />
                  </div>
                  <div className="status-meta">
                    <span>{trainLabel || 'waiting for update...'}</span>
                    <span>{trainPct}%{trainElapsed ? ` | ${trainElapsed}s` : ''}</span>
                  </div>
                  {trainMetrics ? (
                    <div className="status-details">
                      <div><strong>Metrics:</strong> {Object.entries(trainMetrics.metrics || {}).map(([k, v]) => `${k}: ${v}`).join(' | ') || 'none'}</div>
                      <div><strong>Model:</strong> {trainMetrics.model_path || 'unknown'}</div>
                    </div>
                  ) : null}
                </div>
              ) : null}
              {btJobId ? (
                <div className="status-card">
                  <div className="status-card-header">
                    <div>
                      <div className="status-title">Backtest Job</div>
                      <div className="status-id">{btJobId}</div>
                    </div>
                    <span className={`status-chip ${btRunning ? 'running' : 'done'}`}>{btRunning ? 'Running' : 'Finished'}</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill accent" style={{ width: `${btPct}%` }} />
                  </div>
                  <div className="status-meta">
                    <span>{btLabel || 'processing...'}</span>
                    <span>{btPct}%{btElapsed ? ` | ${btElapsed}s` : ''}</span>
                  </div>
                  <div className="status-details">
                    <div>Features | {featuresBadge}</div>
                    <div>Enabled Factors | {enabledFactorCount} / {totalFactors}</div>
                  </div>
                </div>
              ) : null}
              {btSteps.length ? (
                <div className="table-wrapper">
                  <h4>Backtest Steps</h4>
                  <table className="compact-table">
                    <thead>
                      <tr><th>#</th><th>Name</th><th>Duration (s)</th><th>Average (s)</th><th>Delta</th></tr>
                    </thead>
                    <tbody>
                      {btSteps.map((s: any, i: number) => {
                        const stat = stepStats.find((x: any) => x.label === s.label)
                        const avg = stat ? Math.round(stat.avg_seconds) : null
                        const dur = typeof s.duration === 'number' ? s.duration : null
                        const delta = (avg != null && dur != null) ? (dur - avg) : null
                        return (
                          <tr key={i}>
                            <td>{s.idx}/{s.total}</td>
                            <td>{s.label}</td>
                            <td className="text-right">{dur ?? 'n/a'}</td>
                            <td className="text-right">{avg ?? 'n/a'}</td>
                            <td className={`text-right ${delta != null && delta > 5 ? 'text-danger' : ''}`}>{delta ?? 'n/a'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}
              {flowJobId ? (
                <div className="status-card">
                  <div className="status-card-header">
                    <div>
                      <div className="status-title">Flow Job</div>
                      <div className="status-id">{flowJobId}</div>
                    </div>
                    <span className={`status-chip ${flowRunning ? 'running' : 'done'}`}>{flowRunning ? 'Running' : 'Finished'}</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill success" style={{ width: `${flowPct}%` }} />
                  </div>
                  <div className="status-meta">
                    <span>{flowLabel || 'processing...'}</span>
                    <span>{flowPct}%{flowElapsed ? ` | ${flowElapsed}s` : ''}</span>
                  </div>
                </div>
              ) : null}
              {flowSteps.length ? (
                <div className="table-wrapper">
                  <h4>Flow Steps</h4>
                  <table className="compact-table">
                    <thead>
                      <tr><th>#</th><th>Name</th><th>Duration (s)</th><th>Average (s)</th><th>Delta</th></tr>
                    </thead>
                    <tbody>
                      {flowSteps.map((s: any, i: number) => {
                        const stat = stepStats.find((x: any) => x.label === s.label)
                        const avg = stat ? Math.round(stat.avg_seconds) : null
                        const dur = typeof s.duration === 'number' ? s.duration : null
                        const delta = (avg != null && dur != null) ? (dur - avg) : null
                        return (
                          <tr key={i}>
                            <td>{s.idx}/{s.total}</td>
                            <td>{s.label}</td>
                            <td className="text-right">{dur ?? 'n/a'}</td>
                            <td className="text-right">{avg ?? 'n/a'}</td>
                            <td className={`text-right ${delta != null && delta > 5 ? 'text-danger' : ''}`}>{delta ?? 'n/a'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </section>
          </div>
          <div className="backtest-column backtest-column--analytics">
            <div className="analytics-summary">
              <div>
                <span className="analytics-summary-label">Latest Backtest</span>
                <h3 className="analytics-summary-title">{summaryTitle}</h3>
              </div>
              <div className="analytics-summary-source">
                <span>Source</span>
                <strong>{summarySource}</strong>
              </div>
            </div>
            {usingPlaceholderSummary ? (
              <div className="backtest-placeholder-banner">
                Showing demo analytics until your next backtest finishes. Trigger a run to replace these values with live results.
              </div>
            ) : null}
            <div className="analytics-kpi-grid">
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Final Balance</span>
                <span className="analytics-kpi-value">{finalBalanceDisplay}</span>
                <span className="analytics-kpi-footnote">Delta {balanceDeltaDisplay} | {balanceDeltaPctDisplay}</span>
              </div>
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Total Return</span>
                <span className={`analytics-kpi-value ${balanceDeltaPct >= 0 ? 'positive' : 'negative'}`}>{summaryProfit}</span>
                <span className="analytics-kpi-footnote">Starting {startingBalanceDisplay}</span>
              </div>
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Winrate</span>
                <span className="analytics-kpi-value">{winrateDisplay}</span>
                <span className="analytics-kpi-footnote">Trade count | {totalTradesCount}</span>
              </div>
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Expected Return</span>
                <span className="analytics-kpi-value">{expectancyDisplay}</span>
                <span className="analytics-kpi-footnote">Sharpe {sharpeDisplay}</span>
              </div>
                <div className="analytics-kpi">
                  <span className="analytics-kpi-label">Avg Trade Return</span>
                  <span className="analytics-kpi-value">
                    {typeof summary?.metrics?.profit_mean === 'number' ? formatPercent(summary.metrics.profit_mean, 2, true) : summary?.metrics?.profit_mean ?? '--'}
                  </span>
                  <span className="analytics-kpi-footnote">Daily avg | {tradesPerDayDisplay}</span>
                </div>
            </div>
            <div className="analytics-grid">
              <div className="analytics-card analytics-card--wide">
                <div className="analytics-card-header">
                  <h3>Equity Curve</h3>
                  <span className={`delta ${balanceDelta >= 0 ? 'positive' : 'negative'}`}>{balanceDeltaDisplay} | {balanceDeltaPctDisplay}</span>
                </div>
                <Sparkline values={equityValues} />
                <div className="analytics-card-footer">
                  <span>Starting {startingBalanceDisplay}</span>
                  <span>Current {finalBalanceDisplay}</span>
                </div>
              </div>
              {recentDailyValues.length ? (
                <div className="analytics-card">
                  <div className="analytics-card-header">
                    <h3>Trailing {recentDaily.length}-day return</h3>
                    <span className={`delta ${latestDaily >= 0 ? 'positive' : 'negative'}`}>{formatPercent(latestDaily, 2)}</span>
                  </div>
                  <Sparkline values={recentDailyValues} />
                  <div className="analytics-card-footer">
                    <span>Average {formatPercent(avgDaily, 2)}</span>
                    <span>Latest {formatPercent(latestDaily, 2)}</span>
                  </div>
                </div>
              ) : null}
              {exitChartData.length ? (
                <div className="analytics-card">
                  <div className="analytics-card-header">
                    <h3>Exit Contribution</h3>
                  </div>
                  <MiniBarChart data={exitChartData} />
                </div>
              ) : null}
              <div className="analytics-card analytics-card--wide">
                <div className="analytics-card-header">
                  <h3>Pair Performance</h3>
                </div>
                <SimpleTable
                  columns={pairColumns}
                  rows={pairTableRows}
                  empty="No pair data yet"
                />
              </div>
              <div className="analytics-card">
                <div className="analytics-card-header">
                  <h3>Top Trades</h3>
                </div>
                <SimpleTable
                  columns={topTradeColumns}
                  rows={topTradeRows}
                  empty="No trade data yet"
                />
              </div>
              <div className="analytics-card">
                <div className="analytics-card-header">
                  <h3>Worst Trades</h3>
                </div>
                <SimpleTable
                  columns={topTradeColumns}
                  rows={worstTradeRows}
                  empty="No trade data yet"
                />
              </div>
              <div className="analytics-card">
                <div className="analytics-card-header">
                  <h3>Exit Overview</h3>
                </div>
                <SimpleTable
                  columns={exitColumns}
                  rows={exitRows}
                  empty="No exit data yet"
                />
              </div>
              <div className="analytics-card">
                <div className="analytics-card-header">
                  <h3>Risk Metrics</h3>
                </div>
                <SimpleTable
                  columns={drawdownColumns}
                  rows={drawdownRows}
                  empty="No risk data yet"
                />
              </div>
              {comparisonRows.length ? (
                <div className="analytics-card analytics-card--wide">
                  <div className="analytics-card-header">
                    <h3>Strategy Comparison</h3>
                  </div>
                  <SimpleTable
                    columns={comparisonColumns}
                    rows={comparisonRows}
                    empty="No comparison data yet"
                  />
                </div>
              ) : null}
            </div>
          </div>
        </Card>
        <Card>
          <StrategyParams />
        </Card>
      </div>
    )
  }

  const renderTabContent = (): React.ReactNode => {
    switch (activeTab) {
      case 'data':
        return (
          <div className="content-stack">
            <Card>
              <DataManager defaultTimeframe={dlTimeframe} defaultTimerange={dlTimerange} defaultPairs={dlPairs} onJob={refreshJobs} />
            </Card>
          </div>
        )
      case 'expressions':
        return (
          <div className="content-stack">
            <Card>
              <ExpressionsManager onJob={refreshJobs} />
            </Card>
          </div>
        )
      case 'backtest':
        return renderBacktestTab()
      case 'jobs':
        return (
          <div className="content-stack">
            <Card>
              <JobsPanel jobs={jobs} selectedJob={selectedJob} onOpen={openJob} onTerminate={terminateJob} jobLogs={jobLogs} />
            </Card>
          </div>
        )
      case 'runs':
        return (
          <div className="content-stack">
            <Card>
              <RunHistory />
            </Card>
          </div>
        )
      case 'connectors':
        return (
          <div className="content-stack">
            <Card>
              <ConnectorsManager />
            </Card>
          </div>
        )
      case 'secrets':
        return (
          <div className="content-stack">
            <Card>
              <SecretsViewer />
            </Card>
          </div>
        )
      default:
        return null
    }
  }


  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon" aria-hidden>AM</span>
          <div>
            <div className="brand-title">Agent Market</div>
            <div className="brand-subtitle">MCP Control Center</div>
          </div>
        </div>
        <div className="sidebar-api">
          <span>API Base</span>
          <code>{API_BASE}</code>
        </div>
        <nav className="nav">
          {TAB_ITEMS.map(tab => (
            <button
              key={tab.id}
              className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="nav-icon" aria-hidden>{tab.icon}</span>
              <span className="nav-text">
                <span className="nav-label">{tab.label}</span>
                <span className="nav-desc">{tab.description}</span>
              </span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="sidebar-metric">
            <span className="metric-label">Active Job</span>
            <span className="metric-value">{activeJobId}</span>
            <span className="metric-footnote">{activeJobStatus}</span>
          </div>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="topbar-chip">Agent Market | MCP</div>
            <h1>{tabMeta.label}</h1>
            <p className="topbar-subtitle">{tabMeta.description}</p>
          </div>
          <div className="topbar-actions">
            <div className="topbar-badge">
              <span>Observed Jobs</span>
              <strong>{totalJobs}</strong>
            </div>
            <button className="ghost-button" onClick={refreshJobs}>Refresh jobs</button>
          </div>
        </header>
        <section className="metrics-grid">
          <Card className="metric-card">
            <span className="metric-label">Running Jobs</span>
            <span className="metric-value">{runningJobs}</span>
            <span className="metric-footnote">{lastActivity !== '--' ? `Last start | ${lastActivity}` : 'Queue idle'}</span>
          </Card>
          <Card className="metric-card">
            <span className="metric-label">Completed</span>
            <span className="metric-value">{successfulJobs}</span>
            <span className="metric-footnote">{failedJobs ? `${failedJobs} failed` : 'No recent failures'}</span>
          </Card>
          <Card className="metric-card">
            <span className="metric-label">Backtest Status</span>
              <span className="metric-value">{backtestStatus || 'Idle | Awaiting summary'}</span>
              <span className="metric-footnote">{backtestElapsedText || 'Backtest results will appear here'}</span>
          </Card>
          <Card className="metric-card">
            <span className="metric-label">Flow Monitor</span>
              <span className="metric-value">{flowStatus || 'Idle | Awaiting trigger'}</span>
              <span className="metric-footnote">{flowElapsedText}</span>
          </Card>
          <Card className="metric-card highlight">
            <span className="metric-label">Latest PnL</span>
            <span className="metric-value">{summaryProfit}</span>
            <span className="metric-footnote">Features | {featuresBadge}</span>
          </Card>
        </section>
        <section className="content-area">
          {renderTabContent()}
        </section>
      </main>
    </div>
  )
}

