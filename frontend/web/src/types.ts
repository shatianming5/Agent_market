export interface ProgressEvent {
  event: 'progress'
  status?: string
  current?: number
  total?: number
  label?: string
  percent?: number
  ts?: string
  run_id?: string
}

export interface LogEvent {
  event: 'log'
  message?: string
  ts?: string
  run_id?: string
}

export type JobEvent = ProgressEvent | LogEvent | Record<string, unknown>

export interface JobLogEntry {
  line: number
  text: string
  event?: JobEvent
}

export interface JobStatus {
  id: string
  running: boolean
  returncode: number | null
  lines: number
  started_at?: string
  finished_at?: string
}

export interface RunSummary {
  run_id: string
  status?: string
  started_at?: string
  finished_at?: string
  manifest_path?: string
  run_dir?: string
}

export interface RunStep {
  name: string
  status: string
  started_at?: string
  finished_at?: string
  error?: string
}

export interface RunManifest {
  run_id: string
  run_dir?: string
  status: string
  started_at?: string
  finished_at?: string
  error?: string | null
  steps: RunStep[]
  config_snapshot?: string | null
  events_log?: string | null
  artifacts?: Record<string, unknown> | null
  environment?: Record<string, unknown> | null
}
