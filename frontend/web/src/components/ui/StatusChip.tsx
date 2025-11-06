import React from 'react'

type Status = 'default' | 'running' | 'success' | 'warning' | 'danger'

const STATUS_CLASS: Record<Status, string> = {
  default: 'ui-chip--default',
  running: 'ui-chip--running',
  success: 'ui-chip--success',
  warning: 'ui-chip--warning',
  danger: 'ui-chip--danger',
}

type Props = {
  label: React.ReactNode
  tone?: Status
  className?: string
}

export function StatusChip({ label, tone = 'default', className = '' }: Props) {
  const toneClass = STATUS_CLASS[tone] ?? STATUS_CLASS.default
  return <span className={`ui-chip ${toneClass} ${className}`.trim()}>{label}</span>
}

