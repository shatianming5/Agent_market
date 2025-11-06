import React from 'react'

export function Grid({ children, columns = 2, gap = 20, className = '' }: { children: React.ReactNode; columns?: number; gap?: number; className?: string }) {
  const style: React.CSSProperties = {
    display: 'grid',
    gap,
    gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
  }
  return (
    <div className={`ui-grid ${className}`.trim()} style={style}>
      {children}
    </div>
  )
}

export function Stack({ children, gap = 16, className = '' }: { children: React.ReactNode; gap?: number; className?: string }) {
  return (
    <div className={`ui-stack ${className}`.trim()} style={{ display: 'flex', flexDirection: 'column', gap }}>
      {children}
    </div>
  )
}

