import React from 'react'

type TableProps = {
  columns: Array<{ key: string; title: React.ReactNode; align?: 'left' | 'right'; width?: string | number }>
  rows: Array<Record<string, React.ReactNode>>
  empty?: React.ReactNode
  className?: string
}

export function SimpleTable({ columns, rows, empty = null, className = '' }: TableProps) {
  return (
    <table className={`ui-table ${className}`.trim()}>
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col.key} style={{ textAlign: col.align ?? 'left', width: col.width }}>
              {col.title}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.length
          ? rows.map((row, idx) => (
              <tr key={idx}>
                {columns.map((col) => (
                  <td key={col.key} style={{ textAlign: col.align ?? 'left' }}>
                    {row[col.key]}
                  </td>
                ))}
              </tr>
            ))
          : (
            <tr>
              <td className="ui-table__empty" colSpan={columns.length}>
                {empty}
              </td>
            </tr>
            )}
      </tbody>
    </table>
  )
}

