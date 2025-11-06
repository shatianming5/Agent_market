import React from 'react'

type Props = React.PropsWithChildren<unknown>
type BoundaryError = unknown
type State = { hasError: boolean; error?: BoundaryError }

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: BoundaryError): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: BoundaryError, info: unknown) {
    // eslint-disable-next-line no-console
    console.error('UI ErrorBoundary caught:', error, info)
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
          <h2>Something went wrong</h2>
          <div style={{ color: '#c00', marginBottom: 8 }}>{String(this.state.error || '')}</div>
          <p>Please refresh the page or open the Jobs panel to review recent task activity.</p>
        </div>
      )
    }
    return this.props.children ?? null
  }
}
