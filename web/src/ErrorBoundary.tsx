import React from 'react'

type State = { hasError: boolean; error?: any }

export default class ErrorBoundary extends React.Component<React.PropsWithChildren<{}>, State> {
  state: State = { hasError: false }
  static getDerivedStateFromError(error: any): State {
    return { hasError: true, error }
  }
  componentDidCatch(error: any, info: any) {
    // eslint-disable-next-line no-console
    console.error('UI ErrorBoundary caught:', error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
          <h2>界面发生错误</h2>
          <div style={{ color: '#c00', marginBottom: 8 }}>{String(this.state.error || '')}</div>
          <p>请刷新页面或在 Jobs 面板查看后台任务状态。</p>
        </div>
      )
    }
    return this.props.children as any
  }
}

