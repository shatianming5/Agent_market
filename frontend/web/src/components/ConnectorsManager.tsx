import React, { useEffect, useMemo, useState } from 'react'

import {
  createConnector,
  getConnectorLogs,
  getConnectorTypes,
  getConnectors,
  removeConnector,
  runConnectorAction,
  testConnector,
} from '../api'
import type { Connector } from '../api'
import { Card, CardHeader, CardBody, SectionTitle, SectionDescription, StatusChip, Stack } from './ui'

const templates: Record<string, { config: Record<string, any>; credentials: Record<string, any>; metadata?: Record<string, any> }> = {
  evm: {
    config: { timeout: 5 },
    credentials: { rpc_url: 'http://localhost:8545' },
  },
  binance: {
    config: { timeout: 5 },
    credentials: { api_key: 'YOUR_KEY', api_secret: 'YOUR_SECRET' },
    metadata: { job_command: ['echo', 'sync-binance'] },
  },
  okx: {
    config: { timeout: 5 },
    credentials: { api_key: 'YOUR_KEY', api_secret: 'YOUR_SECRET' },
  },
  coingecko: {
    config: {},
    credentials: {},
  },
  covalent: {
    config: {},
    credentials: { api_key: 'YOUR_API_KEY' },
  },
  telegram: {
    config: { default_chat: '@channel' },
    credentials: { bot_token: '123456789:ABCDEF' },
  },
  discord: {
    config: {},
    credentials: { bot_token: 'YOUR_BOT_TOKEN' },
  },
}

function toJsonText(value: Record<string, any> | undefined): string {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return '{}'
  }
}

const statusTone = (status?: string | null) => {
  const value = (status || '').toLowerCase()
  if (value === 'healthy') return { label: 'Healthy', tone: 'success' as const }
  if (value === 'error') return { label: 'Error', tone: 'danger' as const }
  if (value === 'created') return { label: 'Not tested', tone: 'default' as const }
  return { label: status || 'Unknown', tone: 'default' as const }
}

export default function ConnectorsManager() {
  const [types, setTypes] = useState<string[]>([])
  const [connectors, setConnectors] = useState<Connector[]>([])
  const [loading, setLoading] = useState(false)
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [selectedType, setSelectedType] = useState<string>('')
  const [configText, setConfigText] = useState('{}')
  const [credentialsText, setCredentialsText] = useState('{}')
  const [metadataText, setMetadataText] = useState('{}')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [logs, setLogs] = useState<Record<string, any[]>>({})
  const [logsLoading, setLogsLoading] = useState('')
  const [runningAction, setRunningAction] = useState('')

  const sortedConnectors = useMemo(() => {
    return [...connectors].sort((a, b) => b.created_at.localeCompare(a.created_at))
  }, [connectors])

  const loadTemplatesFor = (type: string) => {
    const template = templates[type] || { config: {}, credentials: {} }
    setConfigText(toJsonText(template.config))
    setCredentialsText(toJsonText(template.credentials))
    setMetadataText(toJsonText(template.metadata))
  }

  async function loadData() {
    setLoading(true)
    setError('')
    try {
      const [typeList, connectorList] = await Promise.all([getConnectorTypes(), getConnectors()])
      setTypes(typeList)
      setConnectors(connectorList)
      if (!selectedType && typeList.length) {
        setSelectedType(typeList[0])
        loadTemplatesFor(typeList[0])
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load connector data.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleTypeChange = (type: string) => {
    setSelectedType(type)
    loadTemplatesFor(type)
  }

  async function handleCreate() {
    setMessage('')
    setError('')
    if (!selectedType) {
      setError('Select a connector type.')
      return
    }
    if (!name.trim()) {
      setError('Connector name is required.')
      return
    }
    let parsedConfig: Record<string, any> = {}
    let parsedCredentials: Record<string, any> = {}
    let parsedMetadata: Record<string, any> | undefined
    try {
      parsedConfig = configText.trim() ? JSON.parse(configText) : {}
    } catch {
      setError('Could not parse configuration JSON.')
      return
    }
    try {
      parsedCredentials = credentialsText.trim() ? JSON.parse(credentialsText) : {}
    } catch {
      setError('Could not parse credentials JSON.')
      return
    }
    try {
      parsedMetadata = metadataText.trim() ? JSON.parse(metadataText) : undefined
    } catch {
      setError('Could not parse metadata JSON.')
      return
    }
    setCreating(true)
    try {
      await createConnector({
        name: name.trim(),
        connector_type: selectedType,
        config: parsedConfig,
        credentials: parsedCredentials,
        metadata: parsedMetadata,
      })
      setMessage('Connector created successfully.')
      setName('')
      loadTemplatesFor(selectedType)
      await loadData()
    } catch (err: any) {
      setError(err?.message || 'Failed to create connector.')
    } finally {
      setCreating(false)
    }
  }

  async function handleTest(connectorId: string) {
    setMessage('')
    setError('')
    try {
      const result = await testConnector(connectorId)
      setMessage(`Test completed: ${JSON.stringify(result)}`)
      await loadData()
    } catch (err: any) {
      setError(err?.message || 'Connector test failed.')
    }
  }

  async function handleDelete(connectorId: string) {
    if (!window.confirm('Delete this connector?')) return
    setMessage('')
    setError('')
    try {
      await removeConnector(connectorId)
      setMessage('Connector deleted.')
      await loadData()
    } catch (err: any) {
      setError(err?.message || 'Failed to delete connector.')
    }
  }

  async function loadLogs(connectorId: string) {
    setLogsLoading(connectorId)
    setError('')
    try {
      const entries = await getConnectorLogs(connectorId)
      setLogs((prev) => ({ ...prev, [connectorId]: entries || [] }))
    } catch (err: any) {
      setError(err?.message || 'Failed to load connector logs.')
    } finally {
      setLogsLoading('')
    }
  }

  async function handleRunAction(connectorId: string) {
    setRunningAction(connectorId)
    setMessage('')
    setError('')
    try {
      const result = await runConnectorAction(connectorId)
      setMessage(`Action triggered: ${JSON.stringify(result)}`)
    } catch (err: any) {
      setError(err?.message || 'Failed to execute connector action.')
    } finally {
      setRunningAction('')
    }
  }

  return (
    <div className="connectors-layout">
      <div className="connectors-column connectors-column--form">
        <Card>
          <CardHeader title="Create a connector" subtitle="Follow the assistant steps to register a new connector." />
          <CardBody>
            <Stack gap={16}>
              {error ? <div className="data-lab__status danger">{error}</div> : null}
              {message ? <div className="data-lab__status success">{message}</div> : null}
              <div>
                <SectionTitle>Step 1 - Choose a type</SectionTitle>
                <SectionDescription>Templates help you start with sensible defaults for each connector family.</SectionDescription>
                <select value={selectedType} onChange={(e) => handleTypeChange(e.target.value)}>
                  <option value="">Select a type</option>
                  {types.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </div>
              <label className="form-control">
                <span className="form-label">Name</span>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Example: primary-binance-node"
                />
              </label>
              <div>
                <SectionTitle>Step 2 - Configuration</SectionTitle>
                <SectionDescription>Update shared parameters for the connector (JSON format).</SectionDescription>
                <textarea value={configText} onChange={(e) => setConfigText(e.target.value)} rows={6} spellCheck={false} />
              </div>
              <div>
                <SectionTitle>Step 3 - Credentials</SectionTitle>
                <SectionDescription>Provide sensitive credentials in JSON. Values are encrypted on the server.</SectionDescription>
                <textarea
                  value={credentialsText}
                  onChange={(e) => setCredentialsText(e.target.value)}
                  rows={6}
                  spellCheck={false}
                />
              </div>
              <div>
                <SectionTitle>Optional - Metadata</SectionTitle>
                <SectionDescription>Supply automation metadata such as scheduled jobs or trigger relationships.</SectionDescription>
                <textarea value={metadataText} onChange={(e) => setMetadataText(e.target.value)} rows={4} spellCheck={false} />
              </div>
              <div className="button-group" style={{ justifyContent: 'flex-end' }}>
                <button onClick={handleCreate} disabled={creating}>
                  {creating ? 'Creating...' : 'Create connector'}
                </button>
              </div>
            </Stack>
          </CardBody>
        </Card>
      </div>

      <div className="connectors-column connectors-column--list">
        <Card>
          <CardHeader title="Registered connectors" subtitle="Inspect health, fetch logs and trigger actions." />
          <CardBody>
            {loading ? (
              <div className="expr-empty">Loading connectors...</div>
            ) : sortedConnectors.length === 0 ? (
              <div className="expr-empty">No connectors have been created yet.</div>
            ) : (
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Last tested</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedConnectors.map((connector) => {
                    const tone = statusTone(connector.status)
                    const logItems = logs[connector.id] || []
                    return (
                      <React.Fragment key={connector.id}>
                        <tr>
                          <td>{connector.name}</td>
                          <td>{connector.type}</td>
                          <td>
                            <StatusChip label={tone.label} tone={tone.tone} />
                          </td>
                          <td>{connector.last_tested_at ? new Date(connector.last_tested_at).toLocaleString() : '--'}</td>
                          <td>
                            <div className="connectors-actions">
                              <button onClick={() => handleTest(connector.id)}>Test</button>
                              <button
                                className="ghost-button"
                                onClick={() => loadLogs(connector.id)}
                                disabled={logsLoading === connector.id}
                              >
                                {logsLoading === connector.id ? 'Loading...' : 'View logs'}
                              </button>
                              <button
                                className="ghost-button"
                                onClick={() => handleRunAction(connector.id)}
                                disabled={runningAction === connector.id}
                              >
                                {runningAction === connector.id ? 'Running...' : 'Trigger action'}
                              </button>
                              <button className="ghost-button" onClick={() => handleDelete(connector.id)}>
                                Delete
                              </button>
                            </div>
                          </td>
                        </tr>
                        {logItems.length ? (
                          <tr className="connectors-log-row">
                            <td colSpan={5}>
                              <div className="jobs-logs__panel">
                                <pre>
                                  {logItems
                                    .map((entry) => {
                                      const timestamp = entry.created_at ? new Date(entry.created_at).toLocaleString() : ''
                                      const payload = entry.payload ? JSON.stringify(entry.payload, null, 2) : ''
                                      return `${timestamp} | ${entry.status}: ${payload}`
                                    })
                                    .join('\n')}
                                </pre>
                              </div>
                            </td>
                          </tr>
                        ) : null}
                      </React.Fragment>
                    )
                  })}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
