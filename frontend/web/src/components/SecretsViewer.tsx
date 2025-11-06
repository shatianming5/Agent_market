import React, { useState } from 'react'

import { listSecrets, getSecret } from '../api'
import type { Secret } from '../api'
import { Card, CardHeader, CardBody, SectionTitle, SectionDescription, StatusChip, Stack } from './ui'

export default function SecretsViewer() {
  const [readToken, setReadToken] = useState('')
  const [adminToken, setAdminToken] = useState('')
  const [scope, setScope] = useState('')
  const [secrets, setSecrets] = useState<Secret[]>([])
  const [selected, setSelected] = useState<Secret | null>(null)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)

  async function loadSecrets() {
    setError('')
    setMessage('')
    setSelected(null)
    setLoading(true)
    try {
      const token = readToken || adminToken || undefined
      const items = await listSecrets(token, scope || undefined)
      setSecrets(items)
      if (!items.length) {
        setMessage('No secrets found for the current filters.')
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load secrets.')
    } finally {
      setLoading(false)
    }
  }

  async function handleReveal(secret: Secret) {
    setError('')
    setMessage('')
    try {
      const full = await getSecret(secret.scope, secret.name, adminToken || undefined, true)
      setSelected(full)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to reveal secret. Admin token required.')
    }
  }

  return (
    <div className="secrets-layout">
      <div className="secrets-column">
        <Card>
          <CardHeader title="Secret access" subtitle="Use scoped tokens to browse and retrieve stored secrets." />
          <CardBody>
            {error ? <div className="data-lab__status danger">{error}</div> : null}
            {message ? <div className="data-lab__status info">{message}</div> : null}
            <Stack gap={14}>
              <div>
                <SectionTitle>Access tokens</SectionTitle>
                <SectionDescription>
                  Provide a read-only token or an administrator token. Administrator tokens can decrypt secret values.
                </SectionDescription>
                <div className="secrets-token-grid">
                  <label>
                    Read token
                    <input value={readToken} onChange={(e) => setReadToken(e.target.value)} placeholder="X-Secret-Token" />
                  </label>
                  <label>
                    Admin token
                    <input value={adminToken} onChange={(e) => setAdminToken(e.target.value)} placeholder="Required for reveal" />
                  </label>
                  <label>
                    Scope
                    <input value={scope} onChange={(e) => setScope(e.target.value)} placeholder="Leave empty for all scopes" />
                  </label>
                </div>
              </div>
              <div className="button-group" style={{ justifyContent: 'flex-end' }}>
                <button onClick={loadSecrets} disabled={loading}>
                  {loading ? 'Loading...' : 'Load secrets'}
                </button>
              </div>
            </Stack>
          </CardBody>
        </Card>
      </div>

      <div className="secrets-column secrets-column--list">
        <Card>
          <CardHeader
            title="Secret catalog"
            subtitle="Select a secret to inspect metadata or reveal the stored value."
            actions={<StatusChip label={`${secrets.length} items`} tone="default" />}
          />
          <CardBody>
            {secrets.length === 0 ? (
              <div className="expr-empty">No secrets available. Adjust the filter and try again.</div>
            ) : (
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Scope</th>
                    <th>Name</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {secrets.map((secret) => (
                    <tr key={secret.id} className={selected && selected.id === secret.id ? 'runs-row--active' : undefined}>
                      <td>{secret.scope}</td>
                      <td>{secret.name}</td>
                      <td>{new Date(secret.created_at).toLocaleString()}</td>
                      <td>
                        <div className="connectors-actions">
                          <button className="ghost-button" onClick={() => setSelected(secret)}>
                            Inspect masked
                          </button>
                          <button className="ghost-button" onClick={() => handleReveal(secret)} disabled={!adminToken}>
                            Reveal secret
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Secret detail" subtitle="Review metadata and decrypted values." />
          <CardBody>
            {selected ? (
              <Stack gap={12}>
                <div className="runs-detail-grid">
                  <div>
                    <span className="jobs-detail-label">Scope</span>
                    <span className="jobs-detail-value">{selected.scope}</span>
                  </div>
                  <div>
                    <span className="jobs-detail-label">Name</span>
                    <span className="jobs-detail-value">{selected.name}</span>
                  </div>
                  <div>
                    <span className="jobs-detail-label">Created</span>
                    <span className="jobs-detail-value">{new Date(selected.created_at).toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="jobs-detail-label">Updated</span>
                    <span className="jobs-detail-value">{new Date(selected.updated_at).toLocaleString()}</span>
                  </div>
                </div>
                <div>
                  <SectionTitle>Value</SectionTitle>
                  <div className="jobs-logs__panel">
                    <pre>{selected.value}</pre>
                  </div>
                </div>
              </Stack>
            ) : (
              <div className="expr-empty">Select a secret from the table to view details.</div>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  )
}
