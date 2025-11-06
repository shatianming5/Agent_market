import type { RunSummary, RunManifest } from './types'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8032'

export async function getJSON<T = any>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`)
  if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`)
  return r.json()
}

export async function postJSON<T = any>(path: string, body: any): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`)
  return r.json()
}

export async function putJSON<T = any>(path: string, body: any): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`PUT ${path} failed: ${r.status}`)
  return r.json()
}

export type Agent = { id: string; name: string; created_at: string }
export type Order = { id: string; agent_id: string; side: string; qty: number; status: string; created_at: string }

export async function postEmpty(path: string): Promise<void> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
  if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`)
}

export async function deleteRequest(path: string): Promise<void> {
  const r = await fetch(`${API_BASE}${path}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(`DELETE ${path} failed: ${r.status}`)
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await getJSON<{ runs?: RunSummary[] }>(`/runs`)
  return res?.runs ?? []
}

export async function getRunManifest(runId: string): Promise<RunManifest> {
  return getJSON<RunManifest>(`/runs/${encodeURIComponent(runId)}/manifest`)
}

export async function getRunEvents(runId: string, limit = 200): Promise<any[]> {
  const res = await getJSON<{ events?: any[] }>(`/runs/${encodeURIComponent(runId)}/events?limit=${limit}`)
  return res?.events ?? []
}

export function getRunFileUrl(runId: string, relativePath: string): string {
  const pathParam = encodeURIComponent(relativePath)
  return `${API_BASE}/runs/${encodeURIComponent(runId)}/files?path=${pathParam}`
}

export type Connector = {
  id: string
  name: string
  type: string
  config: Record<string, any>
  status?: string
  last_tested_at?: string
  created_at: string
  updated_at: string
  metadata?: Record<string, any>
}

export async function getConnectorTypes(): Promise<string[]> {
  return getJSON('/connectors/types')
}

export async function getConnectors(): Promise<Connector[]> {
  return getJSON('/connectors')
}

export async function createConnector(payload: {
  name: string
  connector_type: string
  config: Record<string, any>
  credentials: Record<string, any>
  metadata?: Record<string, any>
}): Promise<Connector> {
  return postJSON('/connectors', payload)
}

export async function testConnector(connectorId: string): Promise<any> {
  const r = await fetch(`${API_BASE}/connectors/${connectorId}/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  })
  if (!r.ok) throw new Error(`POST /connectors/${connectorId}/test failed: ${r.status}`)
  return r.json()
}

export async function removeConnector(connectorId: string): Promise<void> {
  await deleteRequest(`/connectors/${connectorId}`)
}


export async function getConnectorLogs(connectorId: string, limit = 20): Promise<any[]> {
  return getJSON(`/connectors/${connectorId}/logs?limit=${limit}`)
}

export async function runConnectorAction(connectorId: string): Promise<any> {
  return postJSON(`/connectors/${connectorId}/run`, {})
}

export type Secret = {
  id: string
  scope: string
  name: string
  value: string
  created_at: string
  updated_at: string
}

export async function listSecrets(token?: string, scope?: string): Promise<Secret[]> {
  const headers: Record<string, string> = {}
  if (token) headers['X-Secret-Token'] = token
  const query = scope ? `?scope=${encodeURIComponent(scope)}` : ''
  const r = await fetch(`${API_BASE}/secrets${query}`, { headers })
  if (!r.ok) throw new Error(`GET /secrets failed: ${r.status}`)
  return r.json()
}

export async function getSecret(scope: string, name: string, token?: string, reveal = false): Promise<Secret> {
  const headers: Record<string, string> = {}
  if (token) headers['X-Secret-Token'] = token
  const query = reveal ? '?reveal=true' : ''
  const r = await fetch(`${API_BASE}/secrets/${scope}/${name}${query}`, { headers })
  if (!r.ok) throw new Error(`GET /secrets/${scope}/${name} failed: ${r.status}`)
  return r.json()
}
