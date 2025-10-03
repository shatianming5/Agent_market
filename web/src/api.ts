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
