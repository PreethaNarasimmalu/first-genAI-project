/**
 * Phase 7 — API client.
 * Talks to the Phase 6 FastAPI backend running on :8000.
 * The Vite dev server proxies /recommend, /cuisines, /locations, /health
 * to localhost:8000 so no CORS issues in development.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? ''

async function _json(res) {
  if (res.ok) return res.json()
  const body = await res.json().catch(() => ({}))
  throw new Error(body.detail ?? `HTTP ${res.status}`)
}

/** POST /recommend — returns RecommendationResponse */
export async function getRecommendations(prefs) {
  // Strip null / undefined / empty-string values before sending
  const payload = Object.fromEntries(
    Object.entries(prefs).filter(([, v]) => v !== null && v !== undefined && v !== '')
  )
  const res = await fetch(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return _json(res)
}

/** GET /cuisines — returns string[] */
export async function getCuisines() {
  return _json(await fetch(`${API_BASE}/cuisines`))
}

/** GET /locations — returns string[] */
export async function getLocations() {
  return _json(await fetch(`${API_BASE}/locations`))
}

/** GET /health */
export async function getHealth() {
  return _json(await fetch(`${API_BASE}/health`))
}
