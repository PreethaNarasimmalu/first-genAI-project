/**
 * Phase 7 tests — API client (src/api/recommend.js)
 *
 * All fetch calls are mocked so tests run fully offline.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { getRecommendations, getCuisines, getLocations, getHealth } from '../api/recommend'

const MOCK_CUISINES  = ['Chinese', 'Italian']
const MOCK_LOCATIONS = ['Koramangala', 'Indiranagar']
const MOCK_RESPONSE  = {
  recommendations: [{
    rank: 1, name: 'Test Restaurant', location: 'Koramangala',
    cuisine: 'North Indian', rating: 4.2, approx_cost: 600,
    why: 'Great match.', highlight: 'Try the biryani.',
  }],
  summary: 'Test Restaurant is a great pick.',
}

function mockOk(body) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
}

function mockFail(status = 400, detail = 'Bad request') {
  return Promise.resolve({ ok: false, json: () => Promise.resolve({ detail }) })
}

beforeEach(() => vi.stubGlobal('fetch', vi.fn()))
afterEach(() => vi.unstubAllGlobals())

// ---------------------------------------------------------------------------
// getCuisines
// ---------------------------------------------------------------------------

describe('getCuisines', () => {
  it('returns the cuisine list on success', async () => {
    fetch.mockReturnValue(mockOk(MOCK_CUISINES))
    const result = await getCuisines()
    expect(result).toEqual(MOCK_CUISINES)
  })

  it('calls the correct endpoint', async () => {
    fetch.mockReturnValue(mockOk([]))
    await getCuisines()
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/cuisines'))
  })

  it('throws when the response is not ok', async () => {
    fetch.mockReturnValue(mockFail(500, 'Server error'))
    await expect(getCuisines()).rejects.toThrow('Server error')
  })
})

// ---------------------------------------------------------------------------
// getLocations
// ---------------------------------------------------------------------------

describe('getLocations', () => {
  it('returns the locations list on success', async () => {
    fetch.mockReturnValue(mockOk(MOCK_LOCATIONS))
    const result = await getLocations()
    expect(result).toEqual(MOCK_LOCATIONS)
  })

  it('calls the correct endpoint', async () => {
    fetch.mockReturnValue(mockOk([]))
    await getLocations()
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/locations'))
  })
})

// ---------------------------------------------------------------------------
// getHealth
// ---------------------------------------------------------------------------

describe('getHealth', () => {
  it('returns health data', async () => {
    fetch.mockReturnValue(mockOk({ status: 'ok', service: 'Restaurant Recommendation API' }))
    const result = await getHealth()
    expect(result.status).toBe('ok')
  })
})

// ---------------------------------------------------------------------------
// getRecommendations
// ---------------------------------------------------------------------------

describe('getRecommendations', () => {
  it('returns recommendation data on success', async () => {
    fetch.mockReturnValue(mockOk(MOCK_RESPONSE))
    const result = await getRecommendations({ location: 'Koramangala' })
    expect(result.recommendations).toHaveLength(1)
    expect(result.summary).toBe('Test Restaurant is a great pick.')
  })

  it('sends a POST request to /recommend', async () => {
    fetch.mockReturnValue(mockOk(MOCK_RESPONSE))
    await getRecommendations({ location: 'Koramangala' })
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/recommend'),
      expect.objectContaining({ method: 'POST' })
    )
  })

  it('sends Content-Type: application/json header', async () => {
    fetch.mockReturnValue(mockOk(MOCK_RESPONSE))
    await getRecommendations({ location: 'Koramangala' })
    expect(fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      })
    )
  })

  it('strips null and empty-string values from the payload', async () => {
    fetch.mockReturnValue(mockOk(MOCK_RESPONSE))
    await getRecommendations({ location: 'Koramangala', cuisine: null, free_text: '' })
    const body = JSON.parse(fetch.mock.calls[0][1].body)
    expect(body).not.toHaveProperty('cuisine')
    expect(body).not.toHaveProperty('free_text')
    expect(body).toHaveProperty('location', 'Koramangala')
  })

  it('throws with the API error detail on failure', async () => {
    fetch.mockReturnValue(mockFail(404, 'No restaurants found matching your preferences.'))
    await expect(getRecommendations({})).rejects.toThrow('No restaurants found')
  })

  it('throws on HTTP 503 (LLM unavailable)', async () => {
    fetch.mockReturnValue(mockFail(503, 'LLM service temporarily unavailable.'))
    await expect(getRecommendations({})).rejects.toThrow('LLM service')
  })

  it('sends the correct JSON body', async () => {
    fetch.mockReturnValue(mockOk(MOCK_RESPONSE))
    await getRecommendations({ location: 'Indiranagar', max_price: 800, min_rating: 4.0 })
    const body = JSON.parse(fetch.mock.calls[0][1].body)
    expect(body).toEqual({ location: 'Indiranagar', max_price: 800, min_rating: 4.0 })
  })
})
