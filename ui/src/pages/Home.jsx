/**
 * Home — main page that wires PreferenceForm ↔ API ↔ ResultsList.
 *
 * Layout (desktop): two-column — form on the left (sticky), results on the right.
 * Layout (mobile):  stacked — form on top, results below.
 */
import { useState } from 'react'
import PreferenceForm from '../components/PreferenceForm'
import ResultsList    from '../components/ResultsList'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorMessage   from '../components/ErrorMessage'
import { getRecommendations } from '../api/recommend'

export default function Home() {
  const [results, setResults]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error,   setError]     = useState(null)

  async function handleSearch(prefs) {
    setLoading(true)
    setError(null)
    try {
      const data = await getRecommendations(prefs)
      setResults(data)
    } catch (err) {
      setError(err.message)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">

      {/* Hero strip */}
      <div className="mb-6 text-center">
        <h2 className="text-2xl sm:text-3xl font-bold text-dark">
          Bengaluru Restaurant Recommendations
        </h2>
        <p className="text-muted text-sm mt-1">
          AI-powered recommendations tailored to your taste — from 51,000+ Bengaluru restaurants
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 items-start">

        {/* Left column — Preference form (sticky on desktop) */}
        <div className="w-full lg:w-80 xl:w-96 flex-shrink-0 lg:sticky lg:top-20">
          <PreferenceForm onSubmit={handleSearch} isLoading={loading} />
        </div>

        {/* Right column — Results */}
        <div className="flex-1 min-w-0">
          {error && (
            <ErrorMessage message={error} onDismiss={() => setError(null)} />
          )}

          {loading ? (
            <LoadingSpinner />
          ) : (
            <ResultsList data={results} />
          )}
        </div>
      </div>
    </main>
  )
}
