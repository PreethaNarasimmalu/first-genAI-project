/**
 * Home — centered single-column layout.
 * Form in the center, results below.
 */
import { useState } from 'react'
import PreferenceForm from '../components/PreferenceForm'
import ResultsList    from '../components/ResultsList'
import LoadingSpinner from '../components/LoadingSpinner'
import ErrorMessage   from '../components/ErrorMessage'
import { getRecommendations } from '../api/recommend'

export default function Home() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

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
    <main className="max-w-2xl mx-auto px-4 sm:px-6 py-10">

      {/* Centered preference form */}
      <div className="mb-10">
        <PreferenceForm onSubmit={handleSearch} isLoading={loading} />
      </div>

      {/* Results below */}
      <div>
        {error && (
          <ErrorMessage message={error} onDismiss={() => setError(null)} />
        )}
        {loading ? (
          <LoadingSpinner />
        ) : (
          <ResultsList data={results} />
        )}
      </div>

    </main>
  )
}
