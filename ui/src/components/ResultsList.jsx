/**
 * ResultsList — renders SummaryBanner + grid of RecommendationCards.
 * Shows an empty-state prompt when there are no results yet.
 */
import RecommendationCard from './RecommendationCard'
import SummaryBanner from './SummaryBanner'

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center gap-3">
      <span className="text-6xl">🍽️</span>
      <h2 className="text-dark text-xl font-bold">Ready to explore?</h2>
      <p className="text-muted text-sm max-w-xs leading-relaxed">
        Fill in your preferences on the left and hit <strong>Find Restaurants</strong> to get
        recommendations tailored to your taste.
      </p>
    </div>
  )
}

export default function ResultsList({ data }) {
  if (!data) return <EmptyState />

  const { recommendations, summary } = data

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center gap-3">
        <span className="text-5xl">🔍</span>
        <h2 className="text-dark text-lg font-bold">No restaurants found</h2>
        <p className="text-muted text-sm">Try relaxing your filters — broader location or higher budget.</p>
      </div>
    )
  }

  return (
    <section aria-label="Restaurant recommendations">
      <SummaryBanner summary={summary} count={recommendations.length} />
      <div className="grid grid-cols-1 gap-4" data-testid="results-grid">
        {recommendations.map(rec => (
          <RecommendationCard key={`${rec.rank}-${rec.name}`} rec={rec} />
        ))}
      </div>
    </section>
  )
}
