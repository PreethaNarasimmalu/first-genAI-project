/**
 * RecommendationCard — compact restaurant card with expandable details.
 *
 * Default view: rank, name, location, rating, cost, cuisine tags.
 * Expanded view: adds "Why this?" and "Must Try" sections.
 *
 * Props:
 *   rec  — { rank, name, location, cuisine, rating, approx_cost, why, highlight }
 */
import { useState } from 'react'

function RatingBadge({ rating }) {
  const color =
    rating >= 4.0 ? 'bg-success text-white' :
    rating >= 3.0 ? 'bg-orange text-white' :
                    'bg-red-500 text-white'
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold ${color}`}>
      ★ {rating.toFixed(1)}
    </span>
  )
}

function RankBadge({ rank }) {
  return (
    <div className="absolute top-3 left-3 w-7 h-7 rounded-full bg-primary text-white text-xs font-bold flex items-center justify-center shadow">
      #{rank}
    </div>
  )
}

export default function RecommendationCard({ rec }) {
  const [expanded, setExpanded] = useState(false)
  const cuisines = rec.cuisine.split(',').map(c => c.trim()).filter(Boolean)

  return (
    <article
      className="bg-white rounded-2xl shadow-card hover:shadow-card-hover relative overflow-hidden border border-border"
      data-testid="recommendation-card"
    >
      {/* Coloured top accent bar */}
      <div className="h-1.5 bg-gradient-to-r from-primary to-orange" />

      <div className="p-5">
        <RankBadge rank={rec.rank} />

        {/* Header row */}
        <div className="flex items-start justify-between gap-2 ml-8">
          <div className="flex-1 min-w-0">
            <h3 className="text-dark font-bold text-base leading-snug truncate">
              {rec.name}
            </h3>
            <p className="text-muted text-xs mt-0.5 flex items-center gap-1">
              <span>📍</span>
              <span>{rec.location}</span>
            </p>
          </div>
          <RatingBadge rating={rec.rating} />
        </div>

        {/* Cuisine tags */}
        <div className="flex flex-wrap gap-1.5 mt-3">
          {cuisines.map(c => (
            <span
              key={c}
              className="px-2 py-0.5 bg-red-50 text-primary border border-red-100 rounded-full text-xs font-medium"
            >
              {c}
            </span>
          ))}
        </div>

        {/* Cost */}
        <p className="text-muted text-xs mt-3 flex items-center gap-1">
          <span>💰</span>
          <span>
            <span className="text-dark font-semibold">₹{rec.approx_cost}</span>
            {' '}for two
          </span>
        </p>

        {/* Expandable details */}
        {expanded && (
          <div className="mt-4 space-y-2">
            <hr className="border-border" />
            <div className="flex items-start gap-2 pt-1">
              <span className="text-sm flex-shrink-0">💡</span>
              <div>
                <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-0.5">Why this?</p>
                <p className="text-dark text-sm leading-relaxed">{rec.why}</p>
              </div>
            </div>
            {rec.highlight && (
              <div className="flex items-start gap-2 bg-orange-50 rounded-lg p-2.5">
                <span className="text-sm flex-shrink-0">⭐</span>
                <div>
                  <p className="text-xs font-semibold text-orange uppercase tracking-wide mb-0.5">Must Try</p>
                  <p className="text-dark text-sm">{rec.highlight}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* See details / Hide details button */}
        <button
          onClick={() => setExpanded(prev => !prev)}
          className="mt-4 w-full text-center text-xs font-semibold text-primary border border-red-200 rounded-lg py-1.5 hover:bg-red-50 transition-colors"
        >
          {expanded ? 'Hide details ▲' : 'See details ▼'}
        </button>
      </div>
    </article>
  )
}
