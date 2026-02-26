/**
 * SummaryBanner — displays the 2-3 sentence LLM-generated conversational
 * summary above the recommendation cards.
 */
export default function SummaryBanner({ summary, count }) {
  return (
    <div className="bg-gradient-to-r from-orange-50 to-red-50 border border-orange-200 rounded-xl p-4 mb-5 flex items-start gap-3">
      <span className="text-2xl flex-shrink-0" role="img" aria-label="sparkle">✨</span>
      <div>
        <p className="text-dark text-sm font-semibold mb-1">
          AI found {count} recommendation{count !== 1 ? 's' : ''} for you
        </p>
        <p className="text-muted text-sm leading-relaxed">{summary}</p>
      </div>
    </div>
  )
}
