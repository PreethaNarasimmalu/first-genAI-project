/**
 * LoadingSpinner — shown while the /recommend API call is in flight.
 */
export default function LoadingSpinner() {
  return (
    <div
      className="flex flex-col items-center justify-center py-24 gap-4"
      role="status"
      aria-label="Loading recommendations"
    >
      {/* Animated ring */}
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-4 border-red-100" />
        <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin" />
        <span className="absolute inset-0 flex items-center justify-center text-xl">
          🍴
        </span>
      </div>

      <div className="text-center">
        <p className="text-dark font-semibold text-base">Finding your perfect restaurants…</p>
        <p className="text-muted text-sm mt-1">Curating recommendations tailored to your taste…</p>
      </div>
    </div>
  )
}
