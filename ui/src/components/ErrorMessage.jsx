/**
 * ErrorMessage — inline error banner shown when /recommend fails.
 */
export default function ErrorMessage({ message, onDismiss }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 bg-red-50 border border-red-200 rounded-xl p-4 mt-4"
    >
      <span className="text-xl flex-shrink-0">⚠️</span>
      <div className="flex-1">
        <p className="text-red-700 font-semibold text-sm">Something went wrong</p>
        <p className="text-red-600 text-sm mt-0.5">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="text-red-400 hover:text-red-600 text-lg leading-none flex-shrink-0"
        >
          ×
        </button>
      )}
    </div>
  )
}
