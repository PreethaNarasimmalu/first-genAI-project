/**
 * Navbar — Zomato-inspired top navigation bar.
 * Red brand bar with logo, tagline, and a subtle food emoji accent.
 */
export default function Navbar() {
  return (
    <header className="bg-primary shadow-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <span className="text-2xl" role="img" aria-label="food">🍽️</span>
            <div>
              <h1 className="text-white text-xl font-bold tracking-tight leading-none">
                FoodieAI
              </h1>
              <p className="text-red-200 text-xs font-medium leading-none mt-0.5">
                Discover · Taste · Enjoy
              </p>
            </div>
          </div>

          {/* Tagline */}
          <div className="hidden md:block text-right">
            <p className="text-red-100 text-sm font-medium">
              AI-powered restaurant recommendations
            </p>
            <p className="text-red-200 text-xs">Bangalore · Powered by Groq</p>
          </div>
        </div>
      </div>
    </header>
  )
}
