/**
 * Navbar — Big Zomato-red hero banner with brand title and tagline.
 */
export default function Navbar() {
  return (
    <header className="w-full text-center shadow-lg" style={{ backgroundColor: '#E23744', padding: '3rem 1rem 3.5rem' }}>
      <h1 className="text-white font-bold tracking-tight" style={{ fontSize: '3.5rem', lineHeight: 1.1 }}>
        Zomato AI
      </h1>
      <p className="font-medium mt-3" style={{ color: '#fecdd3', fontSize: '1.15rem' }}>
        Tell us what you're craving. We'll find the perfect place.
      </p>
    </header>
  )
}
