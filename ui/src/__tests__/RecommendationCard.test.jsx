/**
 * Phase 7 tests — RecommendationCard component
 */
import { render, screen } from '@testing-library/react'
import RecommendationCard from '../components/RecommendationCard'

const MOCK_REC = {
  rank: 1,
  name: 'Trattoria',
  location: 'Indiranagar',
  cuisine: 'Italian, Continental',
  rating: 4.3,
  approx_cost: 900,
  why: 'Cozy Italian spot with a romantic ambiance, well within budget.',
  highlight: 'Try the truffle pasta and the tiramisu.',
}

describe('RecommendationCard', () => {
  it('renders without crashing', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
  })

  it('displays the restaurant name', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('Trattoria')).toBeInTheDocument()
  })

  it('displays the location', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('Indiranagar')).toBeInTheDocument()
  })

  it('displays the cuisine tags', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('Italian')).toBeInTheDocument()
    expect(screen.getByText('Continental')).toBeInTheDocument()
  })

  it('displays the rating', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('★ 4.3')).toBeInTheDocument()
  })

  it('displays the approximate cost', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('₹900')).toBeInTheDocument()
    expect(screen.getByText(/for two/i)).toBeInTheDocument()
  })

  it('displays the "why" text', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText(MOCK_REC.why)).toBeInTheDocument()
  })

  it('displays the highlight dish', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText(MOCK_REC.highlight)).toBeInTheDocument()
  })

  it('displays the rank badge', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByText('#1')).toBeInTheDocument()
  })

  it('has a data-testid for easy selection in integration tests', () => {
    render(<RecommendationCard rec={MOCK_REC} />)
    expect(screen.getByTestId('recommendation-card')).toBeInTheDocument()
  })

  it('handles single-word cuisine without crashing', () => {
    const rec = { ...MOCK_REC, cuisine: 'Chinese' }
    render(<RecommendationCard rec={rec} />)
    expect(screen.getByText('Chinese')).toBeInTheDocument()
  })

  it('renders with rank 2 correctly', () => {
    const rec = { ...MOCK_REC, rank: 2 }
    render(<RecommendationCard rec={rec} />)
    expect(screen.getByText('#2')).toBeInTheDocument()
  })

  it('applies green badge for high rating (≥ 4.0)', () => {
    render(<RecommendationCard rec={{ ...MOCK_REC, rating: 4.5 }} />)
    const badge = screen.getByText('★ 4.5').closest('span')
    expect(badge.className).toContain('bg-success')
  })

  it('applies orange badge for medium rating (3.0–3.9)', () => {
    render(<RecommendationCard rec={{ ...MOCK_REC, rating: 3.5 }} />)
    const badge = screen.getByText('★ 3.5').closest('span')
    expect(badge.className).toContain('bg-orange')
  })
})
