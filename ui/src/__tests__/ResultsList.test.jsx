/**
 * Phase 7 tests — ResultsList component
 */
import { render, screen } from '@testing-library/react'
import ResultsList from '../components/ResultsList'

const MOCK_DATA = {
  recommendations: [
    {
      rank: 1, name: 'Trattoria', location: 'Indiranagar', cuisine: 'Italian',
      rating: 4.3, approx_cost: 900,
      why: 'Cozy Italian spot.', highlight: 'Truffle pasta.',
    },
    {
      rank: 2, name: 'Spice Garden', location: 'Koramangala', cuisine: 'North Indian',
      rating: 4.1, approx_cost: 600,
      why: 'Great for groups.', highlight: 'Butter chicken.',
    },
  ],
  summary: 'Here are the best picks matching your preferences.',
}

describe('ResultsList', () => {
  describe('when data is null (initial state)', () => {
    it('renders the empty state prompt', () => {
      render(<ResultsList data={null} />)
      expect(screen.getByText(/ready to explore/i)).toBeInTheDocument()
    })

    it('shows instruction text', () => {
      render(<ResultsList data={null} />)
      expect(screen.getByText(/fill in your preferences/i)).toBeInTheDocument()
    })
  })

  describe('when recommendations are returned', () => {
    it('renders all recommendation cards', () => {
      render(<ResultsList data={MOCK_DATA} />)
      const cards = screen.getAllByTestId('recommendation-card')
      expect(cards).toHaveLength(2)
    })

    it('renders the summary banner', () => {
      render(<ResultsList data={MOCK_DATA} />)
      expect(screen.getByText(MOCK_DATA.summary)).toBeInTheDocument()
    })

    it('shows the correct recommendation count in the banner', () => {
      render(<ResultsList data={MOCK_DATA} />)
      expect(screen.getByText(/Found 2 recommendations tailored to your taste/i)).toBeInTheDocument()
    })

    it('renders restaurant names', () => {
      render(<ResultsList data={MOCK_DATA} />)
      expect(screen.getByText('Trattoria')).toBeInTheDocument()
      expect(screen.getByText('Spice Garden')).toBeInTheDocument()
    })

    it('has an accessible section label', () => {
      render(<ResultsList data={MOCK_DATA} />)
      expect(screen.getByRole('region', { name: /restaurant recommendations/i })).toBeInTheDocument()
    })
  })

  describe('when recommendations list is empty', () => {
    it('shows the no-results message', () => {
      render(<ResultsList data={{ recommendations: [], summary: '' }} />)
      expect(screen.getByText(/no restaurants found/i)).toBeInTheDocument()
    })
  })
})
