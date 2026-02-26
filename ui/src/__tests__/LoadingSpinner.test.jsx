/**
 * Phase 7 tests — LoadingSpinner component
 */
import { render, screen } from '@testing-library/react'
import LoadingSpinner from '../components/LoadingSpinner'

describe('LoadingSpinner', () => {
  it('renders without crashing', () => {
    render(<LoadingSpinner />)
  })

  it('has role="status" for accessibility', () => {
    render(<LoadingSpinner />)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  it('has an accessible aria-label', () => {
    render(<LoadingSpinner />)
    expect(screen.getByLabelText(/loading/i)).toBeInTheDocument()
  })

  it('shows a descriptive loading message', () => {
    render(<LoadingSpinner />)
    expect(screen.getByText(/finding your perfect restaurants/i)).toBeInTheDocument()
  })

  it('shows a secondary hint message', () => {
    render(<LoadingSpinner />)
    expect(screen.getByText(/Curating recommendations tailored to your taste/i)).toBeInTheDocument()
  })
})
