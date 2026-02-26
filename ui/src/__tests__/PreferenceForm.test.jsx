/**
 * Phase 7 tests — PreferenceForm component
 *
 * The form fetches /cuisines and /locations on mount.
 * We mock those with vi.stubGlobal('fetch', ...) to keep tests fast & offline.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, beforeEach, afterEach } from 'vitest'
import PreferenceForm from '../components/PreferenceForm'

const MOCK_CUISINES  = ['Chinese', 'Italian', 'North Indian']
const MOCK_LOCATIONS = ['Indiranagar', 'Koramangala', 'Whitefield']

function mockFetch(cuisines = MOCK_CUISINES, locations = MOCK_LOCATIONS) {
  vi.stubGlobal('fetch', vi.fn(url => {
    const body = url.includes('/cuisines')  ? cuisines
               : url.includes('/locations') ? locations
               : []
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
  }))
}

beforeEach(() => mockFetch())
afterEach(() => vi.unstubAllGlobals())

describe('PreferenceForm', () => {
  it('renders the form element', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByRole('form', { name: /preference form/i })).toBeInTheDocument()
  })

  it('renders the cuisine dropdown', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByLabelText(/cuisine/i)).toBeInTheDocument()
  })

  it('renders the location dropdown', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByLabelText(/location/i)).toBeInTheDocument()
  })

  it('renders the budget input', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByLabelText(/max budget/i)).toBeInTheDocument()
  })

  it('renders the free text textarea', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByLabelText(/anything specific/i)).toBeInTheDocument()
  })

  it('renders the submit button', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByRole('button', { name: /find restaurants/i })).toBeInTheDocument()
  })

  it('renders the reset button', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument()
  })

  it('populates cuisine options from API', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Italian' })).toBeInTheDocument()
    })
  })

  it('populates location options from API', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'Koramangala' })).toBeInTheDocument()
    })
  })

  it('calls onSubmit when form is submitted', async () => {
    const onSubmit = vi.fn()
    render(<PreferenceForm onSubmit={onSubmit} isLoading={false} />)
    // Wait for dropdowns to populate (mocked fetch resolves asynchronously)
    await waitFor(() => screen.getByRole('option', { name: 'Italian' }))
    // Fill required fields before submitting
    fireEvent.change(screen.getByLabelText(/location/i), { target: { value: 'Indiranagar' } })
    fireEvent.change(screen.getByLabelText(/cuisine/i), { target: { value: 'Italian' } })
    fireEvent.click(screen.getByRole('button', { name: /find restaurants/i }))
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('passes free_text value in the submitted payload', async () => {
    const onSubmit = vi.fn()
    render(<PreferenceForm onSubmit={onSubmit} isLoading={false} />)
    await waitFor(() => screen.getByRole('option', { name: 'Italian' }))
    fireEvent.change(screen.getByLabelText(/location/i), { target: { value: 'Indiranagar' } })
    fireEvent.change(screen.getByLabelText(/cuisine/i), { target: { value: 'Italian' } })
    const textarea = screen.getByLabelText(/anything specific/i)
    await userEvent.type(textarea, 'romantic dinner')
    fireEvent.click(screen.getByRole('button', { name: /find restaurants/i }))
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ free_text: 'romantic dinner' })
    )
  })

  it('disables the submit button while loading', () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={true} />)
    expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled()
  })

  it('shows "Searching…" text while loading', () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={true} />)
    expect(screen.getByText(/searching…/i)).toBeInTheDocument()
  })

  it('resets form fields when Reset is clicked', async () => {
    render(<PreferenceForm onSubmit={() => {}} isLoading={false} />)
    const textarea = screen.getByLabelText(/anything specific/i)
    await userEvent.type(textarea, 'hello')
    expect(textarea.value).toBe('hello')
    fireEvent.click(screen.getByRole('button', { name: /reset/i }))
    expect(textarea.value).toBe('')
  })
})
