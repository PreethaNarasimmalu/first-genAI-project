/**
 * PreferenceForm — Zomato-inspired preference panel.
 *
 * Fetches /cuisines and /locations from the API on mount to populate the
 * dropdowns. On submit calls onSubmit(preferences) with the cleaned payload.
 * Location and Cuisine are required; all other fields are optional.
 */
import { useState, useEffect } from 'react'
import { getCuisines, getLocations } from '../api/recommend'

const MEAL_TYPES = ['Dine-out', 'Delivery', 'Buffet', 'Cafes', 'Desserts', 'Pubs and bars']

function Label({ children, htmlFor, required }) {
  return (
    <label htmlFor={htmlFor} className="block text-xs font-semibold text-muted uppercase tracking-wide mb-1.5">
      {children}
      {required && <span className="text-primary ml-0.5">*</span>}
    </label>
  )
}

function FieldError({ message }) {
  if (!message) return null
  return <p className="text-primary text-xs mt-1">{message}</p>
}

function Select({ id, value, onChange, children, disabled, error }) {
  return (
    <select
      id={id}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className={`w-full border rounded-lg px-3 py-2 text-sm text-dark bg-white
                 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent
                 disabled:opacity-50 disabled:cursor-not-allowed
                 ${error ? 'border-primary' : 'border-border'}`}
    >
      {children}
    </select>
  )
}

function Toggle({ id, checked, onChange, label }) {
  return (
    <label htmlFor={id} className="flex w-full items-center justify-between cursor-pointer">
      <span className="text-sm text-dark">{label}</span>
      <div className="relative flex-shrink-0">
        <input
          id={id}
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
        />
        <div className={`w-10 h-5 rounded-full transition-colors ${checked ? 'bg-primary' : 'bg-gray-200'}`} />
        <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transform transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
      </div>
    </label>
  )
}

export default function PreferenceForm({ onSubmit, isLoading }) {
  const [cuisines, setCuisines]   = useState([])
  const [locations, setLocations] = useState([])
  const [metaLoading, setMetaLoading] = useState(true)
  const [errors, setErrors] = useState({})

  const [form, setForm] = useState({
    cuisine:      '',
    location:     '',
    max_price:    '',
    min_rating:   '',
    online_order: false,
    book_table:   false,
    meal_type:    '',
    free_text:    '',
  })

  // Fetch cuisines + locations on mount
  useEffect(() => {
    Promise.all([getCuisines(), getLocations()])
      .then(([c, l]) => { setCuisines(c); setLocations(l) })
      .catch(() => {})
      .finally(() => setMetaLoading(false))
  }, [])

  function set(key) {
    return e => {
      setForm(f => ({ ...f, [key]: e.target.value }))
      if (errors[key]) setErrors(err => ({ ...err, [key]: '' }))
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    const newErrors = {}
    if (!form.location) newErrors.location = 'Please select a location'
    if (!form.cuisine)  newErrors.cuisine  = 'Please select a cuisine'
    if (Object.keys(newErrors).length) {
      setErrors(newErrors)
      return
    }
    const payload = {
      cuisine:      [form.cuisine],
      location:     form.location,
      max_price:    form.max_price  ? Number(form.max_price)  : undefined,
      min_rating:   form.min_rating ? parseFloat(form.min_rating) : undefined,
      online_order: form.online_order || undefined,
      book_table:   form.book_table  || undefined,
      meal_type:    form.meal_type   || undefined,
      free_text:    form.free_text   || undefined,
    }
    onSubmit(payload)
  }

  function handleReset() {
    setForm({ cuisine: '', location: '', max_price: '', min_rating: '',
              online_order: false, book_table: false, meal_type: '', free_text: '' })
    setErrors({})
  }

  return (
    <form
      onSubmit={handleSubmit}
      aria-label="Restaurant preference form"
      className="bg-white rounded-2xl shadow-card border border-border overflow-hidden"
    >
      <div className="p-5 space-y-4">

        {/* Location + Cuisine — side by side */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label htmlFor="location" required>Location</Label>
            <Select id="location" value={form.location} onChange={set('location')} disabled={metaLoading} error={errors.location}>
              <option value="">Select a location</option>
              {locations.map(l => <option key={l} value={l}>{l}</option>)}
            </Select>
            <FieldError message={errors.location} />
          </div>
          <div>
            <Label htmlFor="cuisine" required>Cuisine</Label>
            <Select id="cuisine" value={form.cuisine} onChange={set('cuisine')} disabled={metaLoading} error={errors.cuisine}>
              <option value="">Select a cuisine</option>
              {cuisines.map(c => <option key={c} value={c}>{c}</option>)}
            </Select>
            <FieldError message={errors.cuisine} />
          </div>
        </div>

        {/* Budget — optional */}
        <div>
          <Label htmlFor="max_price">Max budget (₹ for two)</Label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted text-sm font-medium">₹</span>
            <input
              id="max_price"
              type="number"
              min="1"
              placeholder="e.g. 800 (optional)"
              value={form.max_price}
              onChange={set('max_price')}
              className="w-full border border-border rounded-lg pl-7 pr-3 py-2 text-sm text-dark
                         focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
        </div>

        {/* Min Rating — optional */}
        <div>
          <Label htmlFor="min_rating">Minimum rating <span className="normal-case font-normal text-muted">(optional)</span></Label>
          <div className="flex gap-2">
            {[3.0, 3.5, 4.0, 4.5].map(r => (
              <button
                key={r}
                type="button"
                onClick={() => setForm(f => ({ ...f, min_rating: f.min_rating === r.toFixed(1) ? '' : r.toFixed(1) }))}
                className={`flex-1 py-2 rounded-lg border text-sm font-medium transition-all
                  ${form.min_rating === r.toFixed(1)
                    ? 'bg-primary text-white border-primary'
                    : 'bg-white text-dark border-border hover:border-primary hover:text-primary'}`}
              >
                ★ {r}+
              </button>
            ))}
          </div>
        </div>

        {/* Meal Type — optional */}
        <div>
          <Label htmlFor="meal_type">Meal type <span className="normal-case font-normal text-muted">(optional)</span></Label>
          <Select id="meal_type" value={form.meal_type} onChange={set('meal_type')}>
            <option value="">Any type</option>
            {MEAL_TYPES.map(m => <option key={m} value={m}>{m}</option>)}
          </Select>
        </div>

        {/* Toggles — optional */}
        <div className="space-y-3 pt-1">
          <Toggle
            id="online_order"
            label="Online ordering available"
            checked={form.online_order}
            onChange={v => setForm(f => ({ ...f, online_order: v }))}
          />
          <Toggle
            id="book_table"
            label="Table booking available"
            checked={form.book_table}
            onChange={v => setForm(f => ({ ...f, book_table: v }))}
          />
        </div>

        {/* Free text — optional */}
        <div>
          <Label htmlFor="free_text">Anything specific? <span className="normal-case font-normal text-muted">(optional)</span></Label>
          <textarea
            id="free_text"
            placeholder='e.g. "Romantic rooftop with cocktails"'
            value={form.free_text}
            onChange={set('free_text')}
            rows={3}
            className="w-full border border-border rounded-lg px-3 py-2 text-sm text-dark resize-none
                       focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>

        {/* Actions */}
        <div className="flex gap-2 pt-1">
          <button
            type="submit"
            disabled={isLoading}
            className="flex-1 bg-primary text-white font-semibold py-2.5 rounded-xl text-sm
                       hover:bg-red-600 active:bg-red-700 disabled:opacity-60 disabled:cursor-not-allowed
                       shadow-sm hover:shadow transition-all"
          >
            {isLoading ? 'Searching…' : '🔍 Find Restaurants'}
          </button>
          <button
            type="button"
            onClick={handleReset}
            disabled={isLoading}
            className="px-3 py-2.5 rounded-xl border border-border text-muted text-sm
                       hover:border-primary hover:text-primary disabled:opacity-50"
          >
            Reset
          </button>
        </div>
      </div>
    </form>
  )
}
