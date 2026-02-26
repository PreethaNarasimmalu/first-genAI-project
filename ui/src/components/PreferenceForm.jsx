/**
 * PreferenceForm — Zomato-inspired preference panel.
 *
 * Fetches /cuisines and /locations from the API on mount to populate the
 * dropdowns. On submit calls onSubmit(preferences) with the cleaned payload.
 */
import { useState, useEffect } from 'react'
import { getCuisines, getLocations } from '../api/recommend'

const MEAL_TYPES = ['Dine-out', 'Delivery', 'Buffet', 'Cafes', 'Desserts', 'Pubs and bars']

function Label({ children, htmlFor }) {
  return (
    <label htmlFor={htmlFor} className="block text-xs font-semibold text-muted uppercase tracking-wide mb-1.5">
      {children}
    </label>
  )
}

function Select({ id, value, onChange, children, disabled }) {
  return (
    <select
      id={id}
      value={value}
      onChange={onChange}
      disabled={disabled}
      className="w-full border border-border rounded-lg px-3 py-2 text-sm text-dark bg-white
                 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent
                 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {children}
    </select>
  )
}

function Toggle({ id, checked, onChange, label }) {
  return (
    <label htmlFor={id} className="flex items-center justify-between cursor-pointer group">
      <span className="text-sm text-dark group-hover:text-primary transition-colors">{label}</span>
      <div className="relative">
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
    return e => setForm(f => ({ ...f, [key]: e.target.value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    const payload = {
      cuisine:      form.cuisine    ? [form.cuisine] : undefined,
      location:     form.location   || undefined,
      max_price:    form.max_price  ? Number(form.max_price)  : undefined,
      min_rating:   form.min_rating ? Number(form.min_rating) : undefined,
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
  }

  return (
    <form
      onSubmit={handleSubmit}
      aria-label="Restaurant preference form"
      className="bg-white rounded-2xl shadow-card border border-border overflow-hidden"
    >
      {/* Form header */}
      <div className="bg-primary px-5 py-4">
        <h2 className="text-white font-bold text-base">Find Restaurants</h2>
        <p className="text-red-200 text-xs mt-0.5">Tell us what you're craving</p>
      </div>

      <div className="p-5 space-y-4">

        {/* Cuisine */}
        <div>
          <Label htmlFor="cuisine">Cuisine</Label>
          <Select id="cuisine" value={form.cuisine} onChange={set('cuisine')} disabled={metaLoading}>
            <option value="">Any cuisine</option>
            {cuisines.map(c => <option key={c} value={c}>{c}</option>)}
          </Select>
        </div>

        {/* Location */}
        <div>
          <Label htmlFor="location">Location</Label>
          <Select id="location" value={form.location} onChange={set('location')} disabled={metaLoading}>
            <option value="">Any location</option>
            {locations.map(l => <option key={l} value={l}>{l}</option>)}
          </Select>
        </div>

        {/* Budget */}
        <div>
          <Label htmlFor="max_price">Max budget (₹ for two)</Label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted text-sm font-medium">₹</span>
            <input
              id="max_price"
              type="number"
              min="1"
              placeholder="e.g. 800"
              value={form.max_price}
              onChange={set('max_price')}
              className="w-full border border-border rounded-lg pl-7 pr-3 py-2 text-sm text-dark
                         focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>
        </div>

        {/* Min Rating */}
        <div>
          <Label htmlFor="min_rating">Minimum rating</Label>
          <div className="flex gap-2">
            {[3.0, 3.5, 4.0, 4.5].map(r => (
              <button
                key={r}
                type="button"
                onClick={() => setForm(f => ({ ...f, min_rating: f.min_rating === String(r) ? '' : String(r) }))}
                className={`flex-1 py-1.5 rounded-lg border text-xs font-semibold transition-all
                  ${form.min_rating === String(r)
                    ? 'bg-primary text-white border-primary'
                    : 'bg-white text-muted border-border hover:border-primary hover:text-primary'}`}
              >
                ★ {r}+
              </button>
            ))}
          </div>
        </div>

        {/* Meal Type */}
        <div>
          <Label htmlFor="meal_type">Meal type</Label>
          <Select id="meal_type" value={form.meal_type} onChange={set('meal_type')}>
            <option value="">Any type</option>
            {MEAL_TYPES.map(m => <option key={m} value={m}>{m}</option>)}
          </Select>
        </div>

        {/* Toggles */}
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

        {/* Free text */}
        <div>
          <Label htmlFor="free_text">Anything specific?</Label>
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
