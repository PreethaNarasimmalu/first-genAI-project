/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        primary:  '#E23744',
        orange:   '#FC8019',
        success:  '#3D9B6D',
        dark:     '#1C1C1C',
        muted:    '#696969',
        surface:  '#F8F8F8',
        border:   '#E8E8E8',
      },
      fontFamily: {
        sans: ['Okra', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        card: '0 2px 12px rgba(0,0,0,0.08)',
        'card-hover': '0 4px 20px rgba(0,0,0,0.13)',
      },
    },
  },
  plugins: [],
}
