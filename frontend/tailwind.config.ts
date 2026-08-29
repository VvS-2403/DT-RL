import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'primary': '#1f2937',
        'secondary': '#0b80f6',
        'accent': '#06b6d4',
        gray: {
          900: '#141414',
          800: '#181818',
          700: '#252526',
          600: '#333333',
        },
      },
    },
  },
  plugins: [],
}
export default config
