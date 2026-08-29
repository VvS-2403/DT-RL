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
        'secondary': '#001926',
        'accent': '#fe03e1',
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
