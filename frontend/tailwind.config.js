/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./App.{js,ts,tsx}', './app/**/*.{js,ts,tsx}', './components/**/*.{js,ts,tsx}'],

  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        primary: '#724EDB',
        secondary: '#F0EDFB',
        background: '#FFFFFF',
        overlay: '#FFFFFF4D',
        text: '#000000',
        text_gray: '#857C7C',
        text_weak: '#C7C7C7',
        stroke: '#D3D3D3',
        good: '#FF7D7D',
        bad: '#8484FF',
      },
    },
  },
  plugins: [],
};
