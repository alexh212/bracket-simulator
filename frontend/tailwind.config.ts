import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    borderRadius: {
      none: "0px",
      sm: "14px",
      DEFAULT: "14px",
      md: "14px",
      lg: "14px",
      xl: "14px",
      "2xl": "14px",
      "3xl": "14px",
      full: "9999px",
    },
    extend: {
      fontFamily: {
        sans: ["'Inter'", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
