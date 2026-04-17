import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#121212",
        surface: "#181818",
        sidebar: "#000000",
        accent: "#1DB954",
        text: "#FFFFFF",
        muted: "#B3B3B3",
      },
      fontFamily: {
        sans: ["Inter", "Circular Std", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        glow: "0 18px 40px rgba(0,0,0,0.35)",
      },
      backgroundImage: {
        aurora:
          "radial-gradient(circle at top left, rgba(29,185,84,0.18), transparent 24%), linear-gradient(180deg, #161616 0%, #121212 50%, #0b0b0b 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
