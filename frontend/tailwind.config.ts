import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./tests/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#1d2a2f",
        mist: "#f5f2ea",
        sand: "#e8dcc6",
        leaf: "#6a8f5b",
        coral: "#d16d5b",
        gold: "#c79b46",
      },
      boxShadow: {
        panel: "0 18px 40px rgba(29, 42, 47, 0.12)",
      },
    },
  },
  plugins: [],
};

export default config;

