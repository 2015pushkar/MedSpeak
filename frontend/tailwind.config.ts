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
        ink:   "#0e1f3d",
        mist:  "#eef3fb",
        sand:  "#d6e4f5",
        leaf:  "#2563eb",
        coral: "#e05645",
        gold:  "#d09716",
      },
      boxShadow: {
        panel: "0 18px 40px rgba(14, 31, 61, 0.11)",
      },
    },
  },
  plugins: [],
};

export default config;

