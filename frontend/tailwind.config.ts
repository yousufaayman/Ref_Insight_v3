
import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "#ee2d40", // Red accent for borders
        input: "#ee2d40", // Red for input borders
        ring: "#ffffff", // White for focus rings
        background: "#3a4d65", // Navy blue background
        foreground: "#ffffff", // White text for better readability
        primary: {
          DEFAULT: "#ee2d40", // Red
          foreground: "#ffffff", // White text on red background
        },
        secondary: {
          DEFAULT: "#ffffff", // White
          foreground: "#3a4d65", // Navy blue text on white background
        },
        destructive: {
          DEFAULT: "#ee2d40", // Red for destructive actions
          foreground: "#ffffff", // White text
        },
        muted: {
          DEFAULT: "#ffffff", // White
          foreground: "#3a4d65", // Navy blue text
        },
        accent: {
          DEFAULT: "#ee2d40", // Red for accents
          foreground: "#ffffff", // White text on red background
        },
        popover: {
          DEFAULT: "#3a4d65", // Navy blue for popover background
          foreground: "#ffffff", // White text
        },
        card: {
          DEFAULT: "#ffffff", // White card background
          foreground: "#3a4d65", // Navy blue text on white card
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;

