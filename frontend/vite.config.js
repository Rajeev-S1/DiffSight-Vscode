import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on a rarely used port (falls back to the next free one); /api is proxied to the backend.
const BACKEND = process.env.NEXASCAN_BACKEND || "http://127.0.0.1:38219";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 34517,
    strictPort: false,
    proxy: { "/api": { target: BACKEND, changeOrigin: false } },
  },
  build: { outDir: "dist", sourcemap: false },
});
