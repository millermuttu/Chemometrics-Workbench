import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The server binds an ephemeral port by default; WORKBENCH_PORT pins it so
// this proxy has something to point at. Development and the packaged build
// then differ in origin only - the client always calls /api on its own origin.
//
//   uv run python -m chemometrics_workbench.server   # with WORKBENCH_PORT=8765
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  server: { proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
});
