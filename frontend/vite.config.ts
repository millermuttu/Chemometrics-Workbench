import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The server binds an ephemeral port by default; WORKBENCH_PORT pins it so
// this proxy has something to point at. Development and the packaged build
// then differ in origin only - the client always calls /api on its own origin.
//
//   WORKBENCH_DEV=1 WORKBENCH_PORT=8765 uv run python -m chemometrics_workbench.server
//
// WORKBENCH_DEV=1 because the proxy forwards the browser's Origin, and the
// server refuses any origin but its own unless it is told this is development.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  server: { proxy: { "/api": { target: API_TARGET, changeOrigin: true } } },
});
