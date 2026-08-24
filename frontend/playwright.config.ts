import { defineConfig } from "@playwright/test";

// The walkthrough that is Phase 1.1's exit criterion (#50) lives here. For now
// it is one smoke test, running against the production build rather than the
// dev server, so what is tested is what would ship.
export default defineConfig({
  testDir: "./e2e",
  use: {
    baseURL: "http://127.0.0.1:4173",
    viewport: { width: 1440, height: 900 },
    // No GPU: the container this runs in has no usable one, and chromium takes
    // the whole browser down rather than falling back on its own.
    launchOptions: { args: ["--disable-gpu"] },
  },
  webServer: {
    // --host 127.0.0.1 is not decoration: without it vite preview binds
    // whatever localhost resolves to, which on a CI runner is ::1, and the
    // wait below polls 127.0.0.1 until it times out.
    command: "pnpm preview --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
  },
});
