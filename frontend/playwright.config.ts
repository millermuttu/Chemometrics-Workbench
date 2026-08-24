import { defineConfig } from "@playwright/test";

/** The walkthrough that is Phase 1.1's exit criterion (#50) grows out of these.
 *
 * They run against the stub server serving the built bundle - the same origin,
 * the same handlers and the same token the packaged application will use. A
 * Vite preview server would test a page that talks to nothing.
 */
export default defineConfig({
  testDir: "./e2e",
  use: {
    baseURL: "http://127.0.0.1:8765",
    viewport: { width: 1440, height: 900 },
    // No GPU: the container this runs in has no usable one, and chromium takes
    // the whole browser down rather than falling back on its own.
    launchOptions: { args: ["--disable-gpu"] },
  },
  webServer: {
    // From the repository root, because that is where uv's environment is.
    command: "uv run python stub/server.py",
    cwd: "..",
    url: "http://127.0.0.1:8765/",
    env: { STUB_PORT: "8765", STUB_TOKEN: "e2e-token", STUB_JOB_STEP_SECONDS: "1" },
    // Never reuse: a server left running from a development session holds a
    // different token, and the failure would look like a broken application.
    reuseExistingServer: false,
  },
});
