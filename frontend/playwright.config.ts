import { defineConfig } from "@playwright/test";

/** The walkthrough that is Phase 1.1's exit criterion (#50) grows out of these.
 *
 * They ran against `stub/server.py` until #89 deleted it. They now run against
 * the real application - `chemometrics_workbench.server`, serving the built
 * bundle behind its token, exactly as the packaged application does. A Vite
 * preview server would test a page that talks to nothing.
 *
 * Three servers, because a project is a directory and a state is a project.
 * The stub reached its states through query parameters, which meant one server
 * could be anything on request; a real server is whatever its project holds, so
 * a starting state is now a seeded directory:
 *
 * - **8765 `seeded`** - the imported dataset and the artboard's four-branch
 *   pipeline with every node run. Everything that reads: the shell, spectra,
 *   results, the inspector, the canvas.
 * - **8766 `empty`** - a project with nothing in it. The empty state, and the
 *   imports, which are the tests that change the project they run in.
 * - **8767 `runs`** - the same pipeline plus a branch that cannot be fitted,
 *   and *nothing executed*. Runs really run here, so they can really be
 *   watched, cancelled and failed. Kept apart from 8765 because a failed run
 *   is remembered by the job table and would follow every later test, and
 *   because a pipeline whose arrays are all cached has no work to cancel.
 *
 * The seed runs as part of each server's command so the ordering is the shell's
 * rather than Playwright's, and each directory is recreated every run - one
 * left over from a previous run carries its arrays and its edits.
 */

const ROOT = "/tmp/chemometrics-e2e";

const serve = (name: string, port: string, seed: string) => ({
  // From the repository root, because that is where uv's environment is.
  command: `rm -rf ${ROOT}/${name} && ${seed} && uv run python -m chemometrics_workbench.server`,
  cwd: "..",
  url: `http://127.0.0.1:${port}/`,
  env: {
    CHEMOMETRICS_PROJECT: `${ROOT}/${name}`,
    CHEMOMETRICS_CONFIG_HOME: `${ROOT}/config`,
    WORKBENCH_PORT: port,
    WORKBENCH_TOKEN: "e2e-token",
    WORKBENCH_BUNDLE: "frontend/dist",
  },
  // Never reuse: a server left running from a development session holds a
  // different token and a project someone has been editing, and the failure
  // would look like a broken application.
  reuseExistingServer: false,
  timeout: 240_000,
});

export default defineConfig({
  testDir: "./e2e",
  // One at a time: the three servers are three projects on disk, and a test
  // that presses Run changes what a parallel test would read.
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8765",
    viewport: { width: 1440, height: 900 },
    // No GPU: the container this runs in has no usable one, and chromium takes
    // the whole browser down rather than falling back on its own.
    launchOptions: { args: ["--disable-gpu"] },
  },
  projects: [
    {
      name: "seeded",
      testIgnore: /(empty|runs)\.spec\.ts/,
      use: { baseURL: "http://127.0.0.1:8765" },
    },
    { name: "empty", testMatch: /empty\.spec\.ts/, use: { baseURL: "http://127.0.0.1:8766" } },
    { name: "runs", testMatch: /runs\.spec\.ts/, use: { baseURL: "http://127.0.0.1:8767" } },
  ],
  webServer: [
    serve("seeded", "8765", `uv run python tests/seed_e2e.py ${ROOT}/seeded`),
    serve("empty", "8766", `uv run python tests/seed_e2e.py --empty ${ROOT}/empty`),
    serve("runs", "8767", `uv run python tests/seed_e2e.py --unrun ${ROOT}/runs`),
  ],
});
