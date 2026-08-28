import os from "node:os";
import path from "node:path";

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

/** Where the seeded projects go.
 *
 * `os.tmpdir()` rather than `/tmp`, because this runs on Windows too, and the
 * removal is `--fresh` inside the seed script rather than an `rm -rf` here for
 * the same reason: the command is handed to whatever shell the platform has,
 * and `cmd.exe` has neither.
 */
const ROOT = path.join(os.tmpdir(), "chemometrics-e2e");

/** Seed the project, then serve it - in one process, with no shell operator.
 *
 * This used to be `<seed> && uv run python -m ...server`. Playwright hands the
 * `webServer` command to the platform's shell, which on Windows is `cmd.exe`,
 * and the chain did not survive the trip: seed and server each ran perfectly
 * there on their own - two smoke steps in CI proved it - while the two joined
 * by `&&` would not start at all. So nothing is joined any more, and the
 * directory is not on the command line either: it comes from
 * `CHEMOMETRICS_PROJECT` below, which is the variable the server reads anyway.
 */
const serve = (name: string, port: string, mode: string) => ({
  // From the repository root, because that is where uv's environment is.
  command: ["uv run python tests/seed_e2e.py --serve --fresh", mode].filter(Boolean).join(" "),
  cwd: "..",
  url: `http://127.0.0.1:${port}/`,
  env: {
    CHEMOMETRICS_PROJECT: path.join(ROOT, name),
    CHEMOMETRICS_CONFIG_HOME: path.join(ROOT, "config"),
    WORKBENCH_PORT: port,
    WORKBENCH_TOKEN: "e2e-token",
    WORKBENCH_BUNDLE: path.join("frontend", "dist"),
  },
  // Never reuse: a server left running from a development session holds a
  // different token and a project someone has been editing, and the failure
  // would look like a broken application.
  reuseExistingServer: false,
  // Generous because this seeds a project through the kernels before the
  // server starts, and a Windows runner is slower at all of it than the
  // machine this was written on.
  timeout: 420_000,
});

export default defineConfig({
  testDir: "./e2e",
  // On a runner: the `github` reporter, which turns each failure into a
  // workflow annotation carrying the message and the line it failed on. A
  // three-platform matrix is worth little if the machine that failed is not
  // the machine anyone can read, and an exit code is not a finding. The HTML
  // report and the trace are uploaded beside it for whoever wants to step
  // through it.
  reporter: process.env.CI
    ? [["github"], ["list"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],
  // One at a time: the three servers are three projects on disk, and a test
  // that presses Run changes what a parallel test would read.
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:8765",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    // No GPU: the container this runs in has no usable one, and chromium takes
    // the whole browser down rather than falling back on its own.
    launchOptions: { args: ["--disable-gpu"] },
  },
  projects: [
    {
      name: "seeded",
      testIgnore: /(empty|runs|walkthrough)\.spec\.ts/,
      use: { baseURL: "http://127.0.0.1:8765" },
    },
    { name: "empty", testMatch: /empty\.spec\.ts/, use: { baseURL: "http://127.0.0.1:8766" } },
    { name: "runs", testMatch: /runs\.spec\.ts/, use: { baseURL: "http://127.0.0.1:8767" } },
    {
      // #50's exit criterion starts from a project with nothing in it and
      // imports into it, so it cannot share one with anything - `empty` leaves
      // a dataset behind, and the walkthrough would find it.
      name: "walkthrough",
      testMatch: /walkthrough\.spec\.ts/,
      use: { baseURL: "http://127.0.0.1:8768" },
    },
  ],
  webServer: [
    serve("seeded", "8765", ""),
    serve("empty", "8766", "--empty"),
    serve("runs", "8767", "--unrun"),
    serve("walkthrough", "8768", "--empty"),
  ],
});
