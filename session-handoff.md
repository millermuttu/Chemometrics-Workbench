# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-09-29

---

## Where things stand

**Phase 3 is complete and released as `v0.8.0`**, merged into `main` through #230 and tagged on
2026-09-28. Its completed list is archived at `docs/phase-3/feature_list.json`, and its exit
criterion is recorded in `docs/phase-3/exit-run.md`.

**Phase 4, package and release, is open.** `feature_list.json` is its list: nine entries, issues
#231 to #239, and a `decisions` block recording what was settled on 2026-09-28:
- The exit run is done by a real non-developer the maintainer arranges.
- Packages ship unsigned, with a "How to open this" page.
- The docs site is MkDocs Material (development dependency only), published on GitHub Pages.
- Package formats: Windows `.zip`, Apple Silicon `.dmg`, Linux `.tar.gz`.
- The name stays "Chemometrics Workbench".

**Phase 3's exit criterion**, `PROPOSAL.md` §16: *two models differing only in preprocessing can be
compared step by step; an exported model reproduces application predictions within tolerance in a
clean environment.* `uv run python -m tests.exit_run_phase3` shows both over HTTP on Tecator:

- **Export.** SNV, Savitzky-Golay (11, 2, d1), a 0.25 train/test split, mean centre and PLS with
  5 LV on `fat`. The served `export.py` runs with `python -I` in a `uv venv` that holds only NumPy,
  and matches the application at `docs/model-export.md` §5's `rtol 1e-4`. The largest relative
  difference is 7.97e-6 on the calibration rows and 2.51e-6 on the held-out rows.
- **Comparison.** The window is changed to 15 and the pipeline run again. Both records are read back
  from the history and diffed by `diff.ts`'s rule, rewritten in the script. The diff names `savgol`
  and `window_length` and nothing else.
- **It can fail.** With a tolerance tighter than the float32 store and a second run that also changes
  `polyorder`, the same script reports both halves not met.

## Phase 3, all merged through green pull requests

| Feature | Issue | PR |
| --- | --- | --- |
| Every run kept, and one opened to its record | #209 | #210 |
| The model artifact, one file readable without this application | #211 | #212 |
| A plain JSON model and a standalone prediction snippet | #213 | #214 |
| Two experiments compared step by step | #215 | #216 |
| A model registry, and the first schema change | #219 | #220 |
| An edit-and-restore poll gets a real budget | #222 | #223 |
| One experiment as a standalone HTML report | #224 | #225 |
| The exit criterion demonstrated | #227 | #228 |

**#217 recorded a process mistake.** `feature/215_lineage-comparison` was cut from
`feature/213_json-and-snippet-export` rather than from `dev`, so #216 carried the export commit into
`dev` and #214 then merged as a no-op. **Cut every branch from a freshly pulled `dev`, and check with
`git merge-base --is-ancestor origin/dev <branch>` before opening the pull request.**

## Current work

**Nothing is `in_progress`.** Phase 4 so far, merged through green pull requests:

| Feature | Issue | PR |
| --- | --- | --- |
| Host and Origin checked on every request | #231 | #241 |
| A launcher: the browser opens on the workbench | #232 | #242 |
| PyInstaller builds on three platforms, each smoke-tested | #233 | #244 |
| A release workflow: a tag builds and publishes the three packages | #234 | #246 |
| A documentation site (blocked on the Pages deploy from main) | #235 | #248 |
| Inspector: a refetch no longer wipes a half-typed edit | #250 | #251 |
| Export JSON, Export Python and Save report buttons | #247 | #252 |
| CONTRIBUTING.md completed, docs/adding-a-step.md, Normalise in the step list | #238 | #254 |

The open issues are #235 (blocked on the deploy), #236, #237, #239 and #255.

**The docs site is `blocked`, deliberately.** It builds strict in CI (`docs` job) with screenshots
from `frontend/e2e/docs-screens.spec.ts`, and deploys to Pages from `main` only (`pages` job). The
maintainer chose on 2026-09-29 to wait for the phase-end merge rather than deploy from `dev`. Before
that merge, **Settings → Pages → Source must be "GitHub Actions"**, then check
https://millermuttu.github.io/Chemometrics-Workbench/ answers and mark `docs-site` passing. Locally:
run the screenshot spec, then `uv run mkdocs build --strict`.

**#250, fixed in #251.** A red `e2e (macos-latest)` on a bookkeeping PR was a real bug: the inspector
reset its form on every pipeline refetch, so the refetch after an Apply put the old value back over
a newly typed one. The form now resets on node id and kind only, and `inspector.spec.ts` holds the
app's refetch back with `page.route` so the race runs every time.

**#255, an unexplained Windows e2e failure.** On #254, `pipeline.spec.ts:75` found
`pipeline-canvas` but hidden for 5 s; 68 others passed and the diff touched nothing in the canvas.
The trace is the `playwright-report-windows-latest` artifact of run 36540710418 (kept 7 days, to
2026-10-06); the MCP tools cannot fetch it. The maintainer merged #254 over it. Read that trace
before treating the next Windows red as a flake. This is the third Windows-only e2e failure (#210,
#221's unread one, this).

**Merging:** the maintainer gave a standing "merge when CI is green" on 2026-09-29, for that session.

**#247, `ui-export-buttons`, is done.** `download()` in `api/client.ts` fetches with the token and
saves through a blob URL; `screens/DownloadButton.tsx` shows the server's refusal beside the button.
`e2e/downloads.spec.ts` covers it. The `.cwmodel` download (optional) was not built.

**Releasing.** `git tag -a vX.Y.Z -m "notes"` then push the tag: release.yml builds through the
reusable package.yml, refuses a lightweight tag, and publishes with the annotation, a size table and
a link to docs/how-to-open.md. A hyphenated tag is a pre-release. `v0.9.0-rc1` is the proof and is
public; delete it or keep it as history.

**Packaging.** `uv run pyinstaller packaging/workbench.spec --noconfirm` after `pnpm build` gives
`dist/ChemometricsWorkbench/`; `uv run python -m tests.smoke_package dist/ChemometricsWorkbench`
drives it. The reusable `package.yml` does both on three platforms and uploads the archives. `.gitignore`
ignores `*.spec`, so the spec is un-ignored by name. Download / unpacked: macOS 48 / 95 MB, Windows 62 / 146 MB, Linux 68 / 174 MB.

**The launcher is `python -m chemometrics_workbench`** (`src/chemometrics_workbench/__main__.py`),
and it is the entry script PyInstaller should take. It serves on an ephemeral port and, once the
socket is listening, hands the token URL to `webbrowser.open` **on a daemon thread**: called on the
event loop, a browser that blocks holds the loop that has to answer it, and the first version of the
test deadlocked exactly that way. `python -m chemometrics_workbench.server` stays serve-only, because
Playwright's `seed_e2e --serve`, the exit runs and the dev loop all start it and none should open a
browser. `./run.sh` now execs the launcher. In a frozen application the bundle resolves to
`sys._MEIPASS/frontend/dist`, so the PyInstaller spec has to put the built bundle at that relative
path.

**The auto mode permission check failed for a stretch on 2026-09-28**: every Bash and GitHub write
answered "the classifier gave no verdict". It was transient, and leaving auto mode got past it. If
it happens again, stop retrying (repeated failures end the turn) and say so.

**An unexplained Windows e2e failure.** #221's last run went red on `e2e (windows-latest)` only, on a
markdown-only change. Nobody has read that job's log. If a Windows e2e check goes red again, read the
log first and open an issue for the cause rather than re-running:
https://github.com/millermuttu/Chemometrics-Workbench/actions/runs/35343976638/job/105596198698

**Untracked in the root, still not decided:** `AGENTS.md` (a Codex copy of `CLAUDE.md` that will
drift) and `tecator.csv`. Either gitignore them or commit them.

**`./run.sh` is the way in.** It syncs, installs with **pnpm** — this project has no
`package-lock.json` and `npm ci` refuses it — builds the bundle if there is not one, and launches,
opening the browser and printing `http://127.0.0.1:<port>/?token=<token>`. `--build` forces the
rebuild a changed `frontend/src` needs.

**Scripts worth knowing.** `uv run python -m tests.exit_run` and `tests.exit_run_phase3` each drive
the served application end to end and rewrite their phase's `docs/phase-N/exit-run.md`.
`uv run python -m tests.memory_probe 6000 1200` prints the peak resident memory of a ten-fold branch,
which is the number #176 is judged by.

## Next action

**Every entry with its dependencies met is done.** Left: `published-parity-report` (#236) and
`worked-examples` (#237), which depend on `docs-site` (blocked only on the Pages deploy from main;
ask the maintainer whether they may start before it), #255, and the exit run. The exit run (#239) needs the release workflow and the docs site, and a person the
maintainer arranges.

## What Phase 2 added

All merged through green pull requests on 2026-09-18:

| Feature | Issue | PR |
| --- | --- | --- |
| One version source, an unrun node says so, no `stale` state the server never sends | #181 | #189 |
| Estimator and split nodes edited in the inspector; PLS models a real column | #182 | #190 |
| Phase 2 entered on the list, with the two decisions the review left open | — | #191 |
| Jackson–Mudholkar limit with `h0 <= 0` returned with its caveat | #71 | #192 |
| Train/test split executes, and the canvas offers it | #183 | #194 |
| VIP and the folded coefficient vector are drawn | #184 | #198 |
| The experiment record carries a regression's metrics | #188 | #199 |
| The exit criterion demonstrated on Tecator | #201 | #202 |
| Two-class PLS-DA: specification, kernel, parity claims, screen | #185 | #203 |
| Contribution plots, and the macOS tab-overflow flake | #186, #168 | #204 |
| The Bruker OPUS reader | #187 | #205 |
| A run holds its frontier, not every array it has computed | #176 | #206 |

**Three decisions taken during the phase**, recorded in the archived entries rather than only here:

- **#71.** A non-positive `h0` returns the limit *with a caveat* naming it. Clamping (what `mdatools`
  does) hides that the assumption failed; raising refuses a plot for a dataset that is otherwise fine.
- **The exit criterion.** "Matches reference software within stated tolerance" means the workflow over
  HTTP compared against an independent PLS *on the experiment's own resolved folds*, within the parity
  tolerances. Kernel parity alone is not it — the executor's fold handling is exactly what kernel
  parity cannot see, and #173 was that.
- **PLS-DA is two-class only.** One dummy column is PLS1, which the existing kernel fits and the
  existing parity claims cover. Three classes are PLS2, which `pls-regression.md` §10 defers, and are
  refused by name rather than reduced.

## What Phase 3 has added worth knowing

- **The export splits the chain at the last unfoldable step.** Everything after it folds into the
  coefficient vector; everything up to and including it is carried as a residual chain. SNV, MSC and
  normalise are written out; a baseline is refused by name rather than approximated.
- **The export tolerance is `rtol = 1e-4` and the number is measured, not chosen.** The store is
  float32 and an export computes float64, so `docs/phase-2/exit-run.md`'s 7.09e-6 relative difference
  is the floor. `docs/model-export.md` §5 says so, and says what to do if the store ever goes float64.
- **An artifact is refused by schema version, the way `db.py` refuses a newer database.** A newer
  writer may have added a field whose absence this reader would take as a default.
- **The lineage diff matches nodes by id, so a renamed node reads as a remove plus an add.** That is
  the honest reading: nothing in the model says a rename is not a replacement.
- **`db.SCHEMA_VERSION` is 2, and the upgrade it added is additive only.** A database stamped below it
  gets `create_all` and a re-stamp, which writes a missing table and leaves an existing one alone. A
  column that is added, renamed, retyped or dropped is *not* covered, and shipping one means writing
  real migration machinery rather than widening that branch.
- **Saving a model writes the artifact before the row.** A row pointing at a file that was never
  written is a registry entry nobody can open; a file with no row costs disk. The failure mode was
  chosen, not stumbled into.
- **A feature list note can be wrong.** #219's said the `model` table existed. It did not. Check the
  code before believing a note about what is already built.
- **A test that edits and restores is borrowing the seeded project.** When the restore misses its
  poll budget the edit stays, and the next test fails for a reason that is not its own — which is how
  #222 produced two red tests from one cause. `APPLIED` in `inspector.spec.ts` is the budget, and
  `expect.poll`'s default five seconds is what a loaded macOS runner misses.
- **A red macOS check on a markdown-only change is a flake, and it still has a cause.** #221 went red
  with no source change. Reading the job log found a real ordering problem worth fixing rather than a
  reason to press the button again.
- **The HTML report is reachable over the API only.** `GET /api/experiments/{id}/report.html`
  (`current` or an id) serves the file; no button in the application asks for it yet. Its plots are
  hand-written inline SVG, and it looks the stored result up against the experiment's own pipeline
  snapshot, so a run whose arrays are gone gets a sentence instead of plots. PDF is deferred.
- **CI runs mypy; run it locally too.** #225's first push went red on both `check` jobs for a
  `no-any-return` in a test fixture that ruff and pytest were both happy with. The local gate is
  `uv run ruff check && uv run ruff format --check && uv run mypy && uv run pytest`.
- **Ruff's import sorting merges a fixture import into the line above and strands its `noqa`.**
  `tests/test_report.py` keeps `from tests.test_server import client  # noqa: F401` on its own line
  after an `# isort: split` marker, with `# noqa: F811` on each `client` parameter.
- **The GitHub MCP merge wants the full 40-character sha** in `expectedHeadSha`; a short one is
  refused before anything happens. `git rev-parse HEAD` gives it.
- **A branch cut from another feature branch carries that feature into the pull request.** #216
  merged #213's export commit as a side effect, which nothing in the pull request said.
- **`count()` does not wait.** A Playwright assertion of the form `expect(await x.count())` compares a
  frame rather than a state, and that is what failed on the Windows runner in #210. Use
  `expect.poll(() => x.count())`.

## What Phase 2 left worth knowing

- **A train/test split is one `Fold` and not a partition.** `validate_partition` refuses it,
  `_State.display` returns the one array for a single fold, and `_pls` computes the cross-validated
  block only for more than one fold. `stratify_by` is refused by name until a class column exists.
- **A classification is the regression on a dummy response.** `_fit_pls1` is shared, `task` tells them
  apart, and every model quantity — scores, loadings, VIP, both diagnostics — means the same thing for
  both. The analysis tab keeps every regression panel and swaps one.
- **The seeded e2e project now holds sixteen nodes**, not fifteen: `plsda_d` sits beside `pls_d` below
  the split, and the seeded Tecator CSV carries a `fat_class` column. Five counts in `pipeline.spec.ts`
  and `inspector.spec.ts` moved with it.
- **The `runs` e2e project's last two tests rewrite its pipeline** through the API — one swaps the
  k-fold for a train/test split and drops the failing branch, the other adds a mean-centre-only branch
  so the raw-axis coefficients have a foldable chain to fold. Anything added to that file after them
  sees the rewritten graph.
- **`Run.displays` is a mapping over the store, not a dict of arrays.** A `Run` sits in the job table
  for the life of the process; one holding arrays kept a whole run resident after it finished.
- **An `id()`-based dedupe of a split's repeated array is wrong.** A freed array's address is reused,
  so folds pair with the wrong file. Tried, caught by `test_executor.py`, reverted.

## Older lessons still worth not rediscovering

- **`pnpm build` typechecks `src/__tests__`.** A "prove the test fails on the unfixed code" run that
  reverts the source but keeps the new unit tests fails to *build* and never runs Playwright, and the
  grep afterwards prints nothing — which looks like a pass of the wrong kind. Revert the tests too.
- **Outline buttons are named label plus dim.** `getByRole("button", { name: "SNV", exact: true })`
  matches nothing, because the row's accessible name is `SNV snv`. Use a regex anchored at the start.
- **A regression test is trusted only after it fails on the unfixed code.** Every test added in Phase 2
  was run that way first, and two of them failed for the wrong reason until the setup was fixed.
- **"It changed" is not the claim "it is what I set it to".** #162 shipped green and did nothing,
  because its test asserted a stored position was *not* the pre-drag value and a generated fallback
  satisfies that too.
- **Never `git stash -u` with a pathspec in this checkout.** On 2026-09-17 an untracked `.agents/`
  disappeared that way and the stash's untracked tree was empty.
- **A stray `pkill` takes Playwright's own web servers with it.** `fuser -k -n tcp 8765 8766 8767 8768`
  before a run, never during one.
- **This project uses pnpm**, and `run.sh` said `npm ci` until #160 and failed on any clean checkout.
  Delete the artefact before verifying the code that builds it.

## Three canvas traps, all of which cost time

- **A node's label is built from its parameters.** Editing `SG d1 w11` to a window of 9 renames it
  `SG d1 w9`, and `snv_savgol` — same window, same label — inherits the old name alone. Re-finding an
  edited node by the name it used to have finds a *different* node.
- **Making nodes draggable breaks two things silently.** A drag ends with a click on the node it moved,
  which opens its tab, so the drag has to consume that click; and any header button becomes a drag
  handle without React Flow's `nodrag` class.
- **`fitView` re-fits the viewport on every mount**, so comparing a node's screen box either side of a
  reload compares two zoom levels and fails on a change that did not happen.
