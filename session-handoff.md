# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-11-03 (after v1.3.0)

---

## Where things stand

**`v1.3.0` is released** (2026-11-02): #407 merged dev into main, the annotated tag was pushed, and
the Release workflow (run 41501229040) published
https://github.com/millermuttu/Chemometrics-Workbench/releases/tag/v1.3.0 with the Linux (73.7 MB),
macOS (48.3 MB) and Windows (63.5 MB) packages. Phase 7, modelling breadth, is closed; its exit run is
`docs/phase-7/exit-run.md`, written by `tests/exit_run_phase7.py`, met on all 31 checks. #425 closes the
phase (CLAUDE.md, the release evidence) and closed #383; merged.

**Phase 8 — UI/UX and usability, towards `v1.4.0` — is planned but not opened.** Its issues are open,
one per feature, each with the change and its "done when":

| Issue | Feature id | What |
| --- | --- | --- |
| #410 | `panel-resize-gap` | Widening the sidebar/inspector leaves a gap: `.side`/`.insp` keep fixed widths inside the resized wrapper (`styles/shell.css:23,25`, `Shell.tsx:425,534`). First. |
| #411 | `usability-audit` | Walk the worked examples as a new user; `docs/phase-8/usability-audit.md`; each friction point an issue. Before the rest. |
| #412 | `theme-icon` | Sun/moon icon instead of the "Dark"/"Light" button (`Shell.tsx:400`). |
| #413 | `remember-layout` | Theme, panel widths, collapsed state, open tabs in localStorage per project. |
| #414 | `delete-items` | DELETE experiments, models, dataset versions (refused while in use); rename experiments; in-app confirmation. |
| #415 | `project-switcher` | Header dropdown: recent projects, new, open folder; the server switches project in place; in-app folder browser. |
| #416 | `shortcuts` | Keyboard shortcuts and a `?` overlay. |
| #417 | `run-notices` | Non-blocking notice when a run finishes or fails. |
| #418 | `example-dataset` | Open Tecator or the meat set from the empty project. |
| #419 | `parameter-help` | One-line help and range on every inspector field, from the Pydantic schema. |
| #420 | `error-messages` | Every user-facing message names what failed and what to do. |
| #421 | `accessibility` | axe in e2e, focus order, labels, AA contrast, keyboard canvas. |
| #422 | `design-questions` | Settle DESIGN_BRIEF.md §11 questions 2-4. |
| #423 | `docs-phase-8` | How-tos for projects, deleting, shortcuts; screenshots. |
| #424 | `phase-8-exit-run` | Scripted Playwright walk; timing optional; v1.4.0. |

**Maintainer's decisions for Phase 8 (2026-11-02)** — record them in the new list's `decisions`:
switch projects in place and ask for the project folder; confirm before every delete (no trash, no
undo); run the usability audit; **no undo/redo**; version `v1.4.0`; timing the exit run is optional;
the sidebar gap fix waits for the phase (not a pre-phase fix). Signed packages (#339) carry over, blocked.

**The only other open issue is #339, signed packages: `blocked`** on an Apple Developer ID with
notarisation and a Windows code-signing certificate, which only the maintainer can supply.

## What Phase 7 left behind, worth knowing

- **Arrays are float64 (#402, decision 0007).** The rank tolerance keeps float32's epsilon as the
  data's precision. Exports are held to the parity predictions class (rtol 1e-6, atol 1e-9).
- **Exactly rank-deficient data is now real.** float32 noise used to lift ranks; in float64 a
  synthetic matrix can be exactly rank r. #404: `executor._spe` reports SPE zero, limit 0.0 and a
  caveat when a PCA keeps every component. PLS's chi-squared SPE limit on a zero residual is not
  covered. SIMCA still refuses a full-rank class (its rule needs the Q limit). Test data that should
  not be low-rank needs an irregular term.
- **e2e runs only on the dev-into-main PR (#392)**, so a Phase 7 UI change (the grouped menu) broke
  `empty.spec.ts` unnoticed until #407 (#408). Run the changed spec locally before merging a UI
  feature, and run a spec **file**, not one test: tests in a file share one server's project in order.
- **A macOS runner can fail to be acquired** ("The job was not acquired by Runner of type hosted"):
  a `cancelled` package job, not a code failure. Closing and reopening the PR reran the checks.
- `results_payload` serves each regression's `cross_validated` predictions (#383).

## Development data

All Phase 5 data comes from open, licensed internet sources (decision 0006). It is downloaded into
`dataset/` at the root, which is **gitignored**, so it is only on this machine. Re-download it from the
DOIs listed in `docs/decisions/0006-phase-5-data-sources.md`. The multi-class set is the Quadram fresh-meat
FTIR CSV (3 classes, CC0). Tests that need a file in CI get a small copy in `tests/fixtures/` with its licence.

## How a new estimator lands

Spec in `docs/algorithms/`, reference values, kernel, `parity.check`, `COVERAGE` row; then the Spec
class in `EstimatorSpec`, `has_kernel` and the `_estimator` branch in `executor.py`, the payload in
`api.py`, artifact/export/report (they branch on `result.task`), and on screen `catalogue.ts`,
`nodeLabel`, `parameterLine`, `AnalysisResults.tsx`. `docs/adding-a-step.md` is the checklist.

## Operational notes

**The docs site is live**, deployed by the `pages` job on every push to `main`. Locally: run the
screenshot spec, then `uv run mkdocs build --strict`.

**#250, fixed in #251.** A red `e2e (macos-latest)` on a bookkeeping PR was a real bug: the inspector
reset its form on every pipeline refetch, so the refetch after an Apply put the old value back over
a newly typed one. The form now resets on node id and kind only, and `inspector.spec.ts` holds the
app's refetch back with `page.route` so the race runs every time.

**#351, fixed in #359**: the same `ERR_NO_BUFFER_SPACE` on one `GET /api/projects` showed "server
is not answering". `api/client.ts` `send` now retries a GET twice on a network failure.

**#255, explained and fixed in #262.** The Windows trace showed `net::ERR_NO_BUFFER_SPACE` on the
bundle's CSS: the page rendered unstyled and the canvas div had no size. #221's red had the same
signature. `frontend/index.html` now reloads once on a LINK or SCRIPT load error (sessionStorage
guard). **A Windows-only red that finds an element "hidden" is this; look for the console error first.**
Public job annotations answer without auth: `curl -s https://api.github.com/repos/millermuttu/Chemometrics-Workbench/check-runs/<job id>/annotations`.

**#260's macOS red** was the docs-examples `branch()` reading a port before the canvas's second fit
after a remount; it now waits for the port to stop moving.

**The worked examples are tested three ways** (and since #288 there are three of them, plus two how-tos). `tests/test_examples.py` recomputes every number
`docs/examples/*.md` quotes, over HTTP on `docs/examples/tecator.csv`, and asserts the page prints
it; `frontend/e2e/docs-examples.spec.ts` walks both pages through the screens on a fifth Playwright
server (8769, `examples`) and takes their screenshots; the CSV is asserted byte-equal to
`tecator_csv()` and ships with the Tecator permission note, which its terms require. A kernel change
that moves a quoted number fails the suite until the page is updated.

**#258, fixed in #260.** The canvas refits when the node count changes (not on refetch or drag,
so a user's pan survives), React Flow's `Controls` give a fit-view button, and the add-step menu
opens away from the nearer window edge. `docs-examples.spec.ts` no longer reloads or drops upward.

**Merging:** on 2026-11-02 the maintainer said to merge #405, #406, #409 and #407 once green and to tag v1.3.0; done. No standing approval carries into a new session.

**CI polling:** the unauthenticated `api.github.com` is rate-limited (60 requests an hour) and a
30-second poll exhausts it; read check runs through `mcp__github__pull_request_read` instead.

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

**`./run.sh` is the way in.** It syncs, installs with **pnpm** — this project has no
`package-lock.json` and `npm ci` refuses it — builds the bundle if there is not one, and launches,
opening the browser and printing `http://127.0.0.1:<port>/?token=<token>`. `--build` forces the
rebuild a changed `frontend/src` needs.

**Scripts worth knowing.** `uv run python -m tests.exit_run`, `tests.exit_run_phase3` and `tests.exit_run_phase5` each drive
the served application end to end and rewrite their phase's `docs/phase-N/exit-run.md`.
`uv run python -m tests.memory_probe 6000 1200` prints the peak resident memory of a ten-fold branch,
which is the number #176 is judged by.


## Next action

1. #425 merged (closed #383).
2. Open Phase 8: on a `docs/open-phase-8` branch, archive `feature_list.json` to
   `docs/phase-7/feature_list.json`, write Phase 8's list from #410-#424 with the decisions above, add the
   Phase 8 row to `PROPOSAL.md` §16 and republish the proposal artifact, update CLAUDE.md. Then start
   #410 (sidebar gap), then #411 (audit).
