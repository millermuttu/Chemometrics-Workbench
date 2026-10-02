# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-10-02

---

## Where things stand

**`v1.0.0` is released** (2026-10-01). Phase 4's list is archived at `docs/phase-4/feature_list.json`;
its timed exit run stays `blocked` there, closed on the maintainer's decision, and is not carried forward.

**Phase 5, methods breadth, is open** (2026-10-02). `feature_list.json` is its list: 22 entries, issues
#268 to #289, and a `decisions` block. In short:
- Classifiers: multi-class PLS-DA (on a new PLS2 kernel), SIMCA, LDA, kNN. SVM deferred.
- Outliers are flagged; the user excludes. An exclusion is a derived `DatasetVersion`
  (`excluded_samples`, `derived_from` already exist in `models.py` but nothing honours them yet).
- Variable selection: VIP threshold, iPLS, CARS, each applied as an explicit `select_variables` step.
- Readers: MATLAB `.mat`, SPC, SPA, ASD, each with a real licensed fixture.
- Smoothing: moving average, median, Gaussian, Whittaker.
- One release, `v1.1.0`, at the end.

`PROPOSAL.md` §16 has the Phase 5 row and §6 marks the four formats Phase 5. `AGENTS.md`, `.agents/`,
`.codex/` and the root `tecator.csv` are now gitignored.

## Open questions for the phase

- **A multi-class dataset** for parity and the exit run. Tecator has no classes (binned fat is the
  fallback). Needs a public multi-class NIR/IR set with a usable licence, ideally in one of the new formats.
- **Real SPC, SPA and ASD files** to commit as fixtures: the maintainer's own, or openly licensed ones
  (spectrochempy, specio test data) after a licence check.

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

**#255, explained and fixed in #262.** The Windows trace showed `net::ERR_NO_BUFFER_SPACE` on the
bundle's CSS: the page rendered unstyled and the canvas div had no size. #221's red had the same
signature. `frontend/index.html` now reloads once on a LINK or SCRIPT load error (sessionStorage
guard). **A Windows-only red that finds an element "hidden" is this; look for the console error first.**
Public job annotations answer without auth: `curl -s https://api.github.com/repos/millermuttu/Chemometrics-Workbench/check-runs/<job id>/annotations`.

**#260's macOS red** was the docs-examples `branch()` reading a port before the canvas's second fit
after a remount; it now waits for the port to stop moving.

**#239's protocol is written** (`docs/phase-4/exit-run.md`): observer's sheet, pass rule, a session
block to copy. The maintainer said on 2026-09-30 not to merge to main yet.

**The worked examples are tested three ways.** `tests/test_examples.py` recomputes every number
`docs/examples/*.md` quotes, over HTTP on `docs/examples/tecator.csv`, and asserts the page prints
it; `frontend/e2e/docs-examples.spec.ts` walks both pages through the screens on a fifth Playwright
server (8769, `examples`) and takes their screenshots; the CSV is asserted byte-equal to
`tecator_csv()` and ships with the Tecator permission note, which its terms require. A kernel change
that moves a quoted number fails the suite until the page is updated.

**#258, fixed in #260.** The canvas refits when the node count changes (not on refetch or drag,
so a user's pan survives), React Flow's `Controls` give a fit-view button, and the add-step menu
opens away from the nearer window edge. `docs-examples.spec.ts` no longer reloads or drops upward.

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

**`./run.sh` is the way in.** It syncs, installs with **pnpm** — this project has no
`package-lock.json` and `npm ci` refuses it — builds the bundle if there is not one, and launches,
opening the browser and printing `http://127.0.0.1:<port>/?token=<token>`. `--build` forces the
rebuild a changed `frontend/src` needs.

**Scripts worth knowing.** `uv run python -m tests.exit_run` and `tests.exit_run_phase3` each drive
the served application end to end and rewrite their phase's `docs/phase-N/exit-run.md`.
`uv run python -m tests.memory_probe 6000 1200` prints the peak resident memory of a ten-fold branch,
which is the number #176 is judged by.

## Next action

Pick up `stratified-splits` (#268, priority 1, no dependencies). `sample-exclusion` (#270),
`smoothing-filters`, `pcr`, `pls2-kernel`, `outlier-diagnostics`, `select-variables-step` and the
readers have no dependencies either; take them in priority order, one at a time.
