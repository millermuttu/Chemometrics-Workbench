# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-10-05

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

**#255, explained and fixed in #262.** The Windows trace showed `net::ERR_NO_BUFFER_SPACE` on the
bundle's CSS: the page rendered unstyled and the canvas div had no size. #221's red had the same
signature. `frontend/index.html` now reloads once on a LINK or SCRIPT load error (sessionStorage
guard). **A Windows-only red that finds an element "hidden" is this; look for the console error first.**
Public job annotations answer without auth: `curl -s https://api.github.com/repos/millermuttu/Chemometrics-Workbench/check-runs/<job id>/annotations`.

**#260's macOS red** was the docs-examples `branch()` reading a port before the canvas's second fit
after a remount; it now waits for the port to stop moving.

**#239's protocol is written** (`docs/phase-4/exit-run.md`): observer's sheet, pass rule, a session
block to copy. The maintainer said on 2026-09-30 not to merge to main yet.

**The worked examples are tested three ways** (and since #288 there are three of them, plus two how-tos). `tests/test_examples.py` recomputes every number
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

## Phase 5 so far

Merged through green pull requests on `dev`:

| Feature | Issue | PR |
| --- | --- | --- |
| Stratified K-fold and train/test | #268 | #292 |
| Multi-class groundwork: N-by-N tally, per-class table, `classification.md` | #269 | #294 |
| Sample exclusion: a derived version the pipeline moves onto | #270 | #295 |
| Moving average, median, Gaussian, Whittaker | #271 | #297 |
| Nothing follows an estimator (found during #271) | #296 | #298 |
| A run started during a save runs what was saved (#291's cause) | #291 | #299 |
| PCR | #272 | #300 |
| PLS2 kernel | #273 | #302 |
| Multi-class PLS-DA | #274 | #303 |
| SIMCA | #275 | #304 |
| LDA | #276 | #305 |
| kNN | #277 | #307 |
| SIMCA and kNN export, the chain as an affine map | #306 | #308 |
| Outlier diagnostics: leverage, studentised residuals, FastMCD, flags table | #278 | #310 |
| Exclude flagged samples | #279 | #311 |
| FastMCD speed, outliers off the payload | #314 | #315 |
| `select_variables` step | #280 | #313 |
| VIP selection | #281 | #316 |
| iPLS | #282 | #317 |
| CARS | #283 | #318 |
| Each step gets its input's axis (nested selections) | #312 | #319 |
| MATLAB `.mat` reader | #284 | #320 |
| Galactic SPC reader | #285 | #321 |
| Thermo OMNIC SPA reader (single file, or a zip of them) | #286 | #322 |
| ASD FieldSpec reader (reflectance against the stored white reference) | #287 | #323 |
| Phase 5 docs: classification example, outlier and selection how-tos | #288 | #324 (open) |

## What these left behind, worth knowing

- **Exclusion materialises.** `derive_version` writes the kept rows as a new array; `excluded_samples`
  are the *parent's* row indices. Nothing masks rows in place, so the executor needed no change.
- **Optional fields on an existing spec** use pydantic `exclude_if` (see `KFoldSplit.stratify_by`) so an
  unset field leaves the JSON, and the cache key, as it was.
- **Regression estimators share `_fit_regression`** in `executor.py` with an `estimator=` class
  (`PLS` or `PCR`); `rmsecv_curve` and `cross_validated_predictions` take the same parameter. A result
  carries `method` (`pls`/`pcr`/`plsda`) so a screen can name it. PLS2 and LDA should reuse this.
- **Linear smoothers fold** (`FOLDABLE` in `regression.py`); the median is an exportable residual step
  with its own NumPy body in `export.py`.
- **e2e tests that change a project** go in `walkthrough.spec.ts`, which runs in order on the one server
  whose project a test may change. Wait for the drafts to clear ("No steps yet") or rely on #299 before
  asserting on a run after Save.
- **SIMCA and kNN export** (#308, `model-export.md` §6): the foldable tail is carried as `x @ M + c`,
  measured by pushing zeros and unit vectors through the fitted steps; `null` when empty. `M` is p x p'
  floats, about 20 MB at 1,000 variables. #308 also fixed the fold being told the post-range-selection
  width when a carried step preceded a range selection.
- **SIMCA has no external parity** (R not installed); its claim rests on per-class `decomposition.PCA`.
- **Parity fixture.** `uv run python tests/fixtures/generate_reference_values.py` regenerates
  deterministically; check its diff is only the new entries and the date. The parity report needs the
  whole suite run first (`uv run pytest`, then `uv run python -m tests.parity_report`).

- **Reader fixtures** live in `tests/fixtures/readers/<format>/` with a `LICENSE.md`; the source DOIs
  are in decision 0006. **A `.zip` is OPUS's unless every member is `.spa` or every member is `.asd`**
  (`_archive_of` in `reader_for`); both read members through `readers.spectrum_files`.
- **ASD reflectance is the plain ratio** target / stored white reference, as ViewSpec shows it. The
  Eaton Fire record's published values also carry a panel calibration curve and a VNIR splice; the
  test factors both out. A splice-correction step would be a preprocessing feature, not a reader one.

- **The classification example** runs on `docs/examples/meat.csv` (Quadram, CC0), which averages each
  sample's two runs: the workbench has no grouped CV. Its walk is `e2e/docs-classification.spec.ts`
  on a sixth Playwright server (8770, `classification`); the docs specs share `e2e/docs-helpers.ts`.
  The outlier and selection how-tos (`docs/how-to/`) continue the Tecator PLS example and quote
  numbers `tests/test_examples.py` recomputes.
- **The Step list offers only preprocessing and PCA.** Splits and estimators come from dragging off
  a node's port. A page that tells a reader otherwise is wrong.

## Next action

#324 (docs) merges when CI is green. Then `phase-5-exit-run` (#289), the last feature: its
verification is in `feature_list.json`, and decision 0006 says it runs on the Quadram meat set.
After it, `dev` into `main` and the `v1.1.0` tag.
