# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-10-11 (after v1.2.0)

---

## Where things stand

**`v1.2.0` is released** (2026-10-11): dev merged into main through #368, the annotated tag pushed,
and the Release workflow published https://github.com/millermuttu/Chemometrics-Workbench/releases/tag/v1.2.0
with the Linux, macOS and Windows packages. Phase 6 is closed; its exit run is
`docs/phase-6/exit-run.md`, written by `tests/exit_run_phase6.py` (met; `--tighten 1e-9` is not met).
**No next phase is open.** `feature_list.json` is still Phase 6's list; archive it to
`docs/phase-6/feature_list.json` when Phase 7 opens, as Phase 5's was.

**The only open issue is #339, signed packages: `blocked`** on an Apple Developer ID with
notarisation and a Windows code-signing certificate, which only the maintainer can supply as
repository secrets. Commented on 2026-10-11. It was in the phase but not its exit criterion.

Phase 6, each merged through a green pull request:

| Feature | Issue | PR |
| --- | --- | --- |
| Grouped splits: `group_by`, leave-one-group-out (`metrics-and-validation.md` §8.8) | #329 | #344 |
| The model below a split refitted on every sample; artifact schema 2 | #330 | #347 |
| Nested validation of a variable selection (`variable-selection.md` §8) | #331 | #348 |
| Variable selection from a PLS-DA, on its dummy response (§9) | #332 | #349 |
| Seeded y-permutation test, run as a job (`metrics-and-validation.md` §14) | #333 | #350 |
| Bootstrap intervals for PLS/PCR coefficients and VIP (`pls-regression.md` §16) | #334 | #352 |
| Class-wise outlier diagnostics; flags sorted by rules broken (`outliers.md` §8) | #335 | #353 |
| The axis kind a correctable choice in the import preview; corrections in provenance | #336 | #356 |
| The spectra view draws a set of 60 or fewer (no band) instead of going blank | #355 | #357 |
| A classifier refuses a one-class training fold, naming fold, column and class | #343 | #358 |
| A GET that fails at the network is tried again (Windows `ERR_NO_BUFFER_SPACE`) | #351 | #359 |
| The Step list offers the port menu's splits and estimators | #337 | #360 |
| SVM: hand-written PCA-SVM (SMO), linear or RBF, one-vs-one | #338 | #361 |
| The Decisions section left the docs site (`exclude_docs`) | #346 | #363 |
| macOS e2e: the train/test note's expect given 30 s (payload 1.04 MB) | #354 | #364 |
| Python suite in parallel (xdist, loadfile), `@every_dataset`: 1,219 tests in 83 s to 983 in 22 s | #362 | #365 |
| Phase 6 docs: `examples/validation.md` on `meat-raw.csv`, how-to sections | #340 | #366 |
| Exit run, met; version 1.2.0 | #341 | #367 |
| dev into main; phase closed | #341 | #368, #369 |

`PROPOSAL.md` §16 has the Phase 6 row.

**GitHub attribution:** the maintainer said on 2026-10-05 not to put "Generated with Claude"
footers, session links or Claude `Co-Authored-By` trailers anywhere on GitHub - PRs, issues,
comments or commits. Two earlier commits (`28cd7ec`, `51b8b7e`) carry them and were left as they are.

**Handoffs go straight to `dev`**, no branch and no pull request (maintainer, 2026-10-05; also in
CLAUDE.md).

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

**Merging:** on 2026-10-10 the maintainer said to merge #360 and #361 once green; done. On 2026-10-05 the maintainer set a goal of finishing five issues by the process, which
covered merging #347-#353 once CI was green. No standing approval carries into a new session.

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

## Phase 5, for reference

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
| Phase 5 docs: classification example, outlier and selection how-tos | #288 | #324 |
| Phase 5 exit run, met; version 1.1.0 | #289 | #325 |
| dev into main, `v1.1.0` released | #289 | #326 |
| Exit run: Tecator PCR and selected PLS, lineage (asked by #289, missed at first) | #289 | #328 |

## What Phase 6 has left behind so far, worth knowing

- **The all-sample model (#330).** Every node below a split carries `_State.full`, stored under
  `<key>#all`; the estimator is fitted on it and `_with_held_out` merges fold zero's held-out view.
  Estimator keys include `RESULT_FORMAT`, so a result stored before #330 is a cache miss. Exports and
  folded coefficients refit the chain on `result.rows` (every row). Artifact schema is 2.
- **A dummy response (#332).** `regression.rmsecv_curve`, `selection.ipls/cars/nested` and the VIP
  selector take a 1-D response or a one-hot matrix (PLS2, pooled RMSE); `regression._response`
  decides which. `api._selection_inputs` builds a PLS-DA's dummy from `result.classes`.
- **Permutation jobs (#333)** use `JOBS.submit("permutation:<node>", ...)` so the canvas's running
  check ignores them; the result is `GET /api/permutations/{job_id}`, in memory only.
  `executor.permutation_test_for` reruns `_fitted` on stored fold matrices with a permuted version.
- **Bootstrap (#334)** is synchronous (`GET /results/{id}/bootstrap`); `api._fit_chain` is shared with
  `folded_coefficients`.
- **One-class folds (#343)**: `executor._refuse_one_class_folds` runs in `_estimator` for any spec
  with a `class_column`, so SVM (#338) inherits it.
- **Axis kind (#336)**: `Detection.axis_kind` is offered by the CSV and XLSX readers whenever the
  axis is not an index, and is correctable without being in `correctable`; `readers.read` reapplies
  it to the axis the reader rebuilds. `SourceFile.corrections` holds the fields changed from the sniff.
- **SVM (#338)**: `classification.SVM`, LIBSVM's WSS2 SMO without shrinking, kernel in float64,
  refuses after 1e6 iterations. `gamma=None` is sklearn's "scale". kNN and SVM share
  `executor._scores_classifier`; the result's `svm` dict holds per-pair support positions into
  `scores`, `dual` (a y) and `rho`. Parity adds the `float32_kernel` tolerance class (frozen in
  `test_parity_report.py`). Linear kernel with large C on unscaled scores is slow (Tecator C=100,
  12 s a fit).
- **Class-wise outliers (#335)**: `outliers.class_diagnostics`, `api.classwise_payload`; flags carry
  `n_rules` and are sorted by it for every model, so a test reading the first flag gets the worst.
- **The Python suite runs under pytest-xdist** (`-n auto --dist loadfile`, #362). `loadfile`
  because module-level RNGs make some files order-dependent; `-n 0` for a debugger. `conftest.py`
  pins BLAS to one thread and merges each worker's parity record. Per-dataset parity claims use
  `@every_dataset` in `test_parity.py`: never `pytest.skip` inside one (it skips the later datasets).
- **The validation example** (#340) runs on `docs/examples/meat-raw.csv` (both runs kept; `meat`
  is the first label column because a new classifier takes the first as its class) and its walk is
  `e2e/docs-validation.spec.ts` on a seventh Playwright server (8771, `validation`). A classifier
  tab has no "fitted · held out" note; that is the regression predicted-vs-measured panel only.
- **New parity-style claims** that are tests rather than fixture entries have a `Coverage` row with
  `not_compared` naming the test (nested selection, permutation, bootstrap, class-wise).

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
  sample's two runs: the workbench had no grouped CV until #329 (the exit run, #341, uses the raw set). Its walk is `e2e/docs-classification.spec.ts`
  on a sixth Playwright server (8770, `classification`); the docs specs share `e2e/docs-helpers.ts`.
  The outlier and selection how-tos (`docs/how-to/`) continue the Tecator PLS example and quote
  numbers `tests/test_examples.py` recomputes.
- **The Step list and the port menu are one list** since #337: both take `stepMenu(targets,
  classColumns)`, and a draft may be a split. Neither sets `stratify_by`; the inspector does.

## Next action

Nothing is queued. Ask the maintainer what Phase 7 is (PROPOSAL.md §16 lists genetic-algorithm
selection, the plugin API and self-hosted mode as what remains post-1.0), or for the signing
certificates that unblock #339. When a phase opens: archive `feature_list.json` to
`docs/phase-6/`, write the new list with its decisions, and open its issues.
