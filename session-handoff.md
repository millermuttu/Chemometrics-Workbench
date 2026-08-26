# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-26

---

## Where things stand

**Phase 0 is released and tagged `v0.1.0`. Phase 1.1 is released and tagged `v0.2.0`.** The walking skeleton walks: the React shell and its five core screens, over a token-authenticated stub FastAPI server on `127.0.0.1`, with a Playwright walkthrough green in CI. `dev` was merged to `main` and tagged on 2026-08-26, which closes the question the previous handoff left open.

**Phase 1.2 is open.** `feature_list.json` is now the 1.2 list — fifteen features against issues #76–#90 — and Phase 1.1's list is archived at `docs/phase-1-1/feature_list.json` next to Phase 0's. #76 opened the list and is `passing`.

**1.2's exit criterion:** #50's walkthrough passes against the real backend, on a file the user picks, with `stub/` deleted (#90).

| | Feature | Issue | Depends on |
| --- | --- | --- | --- |
| A | Archive the 1.1 list, open the 1.2 list | #76 | — |
| B | Project directory and array store | #77 | A |
| C | Reader interface and the CSV/TXT reader | #78 | B |
| D | XLSX reader | #79 | C |
| E | JCAMP-DX reader | #80 | C |
| F | Import endpoints | #81 | B, C |
| G | Drop `MSC(reference="supplied")` | #82 | A |
| H | Pipeline executor | #83 | B, G |
| I | Pipeline validator | #84 | H |
| J | Real jobs | #85 | H |
| K | Server-side decimation and the density band | #86 | H |
| L | Results endpoint | #87 | H |
| M | The metrics gap | #88 | — |
| N | HTTP surface complete, stub retired | #89 | F, J, K, L |
| O | 1.2's exit criterion as a test | #90 | I, N |

Chain: `A → B → C → {D, E, F} → H → {I, J, K, L} → N → O`. **M depends on nothing in 1.2**, so it is what to pick up if the chain is ever blocked.

## Current work

**#77 — project directory and the array store — is `in_progress` on `feature/77_project-directory`.** It is the feature everything except #88 waits on: a project is a directory on disk, the arrays are files in it, float32 on disk and float64 at the kernel boundary, and a path registry in the user's config directory stands in for the project list until SQLite arrives in 1.3.

## Next action

Finish #77, then **#78 — the reader interface and the CSV/TXT reader**. CSV/TXT is the format carrying every detection problem — decimal commas, orientation, wavelength headers, metadata columns — so the reader interface is designed against it and then proven by #79 and #80 rather than the other way round.

### Three decisions taken with the maintainer, so they are not re-argued

1. **The project directory and the array store land in 1.2; SQLite stays in 1.3.** Files are real from 1.2 — a restart loses the project *list*, not the data. This spreads the float32/float64 boundary and the §13 envelope across two sub-phases instead of landing them together in 1.3.
2. **All three readers land in 1.2** — CSV/TXT, XLSX and JCAMP-DX, as `PROPOSAL.md` §6 puts them in Phase 1. The reader interface is therefore designed against three formats rather than proven on one; CSV/TXT is still the one carrying the detection problems and should be written first.
3. **`MSC(reference="supplied")` is removed from the enum** (#82), rather than given a field or left to raise. The schema stops advertising something no executor can do, and re-adding it later is additive and needs no migration. This closes the question that has been open since pull request #22.

### What to be careful about in 1.2

- **The frontend should need no changes at all.** If a screen has to be edited to work against the real backend, the 1.1 contract was wrong, and *that is the finding* — record it rather than quietly adjusting the screen.
- **Four affordances die in #89**: `?empty`, `?oversize`, `?failrun` and `X-Stub-Fail`. Grep for them; each has a comment saying it is 1.1-only.
- **Seven handlers in `stub/server.py` carry `Phase 1.2:` markers** naming the issue that replaces each. Grep `Phase 1.2:`.
- **`preprocessing.from_spec` is the executor's seam.** It covers every step the schema can express and needs one thing from outside the schema: the axis for `RangeSelect`, which #77 provides off the `DatasetVersion`.
- **The validator has two warnings specified and nothing emitting them** (#84): a `MeanCentre` or `Autoscale` upstream of a split leaks validation samples into the training statistics and makes RMSECV optimistic; a PLS node with no centring upstream is legal and almost always wrong. `metrics-and-validation.md` §9 and `pls-regression.md` §3. **The 1.1 validate endpoint returns `valid: true` unconditionally** — a stub with a GUESS envelope, and this is what fills it.
- **The metrics gap (#88) is a Phase 0 inheritance**: SEC, SEP, Q² and `coefficients_original_units` are specified in `metrics-and-validation.md` §5–§6 and not implemented. Nothing verified them in #12.
- **Tecator at 100 variables never exercised x-axis decimation.** #86 needs a dataset big enough to actually drop points along the wavelength axis.

## Waiting on the user

- **#71 — what a non-positive `h0` should do** in the Jackson–Mudholkar SPE limit. Gasoline's `h0` is −0.0190; our kernel uses it as computed, `mdatools` clamps it to 0.001. The divergence is recorded and proven; what the kernel *should* do is a specification decision, not a coding one.
- **GitHub default branch is still `main`.** Any pull request opened without an explicit base targets the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

**Answered and recorded, so they are not re-opened:** whether Phase 1.1 warranted its own release (yes — `v0.2.0`, 2026-08-26), `MSC(reference="supplied")` (removed from the enum in #82), whether the import and empty-project screens needed artboards first (no), and where 1.2's persistence lands (project directory in 1.2, SQLite in 1.3).

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #24 (the R references), which changed what the parity report can claim:

- **Two of the three SPE limits are identical within float against R.** Corn and tecator agree with `mdatools` to the last bit — two languages, two authors, the same number. That is stronger than the `chemotools` agreement, which is another Python implementation on the same NumPy.
- **The T² limits differ from `mdatools` by exactly `(n+1)/n`, the same factor `chemotools` differs by.** Two unrelated implementations landing on the identical factor is what turns "our convention" into a demonstrated one. Both are recorded with `record_divergence()`, and both tests assert the factor before recording it.
- **NIPALS and SIMPLS coincide in coefficients to about 1e-12.** `mdatools` is SIMPLS, so this is the first real check of a claim scikit-learn could never test. Weights and loadings are deliberately *not* compared: the same document says they do not coincide.
- **Gasoline's SPE limit is a real divergence with a proven cause** — see #71 above. The test reconstructs their number from our formula plus their clamp *before* recording the divergence, so it stays a convention only for as long as that is really why the two differ.
- **R is not a dependency and CI never runs it.** The values live in `tests/fixtures/r_mdatools_values.json`, committed. Re-deriving them is a documented two-step pass in `CONTRIBUTING.md`.

From #42–#50 (the frontend), which 1.2 must not disturb:

- **The tokens are ported, not retyped, and a test holds them in step.** `frontend/src/styles/tokens.css` carries both palettes from `design/canvas/_base.css`, and `src/__tests__/tokens.test.ts` compares them value by value. Change `_base.css` first, then re-run. `shell.css` is the same idea for geometry.
- **IBM Plex is bundled, never fetched.** An end-to-end test fails on any request that does not go to `127.0.0.1`, which is the offline check.
- **No fixture file is imported by the frontend.** Everything arrives over HTTP. That is what lets 1.2 swap handlers behind unchanged URLs.
- **The tab model is a pure reducer** in `src/shell/tabs.ts`: one transient tab, replaced by the next preview, pinned by a double click. New screens open through it and do not need their own routing.
- **Plotly is driven from the tokens**, read off the DOM at draw time so a theme switch repaints. `src/plot/theme.ts` is the bridge; a plot that keeps Plotly's defaults is how this drifts from the artboards.
- **The canvas is React Flow with a custom node**, and its layout coordinates come from `pipeline_state.json`, outside `Pipeline.content_hash()`. Moving a node must not change the science — and must not invalidate an executor cache entry either.

From #41 and #53 (the fixtures and the stub server), which are the contract 1.2 has to honour:

- **The fixtures are the contract, and they are generated.** `stub/generate_fixtures.py` computes nothing of its own — it calls kernels and reshapes their output. Deterministic: UUID5 of a fixed namespace, every timestamp pinned to `2026-08-24T09:00Z`.
- **The envelope shapes are marked GUESS and the numbers are not.** Every array, metric, confidence limit, content hash and fold index is real and **1.2 must reproduce it**. Pagination, the error body, how a job reports progress and how run-state attaches to a pipeline are guesses 1.2 may change — each is marked at the point it is built.
- **The endpoint paths are the ones 1.2 implements**, so the frontend never changes when the handlers do: `projects`, `projects/{id}/datasets`, `import/preview`, `import`, `pipelines/{id}`, `pipelines/{id}/state`, `pipelines/{id}/validate`, `experiments/{id}`, `experiments/{id}/run`, `jobs/{id}`, `jobs/{id}/cancel`, `spectra/{node_id}`, `results/{node_id}`. All under `/api`, all requiring `Authorization: Bearer <token>`.
- **`spectra/{node_id}` is keyed on the pipeline's node ids**, not dataset ids. `results/{node_id}` likewise takes `pca_a`–`pca_d`. Anything else is a 404 with a body.

From #4 (PLS):

- **The Jackson–Mudholkar SPE limit does not transfer from PCA to PLS.** PLS components are not eigenvectors of the covariance of X, so there is no residual eigenvalue sequence to sum. PLS uses a χ² moment match on the calibration residuals instead. See `pls-regression.md` §9.
- **SNV, MSC and baseline correction cannot be folded into exported coefficients**, because all three depend on the sample being predicted. Savitzky–Golay *can*. This shapes both `coefficients_original_units` in #88 and the export format in #14. See `pls-regression.md` §7.

From #5 (metrics and validation):

- **Our shuffle is not scikit-learn's.** We fix `numpy.random.default_rng` (PCG64); `check_random_state` gives a legacy `RandomState`. The same seed produces different folds. #12's parity cases read the fold indices out of the fixture, and a separate case asserts our splitter would have produced those same indices — keep both halves if either is ever rewritten.
- **RMSEC divides by `n` here.** Degrees-of-freedom corrections live in SEC and SEP. Packages writing `n − A − 1` under the name RMSEC are reporting our SEC.
- **Fold aggregation is pooled residuals, not the mean of per-fold RMSEs.** Any reference value taken from a paper needs its aggregation rule established before it is comparable.

From #6 (reference datasets):

- **Only Tecator is committed.** Corn and gasoline are downloaded on first use into `~/.cache/chemometrics-workbench/datasets` and verified against a pinned SHA-256. The licence finding that decided each one is in `src/chemometrics_workbench/data/<name>/README.md`. Do not "helpfully" commit the raw files.
- **Publishing a Tecator result obliges you to name the instrument and company (Tecator).** A condition of use, not a courtesy.
- **Tecator's wavelength axis is reconstructed, not read** — `linspace(850, 1050, 100)`. If a reference value ever disagrees at the fourth digit on Tecator only, suspect the axis before suspecting the kernel. #80's JCAMP reader reads its axis and never reconstructs it.
- **Corn's targets are identical across `m5`, `mp5` and `mp6`** — the same 80 samples on three spectrometers. That is what makes it the calibration-transfer benchmark, and a test asserts it.
