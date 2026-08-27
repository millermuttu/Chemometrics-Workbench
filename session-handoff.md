# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-27

---

## Where things stand

**Phase 0 is released and tagged `v0.1.0`. Phase 1.1 is released and tagged `v0.2.0`.** The walking skeleton walks: the React shell and its five core screens, over a token-authenticated stub FastAPI server on `127.0.0.1`, with a Playwright walkthrough green in CI.

**Phase 1.2 is thirteen features in, of nineteen.** The project directory and the array store are real, all three readers are written, the schema no longer advertises a step the executor could not run, the executor runs a pipeline from a real dataset and now fits its PCA nodes too, the import endpoints read a real file into a real `DatasetVersion`, the validator emits the two warnings nothing has ever emitted, the Phase 0 metrics gap is closed, a run is a real background job with counted progress, and the spectra payload is decimated by the server. **Every feature in 1.2 is now done except the two that close it: #89 and #90.** Four new issues came out of that work — #97, #99, #101 and #103, all listed below.

**Two features have landed with their HTTP half deferred.** #81's handlers and #87's results payload are written and tested; neither is served, because the swap from the stub is one cut rather than a handler at a time. That is #99, and it lands in #89.

**1.2's exit criterion:** #50's walkthrough passes against the real backend, on a file the user picks, with `stub/` deleted (#90).

| | Feature | Issue | Depends on | Status |
| --- | --- | --- | --- | --- |
| A | Archive the 1.1 list, open the 1.2 list | #76 | — | passing |
| B | Project directory and array store | #77 | A | passing |
| C | Reader interface and the CSV/TXT reader | #78 | B | passing |
| D | XLSX reader | #79 | C | passing |
| E | JCAMP-DX reader | #80 | C | passing |
| F | Import endpoints | #81 | B, C | passing |
| G | Drop `MSC(reference="supplied")` | #82 | A | passing |
| H | Pipeline executor | #83 | B, G | passing |
| H′ | The fixture's `centre_d` array | #97 | H | not started |
| I | Pipeline validator | #84 | H | passing |
| I′ | MSC above a split | #103 | I | not started |
| J | Real jobs | #85 | H | passing |
| K | Server-side decimation and the density band | #86 | H | passing |
| L | Results endpoint | #87 | H | passing |
| M | The metrics gap | #88 | — | passing |
| M′ | The import contract findings | #99 | F, L | not started |
| M″ | Rank through the store | #101 | L | not started |
| N | HTTP surface complete, stub retired | #89 | F, J, K, L | not started |
| O | 1.2's exit criterion as a test | #90 | I, N | not started |

Chain: `A → B → C → {D, E, F} → H → {I, J, K, L} → N → O`. **H′ (#97) blocks nothing**, but #86 and #87 both need its answer before they assert anything about `centre_d`. **M′ (#99) is folded into N and O** rather than done on its own, and now covers #87's endpoint as well as #81's. **M″ (#101) blocks nothing** and is a specification decision, like #71. **I′ (#103) blocks nothing** — it is #84's own convention working: a third warning becomes an issue rather than a widened branch.

## Current work

**Nothing is `in_progress`.** Every Phase 1.2 feature except #89 and #90 merged into `dev` on 2026-08-27 — #81, #82, #83, #84, #85, #86, #87 and #88, as pull requests #100, #96, #98, #104, #106, #107, #102 and #105. All branches are deleted locally and on origin, the tree is clean and `dev` is pushed.

## Next action

**#89 is what is left**, and then #90. Four findings are open — **#97**, **#99**, **#101** and **#103** — of which #97 and #101 are specification decisions waiting on the maintainer, and #103 is small.

**#89 is much bigger than its issue text suggests, and that is the single most important thing to know before starting it.** It now carries:

- the pipeline store, which nothing else has built and which four deferred endpoints all need;
- the import handlers (#81), the results endpoint (#87), the run endpoint (#85) and the spectra endpoint (#86), all written and tested but not served;
- the frontend's file-picker change — the 1.1 import screen discards the file the user picks (#99);
- the four 1.1-only affordances (`?empty`, `?oversize`, `?failrun`, `X-Stub-Fail`);
- a rewrite of the 17 end-to-end tests that assume the fixture project and dataset.

**Consider splitting it before starting.** The pipeline store is a feature on its own and everything else waits on it; doing it as one branch means one pull request that cannot be reviewed in pieces and cannot land half-done. **#97 should be answered before #86's and #87's numbers are asserted end to end in #90.**

### Decisions taken with the maintainer, so they are not re-argued

1. **The project directory and the array store land in 1.2; SQLite stays in 1.3.** Files are real from 1.2 — a restart loses the project *list*, not the data.
2. **All three readers land in 1.2** — done, and #78's interface survived being generalised by #79 and re-proven by #80.
3. **`MSC(reference="supplied")` is removed from the enum** (#82) — done. The kernel keeps the capability; the schema stops claiming a saved pipeline can express it. Re-adding it means adding the spectrum's field too, which is additive and needs no migration. This closes the question open since pull request #22.
4. **Estimator fitting is #87's, not the executor's** (taken in #83). `execute` walks estimator nodes and reports them in `Run.pending_estimators` rather than fitting them, because what a fitted estimator stores — with which diagnostics and which limits — is exactly what #87 specifies. Guessing at it in #83 would have been the invented contract Phase 1.1 existed to avoid.

### What to be careful about in 1.2

- **The frontend should need no changes at all.** If a screen has to be edited to work against the real backend, the 1.1 contract was wrong, and *that is the finding* — record it rather than quietly adjusting the screen. #82 is the shape to copy: the schema narrowed, the fixture regenerated, and the inspector's generated form followed with no screen touched.
- **Four affordances die in #89**: `?empty`, `?oversize`, `?failrun` and `X-Stub-Fail`. Grep for them; each has a comment saying it is 1.1-only.
- **Seven handlers in `stub/server.py` carry `Phase 1.2:` markers** naming the issue that replaces each. Grep `Phase 1.2:`.
- **The validator has two warnings specified and nothing emitting them** (#84): a `MeanCentre` or `Autoscale` upstream of a split leaks validation samples into the training statistics and makes RMSECV optimistic; a PLS node with no centring upstream is legal and almost always wrong. `metrics-and-validation.md` §9 and `pls-regression.md` §3. **The 1.1 validate endpoint returns `valid: true` unconditionally** — a stub with a GUESS envelope, and this is what fills it.
- **The metrics gap (#88) is a Phase 0 inheritance**: SEC, SEP, Q² and `coefficients_original_units` are specified in `metrics-and-validation.md` §5–§6 and not implemented. Nothing verified them in #12.
- **Tecator at 100 variables never exercised x-axis decimation.** #86 needs a dataset big enough to actually drop points along the wavelength axis.

## Waiting on the user

- **#71 — what a non-positive `h0` should do** in the Jackson–Mudholkar SPE limit. Gasoline's `h0` is −0.0190; our kernel uses it as computed, `mdatools` clamps it to 0.001. The divergence is recorded and proven; what the kernel *should* do is a specification decision, not a coding one.
- **GitHub default branch is still `main`.** Any pull request opened without an explicit base targets the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

**Answered and recorded, so they are not re-opened:** whether Phase 1.1 warranted its own release (yes — `v0.2.0`, 2026-08-26), `MSC(reference="supplied")` (removed from the enum in #82, 2026-08-27), whether the import and empty-project screens needed artboards first (no), and where 1.2's persistence lands (project directory in 1.2, SQLite in 1.3).

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #86 (decimation), which #89's spectra endpoint wires up:

- **Min and max per bucket, never a stride**, and the difference is measured rather than asserted: one channel raised between two strided samples reads 1.133 under min/max and **0.707 under a stride at the same budget** — the peak is simply absent from the strided payload. That flicker is what §13's "preserves the visible shape" is about.
- **The bucket extremes are taken over the mean spectrum**, because the axis is shared by every trace. A per-trace axis would mean one x array per spectrum.
- **§13's budget holds at the full envelope**: 20,000 × 4,000 builds its payload in 0.611 s. The committed test runs at 4,000 × 4,000 (0.112 s) to stay polite on a runner.
- **The data is synthetic and generated to a stated shape**, because no committed dataset is wide enough to drop a point — Tecator 240 × 100, corn 80 × 700, gasoline 60 × 401 are all inside the 1,000-point budget. Nothing numerical is claimed from it.
- **`highlighted` is an additive key** carrying its own full-resolution axis. Consuming it needs a frontend change: selection today is a client-side concept in `spectraTraces` and never reaches the server.

From #85 (real jobs), which #89's run endpoint wires up:

- **Cancellation is cooperative and bounded by one node.** The executor is asked between nodes and stops there; a node already running finishes. A user cancelling a ten-fold cross-validation waits for one fold's fit, not for ten. Pre-emptive cancellation would mean a process pool and a copy of every matrix across a pipe.
- **A cancelled run keeps what finished.** Those arrays are complete — the store renames through a temporary file — and a node's key is its recipe and its data, never what ran after it, so resuming does not repeat the work. Asserted equal to a cache-off run.
- **Progress is counted, never interpolated.** Nodes completed over nodes total, reported as each finishes. The 1.1 stub moved a number against the wall clock, which looks identical until a run is slower than the clock expected and the bar sits at 100% while the work continues.
- **The job envelope kept all five 1.1 fields and gained `node_id`**, so the frontend is unchanged after all — the GUESS did not have to change, only grow.
- **A thread, not a process.** Every kernel operation is NumPy and releases the GIL.
- **The job tests are driven by `threading.Event`, never by sleeps**, and were run three times over to confirm it. A concurrency test that sleeps is a flaky test waiting to happen.

From #88 (the metrics gap), which #14's export and #85's metrics both need:

- **A preprocessing chain is folded into coefficients by measuring it, not by restating §7's table.** Every foldable step is affine, so passing the identity through the fitted chain recovers `f(e_j) - f(0)`, and `f(0)` is the offset. Savitzky-Golay's `interp` edge handling comes out exactly right without anyone re-deriving it, and a kernel change cannot leave a stale copy of the rule behind.
- **SNV, MSC and the baselines are refused by type, never by probing.** SNV will not even accept a row of zeros, and treating its identity image as a linear map gives predictions more than 1.0 away from the model's own — a plausible, wrong number, which is the failure the guard exists to prevent.
- **`scikit-learn`'s `PLSRegression.intercept_` is `ȳ`** and does *not* reproduce its own `predict()` — out by `x̄·b`, which is 4.79 on tecator against a response range of 0.9 to 58.5. A parity claim built on that attribute would have failed against a correct kernel. `pls-regression.md` §14 records it; the generator recovers the intercept from `predict()` and asserts it.
- **SEC and SEP are recorded as `unsourced`.** No installed package computes them, and computing them from another package's predictions would test their model rather than our metric. They are checked against §5's identity and hand arithmetic instead — the standing SNV and MSC had before #13.
- **The parity report is 105 comparisons, up from 96**, and `Metrics.extra` carries SEC and SEP, so the schema needed no change.

From #84 (the validator), which #89 and #103 build on:

- **The validation envelope keeps `valid` and `problems` and adds `warnings` beside them.** The GUESS shape is untouched, so the canvas renders unchanged; the structured list carries `code`, `node_id`, `related` and `severity` for a screen that wants to point at the node instead of parsing a sentence.
- **`valid` means "there is nothing to tell you", not "this will run".** Everything runs — `checks.py` is advisory and a test executes a leaky pipeline to completion. Reporting `valid: true` while holding a warning would mean the 1.1 screen said "valid" and dropped the sentence.
- **`Autoscale` counts as centring for both rules**, because it subtracts the column means before it divides.
- **A PCA with no centring is deliberately not warned about.** Only `pls-regression.md` §3 says "almost always wrong", and extending that to PCA would be a preference with no document behind it.
- **`MSC` above a split leaks by the same rule and is #103.** `SNV`, `Normalise`, Savitzky-Golay, the baselines and `RangeSelect` were all checked and are legitimate above a split — none estimates anything across samples — so MSC is the only step in the schema that leaks and is not warned about.

From #87 (the estimators), which #86 and #90 draw on:

- **An estimator result is JSON at `results/<key>.json`, keyed the way arrays are.** Not content-addressed: a key names exactly one result, so the path is derived rather than looked up. Editing a node gives its estimator a new key and a new result while untouched branches keep both.
- **A PCA below a split is fitted on fold zero's training rows.** The fixture's choice, and deliberately not an aggregation — there is no single model over ten folds, and averaging loadings across them is arithmetic no document specifies. The 24 held-out rows are projected through that model and stored beside the calibration ones.
- **`validation` is an additive payload key**, so every array the 1.1 screen reads keeps the length the fixture has. A screen that ignores it renders exactly what it rendered before.
- **`pca_d` is the second casualty of #97** — 3.8e-05 on explained variance, 1.8e-01 on T². Regenerating the fixtures means regenerating `spectra.json` **and** `pca.json`.
- **The reported rank is one too high for any centred matrix** — #101. A centred array read back as float32 has columns that no longer sum to zero, and the SVD finds a hundredth singular value. Only the displayed integer moves; the limits move in the ninth decimal.
- **PLS and PLS-DA have no kernel in the executor** and are reported in `Run.pending_estimators`. What a PLS result carries is #88's subject.

From #81 (the import endpoints), which #89 has to finish:

- **The router lives in `chemometrics_workbench.api` and the stub does not include it.** Wiring it in fails 17 of the 40 end-to-end tests, because the fixture pipeline is built on the fixture dataset version and a real import produces a dataset it knows nothing about. **The swap is one cut in #89, not a handler at a time** — #99.
- **The 1.1 import screen discards the file the user picks.** `onChange={() => preview.mutate({})}`, and the request body is empty. The published contract has no way to say which file to import, so the real endpoints take a multipart upload — `file`, plus `corrections` and an optional `name`. The URLs did not change; the bodies 1.1 left empty did. Also #99.
- **`datasets.json` in the project directory is the dataset index** until SQLite arrives in 1.3, holding `DatasetEntry` — one `Dataset` and its `DatasetVersion`s. It records paths, never values, and a restart reads the dataset list back from it.
- **§13's envelope is reported, not enforced.** `frontend/src/states/envelope.ts` computes it from `n_samples` and `n_variables`, so the server's honest behaviour is to report the shape as read. What `?oversize` did — fabricate a shape — is what breaks that, and nothing replaces it.
- **An upload keeps the user's own filename inside a temporary directory**, because that name is what `SourceFile` records and what the import screen shows. Only the final path component is used, so an upload calling itself `../../project.json` writes into the temporary directory and nothing else.
- **`python-multipart` is a new runtime dependency**, taken because a commit carries a file and its corrections together and that is what a browser sends.

From #83 (the executor), which #84 to #87 all build on:

- **A node below a split has one array per fold, and displays them assembled out of fold.** `metrics-and-validation.md` §9 refits every node downstream of a split on the training fold alone; the array such a node shows takes each sample's row from the fold that held it out. This is why the fixture's `centre_d` is not reproduced — **#97**, measured there rather than argued.
- **Cache keys are a merkle chain, not a flag.** A node's key is its own JSON plus its inputs' keys, with the source keyed on the dataset version. Editing a node changes its key and its descendants' and nothing else's, so staleness is arithmetic. Layout coordinates are not in the model at all, so a node cannot be moved into a cache miss.
- **Every node's output is read back out of the store before its successors see it.** Otherwise a run that hit the cache and a run that recomputed would disagree in the last digits — a cache that changes an answer. The cost is that all numbers carry float32 truncation from the first node on; the fixture comparison measures it at 1.3e-06 worst case, and at 2.8e-05 for `autoscale_c`, where a derivative cancels the signal and autoscaling divides by what is left.
- **`Run.pending_estimators` is the seam #87 picks up.** The executor names the estimator nodes it did not fit rather than skipping them silently.
- **A second `RangeSelect` below a first one fails, by design.** The axis comes from the `DatasetVersion`, and the first selection has already dropped variables, so the second is refused on the shape with its node named. Threading a per-node axis through the walk would make a recipe's meaning depend on where its node sits — a schema question, not an executor one.

From #78–#80 (the readers), which #81 builds on:

- **The reader interface is `sniff`, `read` and `head`.** `sniff` returns a `Detection` with a `Choice` for every guess; `read` takes a `Detection` — the one `sniff` returned, or that one with the user's corrections folded in. That is what stops a correction being displayed and then ignored, and it is the shape #81's preview and commit endpoints have to expose.
- **Everything three formats share lives in `readers/grid.py`.** #79 moved header detection, orientation, column classification, the axis rules and the preview head out of `delimited.py`; `delimited.py` keeps only what text has. #78's 34 tests passing unchanged against that move is the evidence it preserved behaviour.
- **A correction that cannot be applied is refused, not dropped.** A delimiter correction on a workbook fails; JCAMP's `correctable` is empty because nothing in the file is a guess. Silently ignoring one is the same lie as never offering it.
- **Three formats, one dataset, agreeing within 1e-4.** `tecator_subset.csv`, `.xlsx` and `.jdx` carry the same eight samples and twelve channels, and a test asserts the three readers agree. Any new reader should join that comparison rather than get its own private fixture.
- **An axis is read or numbered, never invented.** A headerless text file gets an index axis with a note; JCAMP reads `##FIRSTX`/`##LASTX`/`##NPOINTS` and refuses a file lacking them; `MICROMETERS` is left numbered rather than converted.

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
