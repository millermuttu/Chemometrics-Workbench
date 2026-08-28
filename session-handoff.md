# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-28

---

## Where things stand

**Phase 0 is released and tagged `v0.1.0`. Phase 1.1 is released and tagged `v0.2.0`.**

**Phase 1.2 is sixteen features in, of twenty-one.** #89 is done: there is one server, `stub/` is
deleted, and the whole end-to-end suite runs against the real backend. What is left in the sub-phase
is **#90**, and #90 is **blocked behind #108**.

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
| M′ | The import contract findings | #99 | F, L | **passing** |
| M″ | Rank through the store | #101 | L | not started |
| N | HTTP surface complete, stub retired | #89 | F, J, K, L | **passing** |
| N′ | No endpoint writes a pipeline | **#108** | N | **not started — blocks #90** |
| N″ | The overloaded state is unreachable | **#109** | N | not started |
| O | 1.2's exit criterion as a test | #90 | I, N, **#108** | not started |

## Current work

**Nothing is `in_progress`.** #89 and #99 are done on `feature/89_http-surface-complete`, committed
as `b9dc2c4` and **pushed**. **No pull request is open**: `git push` works over HTTPS, but
`api.github.com` does not resolve from this machine and the GitHub MCP server fails with `ETIMEOUT`,
so nothing that needs the API could be done — see "Waiting on the user".

## What #89 turned out to be

It was much bigger than its issue text, as the last handoff warned, and the reason is worth keeping:
**pointing the end-to-end suite at the real backend found bugs the stub had been hiding.** Four in the
server, one in the frontend contract, all fixed here with regression tests.

1. **Nothing ever wrote `experiment.json`.** `GET /experiments/current` — a published endpoint — answered
   404 for the life of a project however many runs had happened. `executor.experiment_for` and
   `jobs.submit_run` now write the record on every ending: succeeded, failed and cancelled alike,
   because a failed experiment is a result and `models.py` says so on the field.
2. **`pipelines/{id}/state` could not report a failed node.** It knew `complete`, `queued`, `running`
   and `not_run`, so a failed run rendered as a graph of nodes that merely never ran, and the
   artboard's `failed` encoding was a state only the fixture could produce.
3. **The job blamed the wrong node** — the last one to *report progress*, which is precisely the last
   one that finished rather than the one that raised. `ExecutorError` has carried `node_id` as a field
   since #83 for exactly this ("a canvas that wants to mark the node red cannot parse it back out of
   English"); the job now reads it.
4. **A plain `GET` could 500, intermittently.** `open_project` called `_remember` on *every* call,
   rewriting a registry file shared by every project on the machine; a page load asks six questions at
   once and two of them racing on that write failed the open. Worse, the check-then-create in
   `open_project_directory` was unsynchronised, and `create_project` makes `arrays/` before it writes
   `project.json` — so the losers of that race found a directory that was neither empty nor yet a
   project, which made **the first load of a brand-new project a 500**. Both are fixed, and
   `test_concurrent_reads_on_one_project_never_fail` fires thirty concurrent reads at one project.
5. **One frontend change, and it is the finding.** Nothing refetched `pipelines/{id}/state` while a job
   advanced: in 1.1 the fixture had a node permanently `running` and another permanently `failed`, so
   those states were on screen without anything asking for them. Against a real backend they are
   *events*. `Shell.tsx` now invalidates `pipeline-state` when the job's status or `node_id` changes,
   following the poll the job query already runs. **No URL and no payload changed.**

## How the end-to-end suite works now

**A state is a project, so a starting state is a seeded directory.** The stub reached its states
through query parameters, which meant one server could be anything on request. `playwright.config.ts`
now runs **three real servers over three real project directories**, seeded by `tests/seed_e2e.py`:

- **8765 `seeded`** — Tecator imported through the real import handler, the artboard's four-branch
  fourteen-node pipeline with every node executed. Everything that reads.
- **8766 `empty`** — a project with nothing in it. The empty state, and the imports, which are the
  tests that *change* the project they run in.
- **8767 `runs`** — the same pipeline plus a branch that cannot be fitted, with **nothing executed**.
  Runs really run here, so they can be watched, cancelled and failed.

Things that were learned the hard way and should not be rediscovered:

- **Tecator is too fast to watch.** The whole fourteen-node run finishes in about 0.2 s — real work,
  and far quicker than a browser polling four times a second can catch a node `running`. The `runs`
  project therefore carries a **synthetic 2,000 × 800 matrix** (`seed_e2e.synthetic_dataset`),
  generated and stated as such; **no number is claimed from it**, the same footing as #86's fixture.
  4,000 × 1,500 was measured at ~3 s per branch and rejected as impolite on a runner.
- **A cached pipeline has no work to cancel.** This is why the run tests cannot share the seeded
  project, and it is correct behaviour rather than a problem to work around.
- **A failed run is remembered by the job table**, so it would follow every later test on the same
  server. That is the other reason `runs` is its own project.
- **`data/tecator/tecator.txt` is not reader-readable** — a prose header and the 22 principal
  components the file also supplies, which is why `load_tecator` parses it specially. The seed writes
  the data out as a CSV and posts it to the real `/api/import`, so the seeded dataset comes through
  the reader a user's file goes through.
- **The real backend reproduces the fixture's numbers**: 240 × 100, explained variance 68.9 / 28.4 /
  1.6 per cent and 99.9 cumulative, 240 spectra decimated to 60 traces. `spectra.spec.ts` and
  `analysis.spec.ts` pass **unchanged**, which is the strongest evidence the swap kept the contract.

## Next action

**#108, then #90.** #108 is the blocker and it is a real design question, not a detail.

**#108 — no endpoint writes a pipeline.** The step list builds a **client-side draft**; the frontend
has read `pipelines/current` since its first commit and has never posted one, so 1.1 published no way
to save a pipeline and #89 served every URL it did publish without inventing one it did not. The
consequence is that #50's three walkthrough tests are `test.fixme` with the reason in the file: the
SNV, Savitzky-Golay and PCA nodes the walkthrough assembles never reach the server, and the run that
follows has only the source node to execute. **Everything else the walkthrough asserts already passes
against the real backend** — the empty project and the import in `empty.spec.ts`, the run's lifecycle,
cancellation and failure in `runs.spec.ts`, the scores against the kernel's own numbers in
`analysis.spec.ts`. What is missing is the single path through them.

The shape of #108 is a decision to take deliberately: whether the canvas `PUT`s a whole pipeline or
`POST`s node operations, and what either means for #83's merkle cache keys. Layout coordinates must
stay outside `Pipeline.content_hash()` — moving a node must not invalidate a cache entry.

**Then #90**, which is the exit criterion and is mostly un-`fixme`-ing the walkthrough once #108 lands.

Loose findings, none blocking: **#97** and **#101** are specification decisions waiting on the
maintainer, **#103** is small, **#109** is probably a decision to record rather than a test to write.

## Waiting on the user

- **#71 — what a non-positive `h0` should do** in the Jackson–Mudholkar SPE limit. Gasoline's `h0` is −0.0190; our kernel uses it as computed, `mdatools` clamps it to 0.001. The divergence is recorded and proven; what the kernel *should* do is a specification decision, not a coding one.
- **GitHub was unreachable for the whole of the 2026-08-28 session** — `api.github.com` did not resolve and the GitHub MCP server failed with `ETIMEOUT`. Three things follow, and all three are outstanding:
  - **`feature/89_http-surface-complete` is pushed (`b9dc2c4`) but has no pull request.** Open one **with `dev` as the base** — GitHub's default is `main`, which is the release line. `git push` itself works; only the API is unreachable.
  - **Issues #108 and #109 do not exist on GitHub yet.** They are in `feature_list.json` with full notes, and the numbers were reserved by assumption — check what the next free number actually is when GitHub is reachable, and correct both the issues and the references to them in `feature_list.json`, `session-handoff.md` and `frontend/e2e/walkthrough.spec.ts` if they differ.
  - #89 and #99 could not be closed from here.
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

- **The fixtures are the contract, and they were generated.** They computed nothing of their own — the generator called kernels and reshaped their output. They now sit in `tests/fixtures/contract/`, frozen: a payload that has to change shape is a decision recorded in the issue, never a quiet edit to a file there.
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
