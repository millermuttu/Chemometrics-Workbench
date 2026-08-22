# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

**Both estimator kernels are done.** PLS landed with VIP, cross-validation and the diagnostics, and it is the last kernel Phase 0 needs.

Issues #1–#10 and #12 are merged and closed. **#11 is merged but deliberately still open**, blocked on the one verification step that has nothing to verify against. Pull request #26 merged on 2026-08-22 and its branch is deleted locally and on origin. Nothing is in flight.

**The parity fixture is now fully covered.** `parity-results.json` records 66 comparisons, 66 passed, and **`not_compared` is empty** — every comparable entry in the fixture has been checked against a kernel. That was fourteen untested before #11 and eight before #12, and it is the number the report in #14 is allowed to stand on.

**What is left in Phase 0**: `chemotools-eval` (#13), `parity-report` (#14), and `reference-values-r-mdatools` (#24, raised out of #11).

### Why #11 is blocked, and what unblocks it

Unchanged by this session. Its fourth verification step asks that the T² and SPE limits **match published values**, and there are none to match: scikit-learn reports neither limit, and the only external reference is R `mdatools`, whose entries have been `unsourced` since #7 because R is not installed here. **#24 is the only thing that unblocks it** — install R (`conda install -c conda-forge r-base r-mdatools`, no sudo needed) and source those six entries plus three new `hotelling_t2_limit` ones. You were asked during #11 and chose to land it blocked rather than install R mid-feature.

**Note that #12 did not inherit this problem**, and the difference is worth understanding before #14 writes either up. PLS's SPE limit is a χ² moment match on the observed calibration residuals, so it is checked against its own coverage; its T² limit is `pca.md` §7's, shared with PCA, and carries exactly the same gap. What made #12 verifiable where #11 was not is that every quantity in its issue's "done when" had a reference — VIP included, once three entries were added for it.

**The offline question was raised and settled** in an earlier session. Corn and gasoline are downloaded rather than committed. The option of writing to Eigenvector and Prof. Kalivas to ask for redistribution permission was offered and **declined — do not open that issue and do not chase it.**

## Repository

| | |
| --- | --- |
| Current branch | `dev`, clean, in sync with origin |
| Open pull requests | None |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 4 of 15 — #11 (blocked, merged), #13, #14, #24 |
| Feature statuses | 11 × `passing`, 2 × `blocked`, 2 × `not_started` |
| Parity claims | 66 compared, 66 passed, 60 identical-within-float, 5 within tolerance, 1 documented divergence, **0 not compared** |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green (408 tests), enforced by CI on 3.12 and 3.13 |

## Active feature

None is `in_progress`. Two are `blocked`, both on the same thing:

- **`kernel-pca` (#11)** — code merged and green; blocked on its published-limit verification step.
- **`reference-values-r-mdatools` (#24)** — blocked on R not being installed, which is an environment decision rather than a coding task.

## Next action

**`chemotools-eval` (#13) next.** It is the only eligible feature: `parity-report` (#14) depends on it, and #24 becomes eligible the moment you decide to install R.

```bash
git fetch origin
git checkout -b feature/13_chemotools-eval origin/dev
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest   # once, to populate the dataset cache
```

Read the issue before scoping it: it is an *evaluation*, and its output may legitimately be "no, and here is why". Two findings already point at it — **SNV, MSC and the three baseline methods have no external reference at all**, and their fixture entries are `unsourced` for that reason. `chemotools` is the obvious candidate for the first two and `pybaselines` for the third, and adding either is a dependency decision rather than a kernel one.

**#14 is the phase's exit criterion** and it now has everything it needs from the kernels. What it must not do is overstate what the fixture says; the list under "carried forward" below is where the caveats are written down, and three of them are specifically about claims that look stronger than they are.

**One feature at a time.** The protocol allows exactly one `in_progress`, and a `blocked` feature does not count against that.

## Waiting on the user

- **Two questions raised on pull request #22 and still unanswered.** (a) `MSC(reference="supplied")` has no schema field for the reference spectrum — a schema change and a separate issue. (b) `PROPOSAL.md` §7 says PCA and PLS come from scikit-learn, while `pca.md` §3 and `pls-regression.md` §4 are marked normative and specify our own implementations. **Both kernels have now been written to the specifications** and are green against scikit-learn as a reference rather than built on it. The contradiction is now historical rather than blocking, but `PROPOSAL.md` §7 still says something that is not true of the code and should be corrected.
- **GitHub default branch is still `main`.** Any pull request opened without an explicit base targets the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #4 (PLS):

- **The Jackson–Mudholkar SPE limit does not transfer from PCA to PLS.** PLS components are not eigenvectors of the covariance of X, so there is no residual eigenvalue sequence to sum. PLS uses a χ² moment match on the calibration residuals instead. Implemented in #12; see `pls-regression.md` §9.
- **SNV, MSC and baseline correction cannot be folded into exported coefficients**, because all three depend on the sample being predicted. Savitzky–Golay *can*. An exported model carries a residual preprocessing chain plus coefficients, not always a bare coefficient vector. This shapes the export format in #14 and `PROPOSAL.md` §9. See `pls-regression.md` §7.

From #5 (metrics and validation):

- **Our shuffle is not scikit-learn's.** We fix `numpy.random.default_rng` (PCG64); `check_random_state` gives a legacy `RandomState`. The same seed produces different folds. #12's parity cases read the fold indices out of the fixture and hand them to nothing else, and a separate case asserts our splitter would have produced those same indices — keep both halves if either is ever rewritten.
- **RMSEC divides by `n` here.** Degrees-of-freedom corrections live in SEC and SEP. Packages writing `n − A − 1` under the name RMSEC are reporting our SEC.
- **Fold aggregation is pooled residuals, not the mean of per-fold RMSEs.** Any reference value taken from a paper needs its aggregation rule established before it is comparable.

From #6 (reference datasets):

- **Only Tecator is committed.** Corn and gasoline are downloaded on first use into `~/.cache/chemometrics-workbench/datasets` and verified against a pinned SHA-256. The licence finding that decided each one is in `src/chemometrics_workbench/data/<name>/README.md`. Do not "helpfully" commit the raw files.
- **Publishing a Tecator result obliges you to name the instrument and company (Tecator).** A condition of use, not a courtesy. It applies to the parity report in #14.
- **Tecator's wavelength axis is reconstructed, not read.** The loader uses `linspace(850, 1050, 100)`. If a reference value ever disagrees at the fourth digit on Tecator only, suspect the axis before suspecting the kernel.
- **Corn's targets are identical across `m5`, `mp5` and `mp6`** — the same 80 samples on three spectrometers. That is what makes it the calibration-transfer benchmark, and a test asserts it.

From #7 (reference values):

- **`comparable: false` is not decoration.** `tecator.pls.sep.thodberg` is a real published number that is not a parity target — its inputs are principal components the loader discards. The harness skips entries where `comparable` is false and every `status: "unsourced"` entry, which has a `null` value by construction.
- **The R `pls` vignette entry is the most valuable one in the file**, and #12 turned it green: LOO over gasoline rows 0–49, deterministic, and R `pls` computes MSEP as `SSE/nobj` — divisor `n`, the same as ours — so it compares with no correction. All eleven printed values reproduce to their four significant figures. If only one parity claim survives into the report, make it that one.
- **Six R `mdatools` entries are unsourced because R is not installed here.** They are the T² and SPE limits scikit-learn does not provide. Installing R and rerunning the recorded configurations is the single highest-value way to strengthen the fixture, and #11 is where their absence is felt.

From #8 (parity harness):

- **Never write a bare `assert_allclose` in a kernel test.** Call `parity.check(entry_id, ours)`. A comparison made outside the harness is invisible to #14. (#10's edge-only case is the one deliberate exception.)
- **Tolerances are not knobs.** A failing parity test is a finding. If the difference is a convention, use `parity.record_divergence()` and write the reason into the specification's divergence table.
- **A quantity with no entry in `QUANTITY_CLASS` is refused, not guessed at.** #12 added `vip` there, mapped to the coefficient class because VIP accumulates through the same per-component deflation.
- **`parity-results.json` lists what was never compared, and that list is now empty.** Keep it that way: a new fixture entry with no test is a coverage regression the file will report, and a report that only lists what was tested overstates coverage.
- **The identical-within-float threshold is scale-relative**, 32 ulp of the largest reference value, not a fixed `rtol` with `atol=0`.

From #9 (scaling kernels):

- **Kernels import nothing from scikit-learn.** "scikit-learn-compatible" is duck compatibility — `fit`, `transform`, `fit_transform` — and nothing more. Both estimator kernels follow it.
- **Transformers are stateful, and even the stateless ones have a `fit`.** Fitted parameters are the fit set's, always, because held-out samples are pushed through with the *training fold's* parameters.
- **Zero is judged relative to the magnitude of the data, never `== 0.0`.** `_dead_threshold` in `preprocessing.py` is the shared rule, and #12's SPE limit follows it: a full-rank model's residual is 1e-32 of the matrix rather than 0.0, and a quantile of that is a quantile of noise.
- **Two divergences from scikit-learn are deliberate**: autoscale defaults to `ddof=1`, and a zero-variance row or column raises rather than getting a substituted scale of 1.
- **`from_spec` is the executor's seam.** The two things it still needs from outside the schema are the axis for `RangeSelect` and the spectrum for `MSC(reference="supplied")`.

From #10 (smoothing, derivatives and baselines):

- **Savitzky–Golay's edge mode is `interp` and is not configurable**, and this is the finding most likely to come back as a bug report. A recipe recording "Savitzky–Golay, 11, 2, 1" is only reproducible if the mode belongs to the software.
- **The filter is exposed as its `p × p` convolution matrix**, because Savitzky–Golay folds into exported coefficients as `b_raw = M.T @ b_filtered`. #14's export depends on `convolution_matrix()` existing.
- **Derivatives are per variable index.** The schema carries no spacing field. `delta` exists for callers who want per-axis-unit derivatives and who then have to record that themselves.
- **The polynomial baseline maps the index onto [-1, 1] before fitting**, and needs to: a raw index over 700 variables to the fourth power spans 1e11.
- **AsLS converges when the weight vector stops changing**, capped at 20 iterations, and records `n_iterations_` and `converged_` per spectrum — a baseline that hit the cap is a different claim from one that settled.
- **The three baseline methods have no external reference**, so their entries are `unsourced` and they are checked against defining properties. `pybaselines` is the obvious reference and is worth raising alongside #13.
- **SciPy is a runtime dependency, so the Savitzky–Golay parity claim needs its caveat stated**: `scipy.signal` is not on the kernel's code path. Say this in the report — it is the one parity claim where the reference ships with the application.

From #11 (PCA kernel):

- **The corn PCA reference values were being generated with scikit-learn's randomised SVD.** `svd_solver="auto"` picks it whenever the matrix is wider than 500 with few components asked for — true of corn — and its `random_state` is unseeded. The generator now passes `svd_solver="full"` explicitly. **Check the solver before adding any new scikit-learn reference on a wide matrix.**
- **`spe(X)` requires the matrix; `hotelling_t2()` does not.** The model keeps the scores, so T² needs nothing else, but SPE measures the part of X that is not in the model. Both kernels do this the same way.
- **T², SPE and the cumulative explained variance references are our formula on scikit-learn's decomposition**, because scikit-learn reports none of them. An independent *decomposition*, not an independent formula. **The report in #14 must not present them as more than that**, and the same now applies to the three VIP entries #12 added.
- **`arrays.py` holds the array contract.** #12 added `as_float64_vector` there for the response, so a column vector is refused rather than ravelled.
- **`mean(T²) = a(n−1)/n` on the calibration set, exactly**, in both kernels. The cheapest check that the eigenvalue weighting and the divisor in λ agree.
- **An uncentred constant matrix has rank 1, not 0.** Its one component is the mean spectrum.

From #12 (PLS kernel):

- **The specification was wrong about weight orthogonality and has been corrected.** `pls-regression.md` §4 said `w_a'w_b` is not generally zero. **For PLS1 the weights are orthogonal** — ours and scikit-learn's agree to a few ulp — and the section now states the PLS1 and PLS2 cases separately. The non-orthogonality is a PLS2 property, so the assertion in `test_regression.py` will need revisiting when PLS2 lands, and the section says so.
- **Explained variance was being compared before it was defined.** The R `pls` vignette entry was in the fixture from #7, and nothing in `pls-regression.md` defined the quantity. §8 now defines both block shares. **The running total of the y share is exactly the model's R²** because the X-scores are orthogonal, which is why no separate cumulative-R² quantity exists — if you find yourself adding one, that is the reason not to.
- **Three `pls.vip.sklearn` entries were added to the fixture** so that VIP's verification step had something to run against. They are our formula on scikit-learn's own weights, scores and y-loadings, exactly as the PCA T² and SPE entries are.
- **The T² limit is shared, not duplicated.** It moved to a module-level `hotelling_t2_limit(n, a, alpha, samples)` in `decomposition.py`; `PCA.hotelling_t2_limit()` and `PLS.hotelling_t2_limit()` both delegate. The SPE limits are genuinely different formulas and are not shared.
- **Folds are data, never a seed, and this is load-bearing in three places** — the fixture, `validation.folds_from_indices()`, and `ResolvedSplit`. `cross_validated_predictions()` refuses a split that is not a partition of the samples *before* pooling anything, because a sample missing from every validation set shrinks the sum silently.
- **Centring is refitted inside each fold**, and a test pins the direction as well as the difference: leaking the full-set mean into the folds gives a *smaller* RMSECV, which is what makes it tempting.
- **SEC, SEP, Q² and `coefficients_original_units` are specified but not implemented.** Nothing verified them in #12 and the export format that needs the last one is #14's. They are the known gap between `metrics-and-validation.md` §5–§6 and `validation.py`.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name.
- **Corn and gasoline tests skip on a fresh machine.** Run `CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest` once to fetch them. A skipped dataset test is not a failing one, but it is not evidence either.
- **Regenerating the fixture is `uv run python tests/fixtures/generate_reference_values.py`.** Adding a kernel usually means adding reference entries there, then a tolerance class in `QUANTITY_CLASS`, then an entry in `ALGORITHMS` in `test_reference_values.py` if the algorithm name is new. Say in the commit message what moved and why — and check the diff is additive: #12's was 1264 insertions and zero deletions, which is what "no existing value moved" looks like.
- **The kernel tests use synthetic data on purpose.** `test_regression.py` and `test_decomposition.py` need no download, so they are the tests that still run on a fresh machine. Parity is where the real datasets belong.
- **`uv run pytest -m parity` runs the parity suite alone**, and any pytest run that makes a comparison rewrites `parity-results.json` at the repository root. It is gitignored.
- **`tests/test_parity_harness.py` deliberately provokes failures**, and saves and restores the recorder around every case. Keep new harness tests there, not in `test_parity.py`.
- **`scikit-learn` is dev-only**, for the same reason as `rdata`: it is a reference implementation. SciPy is a runtime dependency and kernels may use it, but not as their own reference.
- **`reference_values.json` is 551 KB.** Arrays are stored in full because the harness compares elementwise. Do not "tidy" it into summaries.
- **`rdata` is dev-only and imported lazily** by `load_gasoline`. It pulls in `xarray`, which is not in the recorded stack — transitive dev dependency only.
- **`ruff format --check` also formats Python blocks inside markdown.** A fenced `python` snippet in `docs/` with cosmetic alignment fails CI.
- **`design/canvas` is excluded from ruff**; **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py`; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues and pull requests.
- **The repository on GitHub is `millermuttu/Chemometrics-Workbench`**, not `Chemometrics_toolbox` — the local directory name differs from the remote, and the MCP tools 404 on the wrong one.
- **CI takes about 30 seconds** per matrix job. Check runs appear a few seconds after a push, so a status read immediately after pushing will show `queued` or `in_progress`.
- **`uv.lock` is a universal lockfile.** Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
