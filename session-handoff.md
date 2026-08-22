# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

**Preprocessing is done and PCA is written.** All eight schema steps have a kernel, and PCA lands with both of its diagnostics.

Issues #1–#10 are merged and closed. **#11 is merged but deliberately still open**: the code is complete and green, and the feature is `blocked` on the one verification step that has nothing to verify against. Pull request #25 merged on 2026-08-22 and its branch is deleted locally and on origin. Nothing is in flight.

The parity report has 55 claims in it, 53 of them in `identical_within_float`. The eighteen new ones are PCA — loadings, scores, eigenvalues, explained variance, cumulative explained variance, T² and SPE — on all three datasets.

**What is left in Phase 0**: `kernel-pls` (#12), `chemotools-eval` (#13), `parity-report` (#14), and `reference-values-r-mdatools` (#24, raised out of #11).

### Why #11 is blocked, and what unblocks it

Its fourth verification step asks that the T² and SPE limits **match published values**. There are none to match. scikit-learn reports neither limit, and the only external reference is R `mdatools`, whose entries have been `unsourced` since #7 because R is not installed here. **#24 is the only thing that unblocks it** — install R (`conda install -c conda-forge r-base r-mdatools`, no sudo needed) and source those six entries plus three new `hotelling_t2_limit` ones. You were asked during #11 and chose to land it blocked rather than install R mid-feature.

The limits are not unverified: about α of the calibration samples fall beyond a 1−α limit at three configurations, the exact identity `mean(T²) = a(n−1)/n` holds, and the preconditions and degenerate cases are covered. That is real evidence, and it is not a second implementation.

**The offline question was raised and settled** in an earlier session. Corn and gasoline are downloaded rather than committed. The option of writing to Eigenvector and Prof. Kalivas to ask for redistribution permission was offered and **declined — do not open that issue and do not chase it.**

## Repository

| | |
| --- | --- |
| Current branch | `dev`, clean, in sync with origin |
| Open pull requests | None |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 5 of 15 — #11 (blocked, merged), #12, #13, #14, #24 |
| Feature statuses | 10 × `passing`, 2 × `blocked`, 3 × `not_started` |
| Parity claims | 55 compared, 55 passed, 53 identical-within-float, 1 within tolerance, 1 documented divergence, 8 fixture entries not yet compared — all of them PLS |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green, and enforced by CI on both 3.12 and 3.13 |

## Active feature

None is `in_progress`. Two are `blocked`, both on the same thing:

- **`kernel-pca` (#11)** — code merged and green; blocked on its published-limit verification step.
- **`reference-values-r-mdatools` (#24)** — blocked on R not being installed, which is an environment decision rather than a coding task.

## Next action

**`kernel-pls` (#12) next.** `chemotools-eval` (#13) is also eligible; priority order puts PLS first. #24 is eligible the moment you decide to install R.

```bash
git fetch origin
git checkout -b feature/12_kernel-pls origin/dev
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest   # once, to populate the dataset cache
```

`pls-regression.md` is normative and complete: NIPALS with deflation of both X and y, no internal centring, signs keyed as in `pca.md` §5, coefficients `b = Rq` (already sign-invariant), VIP with its normalisation, and a χ² moment match for the SPE limit rather than Jackson–Mudholkar, because PLS components are not covariance eigenvectors.

**All eight remaining untested fixture entries are PLS**, and they are what #12 turns green: three `pls.coefficients.sklearn`, three `pls.rmsecv_curve.sklearn`, and the two R `pls` vignette entries — one of which, `gasoline.pls.rmsecv_curve.r_pls_vignette`, is the strongest reference in the whole fixture. The two remaining placeholder cases in `test_parity.py` (`test_predictions_follow_from_the_coefficients` and `test_rmsec_follows_from_the_predictions`) call no kernel yet; **rewrite them to call it, keeping the entry id, the tolerance and the tier where they are.** #11 did the same for its six PCA cases and is the worked example.

The two findings most likely to sink #12 quietly are both below: the fold indices must be read out of the fixture rather than reseeded, and every matrix in the fixture was pre-centred before scikit-learn saw it.

**One feature at a time.** The protocol allows exactly one `in_progress`, and a `blocked` feature does not count against that.

**Read the carried-forward findings below before writing a kernel.** Several of them are the difference between a parity test that means something and one that passes while testing nothing.

## Waiting on the user

- **Two questions raised on pull request #22 and still unanswered**, both still open. (a) `MSC(reference="supplied")` has no schema field for the reference spectrum — a schema change and a separate issue. (b) `PROPOSAL.md` §7 says PCA and PLS come from scikit-learn, while `pca.md` §3 and `pls-regression.md` §4 are marked normative and specify our own implementations. **This one now blocks #11 in spirit**: the next feature is exactly the one the contradiction is about. The specs have been followed throughout and #11 will follow them too unless told otherwise.
- **GitHub default branch is still `main`.** Any pull request opened without an explicit base targets the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #4 (PLS):

- **The Jackson–Mudholkar SPE limit does not transfer from PCA to PLS.** PLS components are not eigenvectors of the covariance of X, so there is no residual eigenvalue sequence to sum. PLS uses a χ² moment match on the calibration residuals instead. See `pls-regression.md` §9.
- **SNV, MSC and baseline correction cannot be folded into exported coefficients**, because all three depend on the sample being predicted. Savitzky–Golay *can* — see #10 below. An exported model carries a residual preprocessing chain plus coefficients, not always a bare coefficient vector. This shapes the export format in #14 and `PROPOSAL.md` §9. See `pls-regression.md` §7.

From #5 (metrics and validation):

- **Our shuffle is not scikit-learn's.** We fix `numpy.random.default_rng` (PCG64); `check_random_state` gives a legacy `RandomState`. The same seed produces different folds. **The parity harness must pass our resolved fold indices to sklearn as an explicit `cv` iterable** — seeding both with 42 and comparing is an invalid test, and it will look like it works. This bites #12 directly.
- **RMSEC divides by `n` here.** Degrees-of-freedom corrections live in SEC and SEP. Packages writing `n − A − 1` under the name RMSEC are reporting our SEC, and that is the expected third-significant-figure mismatch in #7's reference values — check the definition before recording a discrepancy as a defect.
- **Fold aggregation is pooled residuals, not the mean of per-fold RMSEs.** Any reference value taken from a paper needs its aggregation rule established before it is comparable.

From #6 (reference datasets):

- **Only Tecator is committed.** Corn and gasoline are downloaded on first use into `~/.cache/chemometrics-workbench/datasets` and verified against a pinned SHA-256. The licence finding that decided each one is in `src/chemometrics_workbench/data/<name>/README.md`. Do not "helpfully" commit the raw files.
- **Publishing a Tecator result obliges you to name the instrument and company (Tecator).** A condition of use, not a courtesy. It applies to the parity report in #14.
- **Tecator's wavelength axis is reconstructed, not read.** The file states a range and a channel count and no axis; the loader uses `linspace(850, 1050, 100)`. Corn and gasoline axes are read from the file itself. If a reference value ever disagrees at the fourth digit on Tecator only, suspect the axis before suspecting the kernel.
- **Corn's targets are identical across `m5`, `mp5` and `mp6`** — the same 80 samples on three spectrometers. That is what makes it the calibration-transfer benchmark, and a test asserts it.

From #7 (reference values):

- **The harness must read fold indices out of the fixture, not reseed.** Every generated RMSECV entry stores explicit `train_indices` and `test_indices` per fold. Handing those to scikit-learn as an explicit `cv` iterable is the only valid comparison; seeding both with 42 produces different folds and *will still pass a badly written test*. This is the finding most likely to be quietly ignored, and #12 is where it matters.
- **Every matrix in the fixture was pre-centred before scikit-learn saw it**, because sklearn centres internally and unconditionally and `PLSRegression` also defaults to `scale=True`. A harness that feeds raw data to our kernel and raw data to sklearn is not testing anything.
- **`comparable: false` is not decoration.** `tecator.pls.sep.thodberg` is a real published number that is not a parity target — its inputs are principal components the loader discards. The harness must skip entries where `comparable` is false, and skip every `status: "unsourced"` entry, which has a `null` value by construction.
- **The R `pls` vignette entry is the most valuable one in the file.** LOO over gasoline rows 0–49, so it is deterministic and has no shuffle stream to reconcile, and R `pls` computes MSEP as `SSE/nobj` — divisor `n`, the same as ours, so it compares with no correction. If only one parity claim survives, make it that one. It is still untested and #12 is what tests it.
- **Six R `mdatools` entries are unsourced because R is not installed here.** They are the T² and SPE limits scikit-learn does not provide. Installing R and `mdatools` and rerunning the recorded configurations is the single highest-value way to strengthen the fixture, and #11 is where their absence will be felt.

From #8 (parity harness):

- **Never write a bare `assert_allclose` in a kernel test.** Call `parity.check(entry_id, ours)`. It picks the tolerance for the quantity's class, aligns signs where the quantity needs it, tags the claim tier and records the result for the report. A comparison made outside the harness is invisible to #14. (#10's edge-only case is the one deliberate exception, and it borrows the harness's own tolerance rather than inventing one.)
- **Tolerances are not knobs.** A failing parity test is a finding. Widening the tolerance to make it pass is the tempting move and it puts a lie in the one artifact this project cannot afford to have lying in it. If the difference is a convention, use `parity.record_divergence()` and write the reason into the specification's divergence table.
- **A quantity with no entry in `QUANTITY_CLASS` is refused, not guessed at.** A new kernel reporting a new quantity must add its tolerance class with a reason first. `test_every_quantity_in_the_fixture_has_a_tolerance_class` fails until it does, including for unsourced entries.
- **The PCA and PLS parity cases are still placeholders.** They compare arithmetic the specs define directly — `T = XP`, `ŷ = Xb`, the eigenvalue and RMSEC definitions — because no estimator kernel exists. #11 and #12 each rewrite their cases to call the kernel; the entry id, tolerance and tier stay put. Do not add a parallel test file.
- **`parity-results.json` lists what was never compared.** Fourteen comparable fixture entries are currently untested, all of them PCA and PLS. That list shrinking is the real measure of progress, and it is what stops the report overstating coverage.
- **The identical-within-float threshold is scale-relative**, 32 ulp of the largest reference value, not a fixed `rtol` with `atol=0`.

From #9 (scaling kernels):

- **Kernels import nothing from scikit-learn.** "scikit-learn-compatible" is duck compatibility — `fit`, `transform`, `fit_transform` — and nothing more. sklearn is the reference implementation the fixtures are generated against; a kernel inheriting `BaseEstimator` would be a wrapper around the thing we claim parity with. Same rule for #11 and #12.
- **Transformers are stateful, and even the stateless ones have a `fit`.** Fitted parameters are the fit set's, always, because `metrics-and-validation.md` §9 pushes held-out samples through with the *training fold's* parameters. And recording the variable count at fit is the only thing that catches a transposed array — a transposed matrix is still 2-D.
- **Zero is judged relative to the magnitude of the data, never `== 0.0`.** The standard deviation of a genuinely constant row of 0.7 is 1.16e-16. `_dead_threshold` in `preprocessing.py` is the shared rule.
- **Two divergences from scikit-learn are deliberate**: autoscale defaults to `ddof=1` where `StandardScaler` is fixed at 0, and a zero-variance row or column raises rather than getting a substituted scale of 1. The parity case passes `ddof=0` explicitly. Do not "fix" either by matching sklearn without asking.
- **SNV and MSC have no external reference** and are checked against their defining identities. Their fixture entries are `unsourced` on purpose. `chemotools` (#13) would fill the gap.
- **Preprocessing reference values live on a 5 × 8 block per dataset.** `PREPROCESS_BLOCK` in `test_parity.py` must stay in step with `PREPROCESS_ROWS`/`PREPROCESS_COLUMNS` in the generator, or the comparison silently runs on two different blocks. `SAVGOL_WINDOW` and `SAVGOL_POLYORDER` are duplicated across the same boundary for the same reason.
- **`from_spec` is the executor's seam.** It now covers every step the schema can express. The two things it still needs from outside the schema are the axis for `RangeSelect` and the spectrum for `MSC(reference="supplied")`.

From #10 (smoothing, derivatives and baselines):

- **Savitzky–Golay's edge mode is `interp` and is not configurable**, and this is the finding most likely to come back as a bug report. `mirror`, `nearest`, `wrap` and `constant` pad the spectrum and give different values within a half-window of each end; several chemometrics tools instead leave that block unfiltered or drop it, which changes the variable count. A recipe recording "Savitzky–Golay, 11, 2, 1" is only reproducible if the mode belongs to the software. `docs/algorithms/smoothing-and-baselines.md` §3 tabulates the alternatives.
- **The filter is exposed as its `p × p` convolution matrix**, not only as filtered values, because `pls-regression.md` §7 makes Savitzky–Golay foldable into exported coefficients as `b_raw = M.T @ b_filtered`. #14's export depends on `convolution_matrix()` existing. SNV, MSC and baseline correction are *not* foldable.
- **Derivatives are per variable index.** The schema carries no spacing field, so a recipe always means per index. `delta` exists for callers who want per-axis-unit derivatives and who then have to record that themselves. The two differ by `delta**deriv`, which the regression coefficients absorb exactly — no model's fit changes, but every quoted number does.
- **The polynomial baseline maps the index onto [-1, 1] before fitting.** Exact in the same sense any affine change of variable is, and needed: a raw index over 700 variables to the fourth power spans 1e11 and the fit is then decided by the least-squares cutoff rather than by the data. A test comparing an index fit against a raw-axis fit failed on exactly that before the scaling was added.
- **AsLS converges when the weight vector stops changing** — an exact fixed point, since the weights are a step function of the residual sign — capped at 20 iterations. `n_iterations_` and `converged_` are recorded per spectrum, because a baseline that hit the cap is a different claim from one that settled.
- **The three baseline methods have no external reference.** Neither SciPy nor scikit-learn implements baseline correction, so their fixture entries are `unsourced` and they are checked against defining properties instead. `pybaselines` is the obvious reference and adding it is a dependency decision, not a kernel one — worth raising alongside #13.
- **SciPy is a runtime dependency, so the Savitzky–Golay parity claim needs its caveat stated**: `scipy.signal` is not on the kernel's code path. SciPy solves a least-squares system per output position; the kernel builds one matrix from the pseudo-inverse of the window's Vandermonde matrix. Say this in the report — it is the one parity claim where the reference ships with the application.
- **Regenerating the fixture perturbs the corn PCA entries in their last bits.** Four of them moved by at most 2.9e-15 on this regeneration with no formula change — LAPACK on a 700-wide matrix. Well inside the decomposition tolerance of 1e-8, but expect it in the diff and do not go looking for a defect.

From #11 (PCA kernel):

- **The corn PCA reference values were being generated with scikit-learn's randomised SVD**, and this is the finding to remember. `svd_solver="auto"` picks it whenever the matrix is wider than 500 with few components asked for — true of corn, false of gasoline and tecator — and its `random_state` is unseeded, so those four entries moved by about 1e-14 on every regeneration. That is what #10's fixture diff showed and attributed to LAPACK; the mechanism was this. The generator now passes `svd_solver="full"` explicitly, and two consecutive regenerations are bit-identical. **Check the solver before adding any new scikit-learn reference on a wide matrix.**
- **`spe(X)` requires the matrix; `hotelling_t2()` does not.** The model keeps the scores, so T² needs nothing else, but SPE measures the part of X that is not in the model and so cannot be recovered from it. The model deliberately does not keep a copy of X.
- **T² and SPE references are our formula on scikit-learn's decomposition**, because scikit-learn reports neither. So is the cumulative explained variance curve. All six entries say so in their notes — an independent *decomposition*, not an independent formula. The report in #14 must not present them as more than that.
- **`arrays.py` holds the array contract now** — 2-D, finite, float64, the caller's array never modified. Both `preprocessing.py` and `decomposition.py` use it. A third kernel adds nothing new here; it imports `as_float64`.
- **`mean(T²) = a(n−1)/n` on the calibration set, exactly.** The cheapest check that the eigenvalue weighting and the divisor in λ agree with each other, and it caught nothing only because both were right.
- **An uncentred constant matrix has rank 1, not 0.** Its one component is the mean spectrum. The "no variance to decompose" error is reachable only from a matrix that is zero after centring.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name, so they only need updating in one place.
- **Corn and gasoline tests skip on a fresh machine.** Run `CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest` once to fetch them; CI sets the same variable and caches on the `SHA256SUMS` files. A skipped dataset test is not a failing one, but it is not evidence either.
- **Regenerating the fixture is `uv run python tests/fixtures/generate_reference_values.py`.** Adding a kernel usually means adding reference entries there, then a tolerance class in `QUANTITY_CLASS`, then an entry in `ALGORITHMS` in `test_reference_values.py` if the algorithm name is new. All three, or a test fails telling you which. Say in the commit message what moved and why — scientific numbers do not move silently.
- **`uv run pytest -m parity` runs the parity suite alone**, and any pytest run that makes a comparison rewrites `parity-results.json` at the repository root. It is gitignored.
- **`tests/test_parity_harness.py` deliberately provokes failures**, and saves and restores the recorder around every case so fabricated numbers never reach the run record. Keep new harness tests there, not in `test_parity.py`.
- **`scikit-learn` is dev-only**, for the same reason as `rdata`: it is a reference implementation, and our kernels must not call it. SciPy is different — it is a runtime dependency and kernels may use it, but not as their own reference (see the #10 finding above).
- **`reference_values.json` is 513 KB.** Arrays are stored in full because the harness compares elementwise. Do not "tidy" it into summaries.
- **`rdata` is dev-only and imported lazily** by `load_gasoline`, because the application never reads R files. It pulls in `xarray`, which is not in the recorded stack — transitive dev dependency only.
- **`ruff format --check` also formats Python blocks inside markdown.** A fenced `python` snippet in `docs/` with cosmetic alignment fails CI. This bit #5.
- **`design/canvas` is excluded from ruff** — it generates design artboards and is not shipped code.
- **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography (en dashes, ×) that those rules flag as homoglyphs.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py` plus the seeding step; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues and pull requests.
- **The repository on GitHub is `millermuttu/Chemometrics-Workbench`**, not `Chemometrics_toolbox` — the local directory name differs from the remote, and the MCP tools 404 on the wrong one.
- **CI takes about 30 seconds** per matrix job, plus the dataset download on a cold cache. Check runs appear a few seconds after a push, so a status read immediately after pushing will show `queued` or `in_progress`.
- **`uv.lock` is a universal lockfile.** Compiled packages list many wheels because it resolves for every supported platform. That is deliberate — the Phase 4 three-platform build matrix depends on it. Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
