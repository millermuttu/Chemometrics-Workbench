# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

**Every kernel is written and the `chemotools` question is decided.** Phase 0 has one deliverable left that was in the original plan — the parity report, #14 — plus two small sourcing features that #13 turned up.

Issues #1–#10, #12 and #13 are merged and closed. **#11 is merged but deliberately still open**, blocked on the one verification step that has nothing to verify against — **and #13 found what verifies it, without R**. Pull requests #26 and #29 merged on 2026-08-22 and their branches are deleted locally and on origin. Nothing is in flight.

**The parity fixture is fully covered by what is in it.** `parity-results.json` records 66 comparisons, 66 passed, and **`not_compared` is empty**. What is missing is not coverage of the entries but *entries*: eight quantities across SNV, MSC, the three baselines and the two PCA limits are `unsourced`, and #27 and #28 now source all of them from `chemotools`.

**What is left in Phase 0**: `reference-values-chemotools` (#27), `reference-values-chemotools-limits` (#28), `parity-report` (#14), and `reference-values-r-mdatools` (#24, no longer on the critical path).

### Why #11 is blocked, and what now unblocks it

Its fourth verification step asks that the T² and SPE limits **match published values**. When it landed there were none: scikit-learn reports neither limit, and the only reference identified was R `mdatools`, which is not installed here (#24).

**#13 found a second reference, and it is a better one.** `chemotools.outliers` reports both limits from a fitted scikit-learn model, and measured against our PCA on all three datasets:

- **Our SPE limit and theirs are the same number** — both Jackson–Mudholkar, agreeing to 2.3e-15 relative.
- **The T² limits differ by exactly `(n+1)/n`**, because theirs is `a(n−1)/(n−a)·F` and our new-sample form is `a(n²−1)/(n(n−a))·F`. That is a documented convention difference with the formula identified, which is what the harness's third tier exists for — not a failure.
- The T² and SPE *statistics* agree with ours to 4.7e-13 and 3.4e-16, which is a second independent check on the diagnostics themselves.

**#28 is the work**, and it is what should flip `kernel-pca` to `passing`. **#24 (R `mdatools`) is no longer the only route** — it stays worth having, because SIMPLS is a genuinely different algorithm where `chemotools` is another Python implementation on the same NumPy, but nothing is waiting on it.

**The offline question was raised and settled** in an earlier session. Corn and gasoline are downloaded rather than committed. The option of writing to Eigenvector and Prof. Kalivas to ask for redistribution permission was offered and **declined — do not open that issue and do not chase it.**

## Repository

| | |
| --- | --- |
| Current branch | `dev`, clean, in sync with origin |
| Open pull requests | None |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 5 — #11 (blocked, merged), #14, #24, #27, #28 |
| Feature statuses | 12 × `passing`, 2 × `blocked`, 3 × `not_started` |
| Parity claims | 66 compared, 66 passed, 60 identical-within-float, 5 within tolerance, 1 documented divergence, **0 not compared** |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green (408 tests), enforced by CI on 3.12 and 3.13 |
| Runtime dependencies | `pydantic`, `numpy`, `scipy`. **`scikit-learn` and `chemotools` are dev-only and must stay that way** (#13) |

## Active feature

None is `in_progress`. Two are `blocked`, and they are no longer blocked on the same thing:

- **`kernel-pca` (#11)** — code merged and green; blocked on its published-limit verification step, which **#28 can now run**. Leave it `blocked` until that has actually been done, not because it is expected to pass.
- **`reference-values-r-mdatools` (#24)** — blocked on R not being installed, which is an environment decision rather than a coding task. No longer blocking anything else.

## Next action

**`reference-values-chemotools` (#27) next**, then #28, then #11's re-verification, then #14. That order is not arbitrary: #27 and #28 both add `chemotools` to the dev group and regenerate the same fixture, so running them in parallel would conflict, and #14 must not be written until the entries it describes exist.

```bash
git fetch origin
git checkout -b feature/27_reference-values-chemotools origin/dev
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest   # once, to populate the dataset cache
```

Read `docs/decisions/0001-chemotools.md` first — it is the decision #27 and #28 implement, and every number they need is in it. Re-derive them with:

```bash
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run --with chemotools==0.4.3 python \
    docs/decisions/0001-chemotools-evidence.py
```

**The trap in #27 is MSC.** It agrees to 1e-10 relative, which is outside the `preprocessing` tolerance class's 1e-12. Widening that class to make it pass would destroy the thing the class exists for — it is tight on purpose, to catch a `ddof` convention or a differently-defined norm. Give MSC its own class with a stated reason, or record the difference. Do not touch `preprocessing`.

The second trap is the SNV entry: generate it at **`ddof=0`**, because `chemotools` uses the population standard deviation where our default is `ddof=1`, and say so in the entry's notes. The autoscale entry already does exactly this against `StandardScaler`, and is the worked example.

**One feature at a time.** The protocol allows exactly one `in_progress`, and a `blocked` feature does not count against that.

## Waiting on the user

- **One question raised on pull request #22 is still unanswered**: `MSC(reference="supplied")` has no schema field for the reference spectrum — a schema change and a separate issue. **Question (b) is closed**: `PROPOSAL.md` §7 said PCA and PLS come from scikit-learn, which has been false since #11, and #13 rewrote the section to say what the repository actually does — scikit-learn and `chemotools` are dev-only reference implementations, and the kernels are ours. Nothing is waiting on it now.
- **`PROPOSAL.md` changed materially in #13** (§7 and the §14 stack summary). The published artifact at the URL below is now behind, and `CLAUDE.md` says it should be republished to the same URL when that happens.
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

From #13 (chemotools evaluation):

- **`scikit-learn` and `chemotools` are dev-only and must not enter `[project.dependencies]`.** `chemotools` requires scikit-learn, and installs 20 MB of which 17 MB is bundled example CSVs the application would never read. `CLAUDE.md`'s toolchain section now says so, and `docs/decisions/0001-chemotools.md` carries the evidence.
- **The decision is per transform.** Reference-only for SNV, MSC and the three baselines, which have no other external reference; neither adopted nor referenced for Savitzky-Golay and normalisation, both redundant with SciPy and scikit-learn. Do not re-argue it from preference — re-run `docs/decisions/0001-chemotools-evidence.py` and argue from numbers.
- **Their Savitzky-Golay defaults to `mode="nearest"`**, ours fixes `"interp"`. Identical in the interior to 1e-16, up to 30% of the largest value apart within a half-window of each end. This is the clearest confirmation that #10 was right to make the mode a property of the software rather than a caller's default.
- **Their SNV uses the population standard deviation**; ours defaults to `ddof=1`. Bit-identical at `ddof=0`, so the entry #27 generates must be at `ddof=0` and must say why.
- **MSC agrees to 1e-10, not to the last bits.** That is outside the `preprocessing` tolerance class and the class must not be widened for it.
- **A constant spectrum returns `NaN` there and raises here.** The one behavioural difference that would matter in an application, and the reason `_dead_threshold` exists.
- **`docs/decisions/` is the new home for decisions taken with evidence.** Numbered and dated, one per decision, with a re-runnable script beside it where the decision rests on measurements.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name.
- **Corn and gasoline tests skip on a fresh machine.** Run `CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest` once to fetch them. A skipped dataset test is not a failing one, but it is not evidence either.
- **Regenerating the fixture is `uv run python tests/fixtures/generate_reference_values.py`.** Adding a kernel usually means adding reference entries there, then a tolerance class in `QUANTITY_CLASS`, then an entry in `ALGORITHMS` in `test_reference_values.py` if the algorithm name is new. Say in the commit message what moved and why — and check the diff is additive: #12's was 1264 insertions and zero deletions, which is what "no existing value moved" looks like.
- **The kernel tests use synthetic data on purpose.** `test_regression.py` and `test_decomposition.py` need no download, so they are the tests that still run on a fresh machine. Parity is where the real datasets belong.
- **`uv run pytest -m parity` runs the parity suite alone**, and any pytest run that makes a comparison rewrites `parity-results.json` at the repository root. It is gitignored.
- **`tests/test_parity_harness.py` deliberately provokes failures**, and saves and restores the recorder around every case. Keep new harness tests there, not in `test_parity.py`.
- **`scikit-learn` is dev-only**, for the same reason as `rdata`: it is a reference implementation. SciPy is a runtime dependency and kernels may use it, but not as their own reference.
- **`uv run --with <package>` is how a package is evaluated without adding it.** #13's evidence script runs `chemotools` that way, so nothing about the evaluation touched `pyproject.toml` or `uv.lock`.
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
