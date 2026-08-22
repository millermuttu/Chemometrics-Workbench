# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

**The first kernels are written and merged, and the parity report has real claims in it.**

Issues #1–#9 are merged into `dev`. Pull request #22 merged on 2026-08-22 and its branch is deleted locally and on origin. Nothing is in flight.

Six preprocessing transformers now exist in `src/chemometrics_workbench/preprocessing.py`, and all fifteen of their parity claims land in `identical_within_float`, the tightest tier. That is the first evidence the whole Phase 0 apparatus was built to produce.

**The offline question was raised and settled.** Corn and gasoline are downloaded rather than committed, and the maintainer asked why, given that this is a local-first project. The answer is that the download happens once per machine and is cached permanently, the shipped application never imports `datasets.py` at all, and the two datasets have no redistribution terms we can establish. The option of writing to Eigenvector and Prof. Kalivas to ask for permission was offered and **declined — do not open that issue and do not chase it.** The download-and-verify path is the answer, not a placeholder for a better one.

## Repository

| | |
| --- | --- |
| Current branch | `dev`, clean, in sync with origin |
| Open pull requests | None |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 5 of 14 remaining |
| Feature statuses | 9 × `passing`, 5 × `not_started` |
| Parity claims | 31 compared, 31 passed, 30 identical-within-float, 1 documented divergence, 14 fixture entries not yet compared |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green, and enforced by CI on both 3.12 and 3.13 |

## Active feature

None. Nothing is `in_progress`.

## Next action

Three features are eligible — `kernels-smoothing` (#10), `kernel-pca` (#11) and `kernel-pls` (#12). Priority order says **`kernels-smoothing` (#10) next**.

```bash
git fetch origin
git checkout -b feature/10_kernels-smoothing origin/dev
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest   # once, to populate the dataset cache
```

#10 is Savitzky–Golay and derivatives. Two things about it are already known: `scipy.signal.savgol_filter` exists and is the obvious reference, and `pls-regression.md` §7 records that Savitzky–Golay **is** foldable into exported coefficients as a banded matrix, unlike SNV and MSC — so the kernel should expose the convolution matrix, not only the filtered result. `from_spec` already raises for `savgol` by name, which is the seam to fill.

**One feature at a time**, even though three are eligible. The protocol allows exactly one `in_progress`.

**Read the carried-forward findings below before writing a kernel.** Several of them are the difference between a parity test that means something and one that passes while testing nothing.

## Waiting on the user

- **Two questions raised on pull request #22 and still unanswered.** Both are recorded there in full. (a) `MSC(reference="supplied")` has no schema field for the reference spectrum — a schema change and a separate issue. (b) `PROPOSAL.md` §7 says PCA and PLS come from scikit-learn, while `pca.md` §3 and `pls-regression.md` §4 are marked normative and specify our own implementations. Both cannot be right; the specs have been followed throughout.
- **GitHub default branch is still `main`.** Any pull request opened without an explicit base targets the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #4 (PLS):

- **The Jackson–Mudholkar SPE limit does not transfer from PCA to PLS.** PLS components are not eigenvectors of the covariance of X, so there is no residual eigenvalue sequence to sum. PLS uses a χ² moment match on the calibration residuals instead. See `pls-regression.md` §9.
- **SNV and MSC cannot be folded into exported coefficients**, because both depend on the sample being predicted. An exported model carries a residual preprocessing chain plus coefficients, not always a bare coefficient vector. This shapes the export format in #14 and `PROPOSAL.md` §9. See `pls-regression.md` §7.

From #5 (metrics and validation):

- **Our shuffle is not scikit-learn's.** We fix `numpy.random.default_rng` (PCG64); `check_random_state` gives a legacy `RandomState`. The same seed produces different folds. **The parity harness (#8) must pass our resolved fold indices to sklearn as an explicit `cv` iterable** — seeding both with 42 and comparing is an invalid test, and it will look like it works.
- **RMSEC divides by `n` here.** Degrees-of-freedom corrections live in SEC and SEP. Packages writing `n − A − 1` under the name RMSEC are reporting our SEC, and that is the expected third-significant-figure mismatch in #7's reference values — check the definition before recording a discrepancy as a defect.
- **Fold aggregation is pooled residuals, not the mean of per-fold RMSEs.** Any reference value taken from a paper needs its aggregation rule established before it is comparable.

From #6 (reference datasets):

- **Only Tecator is committed.** Corn and gasoline are downloaded on first use into `~/.cache/chemometrics-workbench/datasets` and verified against a pinned SHA-256. The licence finding that decided each one is in `src/chemometrics_workbench/data/<name>/README.md`. Do not "helpfully" commit the raw files.
- **Publishing a Tecator result obliges you to name the instrument and company (Tecator).** A condition of use, not a courtesy. It applies to the parity report in #14.
- **Tecator's wavelength axis is reconstructed, not read.** The file states a range and a channel count and no axis; the loader uses `linspace(850, 1050, 100)`. Corn and gasoline axes are read from the file itself. If a reference value ever disagrees at the fourth digit on Tecator only, suspect the axis before suspecting the kernel.
- **Corn's targets are identical across `m5`, `mp5` and `mp6`** — the same 80 samples on three spectrometers. That is what makes it the calibration-transfer benchmark, and a test asserts it.

From #7 (reference values):

- **The harness must read fold indices out of the fixture, not reseed.** Every generated RMSECV entry stores explicit `train_indices` and `test_indices` per fold. Handing those to scikit-learn as an explicit `cv` iterable is the only valid comparison; seeding both with 42 produces different folds and *will still pass a badly written test*. This is the finding most likely to be quietly ignored.
- **Every matrix in the fixture was pre-centred before scikit-learn saw it**, because sklearn centres internally and unconditionally and `PLSRegression` also defaults to `scale=True`. A harness that feeds raw data to our kernel and raw data to sklearn is not testing anything.
- **`comparable: false` is not decoration.** `tecator.pls.sep.thodberg` is a real published number that is not a parity target — its inputs are principal components the loader discards. The harness must skip entries where `comparable` is false, and skip every `status: "unsourced"` entry, which has a `null` value by construction.
- **The R `pls` vignette entry is the most valuable one in the file.** LOO over gasoline rows 0–49, so it is deterministic and has no shuffle stream to reconcile, and R `pls` computes MSEP as `SSE/nobj` — divisor `n`, the same as ours, so it compares with no correction. If only one parity claim survives, make it that one.
- **Six R `mdatools` entries are unsourced because R is not installed here.** They are the T² and SPE limits scikit-learn does not provide. Installing R and `mdatools` and rerunning the recorded configurations is the single highest-value way to strengthen the fixture.

From #8 (parity harness):

- **Never write a bare `assert_allclose` in a kernel test.** Call `parity.check(entry_id, ours)`. It picks the tolerance for the quantity's class, aligns signs where the quantity needs it, tags the claim tier and records the result for the report. A comparison made outside the harness is invisible to #14.
- **Tolerances are not knobs.** A failing parity test is a finding. Widening the tolerance to make it pass is the tempting move and it puts a lie in the one artifact this project cannot afford to have lying in it. If the difference is a convention, use `parity.record_divergence()` and write the reason into the specification's divergence table.
- **A quantity with no entry in `QUANTITY_CLASS` is refused, not guessed at.** A new kernel reporting a new quantity must add its tolerance class with a reason first.
- **The parity suite's current cases are placeholders.** They compare arithmetic the specs define directly — centring, `T = XP`, `ŷ = Xb`, the eigenvalue and RMSEC definitions — because no kernel exists. Each kernel feature rewrites its case to call the kernel; the entry id, tolerance and tier stay put. Do not add a parallel test file.
- **`parity-results.json` lists what was never compared.** Fourteen comparable fixture entries are currently untested. That list shrinking is the real measure of kernel progress, and it is what stops the report overstating coverage.
- **The identical-within-float threshold is scale-relative**, 32 ulp of the largest reference value, not a fixed `rtol` with `atol=0`. A near-zero score would otherwise have to be bit-exact, which no reordering of a sum can promise.

From #9 (scaling kernels):

- **Kernels import nothing from scikit-learn.** "scikit-learn-compatible" is duck compatibility — `fit`, `transform`, `fit_transform` — and nothing more. sklearn is the reference implementation the fixtures are generated against; a kernel inheriting `BaseEstimator` would be a wrapper around the thing we claim parity with. Follow the same pattern in #10, #11 and #12.
- **Transformers are stateful, and even the stateless ones have a `fit`.** Fitted parameters are the fit set's, always, because `metrics-and-validation.md` §9 pushes held-out samples through with the *training fold's* parameters. And recording the variable count at fit is the only thing that catches a transposed array — a transposed matrix is still 2-D.
- **Zero is judged relative to the magnitude of the data, never `== 0.0`.** The standard deviation of a genuinely constant row of 0.7 is 1.16e-16. An exact test was written first and failed on real arithmetic. `_dead_threshold` in `preprocessing.py` is the shared rule.
- **Two divergences from scikit-learn are deliberate**: autoscale defaults to `ddof=1` where `StandardScaler` is fixed at 0, and a zero-variance row or column raises rather than getting a substituted scale of 1. The parity case passes `ddof=0` explicitly. Do not "fix" either by matching sklearn without asking.
- **SNV and MSC have no external reference** and are checked against their defining identities. Their fixture entries are `unsourced` on purpose. `chemotools` (#13) would fill the gap.
- **Preprocessing reference values live on a 5 × 8 block per dataset.** `PREPROCESS_BLOCK` in `test_parity.py` must stay in step with `PREPROCESS_ROWS`/`PREPROCESS_COLUMNS` in the generator, or the comparison silently runs on two different blocks.
- **`from_spec` is the executor's seam.** It raises by name for steps with no kernel yet, and needs the axis for `RangeSelect` and the spectrum for `MSC(reference="supplied")` — neither of which the schema carries.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name, so they only need updating in one place.
- **Corn and gasoline tests skip on a fresh machine.** Run `CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest` once to fetch them; CI sets the same variable and caches on the `SHA256SUMS` files. A skipped dataset test is not a failing one, but it is not evidence either.
- **Regenerating the fixture is `uv run python tests/fixtures/generate_reference_values.py`.** Adding a kernel usually means adding reference entries there, then a tolerance class in `QUANTITY_CLASS`, then an entry in `ALGORITHMS` in `test_reference_values.py` if the algorithm name is new. All three, or a test fails telling you which.
- **`uv run pytest -m parity` runs the parity suite alone**, and any pytest run that makes a comparison rewrites `parity-results.json` at the repository root. It is gitignored.
- **`tests/test_parity_harness.py` deliberately provokes failures**, and saves and restores the recorder around every case so fabricated numbers never reach the run record. Keep new harness tests there, not in `test_parity.py`.
- **`scikit-learn` is dev-only too**, for the same reason as `rdata`: it is a reference implementation, and our kernels must not call it. A kernel that imports sklearn is not a kernel, it is a wrapper.
- **Regenerate the fixture with `uv run python tests/fixtures/generate_reference_values.py`**, and say in the commit message what moved and why. Scientific numbers do not move silently.
- **`reference_values.json` is 403 KB.** Arrays are stored in full because the harness compares elementwise. Do not "tidy" it into summaries.
- **`rdata` is dev-only and imported lazily** by `load_gasoline`, because the application never reads R files. It pulls in `xarray`, which is not in the recorded stack — transitive dev dependency only, worth knowing before someone reaches for it in shipped code.
- **`ruff format --check` also formats Python blocks inside markdown.** A fenced `python` snippet in `docs/` with cosmetic alignment fails CI. This bit #5.
- **`design/canvas` is excluded from ruff** — it generates design artboards and is not shipped code. Roughly 200 of the initial 214 lint errors came from it.
- **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography (en dashes, ×) that those rules flag as homoglyphs.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py` plus the seeding step; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues and pull requests; `/new-branch` will ask for the issue number rather than looking up its title.
- **The repository on GitHub is `millermuttu/Chemometrics-Workbench`**, not `Chemometrics_toolbox` — the local directory name differs from the remote, and the MCP tools 404 on the wrong one.
- **CI takes about 20 seconds** per matrix job, plus the dataset download on a cold cache. Check runs appear a few seconds after a push, so a status read immediately after pushing will show `queued` or `in_progress`.
- **`uv.lock` is a universal lockfile.** Compiled packages list many wheels because it resolves for every supported platform. That is deliberate — the Phase 4 three-platform build matrix depends on it. Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
