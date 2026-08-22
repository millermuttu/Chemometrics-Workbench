# 0001 — `chemotools`: rejected as a runtime dependency, adopted as a dev-only reference

**Date:** 2026-08-22
**Status:** accepted
**Issue:** [#13](https://github.com/millermuttu/Chemometrics-Workbench/issues/13)
**Decides:** `PROPOSAL.md` §7, which listed `chemotools` as provisional pending exactly this evaluation.

---

## Decision

**Per transform, not wholesale:**

| Transform | `chemotools` equivalent | Decision |
| --- | --- | --- |
| SNV | `scatter.StandardNormalVariate` | **Reference only.** Bit-identical to ours at `ddof=0`. |
| MSC | `scatter.MultiplicativeScatterCorrection` | **Reference only.** Agrees to 1e-10 relative. |
| AsLS baseline | `baseline.AsLs` | **Reference only.** Agrees to 1e-9 relative. |
| Rubberband baseline | `baseline.RubberbandCorrection` | **Reference only.** Bit-identical. |
| Polynomial baseline | `baseline.PolynomialCorrection` | **Reference only.** Agrees to 1.5e-14 relative. |
| Savitzky–Golay, derivatives | `derivative.SavitzkyGolay` | **Neither.** Redundant: it wraps `scipy.signal.savgol_filter`, which is already our reference. |
| Normalise `l1`, `l2` | `scale.NormScaler` | **Neither.** Redundant with `sklearn.preprocessing.normalize`, already our reference, and `max` and `area` are not expressible. |
| Mean centring, autoscaling | none | Not offered; `StandardScaler` is already the reference. |
| Range selection | `feature_selection.RangeCut` | **Neither.** A slice. |

**No `chemotools` code runs in the application.** It is not added to `dependencies`; the transforms above stay ours. What is adopted is its use as an **open, versioned reference implementation in the dev group**, alongside `scikit-learn` and `rdata`, for the five quantities that today have **no external reference at all** and are checked only against their defining identities.

Wiring that into the fixture is deliberately *not* part of this evaluation — it is [#27](https://github.com/millermuttu/Chemometrics-Workbench/issues/27). This record decides; that issue implements.

---

## Evidence

Every number below is reproduced by [`0001-chemotools-evidence.py`](0001-chemotools-evidence.py), which is committed beside this record:

```bash
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run --with chemotools==0.4.3 python \
    docs/decisions/0001-chemotools-evidence.py
```

`uv run --with` installs `chemotools` into a throwaway environment, so re-deriving the evidence does not add it to the project. A rerun that disagrees with the tables below is a finding about this record.

They come from `chemotools` 0.4.3 against our kernels on the **same fixture blocks the #9 and #10 reference values were generated on** — the 5 × 8 block of each of corn, gasoline and tecator — with the baselines run on the first five full spectra, since an eight-variable baseline is not a baseline.

### Numerical agreement

| Comparison | corn | gasoline | tecator |
| --- | --- | --- | --- |
| SNV, ours `ddof=0` | 0.0 | 0.0 | 0.0 |
| SNV, ours `ddof=1` (our default) | 1.26e-01 | 1.26e-01 | 1.22e-01 |
| MSC, reference = mean | 3.9e-12 | 7.1e-16 | 1.0e-10 |
| MSC, reference = median | 2.3e-12 | 5.5e-15 | 1.1e-10 |
| AsLS, `lam=1e5`, `p=0.01` | 8.6e-10 | 7.4e-11 | 7.6e-10 |
| Rubberband | 0.0 | 0.0 | 0.0 |
| Polynomial, order 2 | 3.5e-15 | 5.3e-16 | 1.5e-14 |
| Normalise `l1`, `l2` | 0.0 | 0.0 | 0.0 |
| Savitzky–Golay, `mode="interp"`, deriv 0/1/2 | ≤6.8e-17 | ≤6.8e-17 | ≤3.2e-15 |

Differences are the largest absolute difference relative to the largest reference value, so they are comparable with the harness's identical-within-float threshold of 32 ulp ≈ 7.1e-15.

Three of these are worth reading carefully:

- **The SNV difference is a convention, not a defect.** `chemotools` uses the population standard deviation; ours defaults to `ddof=1`, the sample convention used for eigenvalues in `pca.md` §4 and for SEC and SEP in `metrics-and-validation.md` §5. At `ddof=0` the two are bit-identical, which is exactly the relationship we already have with `StandardScaler` and already record as a deliberate divergence.
- **MSC agrees to 1e-10, not to the last bits**, and the gap is real rather than noise: it is 1e-16 on gasoline and 1e-10 on tecator, which is the signature of a differently-arranged regression rather than a different formula. Good enough to be a reference at the preprocessing tolerance class only if that class is widened, which it must not be — #27 must give MSC an honest tolerance or record the difference, not tighten the number.
- **AsLS agrees to 1e-9** across a different solver (`chemotools` offers a banded solver and defaults to it; ours solves the sparse system), a different iteration cap (theirs 100, ours 20) and a different convergence rule. That two independent implementations of Eilers and Boelens land this close is the strongest evidence in this table, and it is the entry our fixture most lacked.

### Edge handling

`chemotools`'s `SavitzkyGolay` **defaults to `mode="nearest"`**, where ours fixes `mode="interp"` and does not expose it (`smoothing-and-baselines.md` §3). On the full spectra with an 11-point window the two agree to 1e-16 in the interior and diverge by up to 2.3e-02 within a half-window of each end — on gasoline's first derivative that is 30% of the largest value. Set `mode="interp"` explicitly and the whole spectrum agrees to 3e-15.

This is the single most important edge finding, and it cuts **for** our position rather than against it: the mode is a per-call parameter there and a property of the software here, because a recipe recording "Savitzky–Golay, 11, 2, 1" is only reproducible if the mode is not a hidden default that can change under it.

The 5 × 8 fixture block is half edge columns at a window of 5, which is why the block comparison shows 1e-3 differences and the interior comparison shows 1e-16. A reference generated only over the interior would not have caught this.

### Documented conventions

Conventions are documented per class in numpydoc docstrings, with formulas, and the source is readable. Two conventions are *not* stated where it matters: `StandardNormalVariate` does not say which `ddof` it uses, and `SavitzkyGolay`'s `mode` default of `"nearest"` differs from `scipy.signal.savgol_filter`'s own default of `"interp"` without saying why. Both had to be established by reading the source or by measurement, which is the same work our own specification documents exist to save a reader.

### dtype and copy behaviour

Equivalent to ours, and good:

| Check | ours | `chemotools` |
| --- | --- | --- |
| float32 in → float64 out | yes | yes |
| Caller's array modified | no | no |
| NaN | `ValueError` naming row and column | `ValueError` (via `sklearn.check_array`) |
| Narrower matrix at `transform` | `ValueError` naming both counts | `ValueError` naming both counts |
| Constant spectrum (zero spread) | `ValueError` naming the samples | **returns `NaN` silently** |

The last row is the only real difference and it is the one that matters for an application: `preprocessing.py`'s rule is that zero is judged relative to the magnitude of the data and refused, because a NaN that reaches a model surfaces three screens later as an unexplained failure.

### Maintenance activity

| | |
| --- | --- |
| Version evaluated | 0.4.3, released 2026-06-30 |
| Latest activity | last push 2026-08-03, three weeks before this evaluation |
| History | created 2023-01-26; eight releases between 2026-03-06 and 2026-06-30 |
| Licence | MIT |
| Stars / forks / open issues | 82 / 20 / 39 |
| Contributors | 8, of which one has 543 commits, one is dependabot, and the remaining six total 6 |

Actively maintained, and **effectively a single-maintainer project**. `PROPOSAL.md` §7 already dropped `process-improve` from the core for exactly that reason — "core numerical results should not depend on it" — and consistency requires the same judgement here. As a *reference* implementation the risk profile is different: a pinned version that we compare against cannot break the application, and if it were abandoned tomorrow the recorded numbers would remain valid, exactly as with `rdata`.

The eight releases in four months also came with visible churn: importing the package emits `FutureWarning`s for three modules that have moved or split, `SavitzkyGolay` carries three deprecated constructor arguments, and two modules describe themselves as experimental with an API that may change.

### Dependency weight

**This is what decides the runtime question.**

- `chemotools` requires `scikit-learn>=1.6`. scikit-learn is **dev-only** here by deliberate decision — it is the reference our kernels are compared against, and a kernel built on it would be a wrapper around the thing we claim parity with. Adopting `chemotools` at runtime moves scikit-learn, `joblib` and `threadpoolctl` into the shipped application.
- Installed size is **20 MB, of which 17 MB is example datasets bundled inside the package** (`chemotools/datasets/data/*.csv`). The application would never read one of them, and `PROPOSAL.md` §14's three-platform PyInstaller bundle would carry all of it.
- Against that: the transforms it would replace are already written, tested, specified and green, and total a few hundred lines.

---

## Why this and not the alternatives

**Adopt at runtime and delete our kernels.** Rejected. It inverts the dependency rule in `PROPOSAL.md` §7 — *take a dependency for the tedious and well-solved; own the scientifically load-bearing and small* — for code that is precisely small and load-bearing. It would also put scikit-learn and 17 MB of unused CSVs into the desktop bundle, and hand the project's numerical conventions (SNV's `ddof`, Savitzky–Golay's edge mode) to a single-maintainer package's defaults, in a project whose entire claim is that those conventions are fixed and documented.

**Reject outright.** Rejected too, and this is the answer that would have been easy. Five of our transforms — SNV, MSC and all three baselines — have **no external reference at all** and are tested only against their defining identities. `chemotools` is an open, MIT-licensed, version-pinnable implementation that agrees with four of the five to between 0 and 1e-9. Refusing it would leave those five entries `unsourced` for no better reason than that the same package was wrong for a different job.

**Adopt for SNV only.** Rejected as too narrow: the same argument that makes it a good reference for SNV makes it a good reference for MSC and the baselines, and the baselines are where we have the least independent evidence.

---

## Consequences

- `PROPOSAL.md` §7 and the §14 stack summary no longer describe `chemotools` as provisional; `CLAUDE.md`'s toolchain section records the decision so it is not re-litigated.
- `chemotools` is **not** in `[project.dependencies]` and must not be added there. Adding it to the dev group and sourcing the five entries is [#27](https://github.com/millermuttu/Chemometrics-Workbench/issues/27).
- The parity report (#14) may describe SNV, MSC and the baselines as compared against an independent implementation **only once #27 has actually done it**. Until then they remain `unsourced`, and the report must say so.
- Our own conventions do not move. SNV keeps `ddof=1` and Savitzky–Golay keeps `mode="interp"`; both differences are recorded as divergences with reasons rather than resolved by matching.

## Found while evaluating, and outside this decision

`chemotools.outliers` provides `HotellingT2` and `QResiduals`, which report **confidence limits** — the quantities scikit-learn does not provide and on which feature `kernel-pca` (#11) has been `blocked` since it landed. Measured against our PCA on all three datasets:

- **Our SPE limit and theirs are the same number**: Jackson–Mudholkar, agreeing to 2.3e-15 relative. Theirs also offers a chi-squared and a percentile variant, which is the divergence `pca.md` §13 already records.
- **Our `T²` limits and theirs differ by a documented convention.** Theirs is `a(n−1)/(n−a)·F`; ours for a new sample is `a(n²−1)/(n(n−a))·F`, so the ratio is exactly `(n+1)/n` — 1.0125 on corn, 1.0167 on gasoline, 1.0042 on tecator. Our beta form for calibration samples has no counterpart there at all.

That is a reference for the exact quantity #11 could not verify, and it does not need R. It is raised as [#28](https://github.com/millermuttu/Chemometrics-Workbench/issues/28) rather than folded in here, and it reduces #24 (R `mdatools`) from the only way to unblock #11 to a second, independent one.
