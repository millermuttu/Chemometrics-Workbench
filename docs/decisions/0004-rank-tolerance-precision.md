# 0004 — The rank tolerance is quoted for the precision the data has, not the one it is computed in

**Date:** 2026-08-29
**Status:** accepted
**Issue:** [#101](https://github.com/millermuttu/Chemometrics-Workbench/issues/101)
**Decides:** `docs/algorithms/pca.md` §9, which stated the tolerance in float64 terms.

---

## Decision

The rank tolerance becomes

$$\text{tol} = \left(\max(n,p)\,\varepsilon_{64} + \varepsilon_{\text{data}}\right)\sigma_1$$

with $\varepsilon_{\text{data}}$ defaulting to $\varepsilon_{64}$. `PCA` takes it as `data_eps`;
the executor passes `PCA.STORED_EPS` ($\varepsilon_{32}$) because every array it fits reaches it
through a float32 store.

## Why two terms rather than one

They are two different errors and they scale differently.

| term | what it is | scaled by |
| --- | --- | --- |
| $\max(n,p)\,\varepsilon_{64}\,\sigma_1$ | rounding the decomposition accumulates | the dimension |
| $\varepsilon_{\text{data}}\,\sigma_1$ | the matrix is not the matrix that was meant | nothing |

The SVD runs in float64 whatever it was handed, so the arithmetic term keeps its usual growth
factor. The data term is a perturbation of the **input**: by Weyl's inequality, perturbing $A$ by
$E$ moves every singular value by at most $\lVert E \rVert_2$, and a float32 round trip has
$\lVert E \rVert_2 \approx \varepsilon_{32}\sigma_1$. A perturbation of $X$ is not an accumulation
over its entries, so it carries no dimension factor.

**This is the part that was got wrong first, and the error was large.** The obvious reading of
#101 — "the store is float32, so the honest tolerance is $\max(n,p)\,\varepsilon_{32}\,\sigma_1$"
— was implemented and measured on Tecator: it reports **rank 66** for a matrix of rank 99,
discarding 33 genuine components. Multiplying the data error by the dimension overshoots by
$\max(n,p)$, and on collinear spectra that is the difference between a threshold below the real
components and one well inside them.

## The measurement, which is what settled it

SNV then mean centring on Tecator, both stored, giving a 240 × 100 centred matrix:

| | value | ratio to $\sigma_1$ |
| --- | --- | --- |
| $\sigma_1$ | 2.513e+01 | 1 |
| $\sigma_{99}$ — the smallest genuine | 1.313e-04 | 5.2e-06 |
| $\sigma_{100}$ — spurious, from the round trip | 2.799e-07 | 1.1e-08 |

| tolerance | value | rank |
| --- | --- | --- |
| $\max(n,p)\,\varepsilon_{64}\,\sigma_1$ (before) | 1.34e-12 | 100 ✗ |
| $\max(n,p)\,\varepsilon_{32}\,\sigma_1$ (first attempt) | 7.19e-04 | 66 ✗ |
| $(\max(n,p)\,\varepsilon_{64} + \varepsilon_{32})\,\sigma_1$ (accepted) | 3.00e-06 | **99** ✓ |

The genuine and spurious values are **470 apart**, so the accepted threshold sits in a wide gap —
10× above the noise, 44× below the signal — rather than balanced between them. The test asserts
that separation, not just the resulting integer, so a future change that narrows the margin is
visible before it is wrong.

## What does not move

Nothing computed in float64 throughout. With $\varepsilon_{\text{data}} = \varepsilon_{64}$ the
tolerance changes by one part in $\max(n,p)$, far below any singular value that was near the
threshold, and the parity suite is unchanged.

Only `rank` was ever visibly wrong. The spurious eigenvalue is sixteen orders below the leading
one; it moved the SPE limit in the ninth decimal and the explained-variance ratio in the eighth.

## Rejected

- **Accept rank 100 and explain it where rank is displayed.** Cheapest, and the issue's own words
  for it were "the least honest": it reports a rank the matrix does not have.
- **Re-centre exactly after reading.** A fix at the wrong layer — the executor papering over a
  question about precision that belongs to the kernel that asks it.
