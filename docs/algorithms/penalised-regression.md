# Penalised regression — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-regression.md`](pls-regression.md), whose centring rule, reported quantities and export these models share; [`metrics-and-validation.md`](metrics-and-validation.md) for folds, the cross-validation protocol and the tuning curve (§15).

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | $n \times p$, the estimator's input, centred by its training rows' column means |
| $y$ | length $n$, the response, centred by its training mean |
| $b$ | length $p$, the coefficients on the centred scale |
| $\alpha$ | the penalty, $\alpha > 0$ |
| $X = U S V^\top$ | the thin SVD, $s_1 \ge s_2 \ge \dots$ the singular values |

Like PLS ([`pls-regression.md`](pls-regression.md) §3), the kernel centres nothing. The executor centres $X$ and $y$ by the training rows' means, as scikit-learn's `fit_intercept=True` does, and adds $\bar y - \bar x^\top b$ back as the intercept.

## 2. Ridge

[#374](https://github.com/millermuttu/Chemometrics-Workbench/issues/374). Ridge minimises

$$\lVert y - Xb \rVert^2 + \alpha \lVert b \rVert^2,$$

the penalty on the coefficients **as they are**, not on standardised ones: a pipeline that wants the penalty to treat every variable alike autoscales before the estimator. The solution is

$$b = V \,\mathrm{diag}\!\left(\frac{s_i}{s_i^2 + \alpha}\right) U^\top y,$$

which is $(X^\top X + \alpha I)^{-1} X^\top y$ without forming $X^\top X$. With $p \gg n$, the normal case for a spectrum, the SVD is of an $n \times n$ problem rather than a $p \times p$ one. Every direction is shrunk, none is dropped: unlike PCR, which keeps $A$ components whole and discards the rest, ridge scales component $i$ by $s_i^2 / (s_i^2 + \alpha)$.

## 3. Elastic Net and Lasso

Specified with [#375](https://github.com/millermuttu/Chemometrics-Workbench/issues/375).

## 4. Degrees of freedom, and why SEC is absent

The effective degrees of freedom of a ridge fit are the trace of its hat matrix, $\sum_i s_i^2 / (s_i^2 + \alpha)$, which lies strictly between 0 and $\mathrm{rank}(X)$ and is not a whole number. SEC's denominator $n - A - 1$ ([`metrics-and-validation.md`](metrics-and-validation.md) §5) counts whole parameters, so for ridge **SEC is absent** rather than computed with a fractional $A$ under a name that would not mean SEC. RMSEC, $R^2$, bias, RMSECV, $Q^2$, RMSEP and SEP are reported as for PLS.

## 5. The tuning grid

The penalty's curve ([`metrics-and-validation.md`](metrics-and-validation.md) §15) is RMSECV at 25 values of $\alpha$, log-spaced over six decades:

$$\alpha_j = s_1^2 \cdot 10^{-6 + j/4}, \qquad j = 0 \dots 24,$$

where $s_1$ is the largest singular value of the centred matrix the model is fitted on. At $\alpha = s_1^2$ even the strongest direction is shrunk by half and the model predicts little more than the mean; six decades below, the penalty barely acts. The grid is fixed by the data before any residual is seen. The $\alpha$ the node was fitted with is marked on the curve and is not replaced by the curve's minimum.

## 6. Parity

scikit-learn's `Ridge(alpha=1.0, fit_intercept=True, solver="svd")`, on the raw matrix of corn, gasoline and Tecator: its coefficients against ours on the centred matrix, and its calibration predictions against ours plus the mean. The claims are `<dataset>.ridge.coefficients.sklearn` and `<dataset>.ridge.predictions.sklearn`, in the `coefficients` and `predictions` tolerance classes. In the executor, RMSECV, every point of the curve and the final model's predictions are checked against scikit-learn's `Ridge` on the run's own folds (`tests/test_executor.py`).

## 7. Export

$b$ is a fixed vector, so a ridge model exports as a PLS does ([`model-export.md`](../model-export.md)): foldable steps folded into the coefficients, residual steps carried, and the snippet reproduces the application's predictions.

## 8. What is not reported

A ridge model has no components, so it has **no scores, loadings, $T^2$, SPE or contributions**, and no VIP. The analysis screen leaves those panels out, outlier diagnostics refuse it by name, and bootstrap intervals, which are specified for PLS and PCR, refuse it too.
