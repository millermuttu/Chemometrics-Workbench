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

[#375](https://github.com/millermuttu/Chemometrics-Workbench/issues/375). Elastic Net minimises scikit-learn's objective, with its scaling,

$$\frac{1}{2n}\lVert y - Xb \rVert^2 + \alpha\,\rho\,\lVert b \rVert_1 + \frac{\alpha (1 - \rho)}{2}\lVert b \rVert^2,$$

where $\rho$ is `l1_ratio`, $0 < \rho \le 1$. $\rho = 1$ is the **Lasso**; the menu offers both, and they are one estimator. The L1 term sets coefficients exactly to zero, so a sparse $b$ is a wavelength selection: the analysis screen says how many of the $p$ variables the model kept.

The smallest $\alpha$ at which every coefficient is zero is

$$\alpha_{\max} = \frac{\max_j |x_j^\top y|}{n \rho}.$$

At or above it the model predicts the training mean. The schema's default $\alpha = 1$ is scikit-learn's, and on many spectra it is above $\alpha_{\max}$ (corn's is about $10^{-2}$, Tecator's about 4): the curve (§5) shows where the useful range is.

### 3.1 Why the solution is not always unique

With $\rho < 1$ the objective is strictly convex and $b$ is unique. The Lasso's need not be when $p > n$ or the columns are collinear, which spectra are; its fitted values $Xb$ are unique even then. At the parity alpha (§6) the Lasso solutions are sparse and agree coefficient by coefficient.

### 3.2 The solver, and how it stops

**FISTA with adaptive restart**, not coordinate descent (maintainer, 2026-10-30). Each step is a gradient step on the smooth part, $\tfrac12 b^\top (X^\top X + \ell_2 I) b - b^\top X^\top y$ in scikit-learn's $n$-scaled units ($\ell_1 = \alpha\rho n$, $\ell_2 = \alpha(1-\rho)n$), with step $1/L$ for $L$ the largest eigenvalue of $X^\top X$ plus $\ell_2$, followed by soft-thresholding at $\ell_1/L$, Nesterov momentum, and O'Donoghue and Candès's restart, which drops the momentum whenever it points uphill. Every step is one product with $X^\top X$, which NumPy does at compiled speed; a coordinate sweep is a Python loop, and on corn coordinate descent needs up to a million sweeps (scikit-learn's own count).

It stops on **scikit-learn's duality gap**, computed every 10 steps: $\text{gap} \le \text{tol}\,\lVert y \rVert^2$, with `tol` $= 10^{-4}$ as scikit-learn's default. A fit that does not reach it within the iteration limit (100,000 for a model, 20,000 for a point of the curve) is **refused, naming $\alpha$**, never reported.

## 4. Degrees of freedom, and why SEC is absent

The effective degrees of freedom of a ridge fit are the trace of its hat matrix, $\sum_i s_i^2 / (s_i^2 + \alpha)$, which lies strictly between 0 and $\mathrm{rank}(X)$ and is not a whole number. SEC's denominator $n - A - 1$ ([`metrics-and-validation.md`](metrics-and-validation.md) §5) counts whole parameters, so for ridge **SEC is absent** rather than computed with a fractional $A$ under a name that would not mean SEC. RMSEC, $R^2$, bias, RMSECV, $Q^2$, RMSEP and SEP are reported as for PLS.

## 5. The tuning grid

The penalty's curve ([`metrics-and-validation.md`](metrics-and-validation.md) §15) is RMSECV at 25 values of $\alpha$, log-spaced from a top down, the top computed from the centred matrix the model is fitted on before any residual is seen:

| Model | Top | Decades |
| --- | --- | --- |
| Ridge | $s_1^2$, the largest squared singular value: even the strongest direction is shrunk by half and the model predicts little more than the mean | 6 |
| Elastic Net, Lasso | $\alpha_{\max}$ (§3), where every coefficient is zero | 3 |

Elastic Net's grid is shorter because a small L1 penalty on collinear spectra converges slowly: three decades below $\alpha_{\max}$ keeps a corn curve to seconds. Each fold fits the grid **from the top down, each fit started from the last**. A fit that does not converge ends that fold's path, and it and every smaller $\alpha$ are **absent** from the curve: a pooled residual missing a fold is not the curve's number. The $\alpha$ the node was fitted with is marked on the curve and is not replaced by the curve's minimum.

## 6. Parity

scikit-learn's `Ridge(alpha=1.0, fit_intercept=True, solver="svd")`, on the raw matrix of corn, gasoline and Tecator: its coefficients against ours on the centred matrix, and its calibration predictions against ours plus the mean. The claims are `<dataset>.ridge.coefficients.sklearn` and `<dataset>.ridge.predictions.sklearn`, in the `coefficients` and `predictions` tolerance classes. In the executor, RMSECV, every point of the curve and the final model's predictions are checked against scikit-learn's `Ridge` on the run's own folds (`tests/test_executor.py`).

Elastic Net and the Lasso are compared with scikit-learn's `ElasticNet(l1_ratio=0.5)` and `Lasso` at $\alpha = 0.1\,\alpha_{\max}$, on the centred matrices of corn, gasoline and Tecator, **both solved to a duality gap of $10^{-12}$**, where the two solvers reach the same optimum by different routes: coefficients and predictions in the `coefficients` and `predictions` classes, twelve claims. At the default gap of $10^{-4}\lVert y\rVert^2$ each solver stops where its route has got to, so the default fit is a **documented divergence**: before it is recorded, its objective is checked to be within that gap of the optimum.

## 7. Export

$b$ is a fixed vector, zeros included, so a penalised model exports as a PLS does ([`model-export.md`](../model-export.md)): foldable steps folded into the coefficients, residual steps carried, and the snippet reproduces the application's predictions.

## 8. What is not reported

A penalised model has no components, so it has **no scores, loadings, $T^2$, SPE or contributions**, and no VIP. The analysis screen leaves those panels out, outlier diagnostics refuse it by name, and bootstrap intervals, which are specified for PLS and PCR, refuse it too.
