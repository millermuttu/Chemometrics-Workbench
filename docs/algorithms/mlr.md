# MLR — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-regression.md`](pls-regression.md), whose centring rule, reported quantities and export MLR shares; [`variable-selection.md`](variable-selection.md), which chooses the variables MLR is usually fitted on; [`pca.md`](pca.md) §9 for the rank tolerance.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | $n \times p$, the estimator's input, centred by its training rows' column means |
| $y$ | length $n$, the response, centred by its training mean |
| $b$ | length $p$, the coefficients on the centred scale |

## 2. The model

[#376](https://github.com/millermuttu/Chemometrics-Workbench/issues/376). Ordinary least squares, $b = \arg\min \lVert y - Xb \rVert^2$, with the intercept $\bar y - \bar x^\top b$ from the centring, as scikit-learn's `LinearRegression(fit_intercept=True)`. It is solved by **QR**: for $X = QR$, $b$ solves $Rb = Q^\top y$ by back substitution, which never forms $X^\top X$ and so loses half as many digits as the normal equations.

MLR is the classic final model **after a variable selection**: a `select_variables` step, or one written by a VIP threshold, iPLS or CARS, above it. On a whole spectrum it is ill-posed (§3).

## 3. When it is refused

$b$ is unique only when $X$ has **full column rank**. A spectrum usually does not: more variables than samples, and every variable collinear with its neighbours. The rank is measured from the singular values with PCA's tolerance ([`pca.md`](pca.md) §9, decision 0004), $(\max(n,p)\,\varepsilon_{64} + \varepsilon_{\text{data}})\,s_1$, with $\varepsilon_{\text{data}} = \varepsilon_{32}$ in the executor because every array reaches it through the float32 store. When the rank is below $p$ the fit is **refused**, naming the rank, the number of variables and the fix: a selection above it, or PCR, which regresses on the rank's worth of components.

Below a split the check is made on **every training fold** before anything is fitted, and the refusal names the fold, as a classifier's one-class fold is ([#343](https://github.com/millermuttu/Chemometrics-Workbench/issues/343)). A selection of $p$ variables needs every training fold to have more than $p$ rows.

## 4. Reported quantities

RMSEC, $R^2$, bias, **SEC** (§5 of [`metrics-and-validation.md`](metrics-and-validation.md), with $A = p$: MLR spends one parameter per variable), RMSECV, $Q^2$, RMSEP and SEP, as for PLS. MLR has no parameter to tune, so it has no tuning curve, and no components, so no scores, loadings, $T^2$, SPE or VIP; the analysis screen leaves those panels out.

## 5. Export

$b$ is a fixed vector, so MLR exports as a PLS does ([`model-export.md`](../model-export.md)); a `select_variables` above it is foldable, so the exported coefficients are on the dataset's own axis with zeros at the variables not selected.

## 6. Parity

scikit-learn's `LinearRegression(fit_intercept=True)` on ten variables spread evenly across the axis of corn, gasoline and Tecator, few enough for each to have full column rank: coefficients and predictions, `<dataset>.mlr.coefficients.sklearn` and `<dataset>.mlr.predictions.sklearn`. In the executor, the final model, SEC and RMSECV below a selection are checked against `LinearRegression` on the run's own folds (`tests/test_executor.py`).
