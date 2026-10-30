# OPLS — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-regression.md`](pls-regression.md), whose centring rule, sign convention, diagnostics and export OPLS shares; [`metrics-and-validation.md`](metrics-and-validation.md) for folds and the tuning curve (§15).

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | $n \times p$, centred by its training rows' column means |
| $y$ | length $n$, centred by its training mean |
| $A$ | the number of **orthogonal** components; the model has $1 + A$ |
| $w, t, p, c$ | the predictive weight, score, loading and $y$ loading |
| $w_o, t_o, p_o$ | an orthogonal component's weight, score and loading |

## 2. The algorithm

[#378](https://github.com/millermuttu/Chemometrics-Workbench/issues/378). Orthogonal PLS for one response (Trygg and Wold, 2002, *J. Chemometrics* 16, 119–128). Starting from $E = X$, for each of the $A$ orthogonal components:

1. $w = E^\top y / \lVert E^\top y \rVert$, $t = E w$, $p = E^\top t / t^\top t$;
2. $w_o$ is the part of $p$ orthogonal to $w$, $p - (w^\top p)\,w$, normalised;
3. $t_o = E w_o$, $p_o = E^\top t_o / t_o^\top t_o$, and $E \leftarrow E - t_o p_o^\top$.

Then the predictive component is fitted on what is left: $w = E^\top y / \lVert E^\top y \rVert$, $t = E w$, $p = E^\top t / t^\top t$, $c = y^\top t / t^\top t$. Every weight follows PLS's sign convention ([`pls-regression.md`](pls-regression.md) §6): its largest-magnitude element is positive.

What is removed is variation in $X$ that is **orthogonal to $y$**: each $t_o$ has zero covariance with $y$. What remains is one component that carries all of $X$'s covariance with $y$, which is why SIMCA users read OPLS's predictive score against its first orthogonal score: the one is what predicts, the other the largest systematic variation that does not.

## 3. As a PLS

The scores are reported **predictive first**, then the orthogonal ones in order, and the model is set up as a PLS so that everything after the fit is PLS's: each score is a linear function of the centred $X$, so there is a rotation matrix $R$ with $T = XR$ (each weight pushed back through the filters before it); the coefficients are $b = R\,[c, 0, \dots, 0]^\top$; $T^2$, SPE, explained variance and their limits are PLS's own methods. The scores are mutually orthogonal and the residual after all $1 + A$ components is PLS's $E$.

## 4. The tuning curve

OPLS with $1 + k$ components **predicts exactly as PLS1 with $1 + k$** (Trygg and Wold's identity for one response), so its RMSECV curve over $k = 0 \dots A$ orthogonal components is PLS1's over $1 \dots 1 + A$, computed by PLS's one-fit-per-fold curve. The curve is labelled by orthogonal components and marked at the $A$ fitted.

## 5. VIP

The orthogonal components explain none of $y$, so PLS's VIP formula gives the **predictive VIP**, $\sqrt{p}\,|w|$ for the predictive weight - what SIMCA reports as VIP for an OPLS model.

## 6. Parity

There is no scikit-learn OPLS. The predictions are compared through the identity of §4 with scikit-learn's `PLSRegression(1 + A, scale=False)`, for $1 + 1$ and $1 + 4$ components on corn, gasoline and Tecator: `<dataset>.opls.predictions_<n>.sklearn_pls_identity`. The scores and loadings themselves have **no external reference**: no open OPLS implementation is installed (R `ropls` needs R), and the published worked example is given as figures. That gap is in the fixture as `tecator.opls.orthogonal_scores.unsourced`, and the scores' defining properties - $T = XR$, mutually orthogonal, orthogonal scores uncorrelated with $y$ - are tested directly.

## 7. Export

Predictions are $X b$ plus the intercept, a fixed vector, so OPLS exports as a PLS does ([`model-export.md`](../model-export.md)).
