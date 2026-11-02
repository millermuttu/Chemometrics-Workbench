# Logistic regression — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`lda.md`](lda.md), [`knn.md`](knn.md) and [`svm.md`](svm.md), whose PCA front this shares and does not restate; [`classification.md`](classification.md) for the tallies.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $t_i$ | sample $i$'s scores on the $A$ principal components of the centred input |
| $K$ | the number of classes, coded $0 \dots K-1$ in Unicode order |
| $W, b$ | the coefficients ($r \times A$) and intercepts ($r$): $r = 1$ for two classes, $K$ for more |
| $C$ | the inverse strength of the L2 penalty |

## 2. The model

[#380](https://github.com/millermuttu/Chemometrics-Workbench/issues/380). PCA-logistic regression: the input is centred and projected on $A$ principal components, as for LDA, kNN and SVM, and an L2-penalised logistic regression is fitted to the scores. The objective is scikit-learn's,

$$\tfrac12 \lVert W \rVert^2 + C \sum_i \ell_i, \qquad \ell_i = -\log \hat p_{i, y_i},$$

the intercepts unpenalised. **Two classes are one binomial model**, $r = 1$, $\hat p_1 = \sigma(t^\top w + b)$, as scikit-learn fits them; **three or more are one multinomial model** over every class, $\hat p = \mathrm{softmax}(W t + b)$. A class with no training sample is refused, and a one-class training fold is refused as for every classifier ([#343](https://github.com/millermuttu/Chemometrics-Workbench/issues/343)).

## 3. The solver

SciPy's L-BFGS-B on the objective and its exact gradient, from zero, stopping when the projected gradient's largest element is below `tol` $= 10^{-4}$ (scikit-learn's default), with no function-change stop. It refuses to report a fit that reaches its 15,000-iteration limit, naming $C$: a smaller $C$ converges faster.

## 4. Probabilities and assignment

$\hat p$ is $n \times K$, columns in class order, every row summing to one; for two classes the columns are $1 - \hat p_1$ and $\hat p_1$. A sample is assigned its most probable class, a tie to the first. Logistic regression is the first classifier here that reports class probabilities: the analysis screen lists them beside each sample's assignment.

## 5. Export

The chain to the scores travels as the other score-space classifiers' does ([`model-export.md`](../model-export.md) §6): the residual chain, the foldable tail as an affine map, `x_mean`, the loadings, then $W$ and $b$. The snippet computes $z = W t + b$, the probabilities, and the most probable class. The model artifact carries $W$, $b$ and $C$.

## 6. Parity

scikit-learn's PCA then `LogisticRegression(C=1, solver="lbfgs")` on the same scores, two classes about the target's median and three at its terciles, on corn, gasoline and Tecator. **Both solved to a gradient of $10^{-10}$**, the coefficients and probabilities agree within the `lbfgs` class (rtol and atol $10^{-5}$: scikit-learn's L-BFGS also stops on a relative change in the objective, and the two quasi-Newton histories differ). At the default tolerance the assignments are **identical** and the probabilities a **documented divergence**, recorded after they are checked to agree within ten times `tol`.
