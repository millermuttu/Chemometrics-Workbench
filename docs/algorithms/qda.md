# QDA — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`lda.md`](lda.md), whose PCA front and discriminant this generalises; [`classification.md`](classification.md) for the tallies.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $t$ | a sample's scores on the $A$ principal components of the centred input |
| $\mu_k, \Sigma_k, \pi_k$ | class $k$'s training mean, covariance (denominator $n_k - 1$) and prior, its share of the training rows |
| $r$ | `reg_param`, $0 \le r \le 1$ |

## 2. The model

[#381](https://github.com/millermuttu/Chemometrics-Workbench/issues/381). PCA-QDA: the input is centred and projected on $A$ principal components, as for LDA, and each class is one Gaussian on the scores with its **own** covariance - LDA with the pooled covariance replaced by one per class. Each $\Sigma_k = V_k \Lambda_k V_k^\top$ is decomposed whole ($A \times A$), and scikit-learn's `reg_param` shrinks its eigenvalues toward one:

$$s_k = (1 - r)\,\lambda_k + r.$$

Where a class has more training samples than components this is exactly scikit-learn's SVD solution (its squared singular values over $n_k - 1$ are these eigenvalues). With fewer, scikit-learn's SVD solver refuses; here the decomposition stays defined, and with $r > 0$ every scaling is positive.

## 3. When it is refused

A class's covariance is **singular** when one of its scalings is negligible beside its largest: $s_{k,\min} \le (A\,\varepsilon_{64} + \varepsilon_{32})\,s_{k,\max}$, the relative test of decision 0004. scikit-learn's test is an absolute $10^{-4}$ on each scaling, which depends on the scores' units: it refuses corn's at $r = 0$, whose smallest score variances are below $10^{-3}$, where this test does not. A class with one training sample has no covariance at all. Either is refused, naming the class, its count and the fix - fewer components, or a positive `reg_param`. Below a split the check runs on **every training fold** first and names the fold, as a one-class fold is ([#343](https://github.com/millermuttu/Chemometrics-Workbench/issues/343)).

## 4. Decision and assignment

scikit-learn's log posterior up to a constant,

$$\delta_k(t) = -\tfrac12\Big(\big\lVert s_k^{-1/2} \odot V_k^\top (t - \mu_k)\big\rVert^2 + \textstyle\sum \log s_k\Big) + \log \pi_k,$$

assigned to the largest, a tie to the first class. Tallied by [`classification.md`](classification.md).

## 5. Export

The chain to the scores travels as the other score-space classifiers' does ([`model-export.md`](../model-export.md) §6): the residual chain, the foldable tail as an affine map, `x_mean`, the loadings, then each class's $\mu_k$, $V_k$, $s_k$ and $\pi_k$. The model artifact carries the same arrays and, in its manifest, $r$ and the priors.

## 6. Parity

scikit-learn's PCA then `QuadraticDiscriminantAnalysis(reg_param=r)` on the same scores, the tercile classes of each target: the decision values (the `predictions` class) and the assignments (exact), at $r = 0$ and $0.1$ on gasoline and Tecator and at $0.1$ on corn - every case scikit-learn fits.
