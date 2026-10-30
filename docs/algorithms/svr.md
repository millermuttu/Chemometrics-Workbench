# SVR — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`svm.md`](svm.md), whose PCA front, kernels, solver and export this shares and does not restate; [`pls-regression.md`](pls-regression.md) for the reported regression quantities; [`metrics-and-validation.md`](metrics-and-validation.md) for folds and the tuning curve (§15).

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $t_i$ | sample $i$'s scores on the $A$ principal components of the centred input |
| $y_i$ | the response, **not** centred |
| $K(t, u)$ | the kernel, linear $t^\top u$ or RBF $\exp(-\gamma\lVert t - u\rVert^2)$ ([`svm.md`](svm.md) §2) |
| $C$, $\varepsilon$ | the cost of a residual outside the tube, and the tube's half-width |

## 2. The model

[#377](https://github.com/millermuttu/Chemometrics-Workbench/issues/377). PCA-SVR: the input is centred and projected on $A$ principal components exactly as for PCA-SVM, and LIBSVM's **epsilon-SVR** is fitted to the scores. Residuals within $\varepsilon$ of the prediction cost nothing; outside, they cost $C$ per unit. The prediction is

$$\hat y(t) = \sum_i (a_i - a_i^*)\,K(t_i, t) - \rho,$$

the sum over the support vectors, those with $a_i \ne a_i^*$. $y$ is not centred: $\rho$ is the intercept. `gamma` unset is scikit-learn's `"scale"`, $1/(A\,\mathrm{var}(T))$ over the training scores.

## 3. The dual, and the solver

The dual has $2n$ multipliers, $a$ then $a^*$. It is the C-SVM dual of [`svm.md`](svm.md) §3 with labels $+1$ for the first $n$ and $-1$ for the second, the kernel repeated in all four blocks, and the linear term $\varepsilon - y$ then $\varepsilon + y$ in place of $-1$. LIBSVM solves both with one solver, so `classification._smo` solves this unchanged once given the linear term: second-order working-set selection, no shrinking, stopping when the largest KKT violation is below `tol` $= 10^{-3}$, refusing after $10^6$ iterations.

## 4. Reported quantities

RMSEC, $R^2$, bias, RMSECV, $Q^2$, RMSEP and SEP, as for PLS. **SEC is absent**: an SVR's number of parameters is not a whole count. As kNN and SVM do, it has the **PCA front's panels** - scores, loadings, explained variance, $T^2$ and SPE - because its input is a PCA. It has no coefficient vector, so the folded-coefficient panel says so instead.

## 5. The tuning curve

Over $C$ ([`metrics-and-validation.md`](metrics-and-validation.md) §15): **13** values log-spaced from $\mathrm{range}(y)/1000$ up to $\mathrm{range}(y)$, the training response's range, **smallest first**. A larger $C$ converges more slowly, so each point may take 20,000 iterations; a $C$ that does not converge ends that fold's path, and it and every larger $C$ are **absent**. The grid is shorter than a penalty's because the solver is a Python loop: on Tecator a linear kernel on unscaled scores takes tens of seconds over it, where the default RBF kernel takes a few.

## 6. Parity

scikit-learn's PCA then `SVR(kernel, C=1, epsilon=0.1, gamma="scale", shrinking=False)` on the same scores, linear and RBF, on corn, gasoline and Tecator. **Both solved to a KKT gap of $10^{-9}$**, the predictions agree within the `float32_kernel` class ([`svm.md`](svm.md) §7: LIBSVM caches the kernel in float32). At the default $10^{-3}$ each stops where its arithmetic has got to, a **documented divergence** recorded after the predictions are checked to agree within ten times `tol` across the response's range.

## 7. Export

As an SVM's pair travels ([`model-export.md`](../model-export.md) §6): the residual chain, the foldable tail as an affine map, then `x_mean`, the loadings, the support vectors in score space, their $a - a^*$ and $\rho$. The model artifact carries the same arrays and, in its manifest, the kernel, $C$, $\varepsilon$, $\gamma$ and $\rho$.
