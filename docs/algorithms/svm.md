# SVM — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md) for the front end; [`classification.md`](classification.md) for class order, the confusion matrix and every metric; [`knn.md`](knn.md) and [`lda.md`](lda.md), which have the same front end for the same reasons.

---

## 1. PCA-SVM, and why

Every SVM here is fitted to the first $A$ principal-component scores of the centred calibration matrix, where $A$ is `n_components`. This has three effects:
- The scores and diagnostics panels show the space the classifier works in.
- The kernel is evaluated on $A$ numbers per sample rather than on hundreds, so the solver's cost does not grow with the spectrum's length.
- The default RBF width (§2) is set by the variance of those scores, so it follows the data rather than its units.

---

## 2. Model

The classes, $C_0 \dots C_{N-1}$ in Unicode order, come from a metadata column, as in `classification.md` §1. Every class must have at least one calibration sample.

1. Centre: $\bar{x}$ is the calibration mean. This is the estimator's own step.
2. Project: a PCA with $A$ components of $X - \bar{x}$, as in `pca.md`. $T = (X - \bar{x}) P$.
3. Kernel, on score rows:
   - `linear`: $K(s, t) = s^\top t$;
   - `rbf`: $K(s, t) = \exp(-\gamma \lVert s - t \rVert^2)$.

   When `gamma` is unset, $\gamma = 1 / (A \operatorname{var}(T))$, where the variance is over every entry of the training scores. This is scikit-learn's `gamma="scale"`. `gamma` has no effect on a linear kernel.
4. **One-vs-one.** For every pair of classes $i < j$, a C-SVM is fitted to the rows of those two classes, with $y = +1$ for class $i$ and $-1$ for class $j$, by solving the dual

   $$\min_a \tfrac{1}{2} a^\top Q a - \mathbf{1}^\top a \quad \text{subject to} \quad 0 \le a \le C,\; y^\top a = 0, \qquad Q_{kl} = y_k y_l K(t_k, t_l).$$

   The pair's decision is $f_{ij}(x) = \sum_k a_k y_k K(t_k, t) - \rho$. The rows with $a_k > 0$ are the pair's support vectors.

---

## 3. Solver

The dual is solved by **SMO with LIBSVM's second-order working-set selection** (Fan, Chen and Lin, *JMLR* 6, 2005), which is also what scikit-learn's `SVC` runs:
- $i$ maximises $-y_t \nabla_t$ over the rows that can move up, and $j$ minimises $-(\text{gap})^2 / \text{curvature}$ over the rows that can move down. Ties go to the later row, as LIBSVM's comparisons make them. A non-positive curvature is floored at $10^{-12}$.
- The two multipliers are updated analytically and clipped to the box, in LIBSVM's order.
- **The solver stops when the maximal KKT violation, $\max_{\text{up}}(-y\nabla) + \max_{\text{low}}(y\nabla)$, falls below `tol` = $10^{-3}$**, scikit-learn's default.
- $\rho$ is the mean of $y_k \nabla_k$ over the free multipliers ($0 < a_k < C$). With none free, it is the midpoint of the interval the KKT conditions leave it in.
- A solve that has not met `tol` after $10^6$ iterations is refused, naming the count. A smaller $C$ converges faster.

**Known divergences from LIBSVM**, and so from scikit-learn:
- No shrinking. Shrinking changes the order of the steps, not the problem.
- The kernel is held in float64. LIBSVM caches $Q$ in float32, so its gradient carries about seven significant digits.

Both solvers stop at a KKT gap of `tol`, not at the optimum, so at the default `tol` their decision values agree only to about `tol`: up to $10^{-3}$ on the parity sets. Assignments are identical there. Solved to `tol` $= 10^{-9}$, both reach the same optimum and the values agree to float32's precision (§7).

---

## 4. Assignment

Each pair votes: $f_{ij}(x) > 0$ is a vote for class $i$, anything else a vote for class $j$. The class with the most votes is assigned, and **a tied vote goes to the first class** in the order of §2. Both rules are LIBSVM's.

The **decision values** are reported one column per pair, in the order $(0,1), (0,2), \dots, (0,N{-}1), (1,2), \dots$. That is scikit-learn's `decision_function_shape="ovo"`, except for two classes, where scikit-learn negates the single column so that positive means $C_1$. Here positive always means the pair's first class.

There is no probability output. scikit-learn's `probability=True` fits Platt scaling by an internal, randomised cross-validation, and an estimate that changes with a seed is not reported.

---

## 5. Reported quantities

- **Classification:** the confusion matrices and metrics of `classification.md`, for calibration (the model refitted on every sample, `metrics-and-validation.md` §9), fold zero's held-out rows (fold zero's model) and the cross-validated set.
- **Panels:** the scores, loadings, $T^2$, SPE and their limits are the PCA front end's, as for LDA and kNN.
- **Model:** $\bar{x}$, the loadings, the kernel, $C$, the $\gamma$ used, and for each pair its support vectors in score space, their $a_k y_k$ and $\rho$.

---

## 6. Cross-validation and export

Each fold's model is fitted on that fold's training rows of its own preprocessed array, its $\gamma$ included when it is unset: the width is the fold's.

The model artifact, the JSON model and the Python snippet carry everything a decision needs: $\bar{x}$, the loadings, the kernel and $\gamma$, and every pair's support vectors, coefficients and $\rho$. As with kNN (`knn.md` §6), the decision needs the whole preprocessed spectrum, so the foldable tail of the chain travels as an explicit affine map (`model-export.md` §6).

---

## 7. Parity

scikit-learn `PCA(svd_solver="full")` on the centred matrix, then `SVC(C=1, gamma="scale", shrinking=False, decision_function_shape="ovo")` on its scores, with the linear and the RBF kernel. The classes are the terciles of `lda.md` §7, on corn, gasoline and Tecator, with $A = 5$. The quantities compared are:
- the assigned class indices at the default `tol`, which must be identical;
- the decision values solved to `tol` $= 10^{-9}$ by both, within the tolerance of LIBSVM's float32 kernel cache;
- the decision values at the default `tol`, recorded as the stopping-rule divergence of §3 after the assignments are checked identical and the values within $10\,$`tol` of the reference.
