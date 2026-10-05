# LDA — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md) for the front end; [`classification.md`](classification.md) for class order, the confusion matrix and every metric; [`pls-da.md`](pls-da.md) §5 for the argmax rule.

---

## 1. PCA-LDA, and why

Fisher's linear discriminant needs the pooled within-class covariance of its inputs to be invertible. A spectrum has hundreds of strongly collinear variables and usually more variables than samples, so that covariance is singular on the raw matrix. **The discriminant is therefore fitted to the first $A$ principal-component scores**, and $A$ (`n_components`) is the user's parameter. Every LDA here is PCA-LDA. There is no raw-variable form.

---

## 2. Model

The classes, $C_0 \dots C_{N-1}$ in Unicode order, come from a metadata column, as in `classification.md` §1. Every class must have at least one calibration sample.

1. Centre: $\bar{x}$ is the column mean of the calibration rows. Like PLS's centring of $y$, this is the estimator's own step, so a pipeline centring node above LDA is harmless but not required.
2. Project: a PCA with $A$ components of $X - \bar{x}$, exactly `pca.md`. $T = (X - \bar{x}) P$.
3. Class means in score space: $\mu_k$, the mean of $T$'s rows of class $k$.
4. Pooled within-class covariance: $S = \frac{1}{n - N} \sum_i (t_i - \mu_{c_i})(t_i - \mu_{c_i})^\top$. This needs $n - N \ge A$, and fewer degrees of freedom are refused, naming them.
5. Priors: $\pi_k = n_k / n$, the calibration proportions.

---

## 3. Discriminant

$$\delta_k(x) = t^\top S^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top S^{-1} \mu_k + \log \pi_k, \qquad t = (x - \bar{x}) P$$

This is linear in $x$. With $B = P S^{-1} M^\top$, a $p \times N$ matrix whose column $k$ is $P S^{-1} \mu_k$, and intercepts $c_k = -\tfrac{1}{2} \mu_k^\top S^{-1} \mu_k + \log \pi_k$:

$$\delta(x) = (x - \bar{x}) B + c$$

---

## 4. Assignment

The largest discriminant wins, ties going to the first class, as `pls-da.md` §5 has it for three or more classes. This holds for two classes as well: LDA has no 0.5 threshold.

---

## 5. Reported quantities

- **Classification:** the confusion matrices and metrics of `classification.md`, for calibration (the model refitted on every sample, `metrics-and-validation.md` §9), fold zero's held-out rows (fold zero's model) and the cross-validated set.
- **Panels:** the scores, loadings, $T^2$, SPE and their limits shown beside the classification are the **PCA front end's**. They describe the space the discriminant works in, not the discriminant.
- **Model:** $B$, $c$ and $\bar{x}$ are stored as the result's `coefficient_matrix`, `y_means` and `x_mean`.

---

## 6. Export

$\delta$ is linear in $x$, so LDA exports exactly as a multi-class PLS-DA does (`docs/model-export.md`): each column of $B$ folds through the foldable tail of the chain, the intercepts carry $c$ and the centring, and the snippet assigns by argmax.

---

## 7. Parity

scikit-learn `PCA(svd_solver="full")` on the centred matrix, followed by `LinearDiscriminantAnalysis()`, with the default `svd` solver and empirical priors, on its scores. The fixture's classes are the target's terciles for corn (`moisture`), gasoline (`octane`) and Tecator (`fat`): `"low"` below the first tercile, `"high"` from the second, and `"mid"` between. $A = 5$. The two quantities compared are:
- the decision function, scikit-learn's `decision_function`, which is $\delta$;
- the assigned class indices.
