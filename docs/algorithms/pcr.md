# PCR — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md), whose decomposition this is built on and does not restate; [`pls-regression.md`](pls-regression.md), whose reported quantities, centring rule and export PCR shares; [`metrics-and-validation.md`](metrics-and-validation.md) for folds and the cross-validation protocol.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | $n \times p$ preprocessed, centred calibration matrix |
| $y$ | Centred response, length $n$ |
| $A$ | Number of principal components regressed on |
| $P$ | $p \times A$ PCA loadings (`pca.md` §4), unit-length and orthogonal |
| $T = XP$ | $n \times A$ scores, mutually orthogonal |
| $q$ | Length-$A$ regression coefficients on the scores |
| $b$ | Length-$p$ regression vector |

---

## 2. Model

A PCA of $X$ with $A$ components, exactly `pca.md`: an SVD of the matrix as supplied, the same rank tolerance and the same sign convention. Then $y$ is regressed on the scores by ordinary least squares, with no intercept, because both blocks are centred:

$$q = (T^\top T)^{-1} T^\top y, \qquad b = P q, \qquad \hat{y} = X b$$

**Centring.** As for PLS (`pls-regression.md` §3), $X$ is centred by a pipeline node and $y$ by the estimator, which adds the training mean back to every prediction. A PCR with no centring node above it gets the same `pls_without_centring` warning as a PLS.

---

## 3. The scores are orthogonal, so the components nest

$T^\top T$ is diagonal, so each coefficient is independent of the others:

$$q_a = \frac{t_a^\top y}{t_a^\top t_a}$$

The first $a$ coefficients of an $A$-component model are therefore exactly the $a$-component model. Within a fold, the RMSECV curve for $A = 1 \dots A_{\max}$ comes from one fit, as for NIPALS (`pls-regression.md` §9).

The components are chosen to explain $X$, not $y$. A component that explains a great deal of $X$ and nothing of $y$ is still retained at its place in the order. This is PCR's known weakness against PLS and is not corrected here: choosing components by their correlation with $y$ is a different method.

---

## 4. Reported quantities

The same quantities as PLS (`pls-regression.md` §13), with these definitions:

| Quantity | PCR's definition |
| --- | --- |
| Scores, loadings, rotations | $T$, $P$, $P$ (the rotations are the loadings) |
| `y_loadings` | $q$ |
| Coefficients | $b = Pq$ |
| $T^2$, SPE and their limits | The PCA's, `pca.md` §7 and §8, Jackson–Mudholkar SPE limit included |
| Metrics | RMSEC, R², SEC, RMSECV, Q², RMSEP and SEP, by `metrics-and-validation.md` |

---

## 5. Explained variance

X: the PCA's, `pca.md` §6. Y, per component: $q_a^2 \, t_a^\top t_a \,/\, y^\top y$. Because the scores are orthogonal these add up, and the cumulative Y variance at $A$ is the calibration $R^2$.

---

## 6. VIP is not reported

VIP weights each variable by its contribution to components chosen *for* $y$. PCR's components are chosen without $y$, so the quantity has no PCR meaning, and the result carries none. The screen shows the coefficients alone.

---

## 7. Export

$b$ is a fixed vector, so a PCR exports exactly as a PLS does (`docs/model-export.md`): foldable steps fold into $b$, and the residual chain is carried.

---

## 8. Parity references

scikit-learn `PCA(svd_solver="full")` followed by `LinearRegression` on its scores, on the centred matrix of corn, gasoline and Tecator with $A = 5$. Two quantities are compared:
- the coefficient vector $b$, given by `components_.T @ coef_`;
- the calibration predictions in the response's original units.

Both are sign-invariant: flipping a component negates $t_a$, $p_a$ and $q_a$ together. They are therefore compared without alignment.

---

## 9. Bootstrap intervals

The coefficients carry bootstrap percentile intervals exactly as `pls-regression.md` §16 specifies for PLS: each resample refits the chain and the PCR, and the folded coefficients are measured. There is no VIP interval, because there is no VIP (§6).
