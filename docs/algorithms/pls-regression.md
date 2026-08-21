# PLS regression — algorithm specification

Status: **normative**. This document fixes the conventions the implementation must follow. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

This is the kernel the project is judged on. A second implementer should be able to reproduce our numbers from this document alone.

Companion documents: [`pca.md`](pca.md) for the shared conventions on centring, signs and diagnostics; `metrics-and-validation.md` for RMSECV and the cross-validation protocol.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | Predictor matrix, $n \times p$ (samples × variables) |
| $y$ | Response vector, $n \times 1$ |
| $A$ | Number of latent variables retained |
| $W$ | Weights, $p \times A$, columns unit length |
| $T$ | X-scores, $n \times A$ |
| $P$ | X-loadings, $p \times A$ |
| $q_a$ | y-loading for component $a$ (scalar in PLS1) |
| $R$ | Rotations (`W*`), $p \times A$ |
| $b$ | Regression coefficients, $p \times 1$ |
| $E_a, f_a$ | X- and y-residuals after $a$ components |

---

## 2. Variant: NIPALS

**NIPALS.** Chosen over SIMPLS for three reasons:

1. **PLS1 needs no inner iteration.** With a single response the weight vector has a closed form per component, so there is no convergence tolerance and no iteration cap — nothing whose numerical settings could silently differ between our results and a reference implementation.
2. **It is the formulation most chemometrics literature and teaching material states**, which matters for a project whose audience is research and academia and whose credibility rests on a published parity report.
3. **Its intermediate quantities are the ones analysts read.** Weights, loadings and per-component deflation are exposed directly rather than reconstructed.

**What SIMPLS would change.** For a **single response the two coincide**: identical X-score subspace, identical regression coefficients, identical predictions. They diverge only for multiple responses, where SIMPLS deflates the cross-product matrix rather than $X$ itself. Since v1 models one response at a time (§10), the choice is presently free of consequence for predictions — but it fixes the weights and loadings a user sees, and those must be reproducible.

This has a useful corollary for parity: **a SIMPLS-based reference such as R `mdatools` is a valid comparison target for coefficients and predictions**, but not for raw weights and loadings.

---

## 3. What PLS does and does not do

**PLS fits the matrices it is given. It performs no centring and no scaling of its own**, exactly as in [`pca.md` §2](pca.md). Centring and scaling are pipeline nodes and therefore appear in the lineage.

Two consequences:

- **`scikit-learn` differs twice over.** `PLSRegression` centres unconditionally *and* defaults to `scale=True`. Parity comparisons must pass pre-centred data and set `scale=False`.
- **PLS on uncentred data is legal here and almost always wrong.** The application warns when a PLS node has no centring step upstream.

The response is centred by the same rule: if $y$ was not centred, the model has no intercept term to absorb the offset and the first component spends itself on the mean.

---

## 4. Algorithm (PLS1)

Initialise $E_0 = X$, $f_0 = y$. For $a = 1 \dots A$:

$$w_a = \frac{E_{a-1}^{\top} f_{a-1}}{\lVert E_{a-1}^{\top} f_{a-1} \rVert} \qquad t_a = E_{a-1} w_a$$

$$p_a = \frac{E_{a-1}^{\top} t_a}{t_a^{\top} t_a} \qquad q_a = \frac{f_{a-1}^{\top} t_a}{t_a^{\top} t_a}$$

**Deflation — both blocks:**

$$E_a = E_{a-1} - t_a p_a^{\top} \qquad f_a = f_{a-1} - q_a t_a$$

Deflating $y$ as well as $X$ is the classical NIPALS formulation and is what `scikit-learn` does. It is stated explicitly because implementations that skip it exist, and the weights differ from the second component onward.

**Properties the implementation must preserve**, and which make useful assertions:

- $\lVert w_a \rVert = 1$
- $t_a^{\top} t_b = 0$ for $a \neq b$ — the X-scores are mutually orthogonal
- $w_a^{\top} w_b$ is *not* generally zero; the weights are not orthogonal, only the scores are

**Stopping.** Iteration stops at $A$ components, or earlier if $\lVert E_{a}^{\top} f_{a} \rVert$ falls below $\varepsilon^{1/2} \cdot \lVert E_0^{\top} f_0 \rVert$, which means the response has been exhausted. Stopping early is reported, never silent.

---

## 5. Rotations and regression coefficients

The weights act on deflated matrices, so they cannot be applied directly to raw $X$. The rotations can:

$$R = W (P^{\top} W)^{-1} \qquad T = X R$$

$P^{\top}W$ is upper triangular with unit diagonal and therefore always invertible, so no pseudo-inverse is needed. Coefficients on the matrix as fitted:

$$b = R\,q \qquad \hat{y} = X b$$

with $q = (q_1 \dots q_A)^{\top}$.

**$b$ is invariant to the sign convention** (§6): flipping a component negates $w_a$, $t_a$, $p_a$ and $q_a$ together, leaving $R q$ unchanged. Coefficients and predictions can therefore be compared against any reference without sign alignment — only scores, loadings and weights need it.

---

## 6. Sign convention

For each component $a$, let $j^{*} = \arg\max_j |w_{ja}|$. If $w_{j^{*}a} < 0$, negate $w_a$, $t_a$, $p_a$ and $q_a$ together. Ties broken by smallest index.

Keyed on the **weights**, since in PLS the weight vector is the spectrum-shaped quantity carrying the correlation structure with the response. Consistent with [`pca.md` §5](pca.md), which keys on loadings for the same reason.

Comparisons of $W$, $T$ and $P$ align signs by inner product with the reference, never by comparing absolute values.

---

## 7. Prediction, and export to original units

### Within the application

$$T_{\text{new}} = X_{\text{new}} R \qquad \hat{y}_{\text{new}} = X_{\text{new}} b$$

$X_{\text{new}}$ must pass through the identical preprocessing chain with **parameters estimated on the calibration set** — the calibration mean and standard deviation, never the new set's. This is a property of the pipeline executor, stated here because getting it wrong yields plausible and entirely wrong predictions.

### Folding preprocessing into exported coefficients

`PROPOSAL.md` §9 promises a portable model: a JSON coefficient vector and a standalone Python snippet. That requires knowing **which preprocessing steps can be folded into $b$ and which must be re-executed**.

**Foldable — affine in the variables, with parameters fixed at calibration time:**

| Step | Folds as |
| --- | --- |
| Mean centring | shifts the intercept |
| Autoscaling | $b_j \rightarrow b_j / s_j$ |
| Range selection | drops coefficients |
| Savitzky–Golay, derivatives | linear convolution; folds as a banded matrix $C$, giving $b \rightarrow C^{\top} b$ |

For centring by $\bar{x}$ and scaling by $s$, with $y$ centred by $\bar{y}$ and scaled by $s_y$:

$$b^{\text{orig}}_j = \frac{s_y\, b_j}{s_j} \qquad b_0 = \bar{y} - \sum_{j} \bar{x}_j\, b^{\text{orig}}_j \qquad \hat{y} = b_0 + X^{\text{raw}} b^{\text{orig}}$$

**Not foldable — the transform depends on the sample being predicted:**

SNV divides each spectrum by *its own* standard deviation. MSC regresses each spectrum against a stored reference. Neither is a fixed linear map on $X$, so both must be re-executed at prediction time and shipped as part of the exported model rather than absorbed into the coefficients.

An exported model therefore carries a **residual preprocessing chain** plus a coefficient vector, not always a bare coefficient vector. A pipeline whose preprocessing is entirely foldable exports as pure coefficients; one containing SNV or MSC does not. **The export format must represent both cases**, and the exported prediction must match the in-application prediction within the stated tolerance, verified in CI.

---

## 8. VIP scores

With $\mathrm{SS}_a = q_a^{2}\, (t_a^{\top} t_a)$, the sum of squares of $y$ explained by component $a$:

$$\mathrm{VIP}_j = \sqrt{\; p \cdot \frac{\sum_{a=1}^{A} \mathrm{SS}_a \left( \dfrac{w_{ja}}{\lVert w_a \rVert} \right)^{2}}{\sum_{a=1}^{A} \mathrm{SS}_a} \;}$$

Since weights are already unit length (§4), $\lVert w_a \rVert = 1$; it is written out because implementations that skip normalisation exist.

**Normalisation.** This form satisfies $\sum_{j=1}^{p} \mathrm{VIP}_j^{2} = p$, so the mean squared VIP is exactly 1. That property — and nothing more fundamental — is the origin of the "VIP greater than 1" selection rule of thumb. The identity is a cheap and effective unit test.

VIP depends on $A$: it is a property of the fitted model, not of the data. Reporting VIP without the component count is meaningless.

Multiple published variants exist. This one is the standard Wold form and is what the parity report compares.

---

## 9. Diagnostics

$T^2$ and SPE follow [`pca.md` §7–8](pca.md), computed on the PLS X-scores and X-residual, with two substitutions:

- $\lambda_a = \dfrac{t_a^{\top} t_a}{n-1}$ — valid because the NIPALS X-scores are orthogonal (§4).
- The X-residual is $E_A$, so $\mathrm{SPE}_i = \lVert E_{A,i} \rVert^{2}$.

**The Jackson–Mudholkar SPE limit does not transfer.** It is derived from the eigenvalues of the discarded PCA subspace, and PLS components are not eigenvectors of the covariance of $X$ — there is no residual eigenvalue sequence to sum. For PLS the SPE limit uses the $\chi^2$ moment match on the observed calibration residuals:

$$\mathrm{SPE}_{\alpha} = g\,\chi^{2}_{\alpha}(h), \qquad g = \frac{v}{2m}, \qquad h = \frac{2m^{2}}{v}$$

where $m$ and $v$ are the sample mean and variance of $\mathrm{SPE}_i$ over the calibration set. This difference from PCA is deliberate and must be stated wherever a PLS SPE limit is drawn.

---

## 10. PLS1 and PLS2

**v1 implements PLS1 only.** `PLSRegressionSpec` carries a single `target`, so one response is modelled at a time. Two responses mean two models, each with its own optimal component count — which is usually what an analyst wants anyway.

**PLS2** requires an inner iteration per component, since the weights and the y-loading are mutually dependent:

> initialise $u$ from a column of $F$; repeat $w = F^{\top}u / \lVert \cdot \rVert$, $t = Ew$, $q = F^{\top}t/(t^{\top}t)$, $u = Fq/(q^{\top}q)$ until $t$ converges.

That introduces a convergence tolerance, an iteration cap, and a dependence on which column seeds $u$ — three numerical settings that must be specified and matched for parity. **Deferred to post-1.0**, and when it lands it needs its own section here, not an extension of §4.

For a single response the two produce identical results, so nothing about the v1 numbers depends on this deferral.

---

## 11. Choosing and reporting the number of components

$A$ is a **user parameter**. There is no automatic selection in v1.

The model reports RMSECV as a function of component count from 1 to $A_{\max}$, so the user can see the curve and choose. Everything else — coefficients, predictions, VIP, diagnostics — is reported for the chosen $A$ only.

$A$ appears in the model name, in the lineage, and in every export. A reported metric without its component count is incomplete.

Selection heuristics (first minimum, one-standard-error rule, Wold's R) are a workflow question, deliberately not fixed here.

---

## 12. Rank, missing values, numerical policy

- **$A$ may not exceed** $\min(n-1, p)$, nor the point where the response is exhausted (§4). Exceeding it is an **error naming both numbers**, never silent truncation.
- **Missing values are rejected**, as in [`pca.md` §10](pca.md). Handle them upstream and visibly.
- **Computation in float64**, whatever the storage dtype. The caller's array is never modified.
- **No random state.** NIPALS PLS1 is deterministic and takes no seed. Cross-validation seeds belong to the split, not to the estimator.

---

## 13. Reported quantities

| Name | Symbol | Defined in |
| --- | --- | --- |
| `weights` | $W$ | §4 |
| `x_scores` | $T$ | §4 |
| `x_loadings` | $P$ | §4 |
| `y_loadings` | $q$ | §4 |
| `rotations` | $R$ | §5 |
| `coefficients` | $b$ | §5 |
| `coefficients_original_units` | $b^{\text{orig}}, b_0$ | §7 |
| `vip` | $\mathrm{VIP}_j$ | §8 |
| `hotelling_t2`, `hotelling_t2_limit` | | §9 |
| `spe`, `spe_limit` | | §9 |
| `n_components` | $A$ | §11 |

Regression metrics (RMSEC, RMSECV, RMSEP, $R^2$, bias) are defined in `metrics-and-validation.md`, not here.

---

## 14. Known divergences from other packages

| Area | Here | Elsewhere |
| --- | --- | --- |
| Centring and scaling | Explicit pipeline steps | `sklearn` centres always, scales by default |
| Variant | NIPALS | `mdatools` uses SIMPLS — identical for one response, different weights and loadings |
| y-deflation | Yes | Some implementations omit it; weights differ from component 2 onward |
| Sign rule | Largest-magnitude weight | Varies; often unspecified |
| VIP | Wold form, $\sum \mathrm{VIP}^2 = p$ | Several published variants |
| SPE limit | $\chi^2$ moment match | PCA-style Jackson–Mudholkar, which does not apply to PLS |
| Multiple responses | Not supported in v1 | PLS2 common elsewhere |

`scikit-learn`'s `coef_` orientation and scaling have changed across releases. Parity must pin the version and assert the orientation rather than assume it.

---

## 15. Deliberately not specified here

- RMSECV, fold assignment and metric definitions — `metrics-and-validation.md`.
- Tolerances and claim tiers — the parity harness.
- PLS-DA — a separate document when it lands; it is a classification wrapper around this algorithm, not a variant of it.
- Variable selection built on VIP — post-1.0.
