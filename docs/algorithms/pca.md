# PCA — algorithm specification

Status: **normative**. This document fixes the conventions the implementation must follow. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Every quantity a PCA model can display in the application is defined here. If a quantity is not defined here, it is not reported.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $X$ | Data matrix, $n \times p$. Rows are samples, columns are variables. Never transposed silently. |
| $n$ | Number of samples |
| $p$ | Number of variables |
| $a$ | Number of retained components |
| $r$ | Rank available for decomposition |
| $T$ | Scores, $n \times a$ |
| $P$ | Loadings, $p \times a$ |
| $\lambda_k$ | Variance associated with component $k$ |

---

## 2. What PCA does and does not do

**PCA fits the matrix it is given. It performs no centring and no scaling of its own.**

Centring and scaling are explicit pipeline steps (`MeanCentre`, `Autoscale`) and appear as nodes in the pipeline graph. This follows from the project's central rule that the pipeline is the complete record of what was done: preprocessing hidden inside an estimator would be absent from the recipe and therefore absent from the lineage.

Two consequences to be aware of:

- **This differs from `scikit-learn`.** `sklearn.decomposition.PCA` centres internally and unconditionally. Parity comparisons must therefore feed it an already-centred matrix, where its own centring is a no-op. Feeding raw data to both and comparing is an invalid test.
- **Fitting PCA on uncentred data is legal here and usually wrong.** The first component then largely captures the mean spectrum. The application must warn when a PCA node has no centring step upstream. The warning is a UI affordance, not a silent correction.

Recommended default for spectral data: mean centring, no scaling. Autoscaling is appropriate when variables carry different units or wildly different variances, which is uncommon for a single spectral block.

---

## 3. Algorithm

**Singular value decomposition**, computed on the matrix as supplied:

$$X = U \Sigma V^{\top}$$

with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$.

**Why SVD rather than NIPALS.** SVD is deterministic, needs no convergence tolerance, is numerically stable for near-collinear spectra, and returns the full component set in one pass. NIPALS earns its place in two situations: extracting a few components from a matrix too large to decompose whole, and tolerating missing values. Neither applies — the performance envelope keeps $X$ in memory, and missing values are rejected (§9).

**Randomised or truncated SVD is not used.** It is not deterministic without a fixed seed, and a decomposition whose result depends on a random draw is not something a parity report can stand behind.

The decomposition is computed by LAPACK through `numpy.linalg.svd` with `full_matrices=False`.

---

## 4. Outputs

$$P = V_{[:,\,1:a]} \qquad T = X P \qquad \lambda_k = \frac{\sigma_k^{2}}{n-1}$$

- **Loadings** $P$ — orthonormal columns, $P^{\top}P = I_a$.
- **Scores** $T$ — equal to $U_{[:,1:a]} \Sigma_{[1:a]}$, but computed as $XP$ so that the same code path projects new samples.
- **$\lambda_k$** — the variance captured by component $k$. The divisor is $n-1$, matching the sample-variance convention used everywhere else in the project.

**All $r$ singular values are retained on the fitted model**, not only the first $a$. Both the explained-variance denominator (§6) and the SPE limit (§8) require the residual eigenvalues, and recomputing them later would mean re-decomposing.

### Projecting new samples

$$T_{\text{new}} = X_{\text{new}} P$$

$X_{\text{new}}$ must have been through the identical preprocessing chain, using parameters estimated on the calibration set — the calibration mean, not the new set's mean. This is a property of the pipeline executor; it is stated here because getting it wrong produces plausible and entirely wrong scores.

---

## 5. Sign convention

The signs of $U$ and $V$ are jointly arbitrary: negating column $k$ of both leaves $X$ unchanged. A convention is required or results are irreproducible across platforms and LAPACK builds.

**Rule.** For each component $k$, let $j^{*} = \arg\max_j |P_{jk}|$, the position of the largest-magnitude loading. If $P_{j^{*}k} < 0$, negate both $P_{[:,k]}$ and $T_{[:,k]}$. Ties in $|P_{jk}|$ are broken by taking the smallest index $j$.

The rule is applied to the **loadings**, because for spectral data the loading is the interpretable, spectrum-shaped vector, and its orientation is what an analyst reads.

This differs from `scikit-learn`'s default `svd_flip(..., u_based_decision=True)`, which decides from $U$. Signs may therefore differ from `sklearn` per component even when both are correct.

**Comparisons must be sign-invariant.** Before comparing a component against a reference, align it: if $\langle P_{[:,k]}^{\text{ours}},\, P_{[:,k]}^{\text{ref}} \rangle < 0$, negate both $P_{[:,k]}$ and $T_{[:,k]}$ of one side. Comparing absolute values instead is not acceptable — it would pass a result whose loading and score signs disagree with each other, which is a real error.

---

## 6. Explained variance

$$\mathrm{EV}_k = \frac{\lambda_k}{\sum_{j=1}^{r} \lambda_j} \qquad \mathrm{CEV}_a = \sum_{k=1}^{a} \mathrm{EV}_k$$

**The denominator sums over all $r$ components, not the $a$ retained ones.** Normalising by the retained components would make cumulative explained variance always reach 100%, which is both useless and a mistake seen in the wild.

The denominator equals the total variance of the matrix PCA was fitted to:

$$\sum_{j=1}^{r} \lambda_j = \frac{1}{n-1}\lVert X \rVert_F^{2}$$

For centred $X$ this is the sum of the column variances. For uncentred $X$ it is not, which is a further reason the application warns about missing centring: explained variance is reported against a total that includes the mean.

Reported as a fraction internally; formatted as a percentage in the UI.

---

## 7. Hotelling's $T^{2}$

Distance within the model plane.

$$T^{2}_i = \sum_{k=1}^{a} \frac{t_{ik}^{2}}{\lambda_k}$$

Equivalently $t_i \Lambda^{-1} t_i^{\top}$ with $\Lambda = \mathrm{diag}(\lambda_1 \dots \lambda_a)$. Components are weighted by their variance, so a large score on a minor component contributes more than the same score on a major one.

### Confidence limits

Two limits, because they answer different questions. **Which one is drawn must be stated in the plot legend, never left implicit.**

**Calibration samples** — samples used to fit the model. Their scores are not independent of the model, so the exact limit is the beta distribution:

$$T^{2}_{\alpha} = \frac{(n-1)^{2}}{n} \, B_{\alpha}\!\left(\frac{a}{2},\ \frac{n-a-1}{2}\right)$$

**New samples** — projected onto an existing model:

$$T^{2}_{\alpha} = \frac{a\,(n^{2}-1)}{n\,(n-a)} \, F_{\alpha}(a,\ n-a)$$

$\alpha$ defaults to 0.05, giving a 95% limit. $n$ and $a$ are always those of the **calibration** model, including when the limit is applied to new samples.

The two converge as $n$ grows. They differ noticeably for small $n$, which is the common case in chemometrics, so both are implemented rather than approximating one with the other.

Requires $n > a + 1$ for the beta form and $n > a$ for the F form. Outside those, no limit is defined and none is drawn.

---

## 8. Squared prediction error (SPE, also written $Q$)

Distance from the model plane.

$$\hat{x}_i = t_i P^{\top} \qquad e_i = x_i - \hat{x}_i \qquad \mathrm{SPE}_i = \lVert e_i \rVert^{2} = \sum_{j=1}^{p} e_{ij}^{2}$$

Reported as the sum of squares, **not** the mean and not the root. Other packages report the root or the mean; converting is trivial but silent disagreement is not.

### Confidence limit — Jackson–Mudholkar

With $\theta_m = \sum_{k=a+1}^{r} \lambda_k^{m}$ for $m \in \{1,2,3\}$ and $h_0 = 1 - \dfrac{2\theta_1\theta_3}{3\theta_2^{2}}$:

$$\mathrm{SPE}_{\alpha} = \theta_1 \left[ \frac{c_{\alpha}\sqrt{2\theta_2 h_0^{2}}}{\theta_1} + 1 + \frac{\theta_2 h_0 (h_0-1)}{\theta_1^{2}} \right]^{1/h_0}$$

where $c_{\alpha}$ is the standard normal deviate at the upper $\alpha$ tail ($c_{0.05} \approx 1.645$).

**This is why all $r$ eigenvalues are retained (§4).** $\theta_m$ sums over the *discarded* components. A model that kept only the first $a$ eigenvalues cannot compute its own SPE limit.

**Box's $\chi^2$ approximation is not used** as the default, but is recorded here because other packages use it and the difference will show up in comparisons:

$$\mathrm{SPE}_{\alpha} = g\,\chi^{2}_{\alpha}(h), \qquad g = \frac{\theta_2}{\theta_1}, \qquad h = \frac{\theta_1^{2}}{\theta_2}$$

**Degenerate case.** When $a = r$ the residual is zero by construction: $\mathrm{SPE}_i = 0$ for all $i$, $\theta_m = 0$, and no limit exists. Report SPE as exactly zero and draw no limit. Do not report a limit computed from an empty sum.

---

## 9. Rank, and $n\_components$ beyond it

Available rank:

$$r = \min(n - 1,\ p) \quad \text{when the data are centred}, \qquad r = \min(n,\ p) \quad \text{otherwise}$$

Centring removes one degree of freedom, which is why the centred case loses a component. Since PCA does not centre (§2), the implementation cannot infer which case applies from the matrix alone; it takes the effective rank from the SVD, counting singular values above

$$\text{tol} = \left(\max(n,p)\,\varepsilon_{64} + \varepsilon_{\text{data}}\right)\sigma_1$$

**Two error terms, added, because they are two different things.** The first is the rounding the decomposition accumulates: it runs in float64 whatever it was handed, and $\max(n,p)$ is the usual growth factor. The second is the matrix not being the matrix that was meant — a perturbation of the *input*, which by Weyl's inequality moves each singular value by at most the norm of the perturbation, $\varepsilon_{\text{data}}\,\sigma_1$. **The data term carries no dimension factor**, because a perturbation of $X$ is not an accumulation over its entries.

$\varepsilon_{\text{data}}$ defaults to $\varepsilon_{64}$, which leaves the tolerance where it has always been to within one part in $\max(n,p)$. A caller that knows the data arrived less precisely says so: the workbench's executor reads every array back through a float32 store (#83), so it passes $\varepsilon_{32}$.

This is not a refinement for its own sake. Without it, a centred matrix that has been through the store reports **one rank too many** — its columns no longer sum to exactly zero, and the SVD finds a spurious hundredth singular value that a float64 tolerance admits (#101). Scaling the data term by $\max(n,p)$ as well overshoots in the other direction, and by much more: on Tecator it discards 33 genuine components and reports rank 66. The two are far apart — measured, the spurious value is $2.8\times10^{-7}$ and the smallest genuine one $1.3\times10^{-4}$ — so the threshold sits in a wide gap rather than on a knife edge.

**If $a > r$, raise an error naming both numbers. Do not silently return fewer components.** Silent truncation makes downstream array shapes unpredictable and hides a user mistake — asking for 40 components from 30 samples is a misunderstanding worth surfacing, not rounding away.

---

## 10. Missing values

**Rejected.** PCA requires a complete matrix. A `NaN` anywhere raises an error naming the offending rows and columns.

Missing data is handled upstream, explicitly: exclude the sample, exclude the variable, or add an imputation step to the pipeline. Each of those appears in the recipe and therefore in the lineage.

Imputing inside PCA — or quietly using NIPALS's tolerance for gaps — would mean two datasets that differ in their missing entries produce results that look equally trustworthy, with nothing in the record showing that one was partly invented.

---

## 11. Numerical policy

- Computation in **float64**, regardless of the storage dtype. Data may be held as float32 for memory; it is promoted before decomposition and results are returned as float64.
- **The caller's array is never modified.**
- **No random state.** SVD is deterministic, so PCA takes no seed. Any future component that needs randomness must take an explicit seed.
- Eigenvalues below the §9 rank tolerance are treated as exactly zero when forming $\theta_m$, to keep the SPE limit from being driven by numerical noise.

---

## 12. Reported quantities

| Name | Symbol | Defined in |
| --- | --- | --- |
| `scores` | $T$ | §4 |
| `loadings` | $P$ | §4 |
| `eigenvalues` | $\lambda_k$ | §4 |
| `explained_variance` | $\mathrm{EV}_k$ | §6 |
| `cumulative_explained_variance` | $\mathrm{CEV}_a$ | §6 |
| `hotelling_t2` | $T^{2}_i$ | §7 |
| `hotelling_t2_limit` | $T^{2}_{\alpha}$ | §7 |
| `spe` | $\mathrm{SPE}_i$ | §8 |
| `spe_limit` | $\mathrm{SPE}_{\alpha}$ | §8 |

`explained_variance` maps to the field of the same name on `Metrics` in the schema; the diagnostics travel with the fitted model artifact.

---

## 13. Known divergences from other packages

Recorded so the parity report can classify them as *differs by documented convention* rather than as failures.

| Area | Here | Elsewhere |
| --- | --- | --- |
| Centring | Explicit pipeline step | `sklearn` centres internally and unconditionally |
| Sign rule | Largest-magnitude **loading** | `sklearn` decides from $U$ by default |
| SPE scale | Sum of squares | Some packages report the mean or its root |
| SPE limit | Jackson–Mudholkar | Some packages use Box's $\chi^{2}$ |
| SPE limit when $h_0 \le 0$ | Used as computed (see [#71](https://github.com/millermuttu/Chemometrics-Workbench/issues/71)) | R `mdatools` clamps $h_0$ to `0.001` |
| $T^2$ limit | Beta for calibration, F for new samples | Some packages use the F form for both |
| Rank overflow | Error | Some packages silently truncate |

---

## 14. Parity references

Comparison targets, all reproducible without a commercial licence:

- `scikit-learn` `PCA` — scores, loadings, explained variance, on pre-centred input.
- R `prcomp(center = FALSE)` and `mdatools` — scores, loadings, and the $T^{2}$ and SPE limits, which `sklearn` does not provide.
- Published values for the reference datasets.

Tolerances and the claim tiers are defined by the parity harness, not here.

---

## 15. Deliberately not specified here

- Tolerance values and comparison mechanics — parity harness.
- How the number of components is chosen — a UI and workflow question, not an algorithmic one.
- Contribution plots — deferred until there is a diagnostic view that needs them.
- Robust or kernel PCA — out of scope for 1.0.
