# Outlier diagnostics — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md) §7 and §8 for Hotelling's $T^2$ and $Q$ (SPE) and their limits; [`pls-regression.md`](pls-regression.md) and [`pcr.md`](pcr.md) for the scores and the fitted values used here.

---

## 1. What this is

A fitted PCA, PLS or PCR already reports $T^2$ and $Q$ with their limits. This adds three diagnostics, all computed on the calibration rows of the fitted model:

| Diagnostic | Of | Asks |
| --- | --- | --- |
| Leverage $h_i$ (§2) | PCA, PLS, PCR | How far is the sample from the centre of the score space? |
| Studentised residual $r_i$ (§3) | PLS, PCR | Is the sample's reference value out of line with its spectrum? |
| Robust distance $d_i^2$ (§4) | PCA, PLS, PCR | Is the sample far from the bulk, measured so the outliers cannot hide each other? |

It also adds a **flags table** (§5) that names, for each sample, every rule it breaks.

**Diagnostics flag; they never remove.** A flagged sample stays in the model until the user excludes it. Excluding it creates a derived dataset version (`PROPOSAL.md` §16, #279) and is never a pipeline step that drops rows on its own.

A PLS-DA, an LDA and a kNN are not diagnosed. Their response is a class, so a studentised residual of a dummy variable is not a meaningful quantity. A SIMCA has one model per class, and its distances are its own decision (`simca.md` §3).

All three are computed from the stored result: its scores, observed values and predictions. A result fitted before this document existed therefore gets the same numbers as a fresh one. They are served at `/results/{node}/outliers` rather than inside the result payload (#314). The robust distance is a search that takes about a second at 3,000 samples, and nothing else on the analysis tab should wait for it.

---

## 2. Leverage

The leverage of sample $i$ is the $i$-th diagonal element of the hat matrix of the design $Z = [\mathbf{1}, T]$, where $T$ holds the calibration scores of the $A$ retained components:

$$h_i = \left[ Z (Z^\top Z)^{-1} Z^\top \right]_{ii}$$

It is computed as $h_i = \sum_j Q_{ij}^2$ from the thin QR decomposition $Z = QR$, which needs neither centred nor orthogonal scores. For a centred model with orthogonal scores, which covers every model here, it reduces to

$$h_i = \frac{1}{n} + \sum_{a=1}^{A} \frac{t_{ia}^2}{t_a^\top t_a}.$$

$\sum_i h_i = A + 1$, so the mean leverage is $(A+1)/n$.

**Limit:** $h_i > 3 (A + 1) / n$, three times the mean.

A design whose intercept and scores are linearly dependent has no leverage, and the kernel refuses it.

---

## 3. Studentised residuals

For a PLS or PCR on one response, $e_i = y_i - \hat{y}_i$ is the calibration residual in the response's units. The **internally** studentised residual is

$$r_i = \frac{e_i}{s \sqrt{1 - h_i}}, \qquad s^2 = \frac{\sum_i e_i^2}{n - A - 1}.$$

$h_i$ is the leverage of §2. The scaling is exact here, not approximate: the calibration fit of PLS1 and of PCR is the least-squares fit of $y$ on $[\mathbf{1}, T]$, because the scores are orthogonal and each $q_a = t_a^\top y / t_a^\top t_a$. So $e$ has covariance $\sigma^2 (I - H)$ under that model.

- With $n - A - 1 \le 0$ there are no residual degrees of freedom. The residuals are then absent, not zero.
- A row with $h_i = 1$ is fitted exactly and has no studentised residual. It is reported as absent.

**Limit:** $|r_i| > 3$.

---

## 4. Robust distance

The $T^2$ of `pca.md` §7 measures distance with the calibration covariance, which the outliers themselves inflate. Several outliers together can pull the mean and inflate the covariance until none of them looks far away; this is called **masking**. The robust distance measures from the **Minimum Covariance Determinant** (MCD) estimate instead. That is the mean and covariance of the $h$ calibration rows whose covariance has the smallest determinant, with

$$h = \left\lceil \tfrac{1}{2}(n + A + 1) \right\rceil,$$

the largest breakdown the MCD allows. The search is FastMCD (Rousseeuw and Van Driessen, 1999):

1. Draw 500 random subsets of $A + 1$ rows from a generator seeded with **0**. A subset whose covariance is singular grows by one random row until it is not.
2. From each subset's mean and covariance, take the $h$ rows with the smallest Mahalanobis distance. Then apply two **C-steps**: refit on those $h$ rows and keep the $h$ nearest again. A C-step never increases the determinant.
3. Iterate the ten subsets with the smallest determinant until the support stops changing, at most 100 steps. Keep the one with the smallest determinant.

The raw estimate is then corrected and reweighted, as scikit-learn's `MinCovDet` does:

4. **Consistency.** With $c(p, \alpha) = \alpha \,/\, F_{\chi^2_{p+2}}\!\left(\chi^2_{p,\alpha}\right)$ (Croux and Haesbroeck 1999), the raw squared distances are divided by $c(A, h/n)$.
5. **Reweighting.** Rows whose corrected squared distance is below $\chi^2_{A,\,0.975}$ give the final location (their mean) and covariance (their maximum-likelihood covariance, times $c(A, 0.975)$).
6. $d_i^2$ is every calibration row's squared Mahalanobis distance from that location under that covariance.

With $A = 1$ the MCD is exact: it is the shortest window of $h$ sorted scores, centred on the mean of its two ends. No random search is used.

Fewer than $A + 2$ rows, or scores of rank below $A$, cannot support the estimate. The robust distances are then absent and the reason is stated.

**Limit:** $d_i^2 > \chi^2_{A,\,0.975}$, the same cut the reweighting uses.

---

## 5. The flags table

A sample is listed when it breaks at least one rule. For each listed sample, the table names every rule it breaks:

| Rule | Fires when |
| --- | --- |
| `t2` | $T^2_i$ above the Hotelling limit at the result's $\alpha$ (`pca.md` §7) |
| `q` | $Q_i$ above the SPE limit at the result's $\alpha$ (`pca.md` §8) |
| `leverage` | $h_i > 3(A+1)/n$ |
| `residual` | $\lvert r_i \rvert > 3$ (PLS and PCR only) |
| `robust` | $d_i^2 > \chi^2_{A,\,0.975}$ |

**Most of these are expected to fire on clean data.** At $\alpha = 0.05$, about one sample in twenty is beyond the $T^2$ limit by construction, and the same holds for each other limit at its own level. A flag says "look at this sample". It does not say "remove it". This is why exclusion is a deliberate act and not a default.

The table lists calibration rows only. Held-out rows are predictions, not part of the fit, and their $T^2$ and $Q$ are shown in the diagnostics panel as before.

---

## 6. Known divergences

| Area | Here | Elsewhere |
| --- | --- | --- |
| Leverage | Hat diagonal of $[\mathbf{1}, T]$, including the $1/n$ term | R `mdatools` reports $T^2$-based leverage without the intercept term; some packages scale it by $n - 1$ |
| Leverage limit | $3(A+1)/n$ | $2(A+1)/n$ (Belsley, Kuh and Welsch) is also common |
| Residuals | Internally studentised | Externally studentised (leave-one-out $s$) in some regression packages |
| MCD search | 500 random $(A+1)$-subsets, Rousseeuw and Van Driessen's starts | scikit-learn starts from 30 random $h$-subsets and, above 500 rows, pools nested subsets. On corn, gasoline and Tecator it stops at a larger determinant for most seeds (§7) |
| MCD above 500 rows | The same single-level search | FastMCD's nested subsets |

---

## 7. Parity

**Robust distance.** The reference is scikit-learn's own implementation of steps 2 to 6: `select_candidates`, which does its C-steps and keeps the smallest determinant, and `MinCovDet`'s `correct_covariance` and `reweight_covariance`. Its starts are 2,000 random $(A + 1)$-subsets drawn by the fixture generator. The MCD is defined by the support with the smallest determinant, and the starting subsets are a search heuristic, so a reference that reaches the minimum is the right one to compare against.

`MinCovDet(random_state=s).fit` alone is not that reference. With its 30 starts it stops at a larger determinant than ours on corn, gasoline and Tecator for most seeds, and on Tecator with $A = 5$ for every one of 300 seeds. The compared quantity is the squared robust distance of every row, on the scores of a 5-component PCA of each centred dataset. It does not change when a component's sign flips, so our PCA's sign convention (`pca.md` §5) and scikit-learn's need no aligning.

**Leverage and studentised residuals: not compared.** The reference this plan named is R `mdatools`, which needs R, and the development environment has none (decision of 2026-10-04). scikit-learn has no influence measures. The claim rests on these checks in `tests/test_outliers.py`:

- The leverage equals the diagonal of $Z (Z^\top Z)^{-1} Z^\top$, formed by an explicit inverse, and sums to $A + 1$.
- For a PLS and a PCR, the calibration predictions equal the least-squares fit on $[\mathbf{1}, T]$. So §3's scaling is the exact one, and the studentised residuals equal the textbook formula recomputed from that fit.
