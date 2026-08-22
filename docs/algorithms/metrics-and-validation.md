# Metrics and validation — specification

Status: **normative**. This document fixes the conventions the implementation must follow. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Every metric name that can appear in the application is defined here as a formula. If a metric is not defined here, it is not reported. This is the document that explains most "why does this not match Unscrambler" reports: metric definitions and fold assignment vary between packages far more than the algorithms themselves do.

Companion documents: [`pca.md`](pca.md) and [`pls-regression.md`](pls-regression.md) define the algorithms and their model-specific diagnostics ($T^2$, SPE, VIP, explained variance). Those are not repeated here.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $n$ | Number of samples in the set a metric is computed over |
| $y_i$ | Reference value for sample $i$, in its original units |
| $\hat{y}_i$ | Predicted value for sample $i$, in the same units |
| $e_i$ | Residual, $y_i - \hat{y}_i$ |
| $\bar{y}$ | Mean of $y$ over the **calibration** set, unless stated otherwise |
| $A$ | Number of latent variables in the model, as in [`pls-regression.md` §4](pls-regression.md) |
| $K$ | Number of folds |
| $\mathcal{T}_k, \mathcal{V}_k$ | Training and validation index sets of fold $k$ |
| $\hat{y}^{(-k)}_i$ | Prediction for sample $i \in \mathcal{V}_k$ from the model fitted on $\mathcal{T}_k$ |
| $h_i$ | Leverage of sample $i$ |

---

## 2. Two rules that apply to every metric

**Every metric is computed on the original units of the response.** Predictions are un-centred and un-scaled back through the response side of the preprocessing chain before any residual is formed. A RMSEC of 0.31 means 0.31 in the units of the reference method, whatever centring or autoscaling the pipeline applied.

Metrics on centred or autoscaled residuals are not reported at all — not under a different name, not in `extra`. Two models with different scaling would otherwise produce two RMSECs that look comparable and are not.

**Every metric carries the model it came from.** For PLS that means the component count $A$; a reported metric without $A$ is incomplete ([`pls-regression.md` §11](pls-regression.md)). For any metric it means the set it was computed over — calibration, cross-validation or an independent test set. The three are never mixed in one number and never plotted on the same series without a label.

---

## 3. Which set each metric is computed over

| Suffix | Set | Model used |
| --- | --- | --- |
| **C** — calibration | The training samples themselves | The final model, fitted on all of them |
| **CV** — cross-validation | Every calibration sample, once | The fold model that did *not* see that sample |
| **P** — prediction | An independent set, never used in fitting | The final model |

RMSEC measures fit. RMSECV estimates prediction error using only the calibration data. RMSEP measures prediction error on data held out entirely. RMSEC is always the smallest of the three, and a large gap between RMSEC and RMSECV is the standard signal of overfitting — which is why all three are reported side by side rather than one being chosen for the user.

---

## 4. Root-mean-square errors

$$\mathrm{RMSEC} = \sqrt{\frac{1}{n_c}\sum_{i=1}^{n_c} \left(y_i - \hat{y}_i\right)^2}
\qquad
\mathrm{RMSEP} = \sqrt{\frac{1}{n_p}\sum_{i=1}^{n_p} \left(y_i - \hat{y}_i\right)^2}$$

$$\mathrm{RMSECV} = \sqrt{\frac{1}{n_c}\sum_{i=1}^{n_c} \left(y_i - \hat{y}^{(-k(i))}_i\right)^2}$$

**The denominator is $n$ in all three** — the number of samples in the set, with no correction for degrees of freedom and no subtraction of the component count. They are root *mean* squared errors, and the mean of $n$ terms divides by $n$.

For RMSECV, $n_c$ is the full calibration set and $k(i)$ is the fold whose validation set contains sample $i$. Under the split rules of §8 every calibration sample is predicted exactly once, so the sum has exactly $n_c$ terms — this is what makes RMSECV comparable with RMSEC on the same scale.

**Degrees-of-freedom corrections live in SEC and SEP (§5), not here.** Packages that write $n - A - 1$ under a quantity called RMSEC are reporting what this document calls SEC. The distinction is real and is the single most common cause of a mismatch in the third significant figure; see §12.

---

## 5. Bias, SEC and SEP

**Bias** is the mean signed residual over the set:

$$\mathrm{bias} = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)$$

Bias is essentially zero on the calibration set of a model with an intercept, by construction. It is informative on a prediction set, where a non-zero bias means a systematic offset — a different instrument, a different batch of reference analyses, a drifted calibration — and is the quantity a slope-and-bias correction would remove.

**SEC and SEP are the bias-corrected, degrees-of-freedom-corrected standard errors.** They describe the *scatter* of the residuals about their own mean, where the RMSEs of §4 describe total error including any offset:

$$\mathrm{SEC} = \sqrt{\frac{1}{n_c - A - 1}\sum_{i=1}^{n_c}\left(e_i - \mathrm{bias}\right)^2}
\qquad
\mathrm{SEP} = \sqrt{\frac{1}{n_p - 1}\sum_{i=1}^{n_p}\left(e_i - \mathrm{bias}\right)^2}$$

The denominators differ deliberately:

- **SEC uses $n_c - A - 1$.** The calibration residuals come from a model that spent $A$ latent variables plus an intercept fitting those same samples, so the naive variance is optimistic and the lost degrees of freedom are subtracted.
- **SEP uses $n_p - 1$.** The prediction samples took no part in the fit; the only parameter estimated from them is the mean subtracted in the bias correction.

On the prediction set the two are tied exactly by $\mathrm{RMSEP}^2 = \mathrm{bias}^2 + \frac{n_p-1}{n_p}\,\mathrm{SEP}^2$, which is a cheap unit test. No such identity holds between RMSEC and SEC, because their denominators differ by more than one.

$n_c - A - 1 \le 0$ makes SEC undefined. Report it as absent and say why; do not fall back to a different denominator, which would silently produce a number that is not SEC.

---

## 6. $R^2$ and $Q^2$

$$R^2 = 1 - \frac{\sum_{i} \left(y_i - \hat{y}_i\right)^2}{\sum_{i} \left(y_i - \bar{y}\right)^2}
\qquad
Q^2 = 1 - \frac{\mathrm{PRESS}}{\sum_{i}\left(y_i - \bar{y}\right)^2}
\qquad
\mathrm{PRESS} = \sum_{i} \left(y_i - \hat{y}^{(-k(i))}_i\right)^2$$

Both sums run over the calibration set; both denominators use the **total sum of squares of the calibration response about the full calibration mean $\bar{y}$**.

Three points that decide whether these numbers match another package:

- **$\bar{y}$ is the mean of the whole calibration set, never a per-fold mean.** Recomputing the mean inside each fold changes $Q^2$, and packages differ on this. Fixing it to the full calibration mean keeps $Q^2$ and $R^2$ on a common denominator, so the two are directly comparable and $Q^2 \le R^2$ has its usual meaning.
- **$R^2$ is the residual form above, not the squared correlation** $\mathrm{corr}(y,\hat{y})^2$. The two coincide for an ordinary least-squares fit with an intercept on the same data, and diverge for predictions on a new set, where the squared correlation is blind to bias and slope error and is therefore flattering. Where the squared correlation is wanted it is reported as `r2_pearson` in `extra`, never as `r2`.
- **$Q^2$ can be negative**, and is not clipped. A negative $Q^2$ means the model predicts held-out samples worse than the calibration mean does, which is a real and useful finding. Reporting it as zero hides a failed model.

There is no degrees-of-freedom-adjusted $R^2$ in v1. With $A$ latent variables from $p \gg n$ collinear predictors the usual adjustment is not well founded, and reporting RMSECV alongside is a more honest answer to the same question.

---

## 7. Fold aggregation: pooled residuals

**RMSECV, PRESS and $Q^2$ pool the residuals across all folds into a single sum, then take one root at the end.** They are *not* the mean of per-fold RMSEs.

$$\mathrm{RMSECV} = \sqrt{\frac{1}{n_c}\sum_{k=1}^{K}\sum_{i \in \mathcal{V}_k} \left(y_i - \hat{y}^{(-k)}_i\right)^2}
\qquad\text{not}\qquad
\frac{1}{K}\sum_{k=1}^{K} \mathrm{RMSE}_k$$

The two give different numbers whenever the folds differ in size or in difficulty. The pooled form is chosen because it weights every *sample* equally rather than every *fold*, which is what "the error you should expect on a new sample" means, and because it stays exactly comparable with RMSEC and RMSEP — the same $n_c$ residuals, the same denominator, the same units.

Per-fold RMSEs are still computed and kept, as `rmsecv_fold_<k>` in `Metrics.extra`, because their spread is what tells a user whether one awkward fold is carrying the whole estimate. They are diagnostic, never the headline number.

**Every calibration sample contributes exactly one residual** to the pooled sum. A split that leaves a sample in no validation fold, or in two, is a bug — assert $\bigsqcup_k \mathcal{V}_k = \{0 \dots n_c-1\}$ as a disjoint union before aggregating.

---

## 8. Fold assignment

A split is fully determined by its `SplitSpec` and $n$. The rules below are sufficient to reproduce one by hand.

### 8.1 Indices

Samples are identified by **positional row index into the dataset version, counting from 0**, in the row order of the file as ingested. The order is pinned by `Experiment.dataset_content_hash`: a dataset whose rows were reordered is a different content hash and therefore a different experiment.

### 8.2 Shuffling and the seed

When `shuffle` is true the index array is permuted by

```python
rng = numpy.random.default_rng(seed)  # PCG64
perm = rng.permutation(n)
```

with `seed` taken from the `SplitSpec` (default 42). When `shuffle` is false, `perm` is the identity `0 … n-1` and the seed is ignored — recorded as ignored rather than silently accepted.

`default_rng` is fixed here because it is the reproducible modern stream. It is **not** the stream `scikit-learn` uses: `check_random_state` produces a legacy `RandomState`, so seeding both with 42 yields different folds. Parity against `scikit-learn` cross-validation therefore passes our resolved fold indices to it as an explicit `cv` iterable, and never seeds the two independently and compares. See §12.

### 8.3 K-fold

With $K$ folds and $n$ samples, let $q = \lfloor n/K \rfloor$ and $r = n \bmod K$. **The first $r$ folds have $q+1$ members and the remaining $K-r$ have $q$**, taken as consecutive slices of `perm`. Fold $k$'s validation set is that slice; its training set is everything else, in ascending index order.

This is `scikit-learn`'s size rule, kept deliberately so that only the permutation differs.

$K > n$ is an error naming both numbers. $K = n$ is leave-one-out and is expressed as `LeaveOneOut`, not as a K-fold with $K = n$.

**Worked example** — $n = 10$, $K = 3$, `shuffle=True`, `seed=42`. Fold sizes are 4, 3, 3 and `perm` is `[5, 6, 0, 7, 3, 2, 4, 9, 1, 8]`, giving:

| Fold | Validation indices | Training indices |
| --- | --- | --- |
| 0 | 0, 5, 6, 7 | 1, 2, 3, 4, 8, 9 |
| 1 | 2, 3, 4 | 0, 1, 5, 6, 7, 8, 9 |
| 2 | 1, 8, 9 | 0, 2, 3, 4, 5, 6, 7 |

Reproducing this table from a fresh implementation is the test that the seeding, the permutation and the size rule all agree.

### 8.4 Leave-one-out

$n$ folds, fold $i$ holding out sample $i$ alone. No shuffle, no seed — LOO is deterministic and its result cannot depend on one.

LOO is exact and expensive: $n$ fits. It is also the most optimistic of the cross-validation schemes for spectral data, because near-duplicate samples — replicate scans, samples from one batch — stay in the training set when their twin is held out. The application says so where LOO is offered; it does not refuse it.

### 8.5 Repeated K-fold

`RepeatedKFoldSplit` runs the §8.3 scheme `n_repeats` times. **Repeat $r$ (counting from 0) uses seed `seed + r`**, always shuffling. Each repeat independently produces one prediction per sample.

Aggregation is **two-level, and does not pool across repeats**: pool the residuals within a repeat to get $\mathrm{RMSECV}_r$ by §7, then report

$$\mathrm{RMSECV} = \frac{1}{R}\sum_{r=0}^{R-1} \mathrm{RMSECV}_r$$

with the sample standard deviation across repeats as `rmsecv_std` in `extra`. The whole point of repeating is to measure how much the estimate moves with the fold assignment; pooling all $R \cdot n$ residuals into one sum would average that spread away, which is the number the user asked to see. This is the one place where the mean-of-RMSEs form is used, and it is a mean over *repeats*, never over folds.

### 8.6 Train/test and external sets

`TrainTestSplit` draws a test set of $\lceil \texttt{test\_size} \cdot n \rceil$ samples as the first slice of `perm`, seeded as in §8.2. The remainder is the calibration set. This is a single-fold split: `ResolvedSplit` holds one entry, and metrics carry the **P** suffix, not **CV**.

`ExternalSet` takes the validation samples from a different dataset version entirely, named by `validation_version_id`. No permutation is involved. Both dataset versions' content hashes are recorded, because an external validation whose file changed is not the validation that was run.

### 8.7 Stratification

`TrainTestSplit.stratify_by` names a metadata column. The rows are grouped by its value, each group is permuted with the same seeded generator, and the test set takes $\lceil \texttt{test\_size} \cdot n_g \rceil$ from each group $g$ — so the test proportion is honoured within every level, up to rounding. A group with fewer than 2 members is an error naming the column and the level.

**K-fold has no stratification in v1.** `KFoldSplit` and `RepeatedKFoldSplit` carry no `stratify_by` field, and inventing one in the executor would put behaviour in the run that is absent from the recipe. Adding it is a schema change and a new issue, not an implementation detail.

Continuous responses are not stratified. Binning a response to balance folds is a defensible technique with an arbitrary bin count that would have to be recorded, and it is out of scope for v1.

---

## 9. What is refitted inside a fold

**Every node downstream of the split node is refitted on the training fold alone.** Centring means the training fold's mean; autoscaling means its standard deviation; the PLS model means its coefficients. The held-out samples are then pushed through that chain with the training fold's parameters, exactly as new samples are at prediction time ([`pls-regression.md` §7](pls-regression.md)).

**Nodes upstream of the split node are fitted once, on everything.** That is legitimate for transforms with no fitted parameters — Savitzky–Golay, derivatives, SNV, unit conversion, range selection — and is a leak for anything that estimates a parameter from the data, above all centring and autoscaling.

Placing a `MeanCentre` or `Autoscale` node upstream of a split therefore leaks the validation samples' contribution into the training statistics and makes RMSECV optimistic. The pipeline validator **warns and names the node**. It does not rewrite the graph: the pipeline is the record of what was done, and silently relocating a node would make the recipe a lie. The warning travels into the experiment record so the number is never read without it.

**The component count $A$ is not re-selected inside folds.** $A$ is a user parameter; every fold model is fitted with the same $A$, and the reported RMSECV is a property of that $A$. RMSECV as a function of $A$ from 1 to $A_{\max}$ is produced from the *same* fold assignment across all component counts, and is stored as `rmsecv_a<A>` keys in `extra` — one split, one pass, one curve. Choosing $A$ by the minimum of that curve and then quoting the same curve's minimum as the model's expected error is optimistic; it is the user's call, and the application does not do it for them.

---

## 10. `ResolvedSplit`: what is stored, and why

`ResolvedSplit` stores, per split node, the **realised index sets**: `train_indices` and `test_indices`, one list per fold, one entry for a plain split.

The indices are stored rather than recomputed from the seed because **a seed is only reproducible against a fixed random number generator implementation**. A NumPy upgrade that changes a stream, a switch from `RandomState` to `default_rng`, a change to the shuffle order in a helper library — any of these silently produces different folds from the same recorded spec, and a rerun would then report a different RMSECV with nothing in the record explaining why. Stored indices survive all of it.

They also make three things possible that a seed alone does not:

- **Auditing a suspicious fold.** The exact samples that produced a bad per-fold RMSE can be listed and inspected.
- **Parity by construction.** The same fold assignment can be handed to `scikit-learn` or R as an explicit index list, so a comparison tests the metric and the model, not two independent shufflers (§8.2).
- **Replay against a changed dataset.** If the content hash no longer matches, the stored indices no longer mean what they meant, and the mismatch is detectable. A recomputed split would quietly succeed on the wrong rows.

The spec is kept too. `SplitSpec` on the node records *intent* — 5-fold, shuffled, seed 42 — which is what a reader needs to understand the experiment; `ResolvedSplit` records *fact*, which is what a rerun needs to reproduce it.

---

## 11. Reported quantities

| Name | Field on `Metrics` | Defined in |
| --- | --- | --- |
| `rmsec` | `rmsec` | §4 |
| `rmsecv` | `rmsecv` | §4, §7 |
| `rmsep` | `rmsep` | §4 |
| `r2` | `r2` | §6 |
| `q2` | `q2` | §6 |
| `bias` | `bias` | §5 |
| `explained_variance` | `explained_variance` | [`pca.md` §6](pca.md) |
| `sec`, `sep` | `extra` | §5 |
| `rmsecv_fold_<k>` | `extra` | §7 |
| `rmsecv_a<A>` | `extra` | §9 |
| `rmsecv_std` | `extra` | §8.5 |
| `r2_pearson` | `extra` | §6 |

`Metrics.accuracy` is a classification field and is out of scope for this document; it is defined when PLS-DA lands.

A metric that could not be computed is `None` — never `0.0`, never `NaN`. RMSEP without a prediction set is absent, not zero, and the UI renders absence as an em dash rather than a number.

---

## 12. Known divergences from other packages

Recorded so the parity report can classify them as *differs by documented convention* rather than as failures.

| Area | Here | Elsewhere |
| --- | --- | --- |
| RMSEC denominator | $n_c$ | Several packages use $n_c - A - 1$; the difference is a factor of $\sqrt{n/(n-A-1)}$, a few percent at typical $n$ |
| SEC / SEP | Bias-corrected, $n_c - A - 1$ and $n_p - 1$ | Some packages use $n$; some report SEC under the name RMSEC |
| Fold aggregation | Pooled residuals | Mean of per-fold RMSEs is common, and differs whenever folds are uneven |
| $Q^2$ denominator | Full calibration mean $\bar{y}$ | Per-fold means are used by some packages |
| $R^2$ | Residual form | Squared Pearson correlation is common, and flatters predictions on a new set |
| Negative $Q^2$ | Reported as is | Often clipped to 0 |
| Shuffle stream | `numpy.random.default_rng` (PCG64) | `scikit-learn` uses legacy `RandomState`; same seed, different folds |
| Fold sizes | First $r$ folds larger | Same as `scikit-learn`; other tools distribute the remainder differently |
| **Leverage-corrected RMSE** | Not reported | Unscrambler and others offer a leverage-corrected error, $e_i/(1 - h_i)$, as a fast approximation to LOOCV |
| Preprocessing inside CV | Refitted per fold, warned when upstream | Frequently fitted once on all data, which makes RMSECV optimistic |
| Repeated K-fold | Mean of per-repeat RMSECVs | Pooling all repeats into one sum is also seen |

**Leverage-corrected RMSE deserves the emphasis.** Where cross-validation refits the model $K$ times, the leverage correction inflates each calibration residual by $1/(1-h_i)$ to approximate what the residual would have been had that sample been held out — one fit instead of $n$. For a linear model with a fixed design it is exactly LOOCV; for PLS it is an approximation, because the latent variables themselves shift when a sample is removed. It is not reported here: it is cheap and it is *nearly* RMSECV, which is precisely what makes it dangerous to display beside a real RMSECV under a similar name. A user comparing our RMSECV against a leverage-corrected number from another package is comparing two different quantities, and this row is the answer to that report.

---

## 13. Deliberately not specified here

- **Tolerances and claim tiers** — the parity harness, not this document.
- **Classification metrics** — accuracy, sensitivity, specificity, confusion matrices. They arrive with PLS-DA.
- **Slope-and-bias correction and other model updating** — post-1.0.
- **Automatic selection of $A$**, including the one-standard-error rule and Wold's R. A workflow question; see [`pls-regression.md` §11](pls-regression.md).
- **Stratified K-fold and continuous-response binning** — a schema change first (§8.7).
- **Nested cross-validation** — needed only once something is tuned inside the loop, and nothing is in v1.
