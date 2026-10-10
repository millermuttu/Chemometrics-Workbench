# PLS-DA — algorithm specification

Status: **normative**. This document fixes the conventions the implementation must follow. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-regression.md`](pls-regression.md), which this document builds on and does not restate — every quantity of the fitted model is that document's; [`metrics-and-validation.md`](metrics-and-validation.md) for folds and the cross-validation protocol.

---

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $c_i$ | The class label of sample $i$, a string from the dataset's class column |
| $\{C_0, C_1\}$ | The two classes, ordered as §3 orders them |
| $y_i$ | The dummy response, $y_i = 1$ if $c_i = C_1$ else $0$ |
| $\hat{y}_i$ | The PLS1 prediction of $y_i$, a real number |
| $\hat{c}_i$ | The assigned class |
| $A$ | Number of latent variables retained |

---

## 2. Scope: two classes, and more

**Two classes are PLS1.** With one dummy column the model is PLS1 on that column, which is exactly `pls-regression.md` §4: the same weights, scores, loadings, coefficients, VIP and diagnostics, computed by the same kernel and covered by the same parity claims. Nothing new is fitted. What is new is the coding of the response (§3), the assignment of a class to a prediction (§5) and the metrics a classification is read by (§6).

**Three or more classes are PLS2** ([#274](https://github.com/millermuttu/Chemometrics-Workbench/issues/274)): one dummy column per class, fitted together by `pls-regression.md` §10, and assigned by the largest prediction (§5). Two classes are *not* fitted this way. Their results are PLS1's, and adding PLS2 changed none of them.

A class column with a single value has nothing to separate and is refused by name.

---

## 3. Dummy coding

The class column is a metadata column of the dataset version: strings, one per sample. Its distinct values are **ordered by Unicode code point** (Python's `sorted`). With two, the second is the positive class:

$$C_0 < C_1 \qquad y_i = \begin{cases} 1 & c_i = C_1 \\ 0 & c_i = C_0 \end{cases}$$

The order is a convention, not a judgement: "the positive class" here means only "the class coded 1", and every metric in §6 is defined against that coding so that a reader can transpose it. The result records `classes = [C_0, C_1]` so the coding is never implicit.

`{0, 1}` rather than `{-1, +1}`: the two codings give the same assignment with the threshold moved from $0.5$ to $0$, and the same confusion matrix. `{0, 1}` is chosen because the dummy response then reads as a class-membership estimate, which is how §5's threshold is explained.

**Three or more classes are one-hot coded.** $Y$ is $n \times N$, with $Y_{ik} = 1$ when $c_i = C_k$ and $0$ otherwise, and `classes = [C_0, …, C_{N-1}]` records the column order.

**The response is centred by the estimator**, as `pls-regression.md` §3 has it for a regression: $y$ is not on the canvas and no `MeanCentre` node can reach it. The executor subtracts the training mean before fitting and adds it back to every prediction, so $\hat{y}$ is on the $\{0, 1\}$ scale.

---

## 4. Model

PLS1 by NIPALS on the preprocessed $X$ and the dummy $y$, per `pls-regression.md` §4 to §9 without exception. Every reported quantity of the model — weights, scores, loadings, rotations, coefficients, VIP, explained variance, $T^2$ and SPE with their limits — is that document's, computed on the dummy response, and carries the same parity claims.

Centring of $X$ is a pipeline node, as everywhere in this project; the validator's `pls_without_centring` warning covers a PLS-DA node as it covers a regression.

---

## 5. Class assignment

$$\hat{c}_i = \begin{cases} C_1 & \hat{y}_i \ge 0.5 \\ C_0 & \hat{y}_i < 0.5 \end{cases}$$

**The threshold is $0.5$, fixed.** It is the midpoint of the two codes. Other rules exist — a threshold fitted to the calibration scores, a Bayesian cut from the two classes' score distributions, a cost-weighted cut — and each is a choice with parameters that would have to be recorded. None is in v1; a package using one is a documented divergence (§10), not a failed comparison. A tie at exactly $0.5$ goes to $C_1$, stated so that it is reproducible rather than because it matters.

**Three or more classes take the largest prediction:** $\hat{c}_i = C_k$ with $k = \arg\max_k \hat{Y}_{ik}$, ties going to the first class in the order of §3. No threshold applies. With two classes this would be the 0.5 cut, except that a tie at exactly 0.5 would go to $C_0$; two classes keep the cut above.

Assignment is applied to whichever predictions are at hand: calibration predictions, the held-out predictions of a fold, and the cross-validated predictions of `metrics-and-validation.md` §7.

---

## 6. Metrics

From a confusion matrix with **rows the observed class and columns the assigned class**, in the order $[C_0, C_1]$:

$$\begin{bmatrix} \mathrm{TN} & \mathrm{FP} \\ \mathrm{FN} & \mathrm{TP} \end{bmatrix}$$

where TP counts samples of $C_1$ assigned $C_1$, TN samples of $C_0$ assigned $C_0$, FP samples of $C_0$ assigned $C_1$, and FN samples of $C_1$ assigned $C_0$.

| Name | Definition |
| --- | --- |
| `accuracy` | $(\mathrm{TP} + \mathrm{TN}) / n$ |
| `sensitivity` | $\mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$ — the fraction of $C_1$ recovered |
| `specificity` | $\mathrm{TN} / (\mathrm{TN} + \mathrm{FP})$ — the fraction of $C_0$ recovered |

These are [`classification.md`](classification.md)'s two-class case; that document also defines the per-class table every classifier reports. A metric whose denominator is zero — a set with no sample of one class — is **absent**, never `0.0` and never `NaN`, per `metrics-and-validation.md` §11. The confusion matrix is always reported; it is what the metrics are computed from and it is complete where they are not.

**Which set.** The three metrics are computed over three sets and named by suffix, following the regression convention of the same document:

| Suffix | Set | Present when |
| --- | --- | --- |
| none | calibration: the rows the model was fitted on | always |
| `_cv` | cross-validated: every sample assigned from the fold that held it out (§7) | below a split with more than one fold |
| `_p` | held-out: the fitted fold's own test rows, pushed through its model | below any split |

The confusion matrices are reported alongside under the same three names: `calibration`, `cross_validation`, `held_out`.

The calibration figures are optimistic in the way RMSEC is; the cross-validated figures are the ones a classifier is judged by. The application reports both and labels them; it does not choose.

---

## 7. Cross-validation

Exactly `metrics-and-validation.md` §7 to §9 on the dummy response: every node below the split is refitted on the training fold, the held-out samples are predicted through that fold's parameters, and the pooled held-out predictions — one per sample — are assigned by §5 to give the `_cv` confusion matrix and metrics.

**The RMSECV curve is reported on the dummy response**, pooled over every column for three or more classes, as the model's selection curve, $A = 1 \ldots A_{\max}$, exactly as for a regression. A misclassification-rate curve would be the more natural thing to look at for a classifier and would also be a step function of $A$ that hides which $A$ is close to a boundary; the continuous curve is kept for that reason, and the `_cv` accuracy at the chosen $A$ is reported beside it. A misclassification curve is a reasonable later addition and not a change to anything here.

The fitted model reported is refitted on every sample, as for PCA and PLS below a split (`metrics-and-validation.md` §9, #330); its `_p` metrics are fold zero's held-out rows, predicted by fold zero's model.

---

## 8. Reported quantities

Everything `pls-regression.md` §13 lists, on the dummy response, plus:

| Name | Defined in |
| --- | --- |
| `classes` | §3, $[C_0, C_1]$ |
| `observed` | §3, the dummy $y$ |
| `predicted`, `held_out_predicted`, `cross_validated_predicted` | §4, $\hat{y}$ on the $\{0, 1\}$ scale |
| `predicted_class`, `held_out_predicted_class` | §5, as indices into `classes` |
| `confusion.calibration`, `confusion.cross_validation`, `confusion.held_out` | §6 |
| `accuracy`, `sensitivity`, `specificity` and their `_cv` and `_p` forms | §6 |

`Metrics.accuracy` on the experiment record is the calibration accuracy of the last estimator, as `Metrics.rmsec` is its calibration RMSE.

---

## 9. Parity references

Two-class PLS-DA is PLS1 on a dummy response, so the model's parity references are `pls-regression.md` §14's, on that response. What is additionally compared, against `scikit-learn`:

- The continuous predictions $\hat{y}$ of `PLSRegression(scale=False)` fitted to the dummy column, in the prediction class.
- The confusion matrix and accuracy from `sklearn.metrics.confusion_matrix` and `accuracy_score` over those predictions thresholded at $0.5$ — an independent tally, not our arithmetic on their numbers.
- The same two over cross-validated predictions on the fixture's stored folds.

The fixture's class column is derived, for each reference dataset, from its fixture target — `moisture` for corn, `octane` for gasoline, `fat` for Tecator: **`"high"` where the target exceeds its median, `"low"` otherwise**, so $C_0 = $ `"high"` and $C_1 = $ `"low"` under §3. The rule is recorded in the fixture entries' notes and repeated in the parity test, which derives the labels the same way rather than reading them from the fixture.

For three or more classes, the model is PLS2 on the one-hot response, and `pls-regression.md` §10.4's parity claims cover it. The assignment is the argmax of those predictions, tallied by `classification.md`, whose metrics are unit-tested against scikit-learn. `test_executor.py` also checks a three-class run against scikit-learn's `PLSRegression` fitted to the same one-hot matrix on the run's own folds.

---

## 10. Known divergences from other packages

| Area | Here | Elsewhere |
| --- | --- | --- |
| Coding | $\{0, 1\}$, threshold $0.5$ | R `mdatools` codes $\{-1, +1\}$ with threshold $0$; the same assignment |
| Threshold | Fixed at $0.5$ | Some packages fit a threshold to the calibration scores or take a Bayesian cut |
| Classes | Two by PLS1 and a 0.5 cut; three or more by PLS2 and the largest prediction | Some packages use PLS2 and argmax for two classes too, which differs only in a tie at exactly 0.5 |
| Class order | Unicode order of the labels | Order of first appearance, or the user's |
| Sensitivity | Recall of the class coded 1 for two classes; per class for every classifier (`classification.md` §3) | |

---

## 11. Deliberately not specified here

- Stratified splitting on the class column is `metrics-and-validation.md` §8.7's, implemented since #268.
- Probabilistic outputs, ROC curves, class-weighted thresholds.
- Which metric to select $A$ by — a workflow question.
