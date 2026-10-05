# Variable selection — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-regression.md`](pls-regression.md) for the model each interval fits and for VIP; [`metrics-and-validation.md`](metrics-and-validation.md) §7 and §9 for RMSECV and why preprocessing is refitted inside each fold.

---

## 0. What a selection is, and how it is applied

A selection is a set of column positions on an estimator's own input. **No kernel applies one.** Each method returns positions, and the user applies them with *Apply selection*, which writes a `select_variables` step (#280) on the estimator's input with a copy of the estimator below it. The selection is therefore part of the recipe and of its lineage. The original estimator stays beside the copy, so the two can be compared.

Three methods produce positions:

| Method | Positions kept |
| --- | --- |
| VIP threshold (#281) | VIP at or above a cut, conventionally 1 (`pls-regression.md` §9) |
| Coefficient threshold (#281) | $\lvert b_j \rvert$ at or above a cut, on the estimator's own axis |
| iPLS (#282) | The union of the intervals forward selection adds (§3) |
| CARS (#283) | The subset of the sampling run with the lowest RMSECV (§6) |

**A selection made on every sample leaks.** iPLS chooses its intervals by cross-validated error, and the cross-validated error of the model built on those intervals is then optimistic, because the same folds chose them. This is the same leak `metrics-and-validation.md` §9 describes for preprocessing, moved one level up. The honest estimate of a selected model's error comes from samples the selection never saw: a held-out set, or a selection repeated inside an outer loop.

**The application measures it** (§8): *Validate (nested)* repeats the selection inside an outer loop and reports the honest error beside the optimistic one.

**The pipeline says so.** *Apply selection* records the method in the step's `chosen_by`. When an estimator below such a step is validated, that is, has a split anywhere above it, `checks.py` raises `selection_shares_samples` on the step and names the estimator. A step written by hand, with no `chosen_by`, is not warned about, and an unset `chosen_by` leaves a step's JSON, and so its content hash, as it was before the field existed.

---

## 1. Intervals

The estimator's input, $p$ variables wide, is cut into $K$ contiguous intervals. They are as equal as $p$ allows: the first $p \bmod K$ intervals are one variable wider than the rest. $K$ runs from 2 to $p$. Intervals are cut by position, not by axis value, so an uneven axis gives intervals of equal variable count but not of equal width in nanometres.

---

## 2. Interval PLS

For each interval $k$, a PLS model (`pls-regression.md`) is cross-validated on that interval's columns alone:

- The folds are the split above the estimator. **Each fold's matrix is that fold's own**: below a split, every preprocessing node is refitted on each training fold, and the interval is cut from each fold's matrix. Centring is refitted inside each fold, as `rmsecv_curve` does.
- RMSECV is computed for $A = 1, \dots, \min(A_{\max}, \text{width})$, where $A_{\max}$ defaults to the estimator's own component count.
- The interval's RMSECV is the curve's **first minimum**, and its component count is the $A$ at which that minimum falls.

The full spectrum is treated the same way, as one interval holding every column, and its RMSECV is the reference line the intervals are read against.

iPLS needs a cross-validation split above the estimator with at least two folds. A single train/test split, or no split, is refused with that reason.

---

## 3. Forward selection

Starting from no intervals:

1. For each interval not yet chosen, cross-validate the union of the chosen intervals and that one, as in §2, and take its RMSECV.
2. Add the interval whose union has the lowest RMSECV. **A tie goes to the earlier interval.**
3. Stop when no interval lowers the RMSECV of the union, or when every interval has been chosen.

The result is the path, meaning each interval added and the union's RMSECV after it, and the selection: every column of every interval on the path. A path's first step is the best single interval, so it is never empty.

---

## 4. Reported quantities

| Quantity | Meaning |
| --- | --- |
| `intervals` | Per interval: its first and last column, the axis values there, its RMSECV and its component count |
| `full` | The full spectrum's RMSECV and component count |
| `steps` | The forward path: each interval added, and the union's RMSECV after it |
| `selected` | The column positions *Apply selection* writes |

---

## 5. Parity

Against scikit-learn 1.9's `PLSRegression(scale=False)`, on the folds the PLS entries already record, with $K = 10$ and $A_{\max} = 5$, on corn, gasoline and Tecator. Every model on every interval and every union is scikit-learn's fit and prediction. The interval rule (§1) and the forward rule (§3) are restated in the fixture generator, because they are this document's definition and not a library's. The compared quantities are:

- each interval's RMSECV, in the metrics class;
- the forward path, which must be identical.

**Not compared against R `mdatools`**, whose `ipls` this plan named, because the development environment has no R (decision of 2026-10-04). `mdatools` also differs by convention: it can select backward as well as forward, it chooses the component count per model by its own rule, and its default intervals are set by width rather than count.

---

## 6. CARS

Competitive adaptive reweighted sampling (Li, Liang, Xu and Cao, 2009, *Anal. Chim. Acta* 648, 77). With $p$ variables on the estimator's input and $N$ sampling runs (default 50):

1. All $p$ variables are retained at the start.
2. For run $i = 1, \dots, N$:
   - Draw a random $80\%$ of the samples (rounded), without replacement.
   - Fit a PLS (`pls-regression.md`) on those samples and the retained variables, centred on the drawn samples. The component count is $\min(A_{\max}$, retained count, drawn count $- 1)$.
   - Weight each retained variable by $w_j = \lvert b_j \rvert / \sum \lvert b \rvert$. If every $b_j$ is zero, the weights are equal.
   - **Forced removal by the exponentially decreasing function**: keep the $m_i = \max(2, \operatorname{round}(r_i p))$ heaviest variables, where $r_i = a e^{-k i}$, $a = (p/2)^{1/(N-1)}$ and $k = \ln(p/2)/(N-1)$. So $r_1 = 1$ and $r_N = 2/p$. Ties go to the earlier position.
   - **Adaptive reweighted sampling**: draw $m_i$ times, with replacement, from those $m_i$ variables with probability proportional to their weights. The distinct variables drawn are the ones retained.
   - Cross-validate a PLS on the retained variables as in §2, and record its RMSECV and component count.
3. The selection is the retained set of the run with the lowest RMSECV, the earliest one on a tie.

**Randomness.** Both the samples drawn and the reweighted sampling come from one `numpy.random.default_rng(seed)`, with seed 0 unless one is given. A seed always gives the same runs. Every fit is cross-validated on the stored per-fold matrices, as iPLS's are. The sampling fits use the first fold's matrix it is given, every row of it. Run from an estimator, that is fold zero's matrix; inside the nested loop of §8, it is the outer fold's own training rows.

**Reported quantities.** Per run: the retained count, the RMSECV and the component count. Also the best run, the seed and the selected positions.

**Parity: not compared.** There is no reference implementation in this environment. CARS is published as MATLAB code, and its Python ports are not established, versioned libraries. Its randomness also makes a value-by-value comparison meaningless unless both sides draw the same numbers. The claim rests on these checks in `tests/test_selection.py` instead: the same seed gives the same runs and a different seed different ones; the retained set never grows and ends at two variables or fewer; and on a synthetic set of 60 noise variables, five of which carry the response, the selection contains all five and fewer than 20 variables in total.

---

## 7. Known divergences

| Area | Here | Elsewhere |
| --- | --- | --- |
| Interval cut | $K$ intervals by variable count | `mdatools` and the original MATLAB toolbox also cut by a fixed width |
| Component count | The first minimum of each curve | Some tools take the first local minimum, or a minimum within one standard error |
| Direction | Forward only | `mdatools` also offers backward elimination |
| CARS components | Capped at $A_{\max}$, chosen per run at the RMSECV curve's first minimum | The original fixes $A$, and chooses it by its own cross-validation |
| CARS sampling | 80% of the samples per run, one seeded generator | The original's fraction is also 80%; its runs are not reproducible across implementations |

---

## 8. Nested validation

[#331](https://github.com/millermuttu/Chemometrics-Workbench/issues/331). A selection made on every sample and then cross-validated on the same folds is scored by the folds that chose it (§0). Nested validation scores it on samples it never saw:

1. **The outer folds** are the split above the estimator, and each outer fold's matrix is its own, as in §2.
2. In each outer fold, the **inner folds** are a K-fold of that fold's training rows, $K = 5$ unless set, shuffled with the given seed (default 0) by `metrics-and-validation.md` §8.3. Under a grouped split (§8.8 there) they are grouped by the same column, so replicates stay together inside as well as outside.
3. **The selection is rerun** on the outer training rows alone, with the inner folds where the method cross-validates: iPLS and CARS as §2-§6, VIP and the coefficient threshold from a PLS fitted on those rows with the estimator's component count, at the same cut.
4. **The selected model** is a PLS on the selected columns with the estimator's component count, capped at the column count, centred on the outer training rows. It predicts the outer held-out rows.
5. The held-out predictions are pooled (`metrics-and-validation.md` §7) into the **outer RMSECV**.

Beside it is the **inner RMSECV**: the same method run on every sample with the outer folds, as *Apply selection* runs it, and the selected model cross-validated on those folds at the estimator's component count. That is the number the applied model will report, and the one §0 calls optimistic.

The inner folds see one outer fold's preprocessing, fitted on all of its training rows; only the estimator's own centring is refitted per inner fold. The outer score is not affected: the outer held-out rows took no part in any of it.

**The outer error is not always the larger.** A method that chooses by cross-validated error (iPLS, CARS) flatters that error, and on Tecator its outer RMSECV is the higher. A VIP or $\lvert b \rvert$ threshold chooses from the fitted model and never looks at a cross-validated error, so its inner number has no built-in optimism, and on Tecator its outer RMSECV comes out slightly lower.

**Reported quantities.** The outer and inner RMSECV, the outer fold count, the inner fold count, the size of the selection made on every sample, and the size of each outer fold's selection.

**Parity.** No library offers this loop as one call. `tests/test_selection.py` rebuilds it with scikit-learn's `PLSRegression`, VIP computed from its weights and iPLS restated with its fits, on the same outer folds, and the selections and outer predictions agree.

---

## 9. From a PLS-DA

[#332](https://github.com/millermuttu/Chemometrics-Workbench/issues/332). Every method above runs from a PLS-DA as well as from a PLS regression, on the PLS-DA's **dummy response** (`pls-da.md` §3): the $\{0, 1\}$ codes for two classes, which is PLS1, and the one-hot $n \times N$ matrix for three or more, which is PLS2 (`pls-regression.md` §10).

- **VIP** is the PLS-DA's own (`pls-regression.md` §8). For PLS2 its per-component weights sum the response sum of squares over every column.
- **iPLS and CARS** score a subset by the dummy RMSECV `pls-da.md` §7 defines: one RMSE pooled over every element of the held-out one-hot predictions, so every class counts in proportion to its samples. Two classes give PLS1's RMSECV exactly.
- **CARS's weights** for PLS2 are each variable's $\lvert b_{jc} \rvert$ summed over the classes, normalised as in §6.
- **The coefficient threshold** needs one coefficient vector, so it is offered for two classes and refused for three or more.
- **Nested validation** (§8) scores the outer held-out rows by the same pooled RMSE.

*Apply selection* writes the same `select_variables` step above a copy of the PLS-DA, as it does for a regression.

**Parity.** `tests/test_selection.py` compares iPLS on a one-hot response with a rebuild in which every fit is scikit-learn's multi-target `PLSRegression`, iterated to its fixed point, on Tecator split into fat terciles. The interval RMSECVs agree and the forward path is identical. `tests/test_regression.py` compares the pooled curve the same way.
