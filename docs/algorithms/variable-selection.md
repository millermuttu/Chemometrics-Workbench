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

**A selection made on every sample leaks.** iPLS chooses its intervals by cross-validated error, and the cross-validated error of the model built on those intervals is then optimistic, because the same folds chose them. This is the same leak `metrics-and-validation.md` §9 describes for preprocessing, moved one level up. The honest estimate of a selected model's error comes from samples the selection never saw: a held-out set, or a selection repeated inside an outer loop.

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

## 6. Known divergences

| Area | Here | Elsewhere |
| --- | --- | --- |
| Interval cut | $K$ intervals by variable count | `mdatools` and the original MATLAB toolbox also cut by a fixed width |
| Component count | The first minimum of each curve | Some tools take the first local minimum, or a minimum within one standard error |
| Direction | Forward only | `mdatools` also offers backward elimination |
