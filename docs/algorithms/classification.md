# Classification metrics — specification

Status: **normative**. This document fixes how every classifier in this project is tallied and read: PLS-DA, from Phase 5 SIMCA, LDA and kNN, and from Phase 6 SVM. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pls-da.md`](pls-da.md), whose §6 is the two-class case of this one; [`metrics-and-validation.md`](metrics-and-validation.md) for the sets a metric is computed over and the "absent, never NaN" rule (§11).

---

## 1. Classes and their order

The class column is a metadata column of the dataset version: strings, one per sample. Its distinct values $C_0 < C_1 < \dots < C_{N-1}$ are **ordered by Unicode code point** (Python's `sorted`). A result records `classes` in that order, and every assignment is an index into it. $N \ge 2$; a column with one value has nothing to separate.

---

## 2. The confusion matrix

An $N \times N$ matrix of counts, **rows the observed class and columns the assigned class**, both in the order of §1:

$$M_{jk} = \#\{\, i : c_i = C_j,\ \hat{c}_i = C_k \,\}$$

The diagonal holds the correct assignments. With $N = 2$ this is `pls-da.md` §6's

$$\begin{bmatrix} \mathrm{TN} & \mathrm{FP} \\ \mathrm{FN} & \mathrm{TP} \end{bmatrix}$$

with $C_1$ as the class coded 1. The matrix is always reported, because it is what every metric below is computed from and it is complete where they are not.

---

## 3. Metrics

**Accuracy**, over the whole set: $\sum_j M_{jj} / \sum_{jk} M_{jk}$.

**Per class**, one class $C_j$ against the rest:

| Name | Definition | Reads as |
| --- | --- | --- |
| `n` | $\sum_k M_{jk}$ | samples of $C_j$ in the set |
| `sensitivity` | $M_{jj} / \sum_k M_{jk}$ | the fraction of $C_j$ assigned $C_j$ (recall) |
| `specificity` | $\sum_{l \ne j}\sum_{k \ne j} M_{lk} \,/\, \sum_{l \ne j}\sum_k M_{lk}$ | the fraction of the other classes not assigned $C_j$ |
| `precision` | $M_{jj} / \sum_l M_{lj}$ | the fraction of samples assigned $C_j$ that are $C_j$ |

These are scikit-learn's `confusion_matrix` and `precision_recall_fscore_support(average=None)`, plus specificity, which it does not report. They are unit-tested against it.

**A metric whose denominator is zero is absent**, never `0.0` and never `NaN` (`metrics-and-validation.md` §11). Examples are a class with no sample in a held-out fold, or a class no sample was assigned to.

No macro or weighted average is reported. Each is a choice about how classes of different sizes count, and the per-class table shows what the choice would hide.

**Two classes keep their headline names.** With $N = 2$ the flat metrics table carries `accuracy`, `sensitivity` and `specificity` exactly as `pls-da.md` §6 defines them: sensitivity and specificity of $C_1$, the class coded 1. With $N > 2$ there is no class coded 1, so the flat table carries `accuracy` alone, and the per-class table is the reading.

---

## 4. Sets

Every metric and matrix is computed over the three sets `pls-da.md` §6 names. The confusion matrices use the keys `calibration`, `cross_validation` and `held_out`; the flat metrics use the suffixes none, `_cv` and `_p`.

The per-class table is derived from the confusion matrix of its set. It is not stored separately, so it cannot disagree with the matrix.
