# kNN — algorithm specification

Status: **normative**. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

Companion documents: [`pca.md`](pca.md) for the front end; [`classification.md`](classification.md) for class order, the confusion matrix and every metric; [`lda.md`](lda.md), which has the same front end for the same reasons.

---

## 1. PCA-kNN, and why

Every kNN here measures distances in the space of the first $A$ principal-component scores of the centred calibration matrix, where $A$ is `n_components`. That has three effects:
- The scores and diagnostics panels show the space the classifier works in.
- The noise in the discarded components does not vote.
- With $A$ equal to the matrix's rank, the distances are exactly the Euclidean distances between the centred spectra, because a rotation preserves them.

---

## 2. Model

The classes, $C_0 \dots C_{N-1}$ in Unicode order, come from a metadata column, as in `classification.md` §1.

1. Centre: $\bar{x}$ is the calibration mean. This is the estimator's own step.
2. Project: a PCA with $A$ components of $X - \bar{x}$, as in `pca.md`. Each calibration score row $t_i$ and its class $c_i$ are kept: they are the neighbours.
3. $k \le n$ is required, and a larger $k$ is refused, naming both.

---

## 3. Classification

For a sample $x$ with scores $t = (x - \bar{x}) P$:

1. Rank every calibration row by squared Euclidean distance $\lVert t - t_i \rVert^2$. **Equal distances keep calibration order**: the earlier row is nearer.
2. The $k$ nearest vote, one vote each. Votes are not weighted by distance.
3. The class with the most votes is assigned. **A tied vote goes to the first class** in the order of §2.

Both tie rules are scikit-learn's, so they are stated rather than left to chance. The reported **vote fractions**, votes divided by $k$, are scikit-learn's `predict_proba`.

---

## 4. Reported quantities

- **Classification:** the confusion matrices and metrics of `classification.md`, for calibration, fold zero's held-out rows and the cross-validated set.
- **Calibration accuracy is resubstitution.** Each calibration sample is its own nearest neighbour at distance zero, so with $k = 1$ the calibration accuracy is 1 by construction. It is reported because it is defined, and **the cross-validated figure is the one to read**.
- **Panels:** the scores, loadings, $T^2$, SPE and their limits are the PCA front end's, as for LDA.
- **Model:** $\bar{x}$, the loadings, the calibration scores (the neighbours), their classes and $k$.

---

## 5. Cross-validation

Each fold's model is fitted on that fold's training rows of its own preprocessed array. The fold's held-out rows find their neighbours among that fold's training rows only.

---

## 6. Export

The model artifact carries everything a decision needs: $\bar{x}$, the loadings, the neighbours' scores and classes, and $k$. **The JSON model and the Python snippet are not offered in this version**, and the export refuses with that reason. As with SIMCA (`simca.md` §7), a kNN decision needs the whole preprocessed spectrum, so the foldable tail of the chain would have to travel as an explicit affine map. That form is tracked as its own issue.

---

## 7. Parity

scikit-learn `PCA(svd_solver="full")` on the centred matrix, then `KNeighborsClassifier(n_neighbors=5)` with uniform weights and the Euclidean metric, on its scores. The classes are the terciles of `lda.md` §7, on corn, gasoline and Tecator, with $A = 5$. The quantities compared are:
- the assigned class indices, which must be identical;
- the vote fractions, `predict_proba`.
