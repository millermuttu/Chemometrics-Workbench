"""Class-modelling and classification kernels, per `docs/algorithms/`.

SIMCA (`simca.md`) is here. Like every kernel module it imports nothing from
scikit-learn: one class model is `decomposition.PCA`, and the rest is the
arithmetic `simca.md` §3 to §5 states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.arrays import as_float64
from chemometrics_workbench.decomposition import PCA

__all__ = [
    "KNN",
    "LDA",
    "SIMCA",
    "SVM",
    "acceptance_table",
    "simca_class_metrics",
    "simca_metrics",
]


@dataclass(frozen=True)
class ClassModel:
    """One class's PCA, its centre, and both limits (`simca.md` §2, §3)."""

    n_samples: int
    mean: NDArray[np.float64]
    pca: PCA
    t2_limit: float
    q_limit: float


class SIMCA:
    """One PCA per class, and a sample accepted by class `k` when both its
    `T^2` and its `Q` to that class are within their limits (`simca.md`).

    `fit(X, codes)` takes each row's class as an index into the classes, in
    the order `classification.md` §1 gives them; `n_classes` is passed so a
    class with no rows in a fold is reported by index rather than skipped.
    """

    models_: list[ClassModel] | None = None

    def __init__(self, n_components: int, alpha: float = 0.05) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = int(n_components)
        self.alpha = float(alpha)

    def fit(self, X: object, codes: object, n_classes: int) -> Self:
        values = as_float64(X, "X")
        labels = np.asarray(codes, dtype=np.intp)
        if labels.shape != (values.shape[0],):
            raise ValueError(f"X has {values.shape[0]} rows and codes has {labels.size}")
        models: list[ClassModel] = []
        for k in range(n_classes):
            rows = values[labels == k]
            if rows.shape[0] <= self.n_components:
                raise ValueError(
                    f"class {k} has {rows.shape[0]} calibration samples, which cannot support "
                    f"{self.n_components} components (simca.md section 2)"
                )
            mean = rows.mean(axis=0)
            pca = PCA(self.n_components).fit(rows - mean)
            try:
                q_limit = pca.spe_limit(self.alpha)
            except ValueError as error:
                raise ValueError(f"class {k} has no Q limit: {error}") from error
            models.append(
                ClassModel(
                    n_samples=int(rows.shape[0]),
                    mean=mean,
                    pca=pca,
                    t2_limit=pca.hotelling_t2_limit(self.alpha, "new"),
                    q_limit=q_limit,
                )
            )
        self.models_ = models
        return self

    def _models(self) -> list[ClassModel]:
        if self.models_ is None:
            raise RuntimeError("SIMCA has not been fitted")
        return self.models_

    def t2(self, X: object) -> NDArray[np.float64]:
        """`n x N`: each sample's `T^2` to each class model (`simca.md` §3)."""
        values = as_float64(X, "X")
        return np.column_stack([m.pca.hotelling_t2(values - m.mean) for m in self._models()])

    def q(self, X: object) -> NDArray[np.float64]:
        """`n x N`: each sample's `Q` (SPE) to each class model (`simca.md` §3)."""
        values = as_float64(X, "X")
        return np.column_stack([m.pca.spe(values - m.mean) for m in self._models()])

    def distances(self, X: object) -> NDArray[np.float64]:
        """`n x N` reduced distances `h = max(T^2 / T^2_lim, Q / Q_lim)` (`simca.md` §3)."""
        models = self._models()
        t2_limits = np.asarray([m.t2_limit for m in models])
        q_limits = np.asarray([m.q_limit for m in models])
        reduced: NDArray[np.float64] = np.maximum(self.t2(X) / t2_limits, self.q(X) / q_limits)
        return reduced

    def accepted(self, X: object) -> NDArray[np.bool_]:
        """`n x N`: whether class `k` accepts each sample, `h_k <= 1` (`simca.md` §4)."""
        decided: NDArray[np.bool_] = self.distances(X) <= 1.0
        return decided


def acceptance_table(codes: object, accepted: object) -> tuple[list[list[int]], list[int]]:
    """`simca.md` §5: counts of each observed class accepted by each model,
    and per observed class how many samples no model accepted."""
    labels = np.asarray(codes, dtype=np.intp)
    decisions = np.asarray(accepted, dtype=bool)
    n_classes = decisions.shape[1]
    table = [
        [int(np.count_nonzero(decisions[labels == j, k])) for k in range(n_classes)]
        for j in range(n_classes)
    ]
    none = [int(np.count_nonzero(~decisions[labels == j].any(axis=1))) for j in range(n_classes)]
    return table, none


def simca_class_metrics(table: list[list[int]], sizes: list[int]) -> list[dict[str, float]]:
    """`simca.md` §5, per class model: sensitivity on its own class, specificity
    on every other, each absent when its denominator is zero."""
    total = sum(sizes)
    out: list[dict[str, float]] = []
    for k, own in enumerate(sizes):
        entry: dict[str, float] = {"n": float(own)}
        if own:
            entry["sensitivity"] = table[k][k] / own
        others = total - own
        if others:
            accepted_others = sum(table[j][k] for j in range(len(sizes)) if j != k)
            entry["specificity"] = 1.0 - accepted_others / others
        out.append(entry)
    return out


def simca_metrics(table: list[list[int]], sizes: list[int], suffix: str = "") -> dict[str, float]:
    """`simca.md` §5, pooled: own-model acceptance and other-model rejection."""
    total = sum(sizes)
    n_classes = len(sizes)
    metrics: dict[str, float] = {}
    if total:
        metrics[f"sensitivity{suffix}"] = sum(table[k][k] for k in range(n_classes)) / total
    pairs = total * (n_classes - 1)
    if pairs:
        accepted_others = sum(
            table[j][k] for j in range(n_classes) for k in range(n_classes) if j != k
        )
        metrics[f"specificity{suffix}"] = 1.0 - accepted_others / pairs
    return metrics


class LDA:
    """PCA-LDA, per `lda.md`: centre X, project it on `n_components` principal
    components, and fit Fisher's linear discriminant to the scores with a
    pooled within-class covariance and the training class proportions as priors.

    The discriminant is linear in X, so the whole model is one `p x N` matrix
    `B` and `N` intercepts: `delta(x) = (x - x_mean) B + c`, assigned to the
    largest. That is the form multi-class PLS-DA exports, and LDA reuses it.
    """

    coefficients_: NDArray[np.float64] | None = None
    intercepts_: NDArray[np.float64] | None = None
    x_mean_: NDArray[np.float64] | None = None

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = int(n_components)
        self.pca_ = PCA(self.n_components)

    def fit(self, X: object, codes: object, n_classes: int) -> Self:
        values = as_float64(X, "X")
        labels = np.asarray(codes, dtype=np.intp)
        if labels.shape != (values.shape[0],):
            raise ValueError(f"X has {values.shape[0]} rows and codes has {labels.size}")
        counts = np.bincount(labels, minlength=n_classes)
        if (counts == 0).any():
            empty = int(np.flatnonzero(counts == 0)[0])
            raise ValueError(f"class {empty} has no calibration samples (lda.md section 2)")
        n_samples = values.shape[0]
        if n_samples - n_classes < self.n_components:
            raise ValueError(
                f"{n_samples} samples in {n_classes} classes leave {n_samples - n_classes} "
                f"degrees of freedom for a pooled covariance of {self.n_components} scores; "
                "reduce n_components (lda.md section 3)"
            )
        x_mean = values.mean(axis=0)
        self.pca_.fit(values - x_mean)
        loadings = self.pca_.loadings_
        assert loadings is not None
        scores = (values - x_mean) @ loadings
        means = np.stack([scores[labels == k].mean(axis=0) for k in range(n_classes)])
        within = scores - means[labels]
        pooled = within.T @ within / (n_samples - n_classes)
        weights = np.linalg.solve(pooled, means.T)  # A x N: S^-1 mu_k
        priors = counts / n_samples
        self.x_mean_ = x_mean
        self.coefficients_ = loadings @ weights
        self.intercepts_ = -0.5 * np.einsum("ka,ak->k", means, weights) + np.log(priors)
        self.scores_ = scores
        return self

    def decision_function(self, X: object) -> NDArray[np.float64]:
        """`n x N`: `(x - x_mean) B + c`, the discriminant per class (`lda.md` §3)."""
        if self.coefficients_ is None or self.intercepts_ is None or self.x_mean_ is None:
            raise RuntimeError("LDA has not been fitted")
        values = as_float64(X, "X")
        scored: NDArray[np.float64] = (
            values - self.x_mean_
        ) @ self.coefficients_ + self.intercepts_
        return scored

    def predict(self, X: object) -> NDArray[np.intp]:
        """The largest discriminant, ties to the first class (`lda.md` §4)."""
        assigned: NDArray[np.intp] = self.decision_function(X).argmax(axis=1)
        return assigned


class KNN:
    """PCA-kNN, per `knn.md`: centre X, project it on `n_components`
    principal components, and assign each sample the majority class of its
    `k` nearest calibration samples by Euclidean distance in score space.

    Ties are scikit-learn's: among equal distances the earlier calibration
    row is nearer, and a tied vote goes to the first class (`knn.md` §3).
    """

    x_mean_: NDArray[np.float64] | None = None
    scores_: NDArray[np.float64] | None = None
    codes_: NDArray[np.intp] | None = None

    def __init__(self, k: int, n_components: int) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = int(k)
        self.n_components = int(n_components)
        self.pca_ = PCA(self.n_components)
        self.n_classes = 0

    def fit(self, X: object, codes: object, n_classes: int) -> Self:
        values = as_float64(X, "X")
        labels = np.asarray(codes, dtype=np.intp)
        if labels.shape != (values.shape[0],):
            raise ValueError(f"X has {values.shape[0]} rows and codes has {labels.size}")
        if self.k > values.shape[0]:
            raise ValueError(
                f"k = {self.k} neighbours were asked of {values.shape[0]} calibration samples "
                "(knn.md section 2)"
            )
        self.x_mean_ = values.mean(axis=0)
        self.pca_.fit(values - self.x_mean_)
        self.scores_ = self.pca_.transform(values - self.x_mean_)
        self.codes_ = labels
        self.n_classes = int(n_classes)
        return self

    def _fitted(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
        if self.x_mean_ is None or self.scores_ is None or self.codes_ is None:
            raise RuntimeError("KNN has not been fitted")
        return self.x_mean_, self.scores_, self.codes_

    def votes(self, X: object) -> NDArray[np.float64]:
        """`n x N`: the fraction of each sample's `k` neighbours in each class."""
        x_mean, scores, codes = self._fitted()
        projected = self.pca_.transform(as_float64(X, "X") - x_mean)
        squared = (
            (projected**2).sum(axis=1)[:, None]
            - 2.0 * projected @ scores.T
            + (scores**2).sum(axis=1)[None, :]
        )
        # Stable, so equal distances keep calibration order (knn.md section 3).
        nearest = np.argsort(squared, axis=1, kind="stable")[:, : self.k]
        counts = np.stack([np.bincount(codes[row], minlength=self.n_classes) for row in nearest])
        fractions: NDArray[np.float64] = counts / self.k
        return fractions

    def predict(self, X: object) -> NDArray[np.intp]:
        """The majority class, a tied vote to the first class (`knn.md` §3)."""
        assigned: NDArray[np.intp] = self.votes(X).argmax(axis=1)
        return assigned


#: LIBSVM's floor for a non-positive curvature in the two-variable step.
_TAU = 1e-12


def _smo(
    kernel: NDArray[np.float64], y: NDArray[np.float64], c: float, tol: float, max_iter: int
) -> tuple[NDArray[np.float64], float]:
    """The C-SVM dual by SMO with LIBSVM's second-order working-set selection
    and without shrinking (`svm.md` §3). `y` is +1/-1. Returns the multipliers
    and `rho`, the decision being `sum_i a_i y_i K(x_i, x) - rho`.

    Ties in both selections go to the later index, as LIBSVM's `>=` and `<=`
    make them, so the two solvers take the same steps.
    """
    a = np.zeros(y.size)
    gradient = -np.ones(y.size)
    diagonal = np.diag(kernel).copy()
    for _ in range(max_iter):
        up = np.flatnonzero(np.where(y > 0, a < c, a > 0))
        low = np.flatnonzero(np.where(y > 0, a > 0, a < c))
        if up.size == 0 or low.size == 0:
            break
        ascent = -y[up] * gradient[up]
        i = int(up[ascent.size - 1 - np.argmax(ascent[::-1])])
        g_max = -y[i] * gradient[i]
        descent = y[low] * gradient[low]
        if g_max + descent.max() < tol:
            break
        gap = g_max + descent
        curvature = diagonal[i] + diagonal[low] - 2.0 * kernel[i, low]
        curvature = np.where(curvature > 0, curvature, _TAU)
        gain = np.where(gap > 0, -(gap**2) / curvature, np.inf)
        if not np.isfinite(gain).any():
            break
        j = int(low[gain.size - 1 - np.argmin(gain[::-1])])

        q_i = y * y[i] * kernel[i]
        q_j = y * y[j] * kernel[j]
        old_i, old_j = a[i], a[j]
        if y[i] != y[j]:
            quad = diagonal[i] + diagonal[j] + 2.0 * q_i[j]
            delta = (-gradient[i] - gradient[j]) / (quad if quad > 0 else _TAU)
            diff = old_i - old_j
            new_i, new_j = old_i + delta, old_j + delta
            if diff > 0:
                if new_j < 0:
                    new_j, new_i = 0.0, diff
                if new_i > c:
                    new_i, new_j = c, c - diff
            else:
                if new_i < 0:
                    new_i, new_j = 0.0, -diff
                if new_j > c:
                    new_j, new_i = c, c + diff
        else:
            quad = diagonal[i] + diagonal[j] - 2.0 * q_i[j]
            delta = (gradient[i] - gradient[j]) / (quad if quad > 0 else _TAU)
            total = old_i + old_j
            new_i, new_j = old_i - delta, old_j + delta
            if total > c:
                if new_i > c:
                    new_i, new_j = c, total - c
                if new_j > c:
                    new_j, new_i = c, total - c
            else:
                if new_j < 0:
                    new_j, new_i = 0.0, total
                if new_i < 0:
                    new_i, new_j = 0.0, total
        gradient += q_i * (new_i - old_i) + q_j * (new_j - old_j)
        a[i], a[j] = new_i, new_j
    else:
        raise ValueError(
            f"the solver did not reach its stopping tolerance in {max_iter} iterations; "
            "a smaller C converges faster (svm.md section 3)"
        )

    signed = y * gradient
    upper, lower = a >= c, a <= 0
    free = ~upper & ~lower
    if free.any():
        rho = float(signed[free].mean())
    else:
        # No multiplier strictly inside the box: the midpoint of the interval
        # the KKT conditions leave rho in, as LIBSVM takes it.
        above = signed[(upper & (y < 0)) | (lower & (y > 0))]
        below = signed[(upper & (y > 0)) | (lower & (y < 0))]
        rho = float((above.min(initial=np.inf) + below.max(initial=-np.inf)) / 2.0)
    return a, rho


class SVM:
    """PCA-SVM, per `svm.md`: centre X, project it on `n_components`
    principal components, and fit a C-SVM to the scores for every pair of
    classes (one-vs-one), assigning by majority vote.

    `gamma=None` is scikit-learn's `"scale"`, `1 / (A var(T))` over the
    training scores. Ties are LIBSVM's: a pair's decision of exactly zero
    votes for its second class, and a tied vote goes to the first class.
    """

    x_mean_: NDArray[np.float64] | None = None
    scores_: NDArray[np.float64] | None = None
    gamma_: float = 0.0

    def __init__(
        self,
        n_components: int,
        kernel: str = "rbf",
        c: float = 1.0,
        gamma: float | None = None,
        tol: float = 1e-3,
        max_iter: int = 1_000_000,
    ) -> None:
        if kernel not in ("linear", "rbf"):
            raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}")
        if c <= 0:
            raise ValueError(f"C must be positive, got {c}")
        if gamma is not None and gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.n_components = int(n_components)
        self.kernel = kernel
        self.c = float(c)
        self.gamma = gamma
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.pca_ = PCA(self.n_components)
        self.n_classes = 0
        #: One per pair `(i, j)`, `i < j`: the support rows (indices into the
        #: training rows), their `a y`, and `rho`.
        self.pairs_: list[dict[str, Any]] = []

    def _kernel(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kernel == "linear":
            linear: NDArray[np.float64] = a @ b.T
            return linear
        squared = (a**2).sum(axis=1)[:, None] - 2.0 * a @ b.T + (b**2).sum(axis=1)[None, :]
        rbf: NDArray[np.float64] = np.exp(-self.gamma_ * np.maximum(squared, 0.0))
        return rbf

    def fit(self, X: object, codes: object, n_classes: int) -> Self:
        values = as_float64(X, "X")
        labels = np.asarray(codes, dtype=np.intp)
        if labels.shape != (values.shape[0],):
            raise ValueError(f"X has {values.shape[0]} rows and codes has {labels.size}")
        counts = np.bincount(labels, minlength=n_classes)
        if (counts == 0).any():
            empty = int(np.flatnonzero(counts == 0)[0])
            raise ValueError(f"class {empty} has no calibration samples (svm.md section 2)")
        self.x_mean_ = values.mean(axis=0)
        self.pca_.fit(values - self.x_mean_)
        scores = self.pca_.transform(values - self.x_mean_)
        self.scores_ = scores
        variance = float(scores.var())
        self.gamma_ = (
            self.gamma
            if self.gamma is not None
            else 1.0 / (scores.shape[1] * variance)
            if variance > 0
            else 1.0
        )
        self.n_classes = int(n_classes)
        self.pairs_ = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                rows = np.flatnonzero((labels == i) | (labels == j))
                y = np.where(labels[rows] == i, 1.0, -1.0)
                block = scores[rows]
                a, rho = _smo(self._kernel(block, block), y, self.c, self.tol, self.max_iter)
                support = a > 0
                self.pairs_.append(
                    {
                        "classes": (i, j),
                        "support": rows[support],
                        "dual": a[support] * y[support],
                        "rho": rho,
                    }
                )
        return self

    def decision_function(self, X: object) -> NDArray[np.float64]:
        """`n x N(N-1)/2`, one column per pair in `(0, 1), (0, 2), ..., (1, 2)`
        order; positive is a vote for the pair's first class (`svm.md` §4)."""
        if self.x_mean_ is None or self.scores_ is None:
            raise RuntimeError("SVM has not been fitted")
        projected = self.pca_.transform(as_float64(X, "X") - self.x_mean_)
        columns = [
            self._kernel(projected, self.scores_[pair["support"]]) @ pair["dual"] - pair["rho"]
            for pair in self.pairs_
        ]
        decisions: NDArray[np.float64] = np.column_stack(columns)
        return decisions

    def votes(self, X: object) -> NDArray[np.intp]:
        """`n x N`: how many pairs voted for each class."""
        decisions = self.decision_function(X)
        counts = np.zeros((decisions.shape[0], self.n_classes), dtype=np.intp)
        everyone = np.arange(decisions.shape[0])
        for column, pair in enumerate(self.pairs_):
            i, j = pair["classes"]
            np.add.at(counts, (everyone, np.where(decisions[:, column] > 0, i, j)), 1)
        return counts

    def predict(self, X: object) -> NDArray[np.intp]:
        """The most votes, a tied vote to the first class (`svm.md` §4)."""
        assigned: NDArray[np.intp] = self.votes(X).argmax(axis=1)
        return assigned
