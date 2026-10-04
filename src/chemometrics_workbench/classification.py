"""Class-modelling and classification kernels, per `docs/algorithms/`.

SIMCA (`simca.md`) is here. Like every kernel module it imports nothing from
scikit-learn: one class model is `decomposition.PCA`, and the rest is the
arithmetic `simca.md` §3 to §5 states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.arrays import as_float64
from chemometrics_workbench.decomposition import PCA

__all__ = ["LDA", "SIMCA", "acceptance_table", "simca_class_metrics", "simca_metrics"]


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
