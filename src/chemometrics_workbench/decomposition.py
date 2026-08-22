"""PCA, and the two distances a scores plot is read against.

The normative document is `docs/algorithms/pca.md`; this module implements it
and nothing more. Every quantity here is defined in one of its sections and
the docstrings say which. Where this module and that document disagree, one of
them is a bug — decide which before changing either.

## What it does not do

**It does not centre.** `pca.md` §2: centring and scaling are explicit
pipeline steps and appear as nodes in the recipe, because the pipeline is the
complete record of what was done and preprocessing hidden inside an estimator
would be absent from the lineage. `sklearn.decomposition.PCA` centres
internally and unconditionally, which is why every matrix in the parity
fixture was centred before scikit-learn saw it.

Fitting on uncentred data is legal here and usually wrong — the first
component then largely captures the mean spectrum, and explained variance is
reported against a total that includes it. Warning about that is the
application's job (a UI affordance), not this module's: a kernel that silently
centred would be lying about what it fitted.

**It knows nothing about the application.** Arrays in, arrays out, no
project, no schema, no I/O — the condition under which this stays usable as a
standalone library.

**It takes no seed.** The decomposition is `numpy.linalg.svd` with
`full_matrices=False`, which is deterministic. Randomised and truncated SVD
are not used: a decomposition whose result depends on a random draw is not
something a parity report can stand behind (§3).

## The sign convention, and why it is not scikit-learn's

The signs of `U` and `V` are jointly arbitrary. §5 fixes ours on the
**loadings**: for each component, if the largest-magnitude loading is
negative, the whole component is negated. Ties go to the smallest index. The
loading is the spectrum-shaped, interpretable vector for spectral data and its
orientation is what an analyst reads, so that is what the rule is keyed on.

scikit-learn decides from `U` instead (`svd_flip(u_based_decision=True)`), so
signs may differ per component with both being correct. Parity comparisons
align by inner product; the harness does it, and comparing absolute values
instead would pass a result whose score and loading signs disagree with each
other.

## Retaining every eigenvalue

`eigenvalues_` holds all `r` of them, not the `a` retained. The explained
variance denominator (§6) and the SPE limit (§8) are both sums over the
*discarded* components: a model that kept only the first `a` cannot compute
its own SPE limit, and normalising explained variance by the retained
components would make it always reach 100%.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, f, norm

from chemometrics_workbench.arrays import as_float64

__all__ = ["PCA"]

LimitFor = Literal["calibration", "new"]


class PCA:
    """Principal component analysis by SVD, per `pca.md`.

    `fit(X)` then `transform(X)`, duck-compatible with a scikit-learn
    transformer and importing nothing from it — the same rule the preprocessing
    kernels follow, and for the same reason: scikit-learn is the reference
    implementation the parity fixture is generated against.

    Fitted attributes, all in `pca.md` §4:

    | | |
    | --- | --- |
    | `loadings_` | `p x a`, orthonormal columns, sign-fixed by §5 |
    | `scores_` | `n x a`, the calibration set's scores |
    | `eigenvalues_` | `r` values, `sigma_k^2/(n-1)` — every component, not the retained ones |
    | `singular_values_` | the `r` singular values above the rank tolerance |
    | `rank_` | `r`, the effective rank found by the SVD |
    | `n_samples_`, `n_variables_` | the calibration shape, which the `T^2` limits need |
    """

    loadings_: NDArray[np.float64] | None = None
    scores_: NDArray[np.float64] | None = None
    eigenvalues_: NDArray[np.float64] | None = None
    singular_values_: NDArray[np.float64] | None = None
    rank_: int | None = None
    n_samples_: int | None = None
    n_variables_: int | None = None

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = int(n_components)

    # ----------------------------------------------------------------------
    # fitting
    # ----------------------------------------------------------------------

    def fit(self, X: object) -> Self:
        values = as_float64(X, "X")
        n_samples, n_variables = values.shape

        # `full_matrices=False` per §3. The matrix is decomposed as supplied:
        # if it should have been centred, that was a pipeline step upstream.
        u, singular_values, vt = np.linalg.svd(values, full_matrices=False)

        # §9: the effective rank is taken from the SVD rather than inferred
        # from the shape, because PCA does not centre and so cannot know
        # whether a degree of freedom was already spent.
        tolerance = max(n_samples, n_variables) * float(np.finfo(np.float64).eps)
        tolerance *= float(singular_values[0]) if singular_values.size else 0.0
        rank = int(np.count_nonzero(singular_values > tolerance))
        if rank == 0:
            raise ValueError(
                "X has no variance to decompose: every singular value is at or below "
                "the rank tolerance. A constant matrix has no principal components."
            )
        if self.n_components > rank:
            raise ValueError(
                f"{self.n_components} components were asked of a matrix of rank {rank}. "
                "Returning fewer would make downstream shapes unpredictable and would "
                "hide the mistake; reduce n_components, or add samples or variables."
            )

        loadings = vt[: self.n_components].T
        # §5: keyed on the largest-magnitude loading, ties to the smallest index.
        # argmax already returns the first maximum, which is the tie rule.
        dominant = np.abs(loadings).argmax(axis=0)
        flips = np.where(loadings[dominant, np.arange(self.n_components)] < 0, -1.0, 1.0)

        self.loadings_ = loadings * flips
        self.singular_values_ = singular_values[:rank]
        self.eigenvalues_ = self.singular_values_**2 / (n_samples - 1)
        self.rank_ = rank
        self.n_samples_ = n_samples
        self.n_variables_ = n_variables
        # §4: computed as XP rather than taken as U*sigma, so that the
        # calibration scores and a new sample's scores travel the same path.
        self.scores_ = values @ self.loadings_
        # U is not kept: every reported quantity is defined from the loadings,
        # and holding an n x r matrix for nothing is how a 240-sample model
        # ends up carrying more state than it uses.
        del u
        return self

    def fit_transform(self, X: object) -> NDArray[np.float64]:
        self.fit(X)
        assert self.scores_ is not None
        return self.scores_.copy()

    def transform(self, X: object) -> NDArray[np.float64]:
        """Project samples onto the fitted loadings — `T = XP`, §4.

        `X` must have been through the identical preprocessing chain with the
        parameters estimated on the *calibration* set. Centring a new sample by
        its own mean produces plausible and entirely wrong scores, which is
        why the preprocessing kernels hold their fitted parameters rather than
        re-estimating on each block.
        """
        loadings = self._fitted_loadings()
        values = self._checked(X)
        scores: NDArray[np.float64] = values @ loadings
        return scores

    # ----------------------------------------------------------------------
    # explained variance, §6
    # ----------------------------------------------------------------------

    def explained_variance_ratio(self) -> NDArray[np.float64]:
        """Per retained component, over the total variance of the fitted matrix.

        The denominator sums all `r` eigenvalues, not the `a` retained ones.
        Normalising by the retained set would make the cumulative curve always
        reach 100%, which is useless and is a mistake seen in the wild.
        """
        eigenvalues = self._fitted_eigenvalues()
        ratio: NDArray[np.float64] = eigenvalues[: self.n_components] / eigenvalues.sum()
        return ratio

    def cumulative_explained_variance(self) -> NDArray[np.float64]:
        """The running total of `explained_variance_ratio()`."""
        cumulative: NDArray[np.float64] = np.cumsum(self.explained_variance_ratio())
        return cumulative

    # ----------------------------------------------------------------------
    # distance within the model plane, §7
    # ----------------------------------------------------------------------

    def hotelling_t2(self, X: object | None = None) -> NDArray[np.float64]:
        """`T^2_i = sum_k t_ik^2 / lambda_k` over the retained components.

        Weighting by the eigenvalue is the point: the same score on a minor
        component is further from the model's centre than on a major one.
        Defaults to the calibration samples.
        """
        eigenvalues = self._fitted_eigenvalues()[: self.n_components]
        scores = self._scores_of(X)
        t2: NDArray[np.float64] = ((scores**2) / eigenvalues).sum(axis=1)
        return t2

    def hotelling_t2_limit(self, alpha: float = 0.05, samples: LimitFor = "calibration") -> float:
        """The confidence limit, in the form that matches what is being plotted.

        Two limits, because they answer different questions, and **which one is
        drawn must be stated in the plot legend rather than left implicit**:

        * `samples="calibration"` — the beta form. A calibration sample's
          scores are not independent of the model that was fitted to them, and
          this is the exact limit for that case.
        * `samples="new"` — the F form, for samples projected onto an existing
          model.

        `n` and `a` are always the *calibration* model's, including for the new
        sample limit. The two converge as `n` grows and differ noticeably for
        small `n`, which is the common case in chemometrics, so both exist
        rather than one approximating the other.
        """
        self._check_alpha(alpha)
        n = self._fitted_n_samples()
        a = self.n_components

        if samples == "calibration":
            if n <= a + 1:
                raise ValueError(
                    f"the beta limit needs n > a + 1; this model has n={n} and a={a}. "
                    "No limit is defined, and none should be drawn."
                )
            quantile = float(beta.ppf(1.0 - alpha, a / 2.0, (n - a - 1) / 2.0))
            return (n - 1) ** 2 / n * quantile
        if samples == "new":
            if n <= a:
                raise ValueError(
                    f"the F limit needs n > a; this model has n={n} and a={a}. "
                    "No limit is defined, and none should be drawn."
                )
            quantile = float(f.ppf(1.0 - alpha, a, n - a))
            return a * (n**2 - 1) / (n * (n - a)) * quantile
        raise ValueError(f"samples must be 'calibration' or 'new', got {samples!r}")

    # ----------------------------------------------------------------------
    # distance from the model plane, §8
    # ----------------------------------------------------------------------

    def spe(self, X: object) -> NDArray[np.float64]:
        """Squared prediction error, `||x_i - t_i P^T||^2`.

        **The sum of squares, not the mean and not the root.** Other packages
        report one of the other two; converting is trivial and silent
        disagreement is not.

        `X` is required, where `hotelling_t2()` defaults to the calibration
        samples, and the asymmetry is not an oversight: the model keeps the
        calibration *scores*, so `T^2` needs nothing else, but SPE measures the
        part of `X` that the model does not contain and therefore cannot be
        recovered from the model. Pass the same preprocessed matrix that was
        passed to `fit()`.

        Exactly zero for every sample when `a == r`, where the model spans the
        whole row space and there is no residual left to measure.
        """
        loadings = self._fitted_loadings()
        values = self._checked(X)
        residual = values - (values @ loadings) @ loadings.T
        squared: NDArray[np.float64] = (residual**2).sum(axis=1)
        return squared

    def spe_limit(self, alpha: float = 0.05) -> float:
        """The Jackson–Mudholkar limit, §8.

        Built from `theta_m = sum_{k>a} lambda_k^m`, sums over the components
        the model *discarded* — which is the whole reason every eigenvalue is
        retained at fit. Box's chi-squared approximation is not used here; it
        is what several other packages use and is recorded in §13 as a known
        divergence so that a comparison against one of them is classified
        rather than failed.

        Raises when `a == r`: the residual is zero by construction, so there is
        no distribution to take a quantile of and no limit should be drawn.
        """
        self._check_alpha(alpha)
        eigenvalues = self._fitted_eigenvalues()
        discarded = eigenvalues[self.n_components :]
        if discarded.size == 0:
            raise ValueError(
                f"this model retains all {self.rank_} components, so every residual is "
                "zero by construction and no SPE limit exists. Report SPE as zero and "
                "draw no limit rather than taking a quantile of an empty sum."
            )

        # §11: eigenvalues below the rank tolerance are already excluded, so
        # theta is never driven by numerical noise from a null direction.
        theta1, theta2, theta3 = (float((discarded**m).sum()) for m in (1, 2, 3))
        h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * theta2**2)
        bracket = float(
            float(norm.ppf(1.0 - alpha)) * np.sqrt(2.0 * theta2 * h0**2) / theta1
            + 1.0
            + theta2 * h0 * (h0 - 1.0) / theta1**2
        )
        if bracket <= 0.0:
            raise ValueError(
                f"the Jackson-Mudholkar bracket came out at {bracket:g}, which has no "
                f"real {1 / h0:g} power. The residual eigenvalue spectrum is too "
                "degenerate for this approximation; report the SPE values without a limit."
            )
        return float(theta1 * bracket ** (1.0 / h0))

    # ----------------------------------------------------------------------
    # shared checks
    # ----------------------------------------------------------------------

    @staticmethod
    def _check_alpha(alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")

    def _unfitted(self) -> RuntimeError:
        return RuntimeError(
            f"{type(self).__name__} has not been fitted. Fit it on the preprocessed "
            "calibration matrix first."
        )

    def _fitted_loadings(self) -> NDArray[np.float64]:
        if self.loadings_ is None:
            raise self._unfitted()
        return self.loadings_

    def _fitted_eigenvalues(self) -> NDArray[np.float64]:
        if self.eigenvalues_ is None:
            raise self._unfitted()
        return self.eigenvalues_

    def _fitted_n_samples(self) -> int:
        if self.n_samples_ is None:
            raise self._unfitted()
        return self.n_samples_

    def _checked(self, X: object) -> NDArray[np.float64]:
        values = as_float64(X, "X")
        if self.n_variables_ is None:
            raise self._unfitted()
        if values.shape[1] != self.n_variables_:
            raise ValueError(
                f"the model was fitted on {self.n_variables_} variables and was given "
                f"{values.shape[1]}. Arrays are n_samples x n_variables and are never "
                "silently transposed; if this is a transposed matrix, transpose it "
                "deliberately."
            )
        return values

    def _scores_of(self, X: object | None) -> NDArray[np.float64]:
        if X is None:
            if self.scores_ is None:
                raise self._unfitted()
            return self.scores_
        return self.transform(X)
