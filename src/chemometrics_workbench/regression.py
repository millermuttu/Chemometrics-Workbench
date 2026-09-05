"""PLS1 regression by NIPALS, and the quantities an analyst reads off it.

The normative document is `docs/algorithms/pls-regression.md`; this module
implements it and nothing more. Every quantity here is defined in one of its
sections and the docstrings say which. Where this module and that document
disagree, one of them is a bug — decide which before changing either.

This is the kernel the project is judged on, so what it deliberately does
*not* do is worth as much as what it does.

## What it does not do

**It does not centre, and it does not scale.** `pls-regression.md` §3:
centring and scaling are explicit pipeline nodes and appear in the lineage.
`sklearn.cross_decomposition.PLSRegression` differs twice over — it centres
unconditionally *and* defaults to `scale=True` — which is why every matrix in
the parity fixture was centred before scikit-learn saw it, with `scale=False`
passed explicitly.

Fitting on uncentred data is legal here and almost always wrong: the response
has no intercept to absorb its offset, so the first component spends itself on
the mean. Warning about that is the application's job, not this module's.

**It knows nothing about the application.** Arrays in, arrays out — no
project, no schema, no I/O. A cross-validated fit takes its folds as index
arrays (`validation.Fold`), never a seed, so a run recorded as a
`ResolvedSplit` is replayed by handing its stored indices back.

**It takes no seed.** NIPALS PLS1 is deterministic and has no inner iteration
(§2): with a single response the weight vector has a closed form per
component, so there is no convergence tolerance whose setting could silently
differ from a reference implementation.

**It models one response** (§10). PLS2 needs an inner iteration with three
numerical settings that would have to be specified and matched for parity, and
is deferred to post-1.0. For a single response NIPALS and SIMPLS agree on
coefficients and predictions, so R `mdatools` is a valid reference for those —
but not for weights and loadings.

## The two things most likely to be got wrong elsewhere

**y is deflated as well as X** (§4). Implementations that skip it exist, and
their weights differ from the second component onward while the first
component looks identical — the kind of difference that passes a cursory
check.

**The SPE limit is not PCA's.** `pca.md` §8 builds Jackson-Mudholkar from the
eigenvalues of the discarded subspace; PLS components are not eigenvectors of
the covariance of X, so there is no residual eigenvalue sequence to sum. §9
uses a chi-squared moment match on the observed calibration residuals instead.
The `T^2` limit, by contrast, transfers unchanged, and is imported from
`decomposition` rather than written twice.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from chemometrics_workbench.arrays import as_float64, as_float64_vector
from chemometrics_workbench.decomposition import LimitFor, check_alpha, hotelling_t2_limit
from chemometrics_workbench.preprocessing import (
    AutoscaleTransformer,
    MeanCentreTransformer,
    RangeSelectTransformer,
    SavitzkyGolayTransformer,
    Transformer,
)
from chemometrics_workbench.validation import Fold, rmse, validate_partition

__all__ = [
    "FOLDABLE",
    "PLS",
    "coefficients_original_units",
    "cross_validated_predictions",
    "rmsecv_curve",
]

#: The preprocessing steps that fold into a coefficient vector, per
#: `pls-regression.md` §7. Each is an affine map on the variables whose
#: parameters were fixed at calibration time, so applying it and then taking a
#: dot product is the same as taking a dot product with transformed
#: coefficients.
#:
#: SNV, MSC and the baselines are absent on purpose and not by oversight: all
#: three depend on the sample being predicted — SNV divides each spectrum by
#: *its own* standard deviation, MSC regresses each against a stored reference
#: — so neither is a fixed linear map on X and both must be re-executed at
#: prediction time. An exported model therefore carries a residual
#: preprocessing chain plus coefficients, not always a bare coefficient vector.
FOLDABLE = (
    MeanCentreTransformer,
    AutoscaleTransformer,
    RangeSelectTransformer,
    SavitzkyGolayTransformer,
)

Block = Literal["x", "y"]


class PLS:
    """PLS1 regression by NIPALS, per `pls-regression.md`.

    `fit(X, y)` then `predict(X_new)`, duck-compatible with a scikit-learn
    estimator and importing nothing from it — the same rule the preprocessing
    kernels and `PCA` follow, and for the same reason: scikit-learn is the
    reference implementation the parity fixture is generated against, so a
    kernel built on it would be a wrapper around the thing we claim parity
    with.

    Fitted attributes, all in `pls-regression.md` §13:

    | | |
    | --- | --- |
    | `weights_` | `p x A`, unit-length columns, sign-fixed by §6 |
    | `x_scores_` | `n x A`, mutually orthogonal (§4) |
    | `x_loadings_` | `p x A` |
    | `y_loadings_` | `A`, the scalar `q_a` per component |
    | `rotations_` | `p x A`, `W(P'W)^-1`; `T = XR` applies to undeflated X (§5) |
    | `coefficients_` | `p`, `b = Rq`, sign-invariant (§5) |
    | `n_components_` | components actually fitted; below the request only if §4 stopped early |
    | `stopped_early_` | whether the response was exhausted before `n_components` |
    | `x_variance_`, `y_variance_` | sum of squares each component takes from each block |
    | `spe_` | the calibration residual `||E_A,i||^2` (§9), kept so the limit needs no X |
    """

    weights_: NDArray[np.float64] | None = None
    x_scores_: NDArray[np.float64] | None = None
    x_loadings_: NDArray[np.float64] | None = None
    y_loadings_: NDArray[np.float64] | None = None
    rotations_: NDArray[np.float64] | None = None
    coefficients_: NDArray[np.float64] | None = None
    x_variance_: NDArray[np.float64] | None = None
    y_variance_: NDArray[np.float64] | None = None
    x_total_variance_: float | None = None
    y_total_variance_: float | None = None
    spe_: NDArray[np.float64] | None = None
    n_components_: int | None = None
    stopped_early_: bool = False
    n_samples_: int | None = None
    n_variables_: int | None = None

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = int(n_components)

    # ----------------------------------------------------------------------
    # fitting, §4
    # ----------------------------------------------------------------------

    def fit(self, X: object, y: object) -> Self:
        values = as_float64(X, "X")
        response = as_float64_vector(y, "y")
        n_samples, n_variables = values.shape
        if response.size != n_samples:
            raise ValueError(
                f"X has {n_samples} samples and y has {response.size}. Arrays are "
                "n_samples x n_variables and are never silently transposed."
            )

        # §12: A may not exceed min(n-1, p). Naming both numbers, because
        # silently truncating would make the reported component count a lie.
        ceiling = min(n_samples - 1, n_variables)
        if self.n_components > ceiling:
            raise ValueError(
                f"{self.n_components} components were asked of a matrix with "
                f"{n_samples} samples and {n_variables} variables, which supports at "
                f"most min(n-1, p) = {ceiling}. Reduce n_components, or add samples."
            )

        residual_x = values.copy()
        residual_y = response.copy()

        # §4's stopping rule, measured against the first cross-product so that
        # it is a relative test and does not depend on the units of X or y.
        initial = float(np.linalg.norm(values.T @ response))
        floor = float(np.sqrt(np.finfo(np.float64).eps)) * initial
        if initial == 0.0:
            raise ValueError(
                "X and y have no covariance at all: X'y is zero, so there is no "
                "direction for a first component to take. Check that both were "
                "centred and that the response is not constant."
            )

        weights: list[NDArray[np.float64]] = []
        scores: list[NDArray[np.float64]] = []
        x_loadings: list[NDArray[np.float64]] = []
        y_loadings: list[float] = []

        for _ in range(self.n_components):
            cross = residual_x.T @ residual_y
            if float(np.linalg.norm(cross)) <= floor:
                self.stopped_early_ = True
                warnings.warn(
                    f"the response was exhausted after {len(weights)} components: X'f "
                    f"fell to {np.linalg.norm(cross):g}, at or below "
                    f"{floor:g}. Fitting stopped there rather than continuing on "
                    "numerical noise; n_components_ records what was actually fitted "
                    "(pls-regression.md §4).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            weight = cross / float(np.linalg.norm(cross))
            # §6: keyed on the largest-magnitude weight, ties to the smallest
            # index — argmax already returns the first maximum. Flipping the
            # weight here rather than afterwards carries the sign through t, p
            # and q together, which is what keeps the four consistent.
            if weight[int(np.abs(weight).argmax())] < 0:
                weight = -weight

            score = residual_x @ weight
            score_ss = float(score @ score)
            x_loading = (residual_x.T @ score) / score_ss
            y_loading = float(residual_y @ score) / score_ss

            # Both blocks are deflated (§4). Skipping the y deflation is the
            # classic divergence and it changes the weights from component 2 on.
            residual_x = residual_x - np.outer(score, x_loading)
            residual_y = residual_y - y_loading * score

            weights.append(weight)
            scores.append(score)
            x_loadings.append(x_loading)
            y_loadings.append(y_loading)

        if not weights:
            raise ValueError(
                "no component could be fitted: the first cross-product is already at "
                "the exhaustion threshold. There is nothing here to regress."
            )

        self.weights_ = np.column_stack(weights)
        self.x_scores_ = np.column_stack(scores)
        self.x_loadings_ = np.column_stack(x_loadings)
        self.y_loadings_ = np.asarray(y_loadings, dtype=np.float64)
        self.n_components_ = len(weights)

        # §5: P'W is upper triangular with unit diagonal, so it is always
        # invertible and no pseudo-inverse is called for. solve() rather than
        # inv() for the usual reason.
        self.rotations_ = self.weights_ @ np.linalg.inv(self.x_loadings_.T @ self.weights_)
        self.coefficients_ = self.rotations_ @ self.y_loadings_

        # Sums of squares each component takes from each block. X is a rank-one
        # outer product per component, so its share is ||t_a||^2 ||p_a||^2; the
        # scores are orthogonal, so y's share is q_a^2 ||t_a||^2 and the shares
        # add up exactly rather than approximately.
        score_sums = np.asarray([float(t @ t) for t in scores])
        self.x_variance_ = score_sums * np.asarray([float(p @ p) for p in x_loadings])
        self.y_variance_ = self.y_loadings_**2 * score_sums
        self.x_total_variance_ = float((values**2).sum())
        self.y_total_variance_ = float(response @ response)

        # §9: the calibration SPE, kept because the chi-squared moment match
        # needs the calibration residuals and the model deliberately keeps no
        # copy of X. n floats, not n x p.
        self.spe_ = (residual_x**2).sum(axis=1)
        self.n_samples_ = n_samples
        self.n_variables_ = n_variables
        return self

    # ----------------------------------------------------------------------
    # using the model, §7
    # ----------------------------------------------------------------------

    def transform(self, X: object) -> NDArray[np.float64]:
        """`T_new = X_new R` (§5, §7).

        `X_new` must have been through the identical preprocessing chain with
        the parameters estimated on the **calibration** set. Centring a new
        sample by its own mean gives plausible and entirely wrong scores.
        """
        scores: NDArray[np.float64] = self._checked(X) @ self._fitted("rotations_")
        return scores

    def predict(self, X: object) -> NDArray[np.float64]:
        """`y_hat = X b` (§5), in the units of the matrix that was fitted.

        The response's original units are restored by the pipeline, not here:
        this kernel was handed a centred `y` and returns predictions on that
        same centred scale. `metrics-and-validation.md` §2 requires every
        metric to be computed in the original units, which is the executor's
        job — and `cross_validated_predictions()` below does it, because it
        owns the centring it applied.
        """
        predictions: NDArray[np.float64] = self._checked(X) @ self._fitted("coefficients_")
        return predictions

    # ----------------------------------------------------------------------
    # VIP, §8
    # ----------------------------------------------------------------------

    def vip(self) -> NDArray[np.float64]:
        """Variable importance in projection, Wold's form (§8).

        Weighted by `SS_a = q_a^2 (t_a't_a)`, the sum of squares of `y` each
        component explains, so a variable is important when it drives the
        components that predict — not merely the components that are large.

        Satisfies `sum_j VIP_j^2 = p` exactly, which is the whole origin of the
        "VIP greater than 1" rule of thumb and is a cheap unit test. **VIP is a
        property of the fitted model, not of the data**: it depends on A, and
        reporting it without the component count is meaningless.
        """
        weights = self._fitted("weights_")
        explained = self._fitted("y_variance_")
        n_variables = weights.shape[0]
        weighted = (weights**2) @ explained
        vip: NDArray[np.float64] = np.sqrt(n_variables * weighted / float(explained.sum()))
        return vip

    # ----------------------------------------------------------------------
    # explained variance
    # ----------------------------------------------------------------------

    def explained_variance_ratio(self, block: Block = "x") -> NDArray[np.float64]:
        """Per component, the share of a block's sum of squares it takes.

        `block="x"` is `||t_a||^2 ||p_a||^2` over the total sum of squares of
        the fitted matrix — the quantity R `pls` prints as `X` in `summary()`.
        `block="y"` is `q_a^2 ||t_a||^2` over the total sum of squares of the
        fitted response, and because the scores are orthogonal its running
        total is exactly the R^2 of the model at that component count.

        Both denominators are the *whole* block, never the part the retained
        components happen to reach; normalising over the retained components
        would make the cumulative curve always reach 100%.
        """
        if block == "x":
            share = self._fitted("x_variance_") / self._fitted_total("x_total_variance_")
        elif block == "y":
            share = self._fitted("y_variance_") / self._fitted_total("y_total_variance_")
        else:
            raise ValueError(f"block must be 'x' or 'y', got {block!r}")
        ratio: NDArray[np.float64] = share
        return ratio

    def cumulative_explained_variance(self, block: Block = "x") -> NDArray[np.float64]:
        """The running total of `explained_variance_ratio()` — the curve read
        off a summary table, and for `block="y"` the R^2 at each A."""
        cumulative: NDArray[np.float64] = np.cumsum(self.explained_variance_ratio(block))
        return cumulative

    # ----------------------------------------------------------------------
    # diagnostics, §9
    # ----------------------------------------------------------------------

    def hotelling_t2(self, X: object | None = None) -> NDArray[np.float64]:
        """`T^2_i = sum_a t_ia^2 / lambda_a` with `lambda_a = t_a't_a/(n-1)` (§9).

        Valid because the NIPALS X-scores are mutually orthogonal (§4) — for a
        score matrix whose columns were correlated this sum would not be a
        Mahalanobis distance at all. Defaults to the calibration samples.
        """
        eigenvalues = self.score_eigenvalues()
        scores = self._fitted("x_scores_") if X is None else self.transform(X)
        t2: NDArray[np.float64] = ((scores**2) / eigenvalues).sum(axis=1)
        return t2

    def hotelling_t2_limit(self, alpha: float = 0.05, samples: LimitFor = "calibration") -> float:
        """`pca.md` §7's limit, unchanged (§9) — see `decomposition.hotelling_t2_limit`."""
        return hotelling_t2_limit(
            self._fitted_n_samples(), self._fitted_n_components(), alpha=alpha, samples=samples
        )

    def spe(self, X: object) -> NDArray[np.float64]:
        """`||x_i - t_i P'||^2`, the squared distance from the model plane (§9).

        The sum of squares, not the mean and not the root — other packages
        report one of the other two, and converting is trivial where silent
        disagreement is not. `X` is required for the same reason as in `PCA`:
        the residual measures the part of X the model does not contain, so it
        cannot be recovered from the model. Pass the same preprocessed matrix
        that was passed to `fit()`.
        """
        values = self._checked(X)
        residual = values - self.transform(values) @ self._fitted("x_loadings_").T
        squared: NDArray[np.float64] = (residual**2).sum(axis=1)
        return squared

    def spe_limit(self, alpha: float = 0.05) -> float:
        """The chi-squared moment match on the calibration residuals (§9).

        **Not Jackson-Mudholkar.** That limit is built from the eigenvalues of
        the discarded PCA subspace, and PLS components are not eigenvectors of
        the covariance of X, so there is no residual eigenvalue sequence to sum
        — a limit computed that way here would be a number with no derivation
        behind it. Instead `g = v/2m` and `h = 2m^2/v` are matched to the mean
        and variance of the observed calibration SPE, and the limit is
        `g * chi2_alpha(h)`. This difference from PCA must be stated wherever a
        PLS SPE limit is drawn.
        """
        check_alpha(alpha)
        observed = self._fitted("spe_")

        # A model that spans the row space leaves a residual that is numerical
        # noise rather than zero — around 1e-32 of the matrix, not 0.0 — and a
        # quantile of noise is a number with nothing behind it. Judged relative
        # to the magnitude of the fitted matrix, never against 0.0, the same
        # rule `preprocessing.py` uses for a dead variable.
        total = self._fitted_total("x_total_variance_")
        floor = max(self._fitted_n_samples(), self._checked_n_variables()) * float(
            np.finfo(np.float64).eps
        )
        if total > 0.0 and float(observed.sum()) / total <= floor:
            raise ValueError(
                f"this model retains {self._fitted_n_components()} components and leaves "
                "no residual: the calibration SPE is numerical noise at "
                f"{float(observed.sum()) / total:g} of the matrix. Report SPE as zero and "
                "draw no limit rather than taking a quantile of noise."
            )

        mean = float(observed.mean())
        variance = float(observed.var(ddof=1))
        if mean <= 0.0 or variance <= 0.0:
            raise ValueError(
                f"the calibration SPE has mean {mean:g} and variance {variance:g}, so "
                "there is no chi-squared distribution to match. A model spanning the "
                "whole row space has no residual: report SPE as zero and draw no limit."
            )
        g = variance / (2.0 * mean)
        h = 2.0 * mean**2 / variance
        return float(g * chi2.ppf(1.0 - alpha, h))

    # ----------------------------------------------------------------------
    # shared checks
    # ----------------------------------------------------------------------

    def _unfitted(self) -> RuntimeError:
        return RuntimeError(
            f"{type(self).__name__} has not been fitted. Fit it on the preprocessed "
            "calibration matrix and its response first."
        )

    def _fitted(self, name: str) -> NDArray[np.float64]:
        value = getattr(self, name)
        if value is None:
            raise self._unfitted()
        array: NDArray[np.float64] = value
        return array

    def _fitted_total(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise self._unfitted()
        return float(value)

    def _fitted_n_samples(self) -> int:
        if self.n_samples_ is None:
            raise self._unfitted()
        return self.n_samples_

    def _checked_n_variables(self) -> int:
        if self.n_variables_ is None:
            raise self._unfitted()
        return self.n_variables_

    def _fitted_n_components(self) -> int:
        if self.n_components_ is None:
            raise self._unfitted()
        return self.n_components_

    def score_eigenvalues(self) -> NDArray[np.float64]:
        """`lambda_a = t_a't_a / (n - 1)`, the variance each component's scores carry (§9).

        Public because it is reported rather than internal: it is what
        `hotelling_t2` divides by, and it is what a T-squared ellipse is drawn
        from — `PCA.eigenvalues_` is the same quantity for the same purpose.
        Publishing it was missed when #142 gave PLS a result, which left the
        ellipse on a regression's scores plot with `NaN` radii (#146).
        """
        scores = self._fitted("x_scores_")
        eigenvalues: NDArray[np.float64] = (scores**2).sum(axis=0) / (self._fitted_n_samples() - 1)
        return eigenvalues

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


# --------------------------------------------------------------------------
# export to original units, pls-regression.md §7
# --------------------------------------------------------------------------


def coefficients_original_units(
    coefficients: object,
    transformers: Sequence[Transformer],
    *,
    n_variables: int,
    y_mean: float = 0.0,
    y_scale: float = 1.0,
) -> tuple[NDArray[np.float64], float]:
    """Coefficients readable against the raw axis, and the intercept they need.

    `PROPOSAL.md` §9 promises a portable model: a coefficient vector and a
    snippet that depends only on NumPy. That requires the preprocessing to be
    folded into `b`, and §7 of `pls-regression.md` says which steps can be —
    centring shifts the intercept, autoscaling divides each coefficient by its
    scale, range selection drops coefficients, and Savitzky-Golay folds as the
    banded matrix of its own convolution.

    Rather than writing those four rules out and keeping them in step with the
    kernels, the chain is **measured**. Every foldable step is affine, so
    passing the identity matrix through the fitted chain recovers the map it
    applies: `f(0)` is the offset and `f(e_j) - f(0)` is what variable `j`
    contributes. Savitzky-Golay's edge handling comes out exactly right without
    anyone re-deriving `mode="interp"`, and a change to a kernel's arithmetic
    cannot leave a stale copy of it here.

    Returns `(b, intercept)` such that `y_hat = intercept + X_raw @ b`, in the
    response's original units. `y_mean` and `y_scale` are the centring and
    scaling applied to the response before fitting, which are not pipeline
    nodes — `cross_validated_predictions` centres `y` by the training fold's
    mean, and this puts that back.

    Raises `ValueError` naming the step when the chain is not foldable, which
    is the "says so when it is not available" §7 asks for. A caller building an
    export catches it and ships the residual chain instead.
    """
    if n_variables < 1:
        raise ValueError(f"n_variables must be at least 1, got {n_variables}")
    for transformer in transformers:
        if not isinstance(transformer, FOLDABLE):
            raise ValueError(
                f"{type(transformer).__name__} cannot be folded into a coefficient "
                "vector: it depends on the sample being predicted, so it is not a fixed "
                "linear map on X and has to be re-executed at prediction time "
                "(pls-regression.md §7). Export it as part of a residual preprocessing "
                "chain instead."
            )

    # ponytail: an n_variables x n_variables probe, which is 128 MB of float64
    # at §13's envelope of 4,000 variables. Passing it in column blocks would
    # bound that if it ever bites; the arithmetic is unchanged either way.
    probe = np.eye(n_variables, dtype=np.float64)
    zero = np.zeros((1, n_variables), dtype=np.float64)
    for transformer in transformers:
        probe = transformer.transform(probe)
        zero = transformer.transform(zero)
    offset = zero[0]
    linear = probe - offset

    weights = as_float64_vector(coefficients, "coefficients")
    if linear.shape[1] != weights.size:
        raise ValueError(
            f"the chain maps {n_variables} variables to {linear.shape[1]}, and the "
            f"coefficient vector has {weights.size}. They are the same number or one of "
            "them is from a different model."
        )

    folded: NDArray[np.float64] = y_scale * (linear @ weights)
    intercept = float(y_mean + y_scale * float(offset @ weights))
    return folded, intercept


# --------------------------------------------------------------------------
# cross-validation, metrics-and-validation.md §7 and §9
# --------------------------------------------------------------------------


def cross_validated_predictions(
    X: object, y: object, folds: list[Fold], n_components: int
) -> NDArray[np.float64]:
    """One held-out prediction per sample, in the response's original units.

    **Centring is refitted on each training fold** (`metrics-and-validation.md`
    §9): the held-out samples are pushed through with the training fold's mean,
    exactly as a new sample is at prediction time. Centring once on everything
    before the split leaks the validation samples into the training statistics
    and makes the estimate optimistic — the pipeline validator warns about that
    where a recipe does it; here it simply is not done.

    `folds` are realised index arrays, never a seed (§10), so a stored
    `ResolvedSplit` replays through `validation.folds_from_indices()`.
    """
    values = as_float64(X, "X")
    response = as_float64_vector(y, "y")
    if response.size != values.shape[0]:
        raise ValueError(f"X has {values.shape[0]} samples and y has {response.size}")
    validate_partition(folds, values.shape[0])

    held_out = np.empty_like(response)
    for fold in folds:
        train_x = values[fold.train]
        train_y = response[fold.train]
        x_mean = train_x.mean(axis=0)
        y_mean = float(train_y.mean())
        model = PLS(n_components).fit(train_x - x_mean, train_y - y_mean)
        held_out[fold.test] = model.predict(values[fold.test] - x_mean) + y_mean
    return held_out


def rmsecv_curve(
    X: object, y: object, folds: list[Fold], max_components: int
) -> NDArray[np.float64]:
    """RMSECV for `A = 1 ... max_components`, from one fold assignment (§9).

    Residuals are pooled across folds and rooted once (§7) — not averaged as
    per-fold RMSEs, which weights every fold equally instead of every sample
    and differs whenever the folds are uneven.

    The same split is used for every component count, so the curve is one
    experiment rather than `A` unrelated ones. Choosing `A` at its minimum and
    then quoting that minimum as the model's expected error is optimistic; that
    is the user's call and the application does not make it for them.
    """
    if max_components < 1:
        raise ValueError(f"a curve needs at least one component, got {max_components}")
    response = as_float64_vector(y, "y")
    return np.asarray(
        [
            rmse(response, cross_validated_predictions(X, response, folds, a))
            for a in range(1, max_components + 1)
        ]
    )
