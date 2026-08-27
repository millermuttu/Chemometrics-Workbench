"""The preprocessing kernels.

Two halves, and they behave differently. The deterministic half — SNV, MSC,
mean centring, autoscaling, normalisation, range selection — has no convention
freedom in its arithmetic, so each of those should reach the tightest parity
tier the harness offers. The other half — Savitzky-Golay smoothing and
derivatives, and baseline correction — is where implementations genuinely
disagree, mostly at the spectrum bounds. Every free choice in it is fixed
here, in the class docstrings and in `docs/algorithms/smoothing-and-baselines.md`,
and recorded in § "Known divergences" at the bottom of this docstring.

## The interface, and why it is not scikit-learn's

Each transformer is `fit(X)` then `transform(X)`, duck-compatible with a
scikit-learn transformer, **without importing scikit-learn**. That is
deliberate. `scikit-learn` is a dev-only dependency of this project because it
is the *reference implementation* the parity fixtures are generated against; a
kernel that inherits from `BaseEstimator` is not a kernel we can claim parity
for, it is a wrapper around the thing we are claiming parity against.

Duck compatibility is what actually buys anything: a `chemotools` transformer
(evaluated in #13) drops in wherever one of these does, and vice versa.

## Why they are stateful at all

`metrics-and-validation.md` §9 requires that every node downstream of a split
is refitted on the training fold alone, and that held-out samples then pass
through with the *training fold's* parameters. A pure function cannot carry
"the calibration mean" across that boundary. So the parameters estimated at
fit time are held on the transformer, and `transform` uses those and never
re-estimates.

The stateless transformers — SNV, normalisation — still have a `fit`, and
still record the variable count, because that is what enforces the shape
contract on later calls.

## Conventions every transformer here follows

- **Array shape is `n_samples × n_variables`, always.** Recorded at fit; a
  later `transform` with a different variable count is an error. This is what
  catches a transposed array, which is otherwise indistinguishable from a
  legitimate one.
- **The caller's array is never modified**, and never returned. Every result
  is freshly allocated.
- **Computation and results are float64.** A float32 array is promoted on
  entry, as in `pca.md` §11: storage dtype is the caller's business, numerical
  results are not.
- **Missing values are rejected**, naming the rows and columns, per
  `pca.md` §10. An all-NaN row is rejected by that rule like any other.
- **A zero scale is an error, not a substituted 1.** See below. "Zero" is
  judged relative to the magnitude of the data, because the standard deviation
  of a genuinely constant row is 1e-16 rather than 0.

## Zero-variance rows and columns

A constant spectrum has no scatter to correct; a constant variable has no
spread to scale by. Both mean dividing by zero.

**This raises, naming the offending rows or columns.** The alternative — the
one `sklearn.preprocessing.StandardScaler` takes, substituting a scale of 1 —
reports a successful autoscale while leaving that column as exact zeros, and
the user never learns a detector channel is dead. That is the failure mode
`pca.md` §10 rejects for missing values, and the same argument applies here.

It is a real divergence from scikit-learn and is recorded as one.

## Known divergences from other packages

**Autoscale `ddof`** is 1 by default, the sample convention used throughout
this project. `StandardScaler` is fixed at `ddof=0`, so parity against it must
pass `ddof=0` explicitly.

**Zero variance** raises here, naming the row or column. `StandardScaler`
substitutes a scale of 1.

**`normalise(norm="area")`** divides by the signed sum of the row. Some tools
integrate against the real axis by the trapezoid rule instead.

**The MSC reference** is estimated at fit and reused, or supplied by the
caller. Some tools re-estimate it on every block transformed, which makes a
prediction depend on what was predicted alongside it.

**Savitzky-Golay edge handling** is `interp` and is not configurable. Packages
defaulting to `mirror` or `nearest` — and packages that simply leave the first
and last half-window untouched — give different values there and the same
values everywhere else. See `SavitzkyGolayTransformer`.

**Savitzky-Golay derivatives are per variable index** unless `delta` is given.
Tools that ask for the axis and divide by its real spacing report a derivative
per nanometre or per wavenumber, which differs by a constant factor.

**Baselines are estimated against the variable index**, not the axis. Exact
for the polynomial method, an assumption of uniform spacing for the other two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

from chemometrics_workbench.arrays import as_float64
from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    BaselineCorrect,
    MeanCentre,
    Normalise,
    PreprocessStep,
    RangeSelect,
    SavitzkyGolay,
)

__all__ = [
    "AutoscaleTransformer",
    "BaselineCorrectTransformer",
    "MSCTransformer",
    "MeanCentreTransformer",
    "NormaliseTransformer",
    "RangeSelectTransformer",
    "SNVTransformer",
    "SavitzkyGolayTransformer",
    "Transformer",
    "from_spec",
]

Norm = Literal["l1", "l2", "max", "area"]
MSCReference = Literal["mean", "median", "supplied"]
BaselineMethod = Literal["asls", "rubberband", "polynomial"]


# --------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------


# A scale that "should" be zero rarely is, in floating point: the standard
# deviation of a genuinely constant row of 0.7 comes out at 1.2e-16, not 0.
# So "dead" is judged relative to the magnitude of the data, eight units in the
# last place of it, with the same floor of 1.0 the parity harness uses and for
# the same reason — a quantity whose values are all tiny must not be held to
# bit-exactness. Anything a real instrument produced will sit orders of
# magnitude above this.
_ZERO_ULPS = 8


def _dead_threshold(magnitude: NDArray[np.float64] | float) -> NDArray[np.float64]:
    floor = np.maximum(np.asarray(magnitude, dtype=np.float64), 1.0)
    return floor * (_ZERO_ULPS * float(np.finfo(np.float64).eps))


def _reject_zero(
    scale: NDArray[np.float64],
    magnitude: NDArray[np.float64] | float,
    axis_name: str,
    what: str,
) -> None:
    """A zero scale is a dead row or column, and is surfaced rather than patched."""
    dead = np.flatnonzero(np.abs(scale) <= _dead_threshold(magnitude))
    if dead.size:
        listed = ", ".join(str(int(i)) for i in dead[:10])
        more = f" and {dead.size - 10} more" if dead.size > 10 else ""
        raise ValueError(
            f"{what} is zero for {axis_name} {listed}{more}. Dividing by it is "
            "undefined, and substituting a scale of 1 would report a successful "
            "transform over a dead channel. Exclude it, or fix the data."
        )


class Transformer(ABC):
    """Base for the preprocessing kernels: `fit` then `transform`.

    Subclasses implement `_fit` and `_transform`. Everything the conventions
    in the module docstring promise — dtype, shape contract, rejection of
    missing values, never touching the caller's array — is enforced here so
    that no kernel has to remember it.
    """

    n_variables_: int | None = None

    def fit(self, X: object) -> Self:
        values = as_float64(X, "X")
        self.n_variables_ = int(values.shape[1])
        self._fit(values)
        return self

    def transform(self, X: object) -> NDArray[np.float64]:
        if self.n_variables_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Even a transformer that "
                "estimates nothing must be fitted, because that is what records the "
                "variable count the shape contract is checked against."
            )
        values = as_float64(X, "X")
        if values.shape[1] != self.n_variables_:
            raise ValueError(
                f"{type(self).__name__} was fitted on {self.n_variables_} variables and "
                f"was given {values.shape[1]}. Arrays are n_samples x n_variables and "
                "are never silently transposed; if this is a transposed matrix, "
                "transpose it deliberately."
            )
        return self._transform(values)

    def fit_transform(self, X: object) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    def _fit(self, X: NDArray[np.float64]) -> None:  # noqa: B027
        """Estimate whatever this transformer carries.

        Deliberately concrete and empty: SNV and normalisation estimate
        nothing, and forcing them to write an empty override would be
        ceremony, not safety.
        """

    @abstractmethod
    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


# --------------------------------------------------------------------------
# scatter correction
# --------------------------------------------------------------------------


class SNVTransformer(Transformer):
    """Standard normal variate: centre and scale each spectrum by its own moments.

    Row-wise and stateless — `pls-regression.md` §7 records the consequence,
    that SNV cannot be folded into exported coefficients precisely because it
    depends on the sample being predicted.

    `ddof` is 1, the sample convention used throughout this project.
    """

    def __init__(self, ddof: int = 1) -> None:
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof}")
        self.ddof = ddof

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if X.shape[1] <= self.ddof:
            raise ValueError(
                f"SNV with ddof={self.ddof} needs more than {self.ddof} variables, "
                f"got {X.shape[1]}."
            )
        spread = X.std(axis=1, ddof=self.ddof)
        _reject_zero(
            spread, np.abs(X).max(axis=1), "sample", "the standard deviation of the spectrum"
        )
        return (X - X.mean(axis=1, keepdims=True)) / spread[:, np.newaxis]


class MSCTransformer(Transformer):
    """Multiplicative scatter correction against a reference spectrum.

    Each spectrum is regressed on the reference, `x = a + b*r`, and returned as
    `(x - a)/b`. The reference is estimated once at fit and reused, so a
    prediction set is corrected against the *calibration* reference — the same
    rule `pls-regression.md` §7 states for every fitted preprocessing
    parameter. Re-estimating it per block would make a prediction depend on
    which other samples happened to be predicted alongside it.

    Like SNV, MSC is not foldable into exported coefficients.
    """

    reference_: NDArray[np.float64] | None = None

    def __init__(
        self,
        reference: MSCReference = "mean",
        reference_spectrum: object | None = None,
    ) -> None:
        self.reference = reference
        if reference == "supplied":
            if reference_spectrum is None:
                raise ValueError(
                    "reference='supplied' needs a reference_spectrum. This is a "
                    "library call only: the schema's MSC step offers 'mean' and "
                    "'median', so a saved pipeline never reaches here."
                )
            supplied = np.asarray(reference_spectrum, dtype=np.float64).ravel()
            if not np.isfinite(supplied).all():
                raise ValueError("reference_spectrum holds non-finite values")
            self._supplied: NDArray[np.float64] | None = supplied
        else:
            if reference_spectrum is not None:
                raise ValueError(
                    f"reference={reference!r} estimates the reference from the data; "
                    "passing reference_spectrum as well would be ignored."
                )
            self._supplied = None

    def _fit(self, X: NDArray[np.float64]) -> None:
        if self.reference == "supplied":
            assert self._supplied is not None
            if self._supplied.size != X.shape[1]:
                raise ValueError(
                    f"reference_spectrum has {self._supplied.size} variables, X has {X.shape[1]}"
                )
            self.reference_ = self._supplied.copy()
        elif self.reference == "median":
            self.reference_ = np.median(X, axis=0)
        else:
            self.reference_ = X.mean(axis=0)

        centred = self.reference_ - self.reference_.mean()
        magnitude = float(np.abs(self.reference_).max())
        if float(np.abs(centred).max()) <= float(_dead_threshold(magnitude)):
            raise ValueError(
                "the MSC reference spectrum is constant, so no spectrum can be "
                "regressed against it. Check the fit set, or supply a reference."
            )

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.reference_ is not None
        reference = self.reference_ - self.reference_.mean()
        centred = X - X.mean(axis=1, keepdims=True)

        slope = (centred @ reference) / float(reference @ reference)
        _reject_zero(slope, 1.0, "sample", "the regression slope against the reference")

        intercept = X.mean(axis=1) - slope * self.reference_.mean()
        return (X - intercept[:, np.newaxis]) / slope[:, np.newaxis]


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------


class MeanCentreTransformer(Transformer):
    """Subtract the column means estimated at fit time.

    The mean is the *fit set's*, always. `pca.md` §4 and
    `metrics-and-validation.md` §9 both turn on this: a held-out sample centred
    by its own mean has had the model's information leaked into it.
    """

    mean_: NDArray[np.float64] | None = None

    def _fit(self, X: NDArray[np.float64]) -> None:
        self.mean_ = X.mean(axis=0)

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.mean_ is not None
        return X - self.mean_


class AutoscaleTransformer(Transformer):
    """Centre by the fit set's column means and scale by its column deviations.

    `ddof` defaults to 1, matching the sample-variance convention used for
    eigenvalues in `pca.md` §4 and for SEC and SEP in
    `metrics-and-validation.md` §5. `sklearn.preprocessing.StandardScaler` is
    fixed at `ddof=0`; parity against it must set `ddof=0` explicitly.

    Appropriate when variables carry different units or wildly different
    variances, which is uncommon for a single spectral block — `pca.md` §2
    recommends mean centring alone for spectra.
    """

    mean_: NDArray[np.float64] | None = None
    scale_: NDArray[np.float64] | None = None

    def __init__(self, ddof: int = 1) -> None:
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof}")
        self.ddof = ddof

    def _fit(self, X: NDArray[np.float64]) -> None:
        if X.shape[0] <= self.ddof:
            raise ValueError(
                f"autoscaling with ddof={self.ddof} needs more than {self.ddof} "
                f"samples, got {X.shape[0]}."
            )
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=self.ddof)
        _reject_zero(
            scale, np.abs(X).max(axis=0), "variable", "the standard deviation of the variable"
        )
        self.scale_ = scale

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.mean_ is not None
        assert self.scale_ is not None
        return (X - self.mean_) / self.scale_


class NormaliseTransformer(Transformer):
    """Scale each spectrum to unit norm. Row-wise and stateless.

    | `norm` | Divisor |
    | --- | --- |
    | `l1` | sum of absolute values |
    | `l2` | Euclidean length |
    | `max` | largest absolute value |
    | `area` | **signed** sum of the values |

    `area` is the discrete integral at unit spacing, not a trapezoid against
    the real axis. All three reference datasets are uniformly spaced, and
    threading the axis into a row-wise transform to gain a factor that cancels
    is not worth the coupling. It is signed, so a spectrum whose values sum to
    zero — a derivative, most likely — is rejected rather than exploded.
    """

    def __init__(self, norm: Norm = "l2") -> None:
        if norm not in ("l1", "l2", "max", "area"):
            raise ValueError(f"unknown norm {norm!r}")
        self.norm = norm

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.norm == "l1":
            divisor = np.abs(X).sum(axis=1)
        elif self.norm == "l2":
            divisor = np.sqrt((X**2).sum(axis=1))
        elif self.norm == "max":
            divisor = np.abs(X).max(axis=1)
        else:
            divisor = X.sum(axis=1)

        _reject_zero(
            divisor, np.abs(X).max(axis=1), "sample", f"the {self.norm} norm of the spectrum"
        )
        normalised: NDArray[np.float64] = X / divisor[:, np.newaxis]
        return normalised


# --------------------------------------------------------------------------
# variable selection
# --------------------------------------------------------------------------


class RangeSelectTransformer(Transformer):
    """Keep the variables whose axis value falls in `[start, end]`.

    Needs the axis, because the bounds are in real units — nanometres, or
    wavenumbers — and not column indices. The axis is a property of the dataset
    and is known when the pipeline is built, so it is supplied at construction
    rather than at every call.

    Both axis directions are handled: wavenumber axes usually descend, and an
    interval is an interval whichever way the numbers run. Bounds are
    inclusive; an interval containing no variable is an error rather than an
    empty matrix, because a zero-width block fails much later and much less
    informatively.
    """

    mask_: NDArray[np.bool_] | None = None

    def __init__(self, start: float, end: float, axis: object) -> None:
        if start >= end:
            raise ValueError(f"start must be less than end; got {start} and {end}")
        self.start = float(start)
        self.end = float(end)
        self.axis = np.asarray(axis, dtype=np.float64).ravel()
        if self.axis.size == 0:
            raise ValueError("range selection needs the variable axis")

    def _fit(self, X: NDArray[np.float64]) -> None:
        if self.axis.size != X.shape[1]:
            raise ValueError(f"axis has {self.axis.size} values but X has {X.shape[1]} variables")
        mask = (self.axis >= self.start) & (self.axis <= self.end)
        if not mask.any():
            raise ValueError(
                f"no variable falls in [{self.start}, {self.end}]; the axis spans "
                f"[{self.axis.min()}, {self.axis.max()}]"
            )
        self.mask_ = mask

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.mask_ is not None
        return X[:, self.mask_]

    def selected_axis(self) -> NDArray[np.float64]:
        """The axis of the variables kept — the dataset's axis must follow the data."""
        if self.mask_ is None:
            raise RuntimeError("RangeSelectTransformer has not been fitted")
        return self.axis[self.mask_]


# --------------------------------------------------------------------------
# from the schema
# --------------------------------------------------------------------------


def from_spec(
    step: PreprocessStep,
    *,
    axis: object | None = None,
) -> Transformer:
    """Build the transformer a `PreprocessStep` describes.

    This is the executor's entry point, and the reason the schema's
    discriminated union is worth having: an unsupported step fails here with
    its name, rather than halfway through a ten-minute cross-validation.

    One step needs something the schema does not carry: `RangeSelect` needs
    the variable axis, which belongs to the dataset rather than to the recipe.
    Everything else is built from the step alone. `MSC(reference="supplied")`
    used to be the second such case; the schema no longer offers it, so there
    is nothing here to raise about.
    """
    match step:
        case SNV():
            return SNVTransformer()
        case MSC():
            return MSCTransformer(step.reference)
        case MeanCentre():
            return MeanCentreTransformer()
        case Autoscale():
            return AutoscaleTransformer(ddof=step.ddof)
        case Normalise():
            return NormaliseTransformer(step.norm)
        case SavitzkyGolay():
            # No spacing field on the step, so the derivative is per variable
            # index. That is the recipe's meaning, and it is documented rather
            # than guessed at from the axis.
            return SavitzkyGolayTransformer(step.window_length, step.polyorder, step.deriv)
        case BaselineCorrect():
            defaults = BaselineCorrectTransformer(step.method)
            return BaselineCorrectTransformer(
                step.method,
                order=defaults.order if step.order is None else step.order,
                lam=defaults.lam if step.lam is None else step.lam,
                p=defaults.p if step.p is None else step.p,
            )
        case RangeSelect():
            if axis is None:
                raise ValueError(
                    "RangeSelect is expressed in axis units and needs the dataset's "
                    "variable axis; pass axis=."
                )
            return RangeSelectTransformer(step.start, step.end, axis)
        case _:
            raise NotImplementedError(f"no kernel for preprocessing step {step.kind!r} yet.")


# --------------------------------------------------------------------------
# smoothing and derivatives
# --------------------------------------------------------------------------


class SavitzkyGolayTransformer(Transformer):
    """Savitzky-Golay smoothing and derivatives, as an explicit linear operator.

    A local polynomial of degree `polyorder` is fitted by least squares to
    every window of `window_length` variables, and the fitted polynomial —
    or its `deriv`-th derivative — is evaluated at the window centre. The
    whole filter is therefore a single matrix `M` with `X_filtered = X @ M.T`,
    and `convolution_matrix()` returns it.

    **That matrix is the point.** `pls-regression.md` §7 records that Savitzky-
    Golay, unlike SNV and MSC, is foldable into exported regression
    coefficients precisely because it does not depend on the sample being
    filtered: `b_raw = M.T @ b_filtered`. A kernel that returned only the
    filtered values would make the export in #14 impossible.

    ## Edge handling: `interp`, and only `interp`

    Within a half-window of either end there is no centred window. The
    convention here is scipy's `mode="interp"`: the polynomial fitted to the
    *first* (or last) full window is evaluated at each of those edge
    positions, rather than at its centre. Nothing is padded, reflected or
    repeated, so no value the filter returns depends on data that was invented.

    The alternatives — `mirror`, `nearest`, `wrap`, `constant` — extend the
    spectrum before filtering and give different numbers at the bounds. That
    disagreement is one of the most common causes of "why does this not match
    my other package" and it is why the mode is fixed here rather than
    exposed: a recipe that records only "Savitzky-Golay, 11, 2, 1" is
    reproducible only if the mode is a property of the software.

    ## Derivative scaling: per index by default

    `delta` is the spacing between neighbouring variables, and the derivative
    is divided by `delta**deriv`. It defaults to 1.0, which makes the result a
    derivative **per variable index**, not per nanometre or per wavenumber.
    The schema's `SavitzkyGolay` step carries no spacing field, so a pipeline
    recipe always means per index; a caller who wants per-axis-unit
    derivatives passes the axis spacing here and must record it themselves.
    Either is defensible — a derivative per index rescales the regression
    coefficients by a constant and changes no model's fit — but they are not
    the same numbers, so which one it is has to be stated.
    """

    matrix_: NDArray[np.float64] | None = None

    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> None:
        # The same rules the schema enforces (`models.SavitzkyGolay`), repeated
        # rather than duplicated: a transformer built directly in a script never
        # passes through the schema, and these are the conditions under which
        # the least-squares fit is defined at all.
        if window_length % 2 == 0:
            raise ValueError(
                f"window_length must be odd, got {window_length}. An even window has "
                "no centre variable to evaluate the fitted polynomial at."
            )
        if window_length < 3:
            raise ValueError(f"window_length must be at least 3, got {window_length}")
        if polyorder < 0 or polyorder >= window_length:
            raise ValueError(
                f"polyorder must be in [0, {window_length - 1}], got {polyorder}. "
                "A polynomial of degree window_length - 1 interpolates the window "
                "exactly and smooths nothing."
            )
        if deriv < 0 or deriv > polyorder:
            raise ValueError(
                f"deriv must be in [0, {polyorder}], got {deriv}. The deriv-th "
                "derivative of a degree-polyorder polynomial is zero beyond it."
            )
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)

    def _weights_at(
        self, offset: float, pseudo_inverse: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Weights on one window that give the deriv-th derivative at `offset`.

        `pseudo_inverse @ window` are the coefficients c_j of the fitted
        polynomial in the window-centred coordinate t. Its d-th derivative at t
        is the sum over j >= d of j!/(j-d)! * t**(j-d) * c_j, and that sum is
        linear in the window, so it collapses into one weight vector.
        """
        powers = np.zeros(self.polyorder + 1)
        for j in range(self.deriv, self.polyorder + 1):
            powers[j] = float(np.prod(np.arange(j - self.deriv + 1, j + 1))) * offset ** (
                j - self.deriv
            )
        weights: NDArray[np.float64] = powers @ pseudo_inverse
        return weights

    def _fit(self, X: NDArray[np.float64]) -> None:
        n_variables = X.shape[1]
        if n_variables < self.window_length:
            raise ValueError(
                f"a window of {self.window_length} needs at least that many variables, "
                f"got {n_variables}. Shorten the window, or select a wider range."
            )

        half = self.window_length // 2
        t = np.arange(self.window_length, dtype=np.float64) - half
        vandermonde = t[:, np.newaxis] ** np.arange(self.polyorder + 1)
        pseudo_inverse = np.linalg.pinv(vandermonde)

        matrix = np.zeros((n_variables, n_variables), dtype=np.float64)
        centre = self._weights_at(0.0, pseudo_inverse)
        for i in range(half, n_variables - half):
            matrix[i, i - half : i + half + 1] = centre

        # The edges, per `mode="interp"`: the first and last full windows,
        # evaluated off-centre instead of at their middles.
        for i in range(half):
            matrix[i, : self.window_length] = self._weights_at(float(i - half), pseudo_inverse)
            matrix[n_variables - 1 - i, -self.window_length :] = self._weights_at(
                float(half - i), pseudo_inverse
            )

        self.matrix_ = matrix / self.delta**self.deriv

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.matrix_ is not None
        filtered: NDArray[np.float64] = X @ self.matrix_.T
        return filtered

    def convolution_matrix(self) -> NDArray[np.float64]:
        """The `p x p` matrix this filter is, edges included.

        `transform(X)` is `X @ M.T`. Folding the filter into exported
        regression coefficients is `M.T @ b` (`pls-regression.md` §7).
        """
        if self.matrix_ is None:
            raise RuntimeError("SavitzkyGolayTransformer has not been fitted")
        return self.matrix_.copy()


# --------------------------------------------------------------------------
# baseline correction
# --------------------------------------------------------------------------


class BaselineCorrectTransformer(Transformer):
    """Estimate a baseline per spectrum and subtract it.

    Row-wise and stateless, like SNV: the baseline is a property of the
    spectrum in front of it, which is what makes baseline correction — again
    like SNV — impossible to fold into exported coefficients.

    | `method` | Baseline |
    | --- | --- |
    | `asls` | Asymmetric least squares, Eilers and Boelens (2005) |
    | `rubberband` | The lower convex hull of the spectrum, linearly interpolated |
    | `polynomial` | A least-squares polynomial of degree `order` through the whole spectrum |

    ## AsLS, and when it stops

    AsLS alternates between solving a penalised least-squares problem

    $(W + \\lambda D_2^{\\top} D_2)\\,z = W y$

    for the baseline $z$, and reweighting: a point above the current baseline
    gets weight `p`, a point below or on it gets `1 - p`. With `p` small the
    baseline is pulled towards the valleys and ignores the peaks, which is the
    whole idea.

    **The convergence criterion is that the weight vector stops changing**, and
    the iteration cap is `max_iter`, default 20. Both are recorded on the
    transformer as `n_iterations_` and `converged_` after a transform, per row,
    because a baseline that hit the cap is a different claim from one that
    settled and the caller has no other way to tell. Hitting the cap is not an
    error — the last iterate is still a usable baseline — but it is reported.

    The weights are a step function of the residual sign, so once the sign
    pattern repeats the iteration is at a fixed point exactly, not
    approximately. That is why the criterion is equality rather than a
    tolerance on `z`.

    ## Defaults

    `lam=1e5` and `p=0.01` are the values Eilers and Boelens use throughout the
    paper. `order=2` for the polynomial method. Every one of them depends on
    the instrument and the sampling density; none is a universal default and
    the schema carries all three so a recipe records what was actually used.

    ## Axis units

    The baseline is estimated against the variable *index*, not the axis. For
    the polynomial method this is exact rather than approximate: polynomials in
    the index and polynomials in any affine axis span the same space, so the
    fitted baseline is identical, and the index is mapped onto [-1, 1] first
    only to keep the fit well conditioned. For AsLS and rubberband it assumes
    uniform spacing, which all three reference datasets have.
    """

    n_iterations_: NDArray[np.int_] | None = None
    converged_: NDArray[np.bool_] | None = None

    def __init__(
        self,
        method: BaselineMethod = "asls",
        *,
        order: int = 2,
        lam: float = 1e5,
        p: float = 0.01,
        max_iter: int = 20,
    ) -> None:
        if method not in ("asls", "rubberband", "polynomial"):
            raise ValueError(f"unknown baseline method {method!r}")
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be strictly between 0 and 1, got {p}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")

        self.method = method
        self.order = int(order)
        self.lam = float(lam)
        self.p = float(p)
        self.max_iter = int(max_iter)

    def _fit(self, X: NDArray[np.float64]) -> None:
        if self.method == "polynomial" and X.shape[1] <= self.order:
            raise ValueError(
                f"a degree-{self.order} baseline needs more than {self.order} variables, "
                f"got {X.shape[1]}; it would interpolate the spectrum and leave zeros."
            )
        if self.method == "asls" and X.shape[1] < 3:
            raise ValueError("AsLS needs at least 3 variables for a second difference")

    def baseline(self, X: object) -> NDArray[np.float64]:
        """The estimated baselines themselves — the plot the UI draws over the raw data."""
        values = as_float64(X, "X")
        rows = [self._baseline_of(row) for row in values]
        iterations = [n for _, n, _ in rows]
        self.n_iterations_ = np.asarray(iterations, dtype=np.int_)
        self.converged_ = np.asarray([c for _, _, c in rows], dtype=np.bool_)
        return np.vstack([b for b, _, _ in rows])

    def _transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return X - self.baseline(X)

    def _baseline_of(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], int, bool]:
        if self.method == "polynomial":
            # The index mapped onto [-1, 1] before the Vandermonde matrix is
            # built. Polynomials in the index and in any affine transform of it
            # span the same space, so this changes no baseline in exact
            # arithmetic — but a raw index over 700 variables raised to the
            # fourth power spans 1e11, and the fit of a high-order baseline is
            # then decided by the least-squares cutoff rather than by the data.
            index = np.linspace(-1.0, 1.0, y.size)
            design = index[:, np.newaxis] ** np.arange(self.order + 1)
            fitted: NDArray[np.float64] = design @ np.linalg.lstsq(design, y, rcond=None)[0]
            return fitted, 1, True
        if self.method == "rubberband":
            return _rubberband(y), 1, True
        return _asls(y, self.lam, self.p, self.max_iter)


def _lower_hull(y: NDArray[np.float64]) -> NDArray[np.int_]:
    """Indices of the lower convex hull of the points (i, y[i]), left to right.

    Andrew's monotone chain, which needs no sort because the abscissae are
    already the indices in order. Ten lines and exact in integer-index
    arithmetic, which is the argument for not reaching for a general convex
    hull routine here.
    """
    hull: list[int] = []
    for i in range(y.size):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            # Cross product of (second - first) and (i - first). Non-positive
            # means `second` sits on or above the chord and is not on the hull.
            cross = (second - first) * (y[i] - y[first]) - (y[second] - y[first]) * (i - first)
            if cross > 0:
                break
            hull.pop()
        hull.append(i)
    return np.asarray(hull, dtype=np.int_)


def _rubberband(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """A taut band stretched under the spectrum: its lower convex hull, interpolated."""
    vertices = _lower_hull(y)
    interpolated: NDArray[np.float64] = np.interp(
        np.arange(y.size, dtype=np.float64), vertices.astype(np.float64), y[vertices]
    )
    return interpolated


def _asls(
    y: NDArray[np.float64], lam: float, p: float, max_iter: int
) -> tuple[NDArray[np.float64], int, bool]:
    """Eilers and Boelens asymmetric least squares. Returns baseline, iterations, converged."""
    n = y.size
    differences = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = lam * (differences.T @ differences)

    weights = np.ones(n)
    baseline = y.copy()
    for iteration in range(1, max_iter + 1):
        system = csc_matrix(diags(weights, format="csc") + penalty)
        baseline = spsolve(system, weights * y)
        updated = np.where(y > baseline, p, 1.0 - p)
        if np.array_equal(updated, weights):
            return baseline, iteration, True
        weights = updated
    return baseline, max_iter, False
