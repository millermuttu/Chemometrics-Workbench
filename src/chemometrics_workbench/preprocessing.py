"""Scaling and scatter-correction kernels.

The deterministic half of preprocessing: SNV, MSC, mean centring,
autoscaling, normalisation and range selection. None of these has any
convention freedom in its arithmetic, so each should reach the tightest parity
tier the harness offers. Where a convention *is* free — autoscaling's `ddof`,
what "area" means for a normalisation — it is fixed here and recorded in §
"Known divergences" at the bottom of this docstring.

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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    MeanCentre,
    Normalise,
    PreprocessStep,
    RangeSelect,
)

__all__ = [
    "AutoscaleTransformer",
    "MSCTransformer",
    "MeanCentreTransformer",
    "NormaliseTransformer",
    "RangeSelectTransformer",
    "SNVTransformer",
    "Transformer",
    "from_spec",
]

Norm = Literal["l1", "l2", "max", "area"]
MSCReference = Literal["mean", "median", "supplied"]


# --------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------


def _as_float64(array: object, name: str) -> NDArray[np.float64]:
    """Promote to float64 and reject anything that is not a 2-D matrix."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D, n_samples x n_variables; got shape {values.shape}. "
            "A single spectrum is a 1 x n_variables matrix, not a 1-D array."
        )
    if not np.isfinite(values).all():
        rows, columns = np.nonzero(~np.isfinite(values))
        raise ValueError(
            f"{name} holds {rows.size} non-finite values, first at "
            f"row {rows[0]}, column {columns[0]}. Missing values are handled upstream "
            "and visibly: exclude the sample, exclude the variable, or add an "
            "imputation step to the pipeline."
        )
    return values


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
        values = _as_float64(X, "X")
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
        values = _as_float64(X, "X")
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
                    "reference='supplied' needs a reference_spectrum. The schema's "
                    "MSC step carries the choice but not the spectrum itself, so the "
                    "caller must pass it."
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
    reference_spectrum: object | None = None,
) -> Transformer:
    """Build the transformer a `PreprocessStep` describes.

    This is the executor's entry point, and the reason the schema's
    discriminated union is worth having: an unsupported step fails here with
    its name, rather than halfway through a ten-minute cross-validation.

    Two steps need something the schema does not carry. `RangeSelect` needs
    the variable axis, which belongs to the dataset rather than to the recipe.
    `MSC(reference="supplied")` needs the reference spectrum, which the schema
    has no field for — a real gap, and a schema change rather than something
    to paper over here.
    """
    match step:
        case SNV():
            return SNVTransformer()
        case MSC():
            return MSCTransformer(step.reference, reference_spectrum)
        case MeanCentre():
            return MeanCentreTransformer()
        case Autoscale():
            return AutoscaleTransformer(ddof=step.ddof)
        case Normalise():
            return NormaliseTransformer(step.norm)
        case RangeSelect():
            if axis is None:
                raise ValueError(
                    "RangeSelect is expressed in axis units and needs the dataset's "
                    "variable axis; pass axis=."
                )
            return RangeSelectTransformer(step.start, step.end, axis)
        case _:
            raise NotImplementedError(
                f"no kernel for preprocessing step {step.kind!r} yet. "
                "Savitzky-Golay and derivatives are #10; baseline correction is "
                "not yet scheduled."
            )
