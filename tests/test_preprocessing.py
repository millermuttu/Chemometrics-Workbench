"""Tests for the scaling and scatter-correction kernels.

The parity claims — that these agree with an independent implementation —
live in `tests/test_parity.py` and go through the harness. What is tested here
is everything parity cannot see: the conventions in the module docstring of
`preprocessing.py`, the defining identities of the two transforms that have no
external reference, and the schema round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

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
from chemometrics_workbench.preprocessing import (
    AutoscaleTransformer,
    BaselineCorrectTransformer,
    MeanCentreTransformer,
    MSCTransformer,
    NormaliseTransformer,
    RangeSelectTransformer,
    SavitzkyGolayTransformer,
    SNVTransformer,
    Transformer,
    from_spec,
)

RNG = np.random.default_rng(20260822)


def _spectra(n: int = 6, p: int = 12) -> np.ndarray:
    """A small block with the shape of real spectra: smooth, positive, varied."""
    axis = np.linspace(0.0, 1.0, p)
    offsets = np.linspace(0.5, 1.5, n)[:, np.newaxis]
    scales = np.linspace(0.8, 1.4, n)[:, np.newaxis]
    base = 2.0 + np.sin(3.0 * axis) + 0.3 * axis**2
    return offsets + scales * base + 0.01 * RNG.normal(size=(n, p))


def _describe(transformer: Transformer) -> str:
    """Test ids: several of the transformers below are the same class twice."""
    name = type(transformer).__name__.removesuffix("Transformer")
    detail = getattr(transformer, "method", None) or getattr(transformer, "deriv", None)
    return name if detail is None else f"{name}-{detail}"


def _all_transformers(axis: np.ndarray | None = None) -> list[Transformer]:
    axis = np.arange(12, dtype=float) if axis is None else axis
    return [
        SNVTransformer(),
        MSCTransformer("mean"),
        MSCTransformer("median"),
        MeanCentreTransformer(),
        AutoscaleTransformer(),
        NormaliseTransformer("l2"),
        RangeSelectTransformer(2.0, 8.0, axis),
        SavitzkyGolayTransformer(5, 2),
        SavitzkyGolayTransformer(5, 2, deriv=1),
        BaselineCorrectTransformer("asls"),
        BaselineCorrectTransformer("rubberband"),
        BaselineCorrectTransformer("polynomial"),
    ]


# --------------------------------------------------------------------------
# conventions every transformer follows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_the_callers_array_is_never_modified(transformer: Transformer) -> None:
    """The issue's second verification step."""
    X = _spectra()
    before = X.copy()

    transformer.fit_transform(X)

    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_float32_input_is_promoted_and_the_result_is_float64(
    transformer: Transformer,
) -> None:
    """Storage dtype is the caller's business; numerical results are not."""
    X = _spectra().astype(np.float32)
    before = X.copy()

    result = transformer.fit_transform(X)

    assert result.dtype == np.float64
    assert X.dtype == np.float32, "the caller's array was cast in place"
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_a_transposed_array_is_rejected(transformer: Transformer) -> None:
    """The issue's third verification step.

    A transposed matrix is still two-dimensional, so shape alone cannot catch
    it. What catches it is the variable count recorded at fit — which is why
    even the stateless transformers have a `fit`.
    """
    X = _spectra(n=6, p=12)
    transformer.fit(X)

    with pytest.raises(ValueError, match="never silently transposed"):
        transformer.transform(X.T)


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_a_one_dimensional_array_is_rejected(transformer: Transformer) -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        transformer.fit(np.arange(12, dtype=float))


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_missing_values_are_rejected_with_their_position(
    transformer: Transformer,
) -> None:
    X = _spectra()
    X[2, 5] = np.nan

    with pytest.raises(ValueError, match="row 2, column 5"):
        transformer.fit(X)


@pytest.mark.parametrize("transformer", _all_transformers(), ids=_describe)
def test_transform_before_fit_is_refused(transformer: Transformer) -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        transformer.transform(_spectra())


def test_an_all_nan_row_is_rejected_like_any_other_missing_value() -> None:
    X = _spectra()
    X[3, :] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        SNVTransformer().fit(X)


def test_infinities_are_rejected_too() -> None:
    X = _spectra()
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        MeanCentreTransformer().fit(X)


# --------------------------------------------------------------------------
# zero variance: the documented behaviour
# --------------------------------------------------------------------------


def test_a_constant_variable_stops_autoscaling_and_names_the_column() -> None:
    """The issue's fourth verification step.

    Not a NaN and not an unhandled division: a `ValueError` naming the column.
    `StandardScaler` substitutes a scale of 1 here, which reports a successful
    autoscale over a dead channel — see `preprocessing.py` on why this project
    does not.
    """
    X = _spectra()
    X[:, 4] = 3.0

    with pytest.raises(ValueError, match="variable 4"):
        AutoscaleTransformer().fit(X)


def test_a_constant_spectrum_stops_snv_and_names_the_row() -> None:
    X = _spectra()
    X[1, :] = 0.7

    with pytest.raises(ValueError, match="sample 1"):
        SNVTransformer().fit_transform(X)


def test_a_row_summing_to_zero_stops_area_normalisation() -> None:
    """`area` is signed, so a derivative-like row is rejected, not exploded."""
    X = _spectra()
    X[0, :] = np.tile([1.0, -1.0], X.shape[1] // 2)

    with pytest.raises(ValueError, match="area norm"):
        NormaliseTransformer("area").fit_transform(X)


def test_many_dead_columns_are_summarised_rather_than_listed_in_full() -> None:
    X = _spectra(n=6, p=20)
    X[:, :12] = 1.0

    with pytest.raises(ValueError, match="and 2 more"):
        AutoscaleTransformer().fit(X)


# --------------------------------------------------------------------------
# fitted parameters are the fit set's, not the transformed set's
# --------------------------------------------------------------------------


def test_mean_centring_uses_the_fit_sets_mean_on_new_samples() -> None:
    """`metrics-and-validation.md` §9. Centring a held-out sample by its own
    mean leaks the model's information into it and makes RMSECV optimistic."""
    train = _spectra(n=8)
    test = _spectra(n=3) + 5.0

    transformer = MeanCentreTransformer().fit(train)
    centred = transformer.transform(test)

    np.testing.assert_allclose(centred, test - train.mean(axis=0))
    assert not np.allclose(centred.mean(axis=0), 0.0), "the test set was re-centred"


def test_autoscaling_uses_the_fit_sets_scale_on_new_samples() -> None:
    train = _spectra(n=8)
    test = _spectra(n=3)

    transformer = AutoscaleTransformer().fit(train)
    scaled = transformer.transform(test)

    expected = (test - train.mean(axis=0)) / train.std(axis=0, ddof=1)
    np.testing.assert_allclose(scaled, expected)


def test_msc_reuses_the_calibration_reference() -> None:
    """Re-estimating per block would make a prediction depend on which other
    samples happened to be predicted alongside it."""
    train = _spectra(n=8)
    test = _spectra(n=3) * 2.0

    transformer = MSCTransformer("mean").fit(train)
    assert transformer.reference_ is not None
    np.testing.assert_allclose(transformer.reference_, train.mean(axis=0))

    transformer.transform(test)
    np.testing.assert_allclose(transformer.reference_, train.mean(axis=0))


def test_autoscale_ddof_changes_the_result() -> None:
    """If it did not, the convention would not be worth carrying in the schema."""
    X = _spectra()
    ddof0 = AutoscaleTransformer(ddof=0).fit_transform(X)
    ddof1 = AutoscaleTransformer(ddof=1).fit_transform(X)

    assert not np.allclose(ddof0, ddof1)
    n = X.shape[0]
    np.testing.assert_allclose(ddof1, ddof0 * np.sqrt((n - 1) / n))


@pytest.mark.parametrize("ddof", [-1, 2])
def test_an_unsupported_ddof_is_refused(ddof: int) -> None:
    with pytest.raises(ValueError, match="ddof must be 0 or 1"):
        AutoscaleTransformer(ddof=ddof)


# --------------------------------------------------------------------------
# defining identities, for the transforms with no external reference
# --------------------------------------------------------------------------


def test_snv_rows_have_zero_mean_and_unit_deviation() -> None:
    """The definition, and the whole of what SNV claims to do."""
    result = SNVTransformer().fit_transform(_spectra())

    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(result.std(axis=1, ddof=1), 1.0)


def test_snv_removes_an_offset_and_a_multiplier() -> None:
    """Which is the scatter it is there to correct."""
    X = _spectra()
    scattered = 3.0 + 2.5 * X

    np.testing.assert_allclose(
        SNVTransformer().fit_transform(scattered),
        SNVTransformer().fit_transform(X),
    )


@pytest.mark.parametrize("reference", ["mean", "median"])
def test_msc_returns_the_reference_when_given_it(reference: str) -> None:
    X = _spectra()
    transformer = MSCTransformer(reference).fit(X)  # type: ignore[arg-type]
    assert transformer.reference_ is not None

    corrected = transformer.transform(transformer.reference_[np.newaxis, :])
    np.testing.assert_allclose(corrected[0], transformer.reference_)


def test_msc_recovers_a_spectrum_scaled_and_shifted_from_the_reference() -> None:
    """The exact model MSC assumes: x = a + b*r, undone as (x - a)/b."""
    reference = _spectra(n=1)[0]
    a, b = 0.4, 1.7
    observed = (a + b * reference)[np.newaxis, :]

    transformer = MSCTransformer("supplied", reference_spectrum=reference).fit(observed)
    np.testing.assert_allclose(transformer.transform(observed)[0], reference)


def test_msc_supplied_needs_a_spectrum_and_the_others_refuse_one() -> None:
    with pytest.raises(ValueError, match="needs a reference_spectrum"):
        MSCTransformer("supplied")

    with pytest.raises(ValueError, match="would be ignored"):
        MSCTransformer("mean", reference_spectrum=np.arange(12, dtype=float))


def test_msc_rejects_a_reference_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="reference_spectrum has 5 variables"):
        MSCTransformer("supplied", reference_spectrum=np.arange(5, dtype=float)).fit(_spectra())


def test_msc_rejects_a_constant_reference() -> None:
    with pytest.raises(ValueError, match="reference spectrum is constant"):
        MSCTransformer("supplied", reference_spectrum=np.ones(12)).fit(_spectra())


@pytest.mark.parametrize(
    ("norm", "measure"),
    [
        ("l1", lambda r: np.abs(r).sum(axis=1)),
        ("l2", lambda r: np.sqrt((r**2).sum(axis=1))),
        ("max", lambda r: np.abs(r).max(axis=1)),
        ("area", lambda r: r.sum(axis=1)),
    ],
)
def test_each_norm_leaves_its_own_measure_at_one(norm: str, measure: object) -> None:
    result = NormaliseTransformer(norm).fit_transform(_spectra())  # type: ignore[arg-type]
    np.testing.assert_allclose(measure(result), 1.0)  # type: ignore[operator]


def test_an_unknown_norm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown norm"):
        NormaliseTransformer("frobenius")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# range selection
# --------------------------------------------------------------------------


def test_range_selection_keeps_the_variables_inside_the_bounds() -> None:
    axis = np.linspace(1000.0, 1100.0, 11)
    transformer = RangeSelectTransformer(1020.0, 1060.0, axis).fit(_spectra(p=11))

    np.testing.assert_allclose(
        transformer.selected_axis(), [1020.0, 1030.0, 1040.0, 1050.0, 1060.0]
    )
    assert transformer.transform(_spectra(p=11)).shape == (6, 5)


def test_range_selection_handles_a_descending_axis() -> None:
    """Wavenumber axes usually descend; an interval is an interval either way."""
    axis = np.linspace(1100.0, 1000.0, 11)
    transformer = RangeSelectTransformer(1020.0, 1060.0, axis).fit(_spectra(p=11))

    np.testing.assert_allclose(
        transformer.selected_axis(), [1060.0, 1050.0, 1040.0, 1030.0, 1020.0]
    )


def test_range_selection_bounds_are_inclusive() -> None:
    axis = np.arange(10, dtype=float)
    transformer = RangeSelectTransformer(2.0, 4.0, axis).fit(_spectra(p=10))
    np.testing.assert_allclose(transformer.selected_axis(), [2.0, 3.0, 4.0])


def test_an_empty_range_is_an_error_rather_than_an_empty_matrix() -> None:
    axis = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match=r"no variable falls in \[20.0, 30.0\]"):
        RangeSelectTransformer(20.0, 30.0, axis).fit(_spectra(p=10))


def test_range_selection_rejects_an_axis_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="axis has 5 values"):
        RangeSelectTransformer(1.0, 3.0, np.arange(5, dtype=float)).fit(_spectra(p=12))


def test_range_selection_needs_ordered_bounds() -> None:
    with pytest.raises(ValueError, match="start must be less than end"):
        RangeSelectTransformer(5.0, 2.0, np.arange(10, dtype=float))


def test_selected_axis_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        RangeSelectTransformer(1.0, 3.0, np.arange(10, dtype=float)).selected_axis()


# --------------------------------------------------------------------------
# the schema round-trip
# --------------------------------------------------------------------------


def test_every_supported_step_round_trips_through_the_schema() -> None:
    """The issue's fifth verification step.

    Serialise the step, parse it back, build the transformer, and confirm the
    parameters survived — because the pipeline is data, and a parameter that
    does not survive JSON is a parameter the lineage does not record.
    """
    axis = np.arange(12, dtype=float)
    X = _spectra()

    cases: list[tuple[PreprocessStep, Transformer]] = [
        (SNV(), SNVTransformer()),
        (MSC(reference="median"), MSCTransformer("median")),
        (MeanCentre(), MeanCentreTransformer()),
        (Autoscale(ddof=0), AutoscaleTransformer(ddof=0)),
        (Normalise(norm="l1"), NormaliseTransformer("l1")),
        (RangeSelect(start=2.0, end=8.0), RangeSelectTransformer(2.0, 8.0, axis)),
        (
            SavitzkyGolay(window_length=5, polyorder=2, deriv=1),
            SavitzkyGolayTransformer(5, 2, deriv=1),
        ),
        (
            BaselineCorrect(method="polynomial", order=2),
            BaselineCorrectTransformer("polynomial", order=2),
        ),
        (BaselineCorrect(method="rubberband"), BaselineCorrectTransformer("rubberband")),
        (
            BaselineCorrect(method="asls", lam=1e4, p=0.02),
            BaselineCorrectTransformer("asls", lam=1e4, p=0.02),
        ),
    ]

    for step, direct in cases:
        parsed = type(step).model_validate_json(step.model_dump_json())
        assert parsed == step

        built = from_spec(parsed, axis=axis)
        assert isinstance(direct, Transformer)
        assert type(built) is type(direct)
        np.testing.assert_allclose(
            built.fit_transform(X),
            direct.fit_transform(X),
            err_msg=f"{type(step).__name__} did not survive the round trip",
        )


def test_autoscale_ddof_survives_the_round_trip() -> None:
    """The parameter most likely to be lost silently, and the one that moves a number."""
    for ddof in (0, 1):
        step = Autoscale.model_validate_json(Autoscale(ddof=ddof).model_dump_json())
        built = from_spec(step)
        assert isinstance(built, AutoscaleTransformer)
        assert built.ddof == ddof


def test_from_spec_needs_the_axis_for_range_selection() -> None:
    """The bounds are in axis units, and the axis belongs to the dataset."""
    with pytest.raises(ValueError, match="needs the dataset's"):
        from_spec(RangeSelect(start=1.0, end=2.0))


def test_the_schema_does_not_offer_an_msc_reference_from_spec_cannot_build() -> None:
    """The gap closed by narrowing the enum rather than by raising in the seam.

    The kernel still takes a supplied reference - it is a legitimate library
    call - but a step has no field for the spectrum, so the schema no longer
    claims a saved pipeline can express one.
    """
    with pytest.raises(ValidationError):
        MSC(reference="supplied")

    reference = _spectra().mean(axis=0)
    kernel = MSCTransformer("supplied", reference_spectrum=reference).fit(_spectra())
    assert kernel.reference_ is not None
    np.testing.assert_allclose(kernel.reference_, reference)


def test_baseline_parameters_left_unset_in_the_schema_take_the_kernel_defaults() -> None:
    """`lam`, `p` and `order` are optional on the step because each applies to
    one method only. An unset one is the kernel's default, and the default is
    documented rather than implicit."""
    built = from_spec(BaselineCorrect(method="asls"))
    assert isinstance(built, BaselineCorrectTransformer)
    assert (built.lam, built.p, built.order) == (1e5, 0.01, 2)

    given = from_spec(BaselineCorrect(method="asls", lam=1e4, p=0.05))
    assert isinstance(given, BaselineCorrectTransformer)
    assert (given.lam, given.p) == (1e4, 0.05)


def test_savitzky_golay_parameters_survive_the_round_trip() -> None:
    for deriv in (0, 1, 2):
        step = SavitzkyGolay.model_validate_json(
            SavitzkyGolay(window_length=7, polyorder=3, deriv=deriv).model_dump_json()
        )
        built = from_spec(step)
        assert isinstance(built, SavitzkyGolayTransformer)
        assert (built.window_length, built.polyorder, built.deriv) == (7, 3, deriv)
        assert built.delta == 1.0, "the schema carries no spacing, so it means per index"


# --------------------------------------------------------------------------
# Savitzky-Golay
# --------------------------------------------------------------------------


def _quadratic(p: int = 12) -> np.ndarray:
    """A quadratic in the variable index, which a polyorder-2 filter fits exactly."""
    index = np.arange(p, dtype=float)
    return (1.0 + 0.5 * index + 0.25 * index**2)[np.newaxis, :]


def test_savitzky_golay_reproduces_a_polynomial_it_can_fit_exactly() -> None:
    """The defining property, and it holds at the edges too.

    A local quadratic fit to data that *is* a quadratic returns the data. Under
    `mode="interp"` that stays true at the first and last variable, because the
    end window's polynomial is the same polynomial. Under a padded edge mode it
    would not, which is what makes this the identity that pins the convention.
    """
    y = _quadratic()
    smoothed = SavitzkyGolayTransformer(5, 2).fit_transform(y)

    np.testing.assert_allclose(smoothed, y, atol=1e-10)


def test_savitzky_golay_derivatives_are_exact_on_a_polynomial_including_at_the_bounds() -> None:
    """d/di of 1 + i/2 + i^2/4 is 1/2 + i/2, and its second derivative is 1/2."""
    p = 12
    index = np.arange(p, dtype=float)
    y = _quadratic(p)

    first = SavitzkyGolayTransformer(5, 2, deriv=1).fit_transform(y)
    second = SavitzkyGolayTransformer(5, 2, deriv=2).fit_transform(y)

    np.testing.assert_allclose(first[0], 0.5 + 0.5 * index, atol=1e-10)
    np.testing.assert_allclose(second[0], 0.5, atol=1e-10)
    # Stated separately: the interior passing while an edge is padded is the
    # failure this issue exists to prevent.
    np.testing.assert_allclose(first[0, [0, -1]], (0.5 + 0.5 * index)[[0, -1]], atol=1e-10)


def test_the_derivative_is_per_index_unless_delta_says_otherwise() -> None:
    """The convention, and the whole of what `delta` changes."""
    y = _quadratic()
    per_index = SavitzkyGolayTransformer(5, 2, deriv=1).fit_transform(y)
    per_unit = SavitzkyGolayTransformer(5, 2, deriv=1, delta=2.5).fit_transform(y)

    np.testing.assert_allclose(per_unit, per_index / 2.5)

    second = SavitzkyGolayTransformer(5, 2, deriv=2, delta=2.5).fit_transform(y)
    np.testing.assert_allclose(
        second, SavitzkyGolayTransformer(5, 2, deriv=2).fit_transform(y) / 2.5**2
    )


def test_smoothing_is_the_convolution_matrix_and_the_matrix_is_what_gets_exported() -> None:
    """`pls-regression.md` §7: Savitzky-Golay folds into exported coefficients.

    Folding is `b_raw = M.T @ b_filtered`, so a prediction from the filtered
    spectra and a prediction from the raw spectra through the folded
    coefficients must be the same number. That is the property the export in
    #14 depends on, and it is why the matrix is public.
    """
    X = _spectra(n=6, p=12)
    transformer = SavitzkyGolayTransformer(5, 2, deriv=1).fit(X)
    matrix = transformer.convolution_matrix()

    filtered = transformer.transform(X)
    np.testing.assert_allclose(filtered, X @ matrix.T)

    coefficients = RNG.normal(size=12)
    np.testing.assert_allclose(filtered @ coefficients, X @ (matrix.T @ coefficients))


def test_the_convolution_matrix_is_banded_in_the_interior() -> None:
    """Nothing outside the window contributes — which is what makes the filter local."""
    matrix = SavitzkyGolayTransformer(5, 2).fit(_spectra(p=12)).convolution_matrix()

    for row in range(2, 10):
        outside = np.r_[matrix[row, : row - 2], matrix[row, row + 3 :]]
        np.testing.assert_allclose(outside, 0.0, atol=0.0)


def test_the_convolution_matrix_before_fit_is_refused() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        SavitzkyGolayTransformer(5, 2).convolution_matrix()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_length": 4, "polyorder": 2}, "must be odd"),
        ({"window_length": 5, "polyorder": 5}, "polyorder must be in"),
        ({"window_length": 5, "polyorder": 1, "deriv": 2}, "deriv must be in"),
        ({"window_length": 5, "polyorder": 2, "delta": 0.0}, "delta must be positive"),
    ],
)
def test_savitzky_golay_refuses_a_filter_that_is_not_defined(
    kwargs: dict[str, int | float], message: str
) -> None:
    """The same conditions the schema enforces, because a transformer built in a
    script never passes through the schema."""
    with pytest.raises(ValueError, match=message):
        SavitzkyGolayTransformer(**kwargs)  # type: ignore[arg-type]


def test_a_window_wider_than_the_spectrum_is_refused() -> None:
    with pytest.raises(ValueError, match="needs at least that many variables"):
        SavitzkyGolayTransformer(11, 2).fit(_spectra(p=8))


# --------------------------------------------------------------------------
# baseline correction
# --------------------------------------------------------------------------


def _peak_on_a_slope(n: int = 100) -> np.ndarray:
    index = np.arange(n, dtype=float)
    return (2.0 + 0.03 * index + 5.0 * np.exp(-((index - 50.0) ** 2) / 32.0))[np.newaxis, :]


def test_a_polynomial_baseline_removes_a_polynomial_of_its_own_degree_exactly() -> None:
    index = np.arange(30, dtype=float)
    drift = (1.0 - 0.2 * index + 0.01 * index**2)[np.newaxis, :]

    corrected = BaselineCorrectTransformer("polynomial", order=2).fit_transform(drift)

    np.testing.assert_allclose(corrected, 0.0, atol=1e-8)


def test_a_polynomial_baseline_is_the_same_against_index_or_any_affine_axis() -> None:
    """Why the kernel never asks for the axis: polynomials in the index and in
    any affine transform of it span the same space, so the fit is identical."""
    X = _peak_on_a_slope()
    ours = BaselineCorrectTransformer("polynomial", order=3).fit_transform(X)

    axis = np.linspace(1000.0, 1100.0, X.shape[1])
    fitted = np.polynomial.Polynomial.fit(axis, X[0], 3)(axis)

    np.testing.assert_allclose(ours, X - fitted[np.newaxis, :], atol=1e-8)


def test_a_rubberband_baseline_never_rises_above_the_spectrum() -> None:
    """The defining property of a band stretched underneath it, and the reason
    a corrected spectrum is non-negative."""
    X = _peak_on_a_slope()
    transformer = BaselineCorrectTransformer("rubberband")
    corrected = transformer.fit_transform(X)

    assert corrected.min() >= 0.0
    assert corrected[0, 0] == 0.0 and corrected[0, -1] == 0.0, "the hull spans both ends"
    assert (np.abs(corrected) < 1e-12).sum() >= 2, "the band touches the spectrum"


def test_a_rubberband_baseline_is_a_straight_line_under_a_single_peak() -> None:
    """With one convex-hull vertex at each end, the band is the chord between them."""
    X = _peak_on_a_slope()
    baseline = BaselineCorrectTransformer("rubberband").fit(X).baseline(X)

    chord = np.linspace(X[0, 0], X[0, -1], X.shape[1])
    np.testing.assert_allclose(baseline[0], chord)


def test_asls_pulls_the_baseline_under_the_peak_and_leaves_the_peak_standing() -> None:
    X = _peak_on_a_slope()
    corrected = BaselineCorrectTransformer("asls").fit_transform(X)

    assert np.abs(corrected[0, :30]).max() < 0.05, "the flat region is not flat"
    assert corrected[0, 50] > 4.75, "the peak was flattened along with the baseline"


def test_asls_records_whether_it_converged_and_how_many_iterations_it_took() -> None:
    """The criterion is that the weights stop changing; the cap is `max_iter`.
    Hitting the cap is not an error, but the caller has no other way to know."""
    X = _peak_on_a_slope()

    settled = BaselineCorrectTransformer("asls")
    settled.fit_transform(X)
    assert settled.converged_ is not None and bool(settled.converged_.all())
    assert settled.n_iterations_ is not None and int(settled.n_iterations_.max()) < 20

    capped = BaselineCorrectTransformer("asls", max_iter=1)
    capped.fit_transform(X)
    assert capped.converged_ is not None and not bool(capped.converged_.any())
    assert capped.n_iterations_ is not None and int(capped.n_iterations_[0]) == 1


def test_the_baseline_itself_is_available_for_plotting() -> None:
    """The UI draws it over the raw spectrum, so it cannot be an internal."""
    X = _peak_on_a_slope()
    transformer = BaselineCorrectTransformer("polynomial", order=1).fit(X)

    np.testing.assert_allclose(transformer.transform(X), X - transformer.baseline(X))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "spline"}, "unknown baseline method"),
        ({"method": "polynomial", "order": -1}, "order must be non-negative"),
        ({"method": "asls", "lam": 0.0}, "lam must be positive"),
        ({"method": "asls", "p": 1.0}, "p must be strictly between 0 and 1"),
        ({"method": "asls", "max_iter": 0}, "max_iter must be at least 1"),
    ],
)
def test_baseline_correction_refuses_parameters_it_cannot_use(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BaselineCorrectTransformer(**kwargs)  # type: ignore[arg-type]


def test_a_polynomial_baseline_that_would_interpolate_the_spectrum_is_refused() -> None:
    with pytest.raises(ValueError, match="would interpolate"):
        BaselineCorrectTransformer("polynomial", order=6).fit(_spectra(p=6))
