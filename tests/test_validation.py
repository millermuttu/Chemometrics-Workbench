"""Tests for the metrics and the fold assignment.

`metrics-and-validation.md` is the document that explains most "why does this
not match Unscrambler" reports: metric definitions and fold assignment vary
between packages far more than the algorithms themselves do. So what is tested
here is mostly the *definitions* — the divisor, the aggregation rule, the
permutation — against the worked examples the specification states, rather
than against another implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.validation import (
    Fold,
    bias,
    folds_from_indices,
    k_fold,
    leave_one_out,
    q2,
    r2,
    rmse,
    sec,
    sep,
    validate_partition,
)

# --------------------------------------------------------------------------
# metrics, §4 to §6
# --------------------------------------------------------------------------


def test_rmse_divides_by_n() -> None:
    """§4: no degrees-of-freedom correction hides under this name.

    A package dividing by `n - A - 1` is reporting what §5 calls SEC, and the
    difference is a few percent at typical n — the single most common cause of
    a third-significant-figure mismatch against another tool.
    """
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = np.array([1.5, 2.5, 2.5, 3.5])
    assert rmse(y, y_hat) == pytest.approx(0.5)


def test_bias_is_the_mean_signed_residual() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert bias(y, y - 0.25) == pytest.approx(0.25)
    assert bias(y, y) == 0.0


def test_r2_is_the_residual_form_and_not_the_squared_correlation() -> None:
    """§6: the two diverge on a prediction set, where the correlation is blind
    to bias and slope error and is therefore flattering. Predictions that are
    perfectly correlated with the response but offset by a constant have an R^2
    below 1 and a squared correlation of exactly 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    offset = y + 1.0

    assert r2(y, offset) < 1.0
    assert float(np.corrcoef(y, offset)[0, 1] ** 2) == pytest.approx(1.0)


def test_a_negative_r2_is_reported_as_it_is() -> None:
    """§6: worse than predicting the mean is a real and useful finding, and
    clipping it to zero hides a failed model."""
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, np.array([5.0, 5.0, 5.0])) < 0.0


def test_r2_on_a_constant_response_is_refused_rather_than_returned() -> None:
    with pytest.raises(ValueError, match="no variance about its mean"):
        r2(np.full(5, 2.0), np.arange(5, dtype=float))


def test_a_metric_over_two_different_sets_is_refused() -> None:
    """§2: the three sets are never mixed in one number."""
    with pytest.raises(ValueError, match="reference values against"):
        rmse(np.arange(5, dtype=float), np.arange(4, dtype=float))


# --------------------------------------------------------------------------
# fold assignment, §8
# --------------------------------------------------------------------------


def test_the_worked_example_from_the_specification() -> None:
    """§8.3, reproduced by hand: n = 10, K = 3, shuffled, seed 42.

    Reproducing this table from a fresh implementation is the test that the
    seeding, the permutation and the size rule all agree — and it is the check
    that catches a NumPy upgrade moving the stream underneath us.
    """
    folds = k_fold(10, 3, seed=42)

    assert [fold.test.tolist() for fold in folds] == [[0, 5, 6, 7], [2, 3, 4], [1, 8, 9]]
    assert folds[0].train.tolist() == [1, 2, 3, 4, 8, 9]
    assert folds[1].train.tolist() == [0, 1, 5, 6, 7, 8, 9]
    assert folds[2].train.tolist() == [0, 2, 3, 4, 5, 6, 7]


def test_the_first_r_folds_are_one_larger() -> None:
    """§8.3: scikit-learn's size rule, kept deliberately so that only the
    permutation differs between us."""
    sizes = [len(fold.test) for fold in k_fold(23, 5)]
    assert sizes == [5, 5, 5, 4, 4]
    assert sum(sizes) == 23


def test_without_shuffling_the_folds_are_consecutive_and_the_seed_is_ignored() -> None:
    """§8.2: the permutation is the identity, and the seed is recorded as
    ignored rather than silently accepted."""
    assert k_fold(9, 3, shuffle=False)[0].test.tolist() == [0, 1, 2]
    assert [f.test.tolist() for f in k_fold(9, 3, shuffle=False, seed=1)] == [
        f.test.tolist() for f in k_fold(9, 3, shuffle=False, seed=99)
    ]


def test_a_different_seed_gives_a_different_assignment() -> None:
    assert k_fold(20, 4, seed=42)[0].test.tolist() != k_fold(20, 4, seed=43)[0].test.tolist()


def test_more_folds_than_samples_is_an_error_naming_both() -> None:
    with pytest.raises(ValueError, match="11 folds were asked of 10 samples"):
        k_fold(10, 11)


def test_fewer_than_two_folds_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        k_fold(10, 1)


def test_leave_one_out_holds_out_one_sample_per_fold() -> None:
    """§8.4: n folds, no shuffle and no seed — its result cannot depend on one."""
    folds = leave_one_out(4)
    assert [fold.test.tolist() for fold in folds] == [[0], [1], [2], [3]]
    assert folds[2].train.tolist() == [0, 1, 3]


def test_stored_indices_round_trip_through_folds_from_indices() -> None:
    """§10: a recorded `ResolvedSplit` is replayed from its index lists alone."""
    original = k_fold(17, 4, seed=7)
    replayed = folds_from_indices(
        [fold.train.tolist() for fold in original], [fold.test.tolist() for fold in original]
    )
    for mine, theirs in zip(original, replayed, strict=True):
        assert np.array_equal(mine.train, theirs.train)
        assert np.array_equal(mine.test, theirs.test)


def test_a_stored_split_with_mismatched_halves_is_refused() -> None:
    with pytest.raises(ValueError, match="A stored split has one of each per fold"):
        folds_from_indices([[0, 1], [2, 3]], [[2, 3]])


# --------------------------------------------------------------------------
# the disjoint-union assertion, §7
# --------------------------------------------------------------------------


def test_every_generated_split_partitions_the_samples() -> None:
    every = ((k_fold(23, 5), 23), (k_fold(23, 5, shuffle=False), 23), (leave_one_out(9), 9))
    for folds, n in every:
        validate_partition(folds, n)


def test_a_sample_predicted_twice_is_caught_before_residuals_are_pooled() -> None:
    """§7: it produces a plausible RMSECV at a denominator that does not know."""
    folds = folds_from_indices([[2], [2]], [[0, 1], [0, 1]])
    with pytest.raises(ValueError, match="do not partition"):
        validate_partition(folds, 3)


def test_a_sample_predicted_by_nobody_is_caught() -> None:
    folds = folds_from_indices([[1, 2]], [[0]])
    with pytest.raises(ValueError, match="do not partition"):
        validate_partition(folds, 3)


def test_a_fold_that_trains_on_its_own_validation_set_is_caught() -> None:
    folds = [Fold(train=np.array([0, 1, 2]), test=np.array([0, 1, 2]))]
    with pytest.raises(ValueError, match="trains on samples it also validates"):
        validate_partition(folds, 3)


def test_a_split_with_no_folds_is_refused() -> None:
    with pytest.raises(ValueError, match="cross-validates nothing"):
        validate_partition([], 10)


# --------------------------------------------------------------------------
# SEC, SEP and Q^2, §5 and §6
# --------------------------------------------------------------------------

RNG = np.random.default_rng(11)


def _response(n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(2.0, 30.0, n)
    return y, y + RNG.normal(scale=0.8, size=n)


def test_the_identity_section_5_states_holds_exactly() -> None:
    """`RMSEP^2 = bias^2 + (n-1)/n * SEP^2` — §5 calls it the cheap unit test."""
    y, y_hat = _response()
    n = y.size

    assert rmse(y, y_hat) ** 2 == pytest.approx(
        bias(y, y_hat) ** 2 + (n - 1) / n * sep(y, y_hat) ** 2, rel=1e-12
    )


def test_no_such_identity_ties_rmsec_to_sec() -> None:
    """Their denominators differ by more than one, which §5 says explicitly."""
    y, y_hat = _response()
    n = y.size
    naive = bias(y, y_hat) ** 2 + (n - 1) / n * sec(y, y_hat, n_components=6) ** 2

    assert rmse(y, y_hat) ** 2 != pytest.approx(naive, rel=1e-6)


def test_sec_and_sep_are_the_same_scatter_over_different_denominators() -> None:
    y, y_hat = _response()
    n = y.size
    for components in (0, 3, 7):
        ratio = sec(y, y_hat, n_components=components) / sep(y, y_hat)
        assert ratio == pytest.approx(np.sqrt((n - 1) / (n - components - 1)), rel=1e-12)


def test_sec_is_computed_by_hand_on_a_case_small_enough_to_check() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_hat = np.array([1.5, 2.0, 2.5, 4.5, 5.0, 5.5])
    # The residuals are [-0.5, 0, 0.5, -0.5, 0, 0.5], so their sum of squares
    # is exactly 1 and each denominator is the only thing left to check.
    # bias is zero here, so the correction subtracts nothing and only the
    # denominator distinguishes the three numbers.
    assert bias(y, y_hat) == pytest.approx(0.0)
    assert rmse(y, y_hat) == pytest.approx(np.sqrt(1.0 / 6))
    assert sep(y, y_hat) == pytest.approx(np.sqrt(1.0 / 5))
    assert sec(y, y_hat, n_components=2) == pytest.approx(np.sqrt(1.0 / 3))


def test_both_are_bias_corrected_so_an_offset_moves_rmse_and_not_them() -> None:
    """That is the whole distinction: total error against scatter about the mean."""
    y, y_hat = _response()
    shifted = y_hat + 4.0

    assert rmse(y, shifted) > rmse(y, y_hat)
    assert sep(y, shifted) == pytest.approx(sep(y, y_hat), rel=1e-12)
    assert sec(y, shifted, n_components=5) == pytest.approx(
        sec(y, y_hat, n_components=5), rel=1e-12
    )
    assert bias(y, shifted) == pytest.approx(bias(y, y_hat) - 4.0)


def test_sec_with_no_degrees_of_freedom_left_is_refused_by_name() -> None:
    """§5: report it as absent, never fall back to another denominator."""
    y, y_hat = _response(n=8)

    assert sec(y, y_hat, n_components=6) > 0.0
    with pytest.raises(ValueError, match="8 samples less 7 components"):
        sec(y, y_hat, n_components=7)
    with pytest.raises(ValueError, match="SEC is undefined"):
        sec(y, y_hat, n_components=20)


def test_sep_needs_two_samples_to_correct_a_bias_with() -> None:
    with pytest.raises(ValueError, match="SEP is undefined for 1 sample"):
        sep([3.0], [2.5])


def test_a_negative_component_count_is_refused_rather_than_widening_the_denominator() -> None:
    y, y_hat = _response()
    with pytest.raises(ValueError, match="must not be negative"):
        sec(y, y_hat, n_components=-1)


def test_q2_is_the_same_formula_as_r2_over_held_out_predictions() -> None:
    """§6 puts both on the total sum of squares about the full calibration mean."""
    y, calibration = _response()
    held_out = y + RNG.normal(scale=2.0, size=y.size)

    assert q2(y, held_out) == r2(y, held_out)
    assert q2(y, held_out) < r2(y, calibration), "held-out predictions are the worse ones"


def test_q2_uses_the_full_calibration_mean_not_a_per_fold_one() -> None:
    """Recomputing the mean inside each fold changes the number; packages differ."""
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    held_out = np.array([1.5, 2.5, 2.5, 10.5, 11.5, 11.5])

    total = float(np.sum((y - y.mean()) ** 2))
    press = float(np.sum((y - held_out) ** 2))
    assert q2(y, held_out) == pytest.approx(1.0 - press / total)

    # Two folds split the two clusters, so a per-fold mean would give a much
    # smaller denominator and a much worse Q^2. This is the number packages
    # disagree on, so it is pinned rather than left implied.
    per_fold_total = sum(
        float(np.sum((y[rows] - y[rows].mean()) ** 2)) for rows in ([0, 1, 2], [3, 4, 5])
    )
    assert per_fold_total < total / 10
    assert q2(y, held_out) != pytest.approx(1.0 - press / per_fold_total)


def test_q2_is_negative_when_the_model_is_worse_than_the_mean_and_is_not_clipped() -> None:
    """§6: reporting it as zero hides a failed model."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    useless = np.array([4.0, 3.0, 2.0, 1.0])

    assert q2(y, useless) < 0.0
