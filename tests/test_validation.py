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
    r2,
    rmse,
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
