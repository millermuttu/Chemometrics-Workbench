"""iPLS (#282), against `docs/algorithms/variable-selection.md`.

The parity claim is in `tests/test_parity.py`; what is here is the shape of
the method: the intervals, the refusals, the forward rule and the positions
Apply writes.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.regression import rmsecv_curve
from chemometrics_workbench.selection import cars, interval_bounds, ipls
from chemometrics_workbench.validation import k_fold, train_test


def test_intervals_are_contiguous_and_the_first_ones_wider() -> None:
    assert interval_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert interval_bounds(4, 4) == [(0, 1), (1, 2), (2, 3), (3, 4)]


@pytest.mark.parametrize("count", [1, 11])
def test_an_interval_count_the_spectrum_cannot_hold_is_refused(count: int) -> None:
    with pytest.raises(ValueError, match="between 2 and 10"):
        interval_bounds(10, count)


def test_one_split_is_refused() -> None:
    x = np.random.default_rng(0).standard_normal((20, 8))
    with pytest.raises(ValueError, match="at least two folds"):
        ipls(x, x[:, 0], train_test(20, 0.25), 4, 2)


def test_a_planted_interval_is_the_first_one_chosen() -> None:
    """Noise everywhere but one interval, whose columns carry y."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((60, 40))
    y = x[:, 20:25] @ np.array([1.0, -0.5, 0.8, 0.3, 0.6]) + 0.05 * rng.standard_normal(60)
    result = ipls(x, y, k_fold(60, 5), 8, 3)
    assert result.steps[0][0] == 4
    assert result.intervals[4].rmsecv == min(i.rmsecv for i in result.intervals)
    assert set(range(20, 25)) <= set(result.selected)


def test_each_step_lowers_the_error_and_the_selection_is_the_path() -> None:
    data = load_tecator()
    x, y = np.asarray(data.spectra), np.asarray(data.targets["fat"])
    folds = k_fold(len(y), 5)
    result = ipls(x, y, folds, 10, 5)
    errors = [rmsecv for _, rmsecv in result.steps]
    assert errors == sorted(errors, reverse=True) and len(set(errors)) == len(errors)
    chosen = sorted(k for k, _ in result.steps)
    expected = [
        c for k in chosen for c in range(result.intervals[k].start, result.intervals[k].stop)
    ]
    assert result.selected == expected
    # The union's error is the curve's minimum on those columns, recomputed.
    curve = rmsecv_curve(x[:, expected], y, folds, 5)
    assert errors[-1] == pytest.approx(float(curve.min()))


def test_a_matrix_per_fold_is_used_fold_by_fold() -> None:
    """Section 2: below a split each fold's matrix is its own."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((30, 12))
    y = x[:, 0] + 0.1 * rng.standard_normal(30)
    folds = k_fold(30, 3)
    same = ipls([x, x, x], y, folds, 3, 2)
    assert same.intervals == ipls(x, y, folds, 3, 2).intervals


# --------------------------------------------------------------------------
# CARS, section 6
# --------------------------------------------------------------------------


def _planted(seed: int = 3) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sixty noise variables, five of which carry y."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((80, 60))
    informative = [7, 19, 33, 41, 52]
    y = x[:, informative] @ np.array([1.0, -0.8, 0.6, 0.9, -0.7]) + 0.05 * rng.standard_normal(80)
    return x, y, informative


def test_cars_is_deterministic_for_a_seed() -> None:
    x, y, _ = _planted()
    folds = k_fold(80, 5)
    first, again = cars(x, y, folds, 5, seed=11), cars(x, y, folds, 5, seed=11)
    assert first.runs == again.runs and first.best == again.best
    assert cars(x, y, folds, 5, seed=12).runs != first.runs


def test_cars_recovers_the_informative_variables() -> None:
    x, y, informative = _planted()
    result = cars(x, y, k_fold(80, 5), 5, n_runs=50)
    assert set(informative) <= set(result.selected)
    assert len(result.selected) < 20, "and leaves most of the noise behind"


def test_cars_shrinks_on_its_schedule_and_reports_every_run() -> None:
    x, y, _ = _planted()
    result = cars(x, y, k_fold(80, 5), 3, n_runs=20)
    sizes = [len(run.variables) for run in result.runs]
    assert len(sizes) == 20
    assert sizes == sorted(sizes, reverse=True), "the retained set never grows"
    assert sizes[-1] <= 2
    assert result.runs[result.best].rmsecv == min(run.rmsecv for run in result.runs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_runs": 1}, "at least two sampling runs"),
        ({"fraction": 1.0}, "between 0 and 1"),
    ],
)
def test_cars_refuses_settings_it_cannot_run(kwargs: dict[str, float], message: str) -> None:
    x, y, _ = _planted()
    with pytest.raises(ValueError, match=message):
        cars(x, y, k_fold(80, 5), 3, **kwargs)  # type: ignore[arg-type]


def test_cars_without_cross_validation_is_refused() -> None:
    x, y, _ = _planted()
    with pytest.raises(ValueError, match="at least two folds"):
        cars(x, y, train_test(80, 0.25), 3)
