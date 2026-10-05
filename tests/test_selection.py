"""iPLS (#282), against `docs/algorithms/variable-selection.md`.

The parity claim is in `tests/test_parity.py`; what is here is the shape of
the method: the intervals, the refusals, the forward rule and the positions
Apply writes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.regression import rmsecv_curve
from chemometrics_workbench.selection import (
    cars,
    cars_selector,
    coefficient_selector,
    interval_bounds,
    ipls,
    ipls_selector,
    nested,
    selected_rmsecv,
    vip_selector,
)
from chemometrics_workbench.validation import Fold, by_group, k_fold, rmse, train_test


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


# --------------------------------------------------------------------------
# nested validation, section 8 (#331)
# --------------------------------------------------------------------------


def _sklearn_pls(x: np.ndarray, y: np.ndarray, a: int) -> Any:
    """Iterated to its fixed point, so a one-hot response agrees with PLS2's SVD."""
    import warnings

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return PLSRegression(n_components=a, scale=False, tol=0.0, max_iter=2000).fit(
            x - x.mean(axis=0), y - y.mean(axis=0)
        )


def _sklearn_vip(model: Any) -> np.ndarray:
    """Wold's VIP from scikit-learn's own weights, scores and y-loadings."""
    w = np.asarray(model.x_weights_)
    t = np.asarray(model.x_scores_)
    q = np.asarray(model.y_loadings_).ravel()
    explained = q**2 * np.sum(t**2, axis=0)
    w = w / np.linalg.norm(w, axis=0)
    vip: np.ndarray = np.sqrt(w.shape[0] * (w**2 @ explained) / explained.sum())
    return vip


def _sklearn_rmsecv_curve(
    x: np.ndarray, y: np.ndarray, folds: list[Fold], a_max: int
) -> list[float]:
    curve = []
    for a in range(1, a_max + 1):
        held = np.empty_like(y)
        for fold in folds:
            xt, yt = x[fold.train], y[fold.train]
            model = _sklearn_pls(xt, yt, a)
            predicted = np.asarray(model.predict(x[fold.test] - xt.mean(axis=0)))
            held[fold.test] = predicted.reshape(held[fold.test].shape) + yt.mean(axis=0)
        curve.append(rmse(y.ravel(), held.ravel()))
    return curve


def _sklearn_ipls(x: np.ndarray, y: np.ndarray, folds: list[Fold], k: int, a_max: int) -> list[int]:
    """§2-§3 restated, every fit scikit-learn's."""
    bounds = [np.arange(start, stop) for start, stop in interval_bounds(x.shape[1], k)]

    def best(columns: np.ndarray) -> float:
        return min(_sklearn_rmsecv_curve(x[:, columns], y, folds, min(a_max, columns.size)))

    chosen: list[int] = []
    current = np.inf
    while len(chosen) < k:
        error, pick = min(
            (best(np.concatenate([bounds[j] for j in sorted([*chosen, c])])), c)
            for c in range(k)
            if c not in chosen
        )
        if error >= current:
            break
        chosen.append(pick)
        current = error
    return [int(j) for c in sorted(chosen) for j in bounds[c]]


def _sklearn_nested(
    x: np.ndarray,
    y: np.ndarray,
    outer: list[Fold],
    choose: Callable[[np.ndarray, np.ndarray, list[Fold]], Any],
    a: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """The §8 loop, rebuilt: inner K-fold of five on each outer training set."""
    predicted = np.empty_like(y)
    selections: list[list[int]] = []
    for fold in outer:
        xt, yt = x[fold.train], y[fold.train]
        columns = np.asarray(choose(xt, yt, k_fold(fold.train.size, 5, seed=0)))
        model = _sklearn_pls(xt[:, columns], yt, min(a, columns.size))
        mean = xt[:, columns].mean(axis=0)
        predicted[fold.test] = (
            np.asarray(model.predict(x[np.ix_(fold.test, columns)] - mean)).ravel() + yt.mean()
        )
        selections.append([int(j) for j in columns])
    return predicted, selections


@pytest.fixture(scope="module")
def tecator_fat() -> tuple[np.ndarray, np.ndarray]:
    tecator = load_tecator()
    return np.asarray(tecator.spectra, dtype=np.float64), np.asarray(
        tecator.targets["fat"], dtype=np.float64
    )


def test_nested_vip_matches_a_scikit_learn_rebuild_on_the_served_folds(
    tecator_fat: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = tecator_fat
    outer = k_fold(len(y), 5, seed=42)
    ours = nested(x, y, outer, vip_selector(5), 5)
    theirs, selections = _sklearn_nested(
        x,
        y,
        outer,
        lambda xt, yt, _: np.flatnonzero(_sklearn_vip(_sklearn_pls(xt, yt, 5)) >= 1.0),
        5,
    )
    assert ours.selected == selections
    np.testing.assert_allclose(ours.predicted, theirs, rtol=1e-6, atol=1e-8)
    assert ours.outer_rmsecv == pytest.approx(rmse(y, theirs), rel=1e-6)


def test_nested_ipls_matches_a_scikit_learn_rebuild_on_the_served_folds(
    tecator_fat: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = tecator_fat
    outer = k_fold(len(y), 4, seed=42)
    ours = nested(x, y, outer, ipls_selector(5, 3), 3)
    theirs, selections = _sklearn_nested(
        x, y, outer, lambda xt, yt, inner: _sklearn_ipls(xt, yt, inner, 5, 3), 3
    )
    assert ours.selected == selections
    np.testing.assert_allclose(ours.predicted, theirs, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("method", ["ipls", "cars"])
def test_the_outer_error_is_at_least_the_inner_on_tecator(
    tecator_fat: tuple[np.ndarray, np.ndarray], method: str
) -> None:
    """§8: a selection chosen by cross-validated error flatters that error.

    VIP is not asserted: it is a filter on the fitted model that never looks at
    a cross-validated error, so nothing makes its inner number optimistic, and
    on Tecator its outer error happens to come out below it (§8)."""
    x, y = tecator_fat
    folds = k_fold(len(y), 5, seed=42)
    if method == "ipls":
        chosen = ipls(x, y, folds, 10, 5).selected
        selector = ipls_selector(10, 5)
    else:
        chosen = cars(x, y, folds, 5, n_runs=20).selected
        selector = cars_selector(5, n_runs=20)
    inner = selected_rmsecv(x, y, folds, chosen, 5)
    outer = nested(x, y, folds, selector, 5).outer_rmsecv
    assert outer >= inner


def test_a_grouped_outer_split_is_grouped_inside_too() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 6))
    y = x[:, 0] + 0.1 * rng.standard_normal(40)
    groups = [f"s{i // 2}" for i in range(40)]
    outer = by_group(groups, lambda m: k_fold(m, 4))
    seen: list[list[Fold]] = []

    def spy(xt: np.ndarray, yt: np.ndarray, inner: list[Fold]) -> list[int]:
        seen.append(inner)
        return [0, 1]

    nested(x, y, outer, spy, 2, groups=groups)
    for fold, inner in zip(outer, seen, strict=True):
        names = [groups[i] for i in fold.train]
        for one in inner:
            assert not {names[i] for i in one.train} & {names[i] for i in one.test}


def test_an_outer_fold_that_selects_nothing_is_refused_by_number() -> None:
    x = np.random.default_rng(2).standard_normal((20, 4))
    with pytest.raises(ValueError, match="outer fold 0's selection kept no variable"):
        nested(x, x[:, 0], k_fold(20, 2), lambda *_: [], 2)


# --------------------------------------------------------------------------
# from a PLS-DA, section 9 (#332)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tecator_terciles() -> tuple[np.ndarray, np.ndarray]:
    """Tecator in three classes by fat, one-hot."""
    tecator = load_tecator()
    fat = np.asarray(tecator.targets["fat"])
    codes = np.digitize(fat, np.quantile(fat, [1 / 3, 2 / 3]))
    return np.asarray(tecator.spectra, dtype=np.float64), np.eye(3)[codes]


def test_ipls_on_a_one_hot_response_matches_a_scikit_learn_rebuild(
    tecator_terciles: tuple[np.ndarray, np.ndarray],
) -> None:
    x, onehot = tecator_terciles
    folds = k_fold(len(onehot), 5, seed=42)
    ours = ipls(x, onehot, folds, 5, 3)
    for interval, (start, stop) in zip(ours.intervals, interval_bounds(100, 5), strict=True):
        reference = min(_sklearn_rmsecv_curve(x[:, start:stop], onehot, folds, 3))
        assert interval.rmsecv == pytest.approx(reference, rel=1e-6)
    assert ours.selected == _sklearn_ipls(x, onehot, folds, 5, 3)


def test_two_classes_are_pls1_on_the_codes(tecator_fat: tuple[np.ndarray, np.ndarray]) -> None:
    """Two classes: the dummy is a vector, and a one-hot of two columns pools to
    the same RMSE, because the columns' residuals are equal and opposite."""
    x, y = tecator_fat
    codes = (y > np.median(y)).astype(np.float64)
    folds = k_fold(len(y), 5, seed=42)
    vector = rmsecv_curve(x, codes, folds, 3)
    pooled = rmsecv_curve(x, np.eye(2)[codes.astype(int)], folds, 3)
    np.testing.assert_allclose(vector, pooled, rtol=1e-9)


def test_cars_and_the_nested_loop_run_on_a_one_hot_response(
    tecator_terciles: tuple[np.ndarray, np.ndarray],
) -> None:
    x, onehot = tecator_terciles
    folds = k_fold(len(onehot), 4, seed=42)
    first = cars(x, onehot, folds, 3, n_runs=8, seed=1)
    assert first == cars(x, onehot, folds, 3, n_runs=8, seed=1) and first.selected
    found = nested(x, onehot, folds, vip_selector(3), 3, inner_splits=3)
    assert len(found.predicted) == onehot.size and found.outer_rmsecv > 0


def test_a_coefficient_cut_on_three_classes_is_refused() -> None:
    x = np.random.default_rng(3).standard_normal((30, 5))
    onehot = np.eye(3)[np.arange(30) % 3]
    with pytest.raises(ValueError, match="one per class"):
        coefficient_selector(2, 0.1)(x, onehot, k_fold(30, 3))
