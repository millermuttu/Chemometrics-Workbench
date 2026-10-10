"""The y-permutation test (#333), against `metrics-and-validation.md` §14.

Parity is with scikit-learn's `permutation_test_score`, handed the same
permutations: its `random_state` is a `RandomState` whose `permutation` draws
from our `default_rng(seed)` stream, and its folds are ours. Two conventions are
bridged rather than ignored. scikit-learn averages a score per fold where we
pool residuals (§7), so the folds here are equal in size, where the two
coincide. And it scores higher-is-better, so the regression is compared on
negative MSE, whose fold mean over equal folds is minus the pooled MSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import permutation_test_score

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.regression import cross_validated_predictions
from chemometrics_workbench.validation import k_fold, permutation_test, rmse

N_PERMUTATIONS = 30
SEED = 7


class _OurStream(np.random.RandomState):
    """A `RandomState` that permutes from `numpy.random.default_rng(seed)`, so
    scikit-learn shuffles exactly as §14 does."""

    def __init__(self, seed: int) -> None:
        super().__init__(0)
        self._rng = np.random.default_rng(seed)

    def permutation(self, x: Any) -> Any:
        return self._rng.permutation(x)


class _PlsDa(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Two-class PLS-DA as pls-da.md defines it, in scikit-learn's interface."""

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components

    def fit(self, x: np.ndarray, y: np.ndarray) -> _PlsDa:
        self.model = PLSRegression(self.n_components, scale=False).fit(x, y.astype(float))
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(self.model.predict(x)).ravel() >= 0.5).astype(int)


@pytest.fixture(scope="module")
def tecator() -> tuple[np.ndarray, np.ndarray]:
    data = load_tecator()
    return np.asarray(data.spectra, dtype=np.float64), np.asarray(
        data.targets["fat"], dtype=np.float64
    )


def test_a_pls_rmsecv_null_matches_permutation_test_score(
    tecator: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = tecator
    folds = k_fold(len(y), 5, seed=42)  # 240 / 5: equal folds
    assert len({fold.test.size for fold in folds}) == 1

    def score(order: np.ndarray) -> float:
        return rmse(y[order], cross_validated_predictions(x, y[order], folds, 5))

    ours = permutation_test(score, len(y), N_PERMUTATIONS, seed=SEED, greater_is_better=False)
    observed, null, p_value = permutation_test_score(
        PLSRegression(5, scale=False),
        x,
        y,
        cv=[(fold.train, fold.test) for fold in folds],
        scoring="neg_mean_squared_error",
        n_permutations=N_PERMUTATIONS,
        random_state=_OurStream(SEED),
    )
    assert ours.observed**2 == pytest.approx(-observed, rel=1e-6)
    np.testing.assert_allclose(np.square(ours.null), -null, rtol=1e-6)
    assert ours.p_value == pytest.approx(p_value)


def test_a_pls_da_accuracy_null_matches_permutation_test_score(
    tecator: tuple[np.ndarray, np.ndarray],
) -> None:
    x, fat = tecator
    codes = (fat > np.median(fat)).astype(int)
    folds = k_fold(len(codes), 5, seed=42)

    def score(order: np.ndarray) -> float:
        labels = codes[order]
        predicted = cross_validated_predictions(x, labels.astype(float), folds, 3) >= 0.5
        return float(np.mean(predicted == labels))

    ours = permutation_test(score, len(codes), N_PERMUTATIONS, seed=SEED, greater_is_better=True)
    observed, null, p_value = permutation_test_score(
        _PlsDa(3),
        x,
        codes,
        cv=[(fold.train, fold.test) for fold in folds],
        scoring="accuracy",
        n_permutations=N_PERMUTATIONS,
        random_state=_OurStream(SEED),
    )
    assert ours.observed == pytest.approx(observed)
    np.testing.assert_allclose(ours.null, null)
    assert ours.p_value == pytest.approx(p_value)


def test_the_null_follows_from_its_seed_and_the_p_value_is_never_zero() -> None:
    values = np.arange(20, dtype=np.float64)

    def score(order: np.ndarray) -> float:
        # The correlation of the response with its own position: highest unpermuted.
        return float(np.corrcoef(values, values[order])[0, 1])

    first = permutation_test(score, 20, 50, seed=3, greater_is_better=True)
    assert first == permutation_test(score, 20, 50, seed=3, greater_is_better=True)
    assert first.null != permutation_test(score, 20, 50, seed=4, greater_is_better=True).null
    assert first.observed == pytest.approx(1.0)
    assert first.p_value == pytest.approx(1 / 51)


def test_no_permutations_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one permutation"):
        permutation_test(lambda order: 0.0, 5, 0, greater_is_better=True)
