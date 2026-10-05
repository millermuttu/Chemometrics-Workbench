"""Outlier diagnostics (#278), against `docs/algorithms/outliers.md`.

Leverage and studentised residuals have no reference implementation in this
environment (section 7), so their claim is here: each equals its textbook
definition, recomputed by a different route. The robust distance's parity
claim is in `tests/test_parity.py`; what is here is its behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.outliers import (
    leverage,
    leverage_limit,
    min_cov_det,
    robust_distance_limit,
    studentised_residuals,
)
from chemometrics_workbench.regression import PCR, PLS


@pytest.fixture(scope="module")
def centred() -> tuple[np.ndarray, np.ndarray]:
    data = load_tecator()
    x = np.asarray(data.spectra)
    y = np.asarray(data.targets["fat"], dtype=np.float64)
    return x - x.mean(axis=0), y - y.mean()


def _hat(scores: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(scores.shape[0]), scores])
    return np.diag(design @ np.linalg.inv(design.T @ design) @ design.T)


# --------------------------------------------------------------------------
# leverage, section 2
# --------------------------------------------------------------------------


def test_leverage_is_the_hat_diagonal_and_sums_to_a_plus_one(
    centred: tuple[np.ndarray, np.ndarray],
) -> None:
    x, _ = centred
    scores = np.asarray(PCA(5).fit(x).scores_)
    h = leverage(scores)
    np.testing.assert_allclose(h, _hat(scores), rtol=1e-10, atol=1e-14)
    assert h.sum() == pytest.approx(6.0)
    # Centred, orthogonal scores: the closed form section 2 gives.
    closed = 1 / x.shape[0] + ((scores**2) / (scores**2).sum(axis=0)).sum(axis=1)
    np.testing.assert_allclose(h, closed, rtol=1e-10)
    assert leverage_limit(x.shape[0], 5) == pytest.approx(3 * 6 / x.shape[0])


def test_leverage_refuses_scores_that_repeat_the_intercept() -> None:
    with pytest.raises(ValueError, match="linearly dependent"):
        leverage(np.column_stack([np.ones(10), np.arange(10.0)]))


# --------------------------------------------------------------------------
# studentised residuals, section 3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", [PLS, PCR])
def test_the_calibration_fit_is_least_squares_on_the_scores(
    centred: tuple[np.ndarray, np.ndarray], estimator: Any
) -> None:
    """Section 3's claim that the scaling is exact: PLS1 and PCR fit y by
    least squares on [1, T], so e has covariance sigma^2 (I - H)."""
    x, y = centred
    model = estimator(6).fit(x, y)
    scores = np.asarray(model.x_scores_)
    design = np.column_stack([np.ones(x.shape[0]), scores])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(model.predict(x), fitted, rtol=1e-8, atol=1e-8)

    e = y - model.predict(x)
    h = _hat(scores)
    s = np.sqrt((e**2).sum() / (x.shape[0] - 6 - 1))
    np.testing.assert_allclose(
        studentised_residuals(e, leverage(scores), 6), e / (s * np.sqrt(1 - h)), rtol=1e-8
    )


def test_no_residual_degrees_of_freedom_is_refused() -> None:
    with pytest.raises(ValueError, match="no residual degrees of freedom"):
        studentised_residuals(np.ones(4), np.full(4, 0.5), 3)


def test_an_exactly_fitted_row_has_no_studentised_residual() -> None:
    r = studentised_residuals([0.0, 1.0, -1.0, 0.5, 0.2], [1.0, 0.3, 0.3, 0.2, 0.2], 1)
    assert np.isnan(r[0]) and np.all(np.isfinite(r[1:]))


# --------------------------------------------------------------------------
# robust distance, section 4
# --------------------------------------------------------------------------


def test_the_robust_distance_sees_a_cluster_the_classical_one_masks() -> None:
    """Masking, section 4: twelve planted outliers pull the mean and inflate
    the covariance until the classical distance misses some of them."""
    rng = np.random.default_rng(7)
    clean = rng.standard_normal((88, 2))
    planted = rng.normal([6.0, 6.0], 0.3, (12, 2))
    values = np.vstack([clean, planted])
    robust = min_cov_det(values)
    limit = robust_distance_limit(2)
    assert np.all(robust.distances[88:] > limit)
    assert not robust.support[88:].any()

    centred = values - values.mean(axis=0)
    classical = (centred @ np.linalg.inv(np.cov(values.T, bias=True)) * centred).sum(axis=1)
    assert np.count_nonzero(classical[88:] > limit) < 12


def test_the_search_is_seeded() -> None:
    values = np.random.default_rng(3).standard_normal((60, 3))
    first, second = min_cov_det(values), min_cov_det(values)
    np.testing.assert_array_equal(first.distances, second.distances)


def test_one_dimension_is_exact_and_agrees_with_scikit_learn() -> None:
    """Section 4: A = 1 needs no search. scikit-learn's univariate MCD is the
    same shortest-half rule, and deterministic, so it compares directly."""
    values = np.random.default_rng(11).standard_normal((50, 1))
    values[:5] += 8.0
    ours = min_cov_det(values)
    theirs = MinCovDet(random_state=0).fit(values)
    np.testing.assert_allclose(ours.distances, theirs.dist_, rtol=1e-12)
    np.testing.assert_allclose(ours.location, theirs.location_, rtol=1e-12)


def test_too_few_rows_for_a_robust_covariance_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot support a robust covariance"):
        min_cov_det(np.random.default_rng(0).standard_normal((4, 3)))


def test_the_limit_is_the_chi_squared_quantile() -> None:
    assert robust_distance_limit(5) == pytest.approx(chi2.ppf(0.975, 5))


# --------------------------------------------------------------------------
# class-wise diagnostics, section 8 (#335)
# --------------------------------------------------------------------------


def test_class_wise_diagnostics_match_an_independent_pca_per_class() -> None:
    """Each class against its own PCA: T² and Q from scikit-learn's PCA of that
    class's rows, leverage from the hat diagonal of its scores, limits at that
    class's n (pca.md sections 7 and 8, outliers.md section 2)."""
    from sklearn.decomposition import PCA as Reference

    from chemometrics_workbench.outliers import class_diagnostics, leverage_limit

    tecator = load_tecator()
    x = np.asarray(tecator.spectra, dtype=np.float64)
    fat = np.asarray(tecator.targets["fat"])
    codes = np.digitize(fat, np.quantile(fat, [1 / 3, 2 / 3]))
    found = class_diagnostics(x, codes, 4)

    assert found.n_components == [4, 4, 4]
    for code in range(3):
        rows = np.flatnonzero(codes == code)
        centred = x[rows] - x[rows].mean(axis=0)
        reference = Reference(4, svd_solver="full").fit(centred)
        scores = reference.transform(centred)
        t2 = np.sum(scores**2 / reference.explained_variance_, axis=1)
        q = np.sum((centred - scores @ reference.components_) ** 2, axis=1)
        np.testing.assert_allclose(found.t2[rows], t2, rtol=1e-6)
        np.testing.assert_allclose(found.q[rows], q, rtol=1e-6, atol=1e-12)
        z = np.column_stack([np.ones(rows.size), scores])
        hat = np.diag(z @ np.linalg.inv(z.T @ z) @ z.T)
        np.testing.assert_allclose(found.leverage[rows], hat, rtol=1e-8)
        assert set(found.leverage_limit[rows]) == {leverage_limit(rows.size, 4)}
        own = PCA(4).fit(centred)
        assert set(found.t2_limit[rows]) == {own.hotelling_t2_limit(0.05)}
        assert set(found.q_limit[rows]) == {own.spe_limit(0.05)}


def test_a_class_too_small_for_a_pca_is_refused_by_code() -> None:
    from chemometrics_workbench.outliers import class_diagnostics

    x = np.random.default_rng(0).standard_normal((12, 5))
    codes = np.array([0] * 10 + [1] * 2)
    with pytest.raises(ValueError, match="class 1 has 2 samples"):
        class_diagnostics(x, codes, 2)
