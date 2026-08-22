"""Tests for the PLS kernel.

The parity claims — that coefficients, predictions, VIP, RMSEC, R^2 and the
RMSECV curve agree with scikit-learn and with the R `pls` vignette — live in
`tests/test_parity.py` and go through the harness. What is tested here is
everything parity cannot see: the conventions `pls-regression.md` fixes, the
identities each quantity satisfies by definition, and the diagnostics, whose
limits have no reference implementation available here and are checked against
the distributions they claim to come from instead.

Synthetic data throughout, with the shape of real spectra. A kernel test that
needs a downloaded dataset is a kernel test that skips on a fresh machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.models import ResolvedSplit
from chemometrics_workbench.regression import PLS, cross_validated_predictions, rmsecv_curve
from chemometrics_workbench.validation import (
    folds_from_indices,
    k_fold,
    leave_one_out,
    rmse,
    validate_partition,
)

RNG = np.random.default_rng(20260822)


def _spectra_and_response(
    n: int = 40, p: int = 15, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """A block with the shape of real spectra and a response that is genuinely in it."""
    axis = np.linspace(0.0, 1.0, p)
    latent = RNG.normal(size=(n, 4)) * np.array([4.0, 2.0, 0.8, 0.2])
    shapes = np.vstack([np.sin((k + 1) * 3.0 * axis) for k in range(4)])
    X = 2.0 + latent @ shapes + 0.01 * RNG.normal(size=(n, p))
    y = latent @ np.array([1.5, -0.7, 0.3, 0.05]) + noise * RNG.normal(size=n)
    return X, y


def _centred(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X - X.mean(axis=0), y - y.mean()


def _fitted(a: int = 3, **kwargs: object) -> tuple[PLS, np.ndarray, np.ndarray]:
    X, y = _spectra_and_response(**kwargs)  # type: ignore[arg-type]
    Xc, yc = _centred(X, y)
    return PLS(a).fit(Xc, yc), Xc, yc


# --------------------------------------------------------------------------
# what PLS does and does not do, §3 and §12
# --------------------------------------------------------------------------


def test_pls_centres_nothing() -> None:
    """`pls-regression.md` §3: centring is a pipeline step with a node in the
    lineage, never something the estimator does for itself — which is where it
    differs twice over from `PLSRegression`, whose centring is unconditional
    and whose scaling is on by default."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)

    on_raw = PLS(3).fit(X, y)
    on_centred = PLS(3).fit(Xc, yc)

    assert not np.allclose(on_raw.predict(Xc), on_centred.predict(Xc))


def test_the_callers_arrays_are_never_modified() -> None:
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    before_x, before_y = Xc.copy(), yc.copy()

    model = PLS(3).fit(Xc, yc)
    model.predict(Xc)
    model.spe(Xc)

    assert np.array_equal(Xc, before_x)
    assert np.array_equal(yc, before_y)


def test_float32_input_is_promoted_and_results_are_float64() -> None:
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)

    model = PLS(2).fit(Xc.astype(np.float32), yc.astype(np.float32))

    assert model.coefficients_ is not None
    assert model.coefficients_.dtype == np.float64
    assert model.predict(Xc).dtype == np.float64


def test_missing_values_are_rejected_on_both_blocks() -> None:
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)

    with_hole = Xc.copy()
    with_hole[2, 3] = np.nan
    with pytest.raises(ValueError, match="row 2, column 3"):
        PLS(2).fit(with_hole, yc)

    missing_response = yc.copy()
    missing_response[4] = np.nan
    with pytest.raises(ValueError, match="position 4"):
        PLS(2).fit(Xc, missing_response)


def test_a_response_of_the_wrong_length_is_rejected() -> None:
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    with pytest.raises(ValueError, match="samples and y has"):
        PLS(2).fit(Xc, yc[:-1])


def test_a_column_vector_response_is_refused_rather_than_ravelled() -> None:
    """An `n x 1` array and a `1 x n` array print alike and mean opposite things."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    with pytest.raises(ValueError, match="must be 1-D"):
        PLS(2).fit(Xc, yc.reshape(-1, 1))


def test_a_transposed_matrix_is_rejected_on_predict() -> None:
    model, Xc, _ = _fitted()
    with pytest.raises(ValueError, match="fitted on 15 variables"):
        model.predict(Xc.T)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.predict(np.zeros((2, 15))),
        lambda m: m.transform(np.zeros((2, 15))),
        lambda m: m.vip(),
        lambda m: m.hotelling_t2(),
        lambda m: m.spe_limit(),
        lambda m: m.explained_variance_ratio(),
    ],
)
def test_anything_before_fit_is_refused(call: object) -> None:
    with pytest.raises(RuntimeError, match="has not been fitted"):
        call(PLS(2))  # type: ignore[operator]


def test_pls_is_deterministic_and_takes_no_seed() -> None:
    """§12: NIPALS PLS1 has no random state and no inner iteration."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    first = PLS(4).fit(Xc, yc)
    second = PLS(4).fit(Xc, yc)

    assert first.coefficients_ is not None and second.coefficients_ is not None
    assert np.array_equal(first.coefficients_, second.coefficients_)


@pytest.mark.parametrize("n_components", [0, -1])
def test_a_meaningless_component_count_is_refused(n_components: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        PLS(n_components)


def test_more_components_than_the_matrix_supports_is_an_error_naming_both() -> None:
    """§12: the ceiling is min(n-1, p), and silent truncation would make the
    reported component count a lie."""
    X, y = _spectra_and_response(n=8, p=15)
    Xc, yc = _centred(X, y)
    with pytest.raises(ValueError, match=r"min\(n-1, p\) = 7"):
        PLS(8).fit(Xc, yc)


# --------------------------------------------------------------------------
# the algorithm's own identities, §4 and §5
# --------------------------------------------------------------------------


def test_weights_are_unit_length() -> None:
    model, _, _ = _fitted(4)
    assert model.weights_ is not None
    assert np.allclose(np.linalg.norm(model.weights_, axis=0), 1.0)


def test_scores_and_weights_are_both_orthogonal() -> None:
    """§4: `t_a't_b = 0` for a != b, and for **PLS1** the weights are orthogonal too.

    The score orthogonality is what makes `lambda_a = t_a't_a/(n-1)` a variance
    and the `T^2` of §9 a Mahalanobis distance; a kernel whose scores were
    correlated would still produce a plausible-looking `T^2` that measured
    nothing. The weight orthogonality is a property of the single-response case
    specifically — it does not hold for PLS2, which is why §4 states the two
    separately."""
    model, _, _ = _fitted(4)
    assert model.x_scores_ is not None and model.weights_ is not None

    gram = model.x_scores_.T @ model.x_scores_
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() < 1e-10 * np.abs(np.diag(gram)).max()

    assert np.abs(model.weights_.T @ model.weights_ - np.eye(4)).max() < 1e-10


def test_both_blocks_are_deflated() -> None:
    """§4: X is deflated by `t_a p_a'` and y by `q_a t_a`.

    Reconstructing X from the scores and loadings and y from the scores and
    y-loadings is the identity that fails if either deflation is skipped."""
    model, Xc, _ = _fitted(4)
    assert model.x_scores_ is not None
    assert model.x_loadings_ is not None
    assert model.y_loadings_ is not None
    assert model.spe_ is not None

    residual_x = Xc - model.x_scores_ @ model.x_loadings_.T
    assert np.allclose((residual_x**2).sum(axis=1), model.spe_)
    assert np.allclose(model.x_scores_ @ model.y_loadings_, model.predict(Xc))


def test_the_rotation_matrix_makes_scores_computable_from_undeflated_x() -> None:
    """§5: `P'W` is upper triangular with unit diagonal, so `R = W(P'W)^-1`
    always exists and `T = XR` reaches the same scores the deflation produced."""
    model, Xc, _ = _fitted(4)
    assert model.x_loadings_ is not None and model.weights_ is not None
    assert model.x_scores_ is not None

    triangular = model.x_loadings_.T @ model.weights_
    assert np.allclose(np.diag(triangular), 1.0)
    assert np.abs(np.tril(triangular, -1)).max() < 1e-12
    assert np.allclose(model.transform(Xc), model.x_scores_)


def test_coefficients_are_the_rotations_times_the_y_loadings() -> None:
    """§5: `b = Rq`, and `y_hat = Xb` is the same prediction the scores give."""
    model, Xc, _ = _fitted(3)
    assert model.rotations_ is not None and model.y_loadings_ is not None
    assert model.coefficients_ is not None
    assert np.allclose(model.coefficients_, model.rotations_ @ model.y_loadings_)
    assert np.allclose(model.predict(Xc), Xc @ model.rotations_ @ model.y_loadings_)


def test_a_full_rank_model_reproduces_the_least_squares_fit() -> None:
    """With as many components as the matrix supports, PLS spans the same row
    space as ordinary least squares and must reach the same predictions."""
    X, y = _spectra_and_response(n=40, p=8)
    Xc, yc = _centred(X, y)
    model = PLS(8).fit(Xc, yc)
    ols, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    assert np.allclose(model.predict(Xc), Xc @ ols, atol=1e-8)


# --------------------------------------------------------------------------
# the sign convention, §6
# --------------------------------------------------------------------------


def test_the_largest_magnitude_weight_of_every_component_is_positive() -> None:
    model, _, _ = _fitted(4)
    assert model.weights_ is not None
    dominant = np.abs(model.weights_).argmax(axis=0)
    assert (model.weights_[dominant, np.arange(4)] > 0).all()


def test_the_sign_rule_leaves_the_coefficients_alone() -> None:
    """§5: flipping a component negates `w`, `t`, `p` and `q` together, so `Rq`
    is unchanged — which is why coefficients and predictions are compared
    against a reference without sign alignment.

    Negating the response negates every weight before the rule is applied, so
    the rule flips each component back and only the sign of `b` follows the
    sign of `y`."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)

    positive = PLS(3).fit(Xc, yc)
    negative = PLS(3).fit(Xc, -yc)

    assert positive.weights_ is not None and negative.weights_ is not None
    dominant = np.abs(negative.weights_).argmax(axis=0)
    assert (negative.weights_[dominant, np.arange(3)] > 0).all()
    assert np.allclose(np.asarray(negative.coefficients_), -np.asarray(positive.coefficients_))


# --------------------------------------------------------------------------
# stopping early, §4
# --------------------------------------------------------------------------


def test_an_exhausted_response_stops_the_fit_and_says_so() -> None:
    """§4: stopping early is reported, never silent.

    A matrix built from two orthogonal directions whose response *is* the first
    of them has nothing left after one component, and continuing would fit
    numerical noise and report a component count the model does not have."""
    latent = RNG.normal(size=(40, 2))
    # Orthonormal *and* already centred: every column of Q lies in the span of
    # a centred matrix, so the construction survives the centring step and the
    # response really is exhausted by one component rather than nearly so.
    scores = np.linalg.qr(latent - latent.mean(axis=0))[0] * 10.0
    loadings = np.linalg.qr(RNG.normal(size=(12, 2)))[0]
    Xc = scores @ loadings.T
    yc = scores[:, 0]

    with pytest.warns(RuntimeWarning, match="response was exhausted"):
        model = PLS(4).fit(Xc, yc)

    assert model.stopped_early_
    assert model.n_components_ == 1
    assert np.allclose(model.predict(Xc), yc, atol=1e-8)


def test_a_response_with_no_covariance_at_all_is_refused() -> None:
    X, _ = _spectra_and_response()
    Xc = X - X.mean(axis=0)
    with pytest.raises(ValueError, match="no covariance at all"):
        PLS(2).fit(Xc, np.zeros(Xc.shape[0]))


# --------------------------------------------------------------------------
# VIP, §8
# --------------------------------------------------------------------------


def test_the_squared_vip_scores_sum_to_the_variable_count() -> None:
    """§8: `sum_j VIP_j^2 = p` exactly, which is the whole origin of the
    "VIP greater than 1" rule of thumb and the cheapest check that the
    normalisation was not dropped."""
    for a in (1, 2, 4):
        model, _, _ = _fitted(a)
        vip = model.vip()
        assert float((vip**2).sum()) == pytest.approx(vip.size)
        assert (vip >= 0).all()


def test_vip_depends_on_the_component_count() -> None:
    """§8: VIP is a property of the fitted model, not of the data. Reporting it
    without A is meaningless, and this is why."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    assert not np.allclose(PLS(1).fit(Xc, yc).vip(), PLS(3).fit(Xc, yc).vip())


# --------------------------------------------------------------------------
# explained variance
# --------------------------------------------------------------------------


def test_the_cumulative_y_variance_is_the_models_r_squared() -> None:
    """The X-scores are orthogonal, so the y sums of squares add up exactly and
    the running total at component `a` is the R^2 of the `a`-component model."""
    X, y = _spectra_and_response()
    Xc, yc = _centred(X, y)
    model = PLS(4).fit(Xc, yc)

    cumulative = model.cumulative_explained_variance("y")
    for a in range(1, 5):
        predictions = PLS(a).fit(Xc, yc).predict(Xc)
        r_squared = 1.0 - np.sum((yc - predictions) ** 2) / np.sum(yc**2)
        assert cumulative[a - 1] == pytest.approx(r_squared)


def test_explained_variance_is_normalised_over_the_whole_block() -> None:
    """Normalising over the retained components would put every curve at 100%."""
    model, _, _ = _fitted(3)
    assert model.cumulative_explained_variance("x")[-1] < 1.0
    assert model.cumulative_explained_variance("y")[-1] < 1.0


def test_an_unknown_block_is_refused() -> None:
    model, _, _ = _fitted(2)
    with pytest.raises(ValueError, match="block must be"):
        model.explained_variance_ratio("z")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# diagnostics, §9
# --------------------------------------------------------------------------


def test_the_mean_calibration_t2_is_exactly_a_times_n_minus_one_over_n() -> None:
    """The cheapest check that the eigenvalue weighting and the divisor in
    lambda agree with each other (§9, `pca.md` §7)."""
    for a in (1, 3, 5):
        model, _, _ = _fitted(a, n=40, p=15)
        t2 = model.hotelling_t2()
        assert float(t2.mean()) == pytest.approx(a * (40 - 1) / 40)


def test_spe_is_the_residual_of_the_calibration_matrix() -> None:
    model, Xc, _ = _fitted(3)
    assert model.spe_ is not None
    assert np.allclose(model.spe(Xc), model.spe_)


def test_spe_and_t2_are_computed_from_the_same_projection() -> None:
    """A held-out sample's distances travel the same path a calibration
    sample's do — through `T = XR`, with the fitted model's parameters."""
    model, Xc, _ = _fitted(3)
    held_out = Xc[:5] + 0.01 * RNG.normal(size=(5, Xc.shape[1]))

    scores = model.transform(held_out)
    assert model.x_loadings_ is not None
    residual = held_out - scores @ model.x_loadings_.T
    assert np.allclose(model.spe(held_out), (residual**2).sum(axis=1))


def test_the_spe_limit_covers_about_alpha_of_the_calibration_samples() -> None:
    """§9: a chi-squared moment match on the observed residuals, so the
    coverage is what it claims to be only if `g` and `h` were matched to the
    right two moments."""
    model, Xc, _ = _fitted(2, n=200, p=15)
    beyond = float((model.spe(Xc) > model.spe_limit(0.05)).mean())
    assert 0.0 < beyond < 0.15


def test_a_tighter_alpha_gives_a_higher_spe_limit() -> None:
    model, _, _ = _fitted(2)
    assert model.spe_limit(0.01) > model.spe_limit(0.05) > model.spe_limit(0.20)


def test_the_pls_spe_limit_is_not_the_pca_one() -> None:
    """§9 states the difference and it is worth asserting, because a limit
    computed the Jackson-Mudholkar way here would be a number with no
    derivation behind it: PLS components are not eigenvectors of the covariance
    of X, so there is no residual eigenvalue sequence to sum."""
    from chemometrics_workbench.decomposition import PCA

    X, y = _spectra_and_response(n=60, p=15)
    Xc, yc = _centred(X, y)
    pls = PLS(3).fit(Xc, yc)
    pca = PCA(3).fit(Xc)

    assert pls.spe_limit(0.05) != pytest.approx(pca.spe_limit(0.05), rel=1e-3)


def test_a_model_with_no_residual_left_is_refused_a_spe_limit() -> None:
    """A full-rank model's residual is zero by construction, so there is no
    distribution to take a quantile of and no limit should be drawn."""
    X, y = _spectra_and_response(n=40, p=8)
    Xc, yc = _centred(X, y)
    model = PLS(8).fit(Xc, yc)
    with pytest.raises(ValueError, match="leaves no residual"):
        model.spe_limit()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_alpha_is_refused(alpha: float) -> None:
    model, _, _ = _fitted(2)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model.spe_limit(alpha)


def test_the_t2_limits_are_the_shared_ones() -> None:
    """§9 takes `pca.md` §7 unchanged, so the two must agree for the same n and a."""
    from chemometrics_workbench.decomposition import hotelling_t2_limit

    model, _, _ = _fitted(3, n=40, p=15)
    assert model.hotelling_t2_limit(0.05) == hotelling_t2_limit(40, 3, 0.05)
    assert model.hotelling_t2_limit(0.05, "new") == hotelling_t2_limit(40, 3, 0.05, "new")


# --------------------------------------------------------------------------
# cross-validation, metrics-and-validation.md §7 and §9
# --------------------------------------------------------------------------


def test_every_sample_is_predicted_by_a_model_that_did_not_see_it() -> None:
    X, y = _spectra_and_response(n=30)
    folds = k_fold(30, 5)
    held_out = cross_validated_predictions(X, y, folds, 3)

    for fold in folds:
        x_mean = X[fold.train].mean(axis=0)
        y_mean = float(y[fold.train].mean())
        model = PLS(3).fit(X[fold.train] - x_mean, y[fold.train] - y_mean)
        assert np.allclose(held_out[fold.test], model.predict(X[fold.test] - x_mean) + y_mean)


def test_centring_is_refitted_inside_each_fold() -> None:
    """`metrics-and-validation.md` §9: centring on everything before the split
    leaks the validation samples into the training statistics and makes the
    estimate optimistic. Doing it correctly gives a *larger* RMSECV, and this
    case pins the direction as well as the difference."""
    X, y = _spectra_and_response(n=30)
    folds = k_fold(30, 5)
    honest = rmse(y, cross_validated_predictions(X, y, folds, 3))

    leaked_x = X - X.mean(axis=0)
    leaked_y = y - y.mean()
    leaked = np.empty_like(y)
    for fold in folds:
        model = PLS(3).fit(leaked_x[fold.train], leaked_y[fold.train])
        leaked[fold.test] = model.predict(leaked_x[fold.test]) + y.mean()

    assert honest > rmse(y, leaked)


def test_rmsecv_pools_residuals_rather_than_averaging_fold_rmses() -> None:
    """§7: the two differ whenever the folds are uneven in size or difficulty,
    and the pooled form is the one that stays comparable with RMSEC."""
    X, y = _spectra_and_response(n=31)
    folds = k_fold(31, 4)
    held_out = cross_validated_predictions(X, y, folds, 3)

    pooled = rmse(y, held_out)
    mean_of_folds = float(np.mean([rmse(y[f.test], held_out[f.test]) for f in folds]))

    assert pooled == pytest.approx(rmsecv_curve(X, y, folds, 3)[2])
    assert pooled != pytest.approx(mean_of_folds, rel=1e-6)


def test_the_curve_uses_one_fold_assignment_for_every_component_count() -> None:
    """§9: one split, one pass, one curve — so the shape of the curve is a
    property of the model and not of `A` unrelated experiments."""
    X, y = _spectra_and_response(n=30)
    folds = k_fold(30, 5)
    curve = rmsecv_curve(X, y, folds, 4)

    for a in range(1, 5):
        assert curve[a - 1] == pytest.approx(rmse(y, cross_validated_predictions(X, y, folds, a)))


def test_a_cross_validated_fit_replays_from_a_stored_resolved_split_alone() -> None:
    """The reproducibility claim, end to end (§10).

    A run is recorded as a `ResolvedSplit` — index lists, not a seed — and
    rerunning it must reproduce the metric **exactly**, bit for bit, from those
    indices and nothing else. The stored form survives a change of random
    number generator, which a seed does not.
    """
    X, y = _spectra_and_response(n=30)
    folds = k_fold(30, 5, seed=42)
    original = rmse(y, cross_validated_predictions(X, y, folds, 3))

    recorded = ResolvedSplit(
        node_id="split-1",
        train_indices=[fold.train.tolist() for fold in folds],
        test_indices=[fold.test.tolist() for fold in folds],
    )
    replayed = folds_from_indices(recorded.train_indices, recorded.test_indices)
    validate_partition(replayed, 30)

    assert rmse(y, cross_validated_predictions(X, y, replayed, 3)) == original


def test_leave_one_out_predicts_every_sample_from_the_other_n_minus_one() -> None:
    X, y = _spectra_and_response(n=25)
    held_out = cross_validated_predictions(X, y, leave_one_out(25), 2)
    assert held_out.shape == y.shape
    assert rmse(y, held_out) > 0.0


def test_a_split_that_is_not_a_partition_is_refused_before_anything_is_pooled() -> None:
    X, y = _spectra_and_response(n=20)
    folds = folds_from_indices([list(range(10, 20))], [list(range(0, 10))])
    with pytest.raises(ValueError, match="do not partition"):
        cross_validated_predictions(X, y, folds, 2)


def test_a_curve_with_no_components_is_refused() -> None:
    X, y = _spectra_and_response(n=20)
    with pytest.raises(ValueError, match="at least one component"):
        rmsecv_curve(X, y, k_fold(20, 4), 0)
