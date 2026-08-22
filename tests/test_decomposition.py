"""Tests for the PCA kernel.

The parity claims — that scores, loadings, eigenvalues, explained variance,
`T^2` and SPE agree with an independent decomposition — live in
`tests/test_parity.py` and go through the harness. What is tested here is
everything parity cannot see: the conventions `pca.md` fixes, the identities
each quantity satisfies by definition, and the two confidence limits, which
have no reference implementation available here and are checked against the
distributions they claim to come from instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.decomposition import PCA

RNG = np.random.default_rng(20260822)


def _spectra(n: int = 30, p: int = 12) -> np.ndarray:
    """A small block with the shape of real spectra: smooth, positive, correlated."""
    axis = np.linspace(0.0, 1.0, p)
    latent = RNG.normal(size=(n, 4)) * np.array([4.0, 2.0, 0.8, 0.2])
    shapes = np.vstack([np.sin((k + 1) * 3.0 * axis) for k in range(4)])
    block: np.ndarray = 2.0 + latent @ shapes + 0.01 * RNG.normal(size=(n, p))
    return block


def _centred(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0)


def _structured(n: int, p: int, a: int, seed: int) -> np.ndarray:
    """Centred data with `a + 6` real directions of decreasing variance.

    The discarded components have to carry genuine variance, or the SPE limit
    is a quantile of numerical noise and tells us nothing.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, a + 6)) * np.linspace(5.0, 0.4, a + 6)
    basis = np.linalg.qr(rng.normal(size=(p, a + 6)))[0]
    return _centred(latent @ basis.T)


# --------------------------------------------------------------------------
# what PCA does and does not do, §2 and §11
# --------------------------------------------------------------------------


def test_pca_does_not_centre() -> None:
    """`pca.md` §2: centring is a pipeline step, never something the estimator
    does for itself, because preprocessing hidden inside an estimator is absent
    from the recipe and therefore from the lineage."""
    X = _spectra()

    on_raw = PCA(3).fit(X)
    on_centred = PCA(3).fit(_centred(X))

    assert on_raw.loadings_ is not None and on_centred.loadings_ is not None
    assert not np.allclose(on_raw.loadings_, on_centred.loadings_), (
        "fitting raw and centred data gave the same loadings, so something centred on its own"
    )
    # The uncentred first component largely captures the mean spectrum, which
    # is the reason the application warns about a PCA node with no centring
    # upstream. The centred model's first component cannot: the mean is gone.
    mean_direction = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
    raw_alignment = abs(float(on_raw.loadings_[:, 0] @ mean_direction))
    centred_alignment = abs(float(on_centred.loadings_[:, 0] @ mean_direction))
    assert raw_alignment > 0.9 > centred_alignment


def test_the_callers_array_is_never_modified() -> None:
    X = _spectra()
    before = X.copy()

    model = PCA(3).fit(X)
    model.transform(X)
    model.spe(X)

    np.testing.assert_array_equal(X, before)


def test_float32_input_is_promoted_and_results_are_float64() -> None:
    X = _spectra().astype(np.float32)

    model = PCA(3).fit(X)

    assert model.loadings_ is not None and model.loadings_.dtype == np.float64
    assert model.transform(X).dtype == np.float64
    assert X.dtype == np.float32, "the caller's array was cast in place"


def test_missing_values_are_rejected_with_their_position() -> None:
    """`pca.md` §10. Imputing inside PCA would make a partly invented result
    look exactly as trustworthy as a complete one."""
    X = _spectra()
    X[2, 5] = np.nan

    with pytest.raises(ValueError, match="row 2, column 5"):
        PCA(3).fit(X)


def test_a_one_dimensional_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        PCA(2).fit(np.arange(12, dtype=float))


def test_a_transposed_array_is_rejected_on_transform() -> None:
    X = _spectra(n=30, p=12)
    model = PCA(3).fit(X)

    with pytest.raises(ValueError, match="never silently transposed"):
        model.transform(X.T)


def test_anything_before_fit_is_refused() -> None:
    model = PCA(2)
    for call in (
        lambda: model.transform(_spectra()),
        lambda: model.explained_variance_ratio(),
        lambda: model.hotelling_t2(),
        lambda: model.spe(_spectra()),
        lambda: model.hotelling_t2_limit(),
        lambda: model.spe_limit(),
    ):
        with pytest.raises(RuntimeError, match="has not been fitted"):
            call()


def test_pca_is_deterministic_and_takes_no_seed() -> None:
    """`pca.md` §3: SVD is deterministic, and randomised SVD is not used —
    a decomposition whose result depends on a random draw is not something a
    parity report can stand behind."""
    X = _spectra()

    first = PCA(4).fit(X)
    second = PCA(4).fit(X)

    assert first.loadings_ is not None and second.loadings_ is not None
    np.testing.assert_array_equal(first.loadings_, second.loadings_)
    np.testing.assert_array_equal(first.scores_, second.scores_)


# --------------------------------------------------------------------------
# the outputs, §4
# --------------------------------------------------------------------------


def test_loadings_are_orthonormal() -> None:
    model = PCA(4).fit(_centred(_spectra()))

    assert model.loadings_ is not None
    np.testing.assert_allclose(model.loadings_.T @ model.loadings_, np.eye(4), atol=1e-12)


def test_scores_are_the_matrix_projected_onto_the_loadings() -> None:
    """`pca.md` §4: T = XP, computed that way rather than taken as U*sigma, so
    that calibration scores and a new sample's scores travel the same path."""
    X = _centred(_spectra())
    model = PCA(3).fit(X)

    assert model.loadings_ is not None
    np.testing.assert_allclose(model.scores_, X @ model.loadings_)  # type: ignore[arg-type]
    np.testing.assert_allclose(model.transform(X), model.scores_)  # type: ignore[arg-type]


def test_eigenvalues_are_the_sample_variance_of_the_scores() -> None:
    """`pca.md` §4: lambda_k = sigma_k^2/(n-1), which is var(t_k) at ddof=1."""
    X = _centred(_spectra())
    model = PCA(4).fit(X)

    assert model.eigenvalues_ is not None and model.scores_ is not None
    np.testing.assert_allclose(model.eigenvalues_[:4], model.scores_.var(axis=0, ddof=1))


def test_every_eigenvalue_is_kept_not_only_the_retained_ones() -> None:
    """`pca.md` §4. §6's denominator and §8's limit are both sums over the
    components the model discarded, so a model holding only the first `a`
    could not compute its own SPE limit."""
    X = _centred(_spectra(n=30, p=12))
    model = PCA(2).fit(X)

    assert model.eigenvalues_ is not None and model.rank_ is not None
    assert model.eigenvalues_.size == model.rank_ > 2
    # Centring spends one degree of freedom, so the rank is n-1 or p, whichever
    # is smaller (§9).
    assert model.rank_ == min(X.shape[0] - 1, X.shape[1])


def test_projecting_a_held_out_sample_is_stable_and_is_a_plain_projection() -> None:
    """The issue's fifth verification step.

    The model is fitted once and the held-out block projected twice; the result
    is bit-identical, and equals `X_new @ P` with the *calibration* loadings.
    """
    train = _centred(_spectra(n=30))
    held_out = _spectra(n=5) - _spectra(n=30).mean(axis=0)

    model = PCA(3).fit(train)
    first = model.transform(held_out)
    second = model.transform(held_out)

    np.testing.assert_array_equal(first, second)
    assert model.loadings_ is not None
    np.testing.assert_allclose(first, held_out @ model.loadings_)


# --------------------------------------------------------------------------
# the sign convention, §5
# --------------------------------------------------------------------------


def test_the_largest_magnitude_loading_of_every_component_is_positive() -> None:
    """`pca.md` §5, the rule itself. Keyed on the loading rather than on U,
    because the loading is the spectrum-shaped vector an analyst reads."""
    model = PCA(5).fit(_centred(_spectra(n=40, p=20)))

    assert model.loadings_ is not None
    for k in range(5):
        column = model.loadings_[:, k]
        assert column[np.abs(column).argmax()] > 0


def test_the_sign_rule_survives_a_negated_input() -> None:
    """Negating X negates every score and leaves every loading where the rule
    put it — which is what makes the convention worth having."""
    X = _centred(_spectra())

    positive = PCA(3).fit(X)
    negated = PCA(3).fit(-X)

    assert positive.loadings_ is not None and negated.loadings_ is not None
    assert positive.scores_ is not None and negated.scores_ is not None
    np.testing.assert_allclose(negated.loadings_, positive.loadings_, atol=1e-12)
    np.testing.assert_allclose(negated.scores_, -positive.scores_, atol=1e-12)


def test_a_tie_in_the_largest_loading_is_broken_by_the_smaller_index() -> None:
    """Two variables of exactly equal magnitude in a component is contrived,
    and the rule still has to be deterministic when it happens."""
    # A 2 x 2 rotation of a symmetric pair gives loadings of equal magnitude.
    X = np.array([[1.0, -1.0], [-1.0, 1.0], [2.0, -2.0], [-2.0, 2.0]])
    model = PCA(1).fit(X)

    assert model.loadings_ is not None
    np.testing.assert_allclose(np.abs(model.loadings_[:, 0]), np.abs(model.loadings_[0, 0]))
    assert model.loadings_[0, 0] > 0, "the tie should have been broken by the smaller index"


# --------------------------------------------------------------------------
# explained variance, §6
# --------------------------------------------------------------------------


def test_explained_variance_is_normalised_over_every_component() -> None:
    """`pca.md` §6: the denominator is the total variance of the fitted matrix.

    Normalising by the retained components would make the cumulative curve
    reach 100% every time, which is both useless and a mistake seen in the wild.
    """
    X = _centred(_spectra(n=30, p=12))
    model = PCA(2).fit(X)

    ratio = model.explained_variance_ratio()
    assert ratio.size == 2
    assert ratio.sum() < 1.0

    assert model.eigenvalues_ is not None
    np.testing.assert_allclose(model.eigenvalues_.sum(), X.var(axis=0, ddof=1).sum())
    np.testing.assert_allclose(ratio, model.eigenvalues_[:2] / model.eigenvalues_.sum())


def test_cumulative_explained_variance_is_the_running_total() -> None:
    model = PCA(4).fit(_centred(_spectra()))

    ratio = model.explained_variance_ratio()
    cumulative = model.cumulative_explained_variance()

    np.testing.assert_allclose(cumulative, np.cumsum(ratio))
    assert np.all(np.diff(cumulative) >= 0.0)
    np.testing.assert_allclose(cumulative[0], ratio[0])


def test_a_full_rank_model_explains_everything() -> None:
    """The one case where the cumulative curve does reach 1, and it is a
    property of the model rather than of the normalisation."""
    X = _centred(_spectra(n=8, p=5))
    model = PCA(5).fit(X)

    np.testing.assert_allclose(model.cumulative_explained_variance()[-1], 1.0)


# --------------------------------------------------------------------------
# Hotelling's T-squared, §7
# --------------------------------------------------------------------------


def test_hotelling_t2_is_the_eigenvalue_weighted_sum_of_squared_scores() -> None:
    X = _centred(_spectra())
    model = PCA(3).fit(X)

    assert model.scores_ is not None and model.eigenvalues_ is not None
    expected = ((model.scores_**2) / model.eigenvalues_[:3]).sum(axis=1)

    np.testing.assert_allclose(model.hotelling_t2(), expected)
    np.testing.assert_allclose(model.hotelling_t2(X), expected)


def test_the_mean_calibration_t2_is_exactly_a_times_n_minus_one_over_n() -> None:
    """An identity, not an approximation: sum_i t_ik^2 = (n-1)lambda_k for
    every component, so the T^2 column sums to `a` and its mean is a(n-1)/n.

    It is the cheapest check that the eigenvalue weighting and the divisor in
    lambda agree with each other — get either wrong and this drifts.
    """
    for n, a in ((30, 3), (40, 5)):
        X = _centred(_spectra(n=n, p=12))
        model = PCA(a).fit(X)
        np.testing.assert_allclose(model.hotelling_t2().mean(), a * (n - 1) / n)


def test_the_two_t2_limits_differ_and_both_cover_about_alpha() -> None:
    """`pca.md` §7. Two limits because they answer different questions; the
    beta form is exact for the samples the model was fitted to, the F form is
    for samples projected onto an existing model. They converge as n grows,
    and both are implemented rather than one approximating the other.
    """
    X = _structured(n=200, p=10, a=3, seed=11)
    model = PCA(3).fit(X)

    calibration = model.hotelling_t2_limit()
    new_samples = model.hotelling_t2_limit(samples="new")
    assert new_samples > calibration, "the limit for new samples is the looser of the two"

    exceeded = float((model.hotelling_t2() > calibration).mean())
    assert 0.02 < exceeded < 0.10, f"a 95% limit that {exceeded:.1%} of samples exceed"


def test_a_tighter_alpha_gives_a_higher_limit() -> None:
    model = PCA(3).fit(_structured(n=200, p=10, a=3, seed=12))

    assert model.hotelling_t2_limit(alpha=0.01) > model.hotelling_t2_limit(alpha=0.05)
    assert model.spe_limit(alpha=0.01) > model.spe_limit(alpha=0.05)


def test_a_limit_with_too_few_samples_is_refused_rather_than_drawn() -> None:
    """§7: the beta form needs n > a + 1 and the F form n > a. Outside those,
    no limit is defined and none should be drawn."""
    model = PCA(4).fit(_centred(_spectra(n=5, p=12)))

    with pytest.raises(ValueError, match=r"beta limit needs n > a \+ 1"):
        model.hotelling_t2_limit()

    # n = 5, a = 4: the F form is still defined, one sample short of the beta.
    assert model.hotelling_t2_limit(samples="new") > 0

    # n = a needs an uncentred fit to reach at all: centring costs the degree of
    # freedom that would have made the rank large enough.
    square = PCA(4).fit(_spectra(n=4, p=12))
    with pytest.raises(ValueError, match="F limit needs n > a"):
        square.hotelling_t2_limit(samples="new")


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
def test_an_impossible_alpha_is_refused(alpha: float) -> None:
    model = PCA(2).fit(_centred(_spectra()))

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.hotelling_t2_limit(alpha=alpha)
    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        model.spe_limit(alpha=alpha)


def test_an_unknown_limit_audience_is_refused() -> None:
    model = PCA(2).fit(_centred(_spectra()))

    with pytest.raises(ValueError, match="must be 'calibration' or 'new'"):
        model.hotelling_t2_limit(samples="everyone")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# SPE, §8
# --------------------------------------------------------------------------


def test_spe_is_the_sum_of_squares_of_the_residual() -> None:
    """`pca.md` §8: the sum of squares, not the mean and not the root. Other
    packages report one of the other two and converting is trivial; silent
    disagreement is not."""
    X = _centred(_spectra())
    model = PCA(3).fit(X)

    assert model.loadings_ is not None
    residual = X - model.transform(X) @ model.loadings_.T

    np.testing.assert_allclose(model.spe(X), (residual**2).sum(axis=1))
    assert not np.allclose(model.spe(X), (residual**2).mean(axis=1))


def test_spe_and_t2_split_the_total_distance_between_them() -> None:
    """Everything in the row space is either in the model plane or normal to
    it, so the residual sum of squares plus the reconstruction's equals the
    row's. This is what makes the two diagnostics complementary rather than
    two views of the same number.
    """
    X = _centred(_spectra())
    model = PCA(3).fit(X)

    reconstruction = (model.transform(X) ** 2).sum(axis=1)
    np.testing.assert_allclose(model.spe(X) + reconstruction, (X**2).sum(axis=1))


def test_a_full_rank_model_has_zero_residual_and_no_limit() -> None:
    """`pca.md` §8, the degenerate case: report SPE as exactly zero and draw no
    limit rather than taking a quantile of an empty sum."""
    X = _centred(_spectra(n=8, p=5))
    model = PCA(5).fit(X)

    np.testing.assert_allclose(model.spe(X), 0.0, atol=1e-24)

    with pytest.raises(ValueError, match="no SPE limit exists"):
        model.spe_limit()


def test_the_spe_limit_covers_about_alpha_of_the_calibration_samples() -> None:
    """The Jackson–Mudholkar limit has no reference implementation available
    here — the R `mdatools` fixture entries are unsourced because R is not
    installed — so what is checked is the claim it makes: that about `alpha` of
    the samples it was built from lie beyond it.
    """
    for seed, (n, p, a) in enumerate(((200, 10, 3), (500, 20, 5), (300, 15, 4))):
        model = PCA(a).fit(_structured(n=n, p=p, a=a, seed=seed))
        exceeded = float(
            (model.spe(_structured(n=n, p=p, a=a, seed=seed)) > model.spe_limit()).mean()
        )
        assert 0.01 < exceeded < 0.10, f"a 95% SPE limit that {exceeded:.1%} of samples exceed"


def test_the_spe_limit_is_built_from_the_discarded_eigenvalues_alone() -> None:
    """§8, written out here because it is the reason §4 keeps every eigenvalue.

    Two models over the same data retaining different numbers of components
    have different residual spectra and therefore different limits, and the
    limit falls as more components are retained.
    """
    X = _structured(n=200, p=10, a=3, seed=13)

    limits = [PCA(a).fit(X).spe_limit() for a in (2, 3, 4)]

    assert limits[0] > limits[1] > limits[2]


# --------------------------------------------------------------------------
# rank, §9
# --------------------------------------------------------------------------


def test_asking_for_more_components_than_the_rank_is_an_error() -> None:
    """`pca.md` §9: naming both numbers, and never silently truncating.

    Asking for 40 components from 30 samples is a misunderstanding worth
    surfacing; silently returning fewer makes downstream shapes unpredictable.
    """
    X = _centred(_spectra(n=8, p=5))

    with pytest.raises(ValueError, match="6 components were asked of a matrix of rank 5"):
        PCA(6).fit(X)


def test_a_rank_deficient_matrix_is_measured_by_its_singular_values() -> None:
    """Duplicated variables cost rank, and the rank has to come from the SVD:
    PCA does not centre, so it cannot tell from the shape alone whether a
    degree of freedom was already spent."""
    X = _centred(_spectra(n=20, p=6))
    X[:, 3] = X[:, 1]
    X[:, 5] = X[:, 0] + X[:, 2]

    model = PCA(3).fit(X)

    # Six columns, two of them exact combinations of the others.
    assert model.rank_ == 4


def test_a_matrix_with_nothing_left_after_centring_has_no_components_at_all() -> None:
    """Identical spectra centre to exact zeros, and a zero matrix has no
    direction of greatest variance to find. Note that the *uncentred* constant
    matrix has rank 1 — its one component is the mean spectrum, which is §2's
    point about fitting uncentred data made as sharply as it can be made."""
    constant = np.full((6, 4), 2.5)
    assert PCA(1).fit(constant).rank_ == 1

    with pytest.raises(ValueError, match="no variance to decompose"):
        PCA(1).fit(_centred(constant))


@pytest.mark.parametrize("n_components", [0, -1])
def test_a_meaningless_component_count_is_refused(n_components: int) -> None:
    with pytest.raises(ValueError, match="n_components must be at least 1"):
        PCA(n_components)
