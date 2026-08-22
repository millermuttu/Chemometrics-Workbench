"""The parity suite. Every claim the report in #14 renders is made here.

Run it alone with `uv run pytest -m parity`; it also runs as part of the full
suite, and writes `parity-results.json` either way.

The preprocessing cases call the kernels in
`chemometrics_workbench.preprocessing`, and the PCA cases call
`chemometrics_workbench.decomposition.PCA`. The PLS cases do not yet call a
kernel — that is #12 — so what they compare is arithmetic the specification
defines directly: prediction from coefficients, and the RMSEC denominator.
Each is still a real comparison against a real fixture entry at a real
tolerance, and each is rewritten to call its kernel when one lands. The entry
id, the tolerance and the tier stay as they are when that happens.

Every case takes the same shape: compute a quantity, hand it to
`parity.check()` with the fixture entry it should match, and let the harness
decide the tolerance, the sign handling and the claim tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_corn, load_gasoline, load_tecator
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.preprocessing import (
    AutoscaleTransformer,
    MeanCentreTransformer,
    NormaliseTransformer,
    SavitzkyGolayTransformer,
)
from tests import parity

pytestmark = pytest.mark.parity

DATASETS = ("corn", "gasoline", "tecator")

LOADERS = {"corn": load_corn, "gasoline": load_gasoline, "tecator": load_tecator}


@pytest.fixture(scope="module")
def fixture_entries() -> dict[str, dict[str, object]]:
    return parity.entries_by_id()


# The submatrix the preprocessing references were generated on. Recorded in
# the fixture notes; repeated here because a mismatch would compare two
# different blocks and pass.
PREPROCESS_BLOCK = (slice(0, 5), slice(0, 8))

# The Savitzky-Golay configuration the references were generated at. Same rule
# as the block: repeated from the generator because a mismatch would compare a
# different filter and pass.
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2


def _block(name: str) -> np.ndarray:
    return LOADERS[name]().spectra[PREPROCESS_BLOCK]


def _centred(name: str) -> np.ndarray:
    """Mean centring, through the kernel that is supposed to do it.

    `pca.md` §2: centring is an explicit step, never something the algorithm
    does for itself. Every PCA and PLS comparison below depends on this being
    right, which is what makes it a useful thing to fail first.
    """
    return MeanCentreTransformer().fit_transform(LOADERS[name]().spectra)


# --------------------------------------------------------------------------
# preprocessing kernels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_mean_centring_matches_the_reference(dataset: str) -> None:
    ours = MeanCentreTransformer().fit_transform(_block(dataset))
    assert parity.check(f"{dataset}.preprocess.mean_centred.sklearn", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_autoscaling_matches_the_reference_at_its_ddof(dataset: str) -> None:
    """`ddof=0` is passed explicitly, because StandardScaler offers no choice.

    Our default is `ddof=1`, the sample convention used for eigenvalues in
    `pca.md` §4 and for SEC and SEP in `metrics-and-validation.md` §5. Comparing
    our default against StandardScaler would fail, and it would be a convention
    difference rather than a defect — so the comparison is made at the
    reference's convention and the divergence is recorded in the module
    docstring of `preprocessing.py`.
    """
    ours = AutoscaleTransformer(ddof=0).fit_transform(_block(dataset))
    assert parity.check(f"{dataset}.preprocess.autoscaled_ddof0.sklearn", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalisation_matches_the_reference(dataset: str, norm: str) -> None:
    ours = NormaliseTransformer(norm).fit_transform(_block(dataset))  # type: ignore[arg-type]
    assert parity.check(f"{dataset}.preprocess.normalised_{norm}.sklearn", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savitzky_golay_matches_the_reference(dataset: str, deriv: int) -> None:
    """Smoothing and both derivatives, against SciPy at `mode="interp"`.

    SciPy is a runtime dependency of this project, but `scipy.signal` is not on
    the kernel's code path: SciPy solves a least-squares system per output
    position, and the kernel builds one convolution matrix from the
    pseudo-inverse of the window's Vandermonde matrix. The comparison is
    between two routes to the same filter, which is what makes it worth making.

    The block is 8 variables wide and the half-window is 2, so **four of every
    eight columns are edge columns** — this case is as much a test of the
    `interp` edge convention as of the interior.
    """
    ours = SavitzkyGolayTransformer(SAVGOL_WINDOW, SAVGOL_POLYORDER, deriv).fit_transform(
        _block(dataset)
    )
    assert parity.check(f"{dataset}.preprocess.savgol_deriv{deriv}.scipy", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savitzky_golay_agrees_with_the_reference_at_the_first_and_last_variable(
    dataset: str, deriv: int, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """The bound columns alone, asserted separately and deliberately.

    The case above would still pass if the edges agreed by accident while the
    interior carried the whole comparison. This one compares nothing but the
    first and last variable, which is where `interp`, `mirror` and `nearest`
    disagree, and it is the case that fails if the edge convention ever moves.
    """
    reference = parity.as_array(
        fixture_entries[f"{dataset}.preprocess.savgol_deriv{deriv}.scipy"]["value"]
    )
    ours = SavitzkyGolayTransformer(SAVGOL_WINDOW, SAVGOL_POLYORDER, deriv).fit_transform(
        _block(dataset)
    )
    edges = [0, -1]
    np.testing.assert_allclose(
        ours[:, edges],
        reference[:, edges],
        rtol=parity.TOLERANCES["smoothing"].rtol,
        atol=parity.TOLERANCES["smoothing"].atol,
    )


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------


N_COMPONENTS = 5

# One fit per dataset, reused across the PCA cases below. The model is what is
# being compared from six angles, so fitting it six times would only be slower.


@pytest.fixture(scope="module")
def pca_models() -> dict[str, PCA]:
    """`pca.md` §2: the kernel never centres, so the centring is a step here."""
    return {name: PCA(N_COMPONENTS).fit(_centred(name)) for name in DATASETS}


@pytest.mark.parametrize("dataset", DATASETS)
def test_pca_loadings_match_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §4 and §5. Our sign rule is the largest-magnitude loading;
    scikit-learn's is decided from U, so the harness aligns before comparing."""
    result = parity.check(f"{dataset}.pca.loadings.sklearn", pca_models[dataset].loadings_)
    assert result.passed
    assert result.sign_aligned, "loadings are sign-invariant and must be aligned"


@pytest.mark.parametrize("dataset", DATASETS)
def test_pca_scores_match_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §4: T = XP, computed through the same path a new sample takes."""
    result = parity.check(f"{dataset}.pca.scores.sklearn", pca_models[dataset].scores_)
    assert result.passed
    assert result.sign_aligned, "scores are sign-invariant and must be aligned"


@pytest.mark.parametrize("dataset", DATASETS)
def test_pca_eigenvalues_match_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §4: lambda_k = sigma_k^2/(n-1), the sample-variance convention.

    Sign-invariant by construction — squaring removes the sign — so this is
    also the case that would catch a sign convention leaking into a quantity
    that should not have one. The fixture holds the first five; the model keeps
    all r, because §6 and §8 are sums over the ones it discarded.
    """
    eigenvalues = pca_models[dataset].eigenvalues_
    assert eigenvalues is not None
    result = parity.check(f"{dataset}.pca.eigenvalues.sklearn", eigenvalues[:N_COMPONENTS])
    assert result.passed
    assert not result.sign_aligned


@pytest.mark.parametrize("dataset", DATASETS)
def test_pca_explained_variance_ratio_matches_the_reference(
    dataset: str, pca_models: dict[str, PCA]
) -> None:
    """`pca.md` §6: the denominator is the total over all r components.

    Normalising by the five retained ones instead would make the cumulative
    curve reach 100% every time, and this is the case that catches it — the
    reference sums to well under 1 on every dataset.
    """
    ratio = pca_models[dataset].explained_variance_ratio()
    result = parity.check(f"{dataset}.pca.explained_variance_ratio.sklearn", ratio)
    assert result.passed
    assert ratio.sum() < 1.0, "five components explaining everything means the wrong denominator"


@pytest.mark.parametrize("dataset", DATASETS)
def test_pca_cumulative_explained_variance_matches_the_reference(
    dataset: str, pca_models: dict[str, PCA]
) -> None:
    """`pca.md` §6, the curve the component-count decision is read off.

    The reference is the running total of the entry above rather than an
    independently sourced quantity, so this claim is worth exactly one thing:
    that our curve is a cumulative sum of ratios taken over the right
    denominator. A curve reaching 1.0 at the last retained component would be
    the classic wrong denominator, and it is asserted separately here.
    """
    cumulative = pca_models[dataset].cumulative_explained_variance()
    result = parity.check(f"{dataset}.pca.cumulative_explained_variance.sklearn", cumulative)
    assert result.passed
    assert cumulative[-1] < 1.0


@pytest.mark.parametrize("dataset", DATASETS)
def test_hotelling_t2_matches_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §7, on the calibration samples.

    scikit-learn reports no T^2, so the reference is computed in the generator
    from *its* scores and eigenvalues by the definition. What that tests is our
    decomposition against theirs, carried through a formula both sides agree
    on — worth having, and worth not overstating in the report.
    """
    result = parity.check(f"{dataset}.pca.hotelling_t2.sklearn", pca_models[dataset].hotelling_t2())
    assert result.passed
    assert not result.sign_aligned, "T^2 squares every score, so it carries no sign"


@pytest.mark.parametrize("dataset", DATASETS)
def test_spe_matches_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §8: the sum of squares of the residual, not its mean or root."""
    result = parity.check(f"{dataset}.pca.spe.sklearn", pca_models[dataset].spe(_centred(dataset)))
    assert result.passed
    assert not result.sign_aligned


# --------------------------------------------------------------------------
# PLS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_predictions_follow_from_the_coefficients(
    dataset: str, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """`pls-regression.md` §5: y_hat = Xb, un-centred back to original units.

    `metrics-and-validation.md` §2 requires the response's original units, so
    the calibration mean goes back on before anything is compared. Coefficients
    are sign-invariant already (§5), so no alignment is involved here.
    """
    target = parity.load_fixture()["targets"][dataset]
    dataset_object = LOADERS[dataset]()
    y = dataset_object.targets[target]

    coefficients = parity.as_array(fixture_entries[f"{dataset}.pls.coefficients.sklearn"]["value"])
    predictions = _centred(dataset) @ coefficients + y.mean()

    result = parity.check(f"{dataset}.pls.predictions.sklearn", predictions)
    assert result.passed
    assert not result.sign_aligned


@pytest.mark.parametrize("dataset", DATASETS)
def test_rmsec_follows_from_the_predictions(
    dataset: str, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """`metrics-and-validation.md` §4: the divisor is n, not n - A - 1.

    A package dividing by n - A - 1 under this name is reporting our SEC, and
    on these datasets the difference is a few percent — comfortably outside
    the metric tolerance, so this case would catch it.
    """
    target = parity.load_fixture()["targets"][dataset]
    y = LOADERS[dataset]().targets[target]
    predictions = parity.as_array(fixture_entries[f"{dataset}.pls.predictions.sklearn"]["value"])
    rmsec = np.sqrt(np.mean((y - predictions) ** 2))

    assert parity.check(f"{dataset}.pls.rmsec.sklearn", rmsec).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_r2_is_the_residual_form(
    dataset: str, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """`metrics-and-validation.md` §6: residual form, not squared correlation.

    The two coincide for a least-squares fit on the same data, which is what
    this case is, so passing here does not distinguish them. The distinction
    bites on a prediction set and is tested when one exists.
    """
    target = parity.load_fixture()["targets"][dataset]
    y = LOADERS[dataset]().targets[target]
    predictions = parity.as_array(fixture_entries[f"{dataset}.pls.predictions.sklearn"]["value"])
    r2 = 1.0 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)

    assert parity.check(f"{dataset}.pls.r2.sklearn", r2).passed


# --------------------------------------------------------------------------
# the third tier
# --------------------------------------------------------------------------


def test_tecator_published_sep_is_a_documented_divergence() -> None:
    """The one value in the fixture that is real, published and not comparable.

    `tecator.pls.sep.thodberg` is 2.78 from a linear model on ten of the 22
    principal components the file supplies and `load_tecator()` discards. It
    belongs in the report — it is the only published number travelling with
    that dataset — but as a documented divergence, never as a number we failed
    to reproduce.
    """
    result = parity.record_divergence(
        "tecator.pls.sep.thodberg",
        reason=(
            "Not reproducible from what we model. The reference is a linear model on "
            "ten of the 22 principal components supplied inside tecator.txt, computed "
            "by the original authors in 1992 on subset C; load_tecator() discards that "
            "block because it is preprocessing rather than data. The file also never "
            "states its SEP denominator. Recorded for context, not as a target."
        ),
    )
    assert result.tier is parity.Tier.DOCUMENTED_DIVERGENCE
    assert result.passed


def test_a_non_comparable_entry_cannot_be_checked_numerically() -> None:
    """The flag is load-bearing: check() refuses rather than quietly comparing."""
    with pytest.raises(ValueError, match="comparable=false"):
        parity.check("tecator.pls.sep.thodberg", 2.78)
