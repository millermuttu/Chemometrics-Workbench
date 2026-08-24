"""The parity suite. Every claim the report in #14 renders is made here.

Run it alone with `uv run pytest -m parity`; it also runs as part of the full
suite, and writes `parity-results.json` either way.

Every case now calls a kernel: the preprocessing cases call
`chemometrics_workbench.preprocessing`, the PCA cases call
`chemometrics_workbench.decomposition.PCA`, and the PLS cases call
`chemometrics_workbench.regression.PLS` and the cross-validation in the same
module. The two published R `pls` claims are the strongest in the file — the
vignette's leave-one-out RMSECV curve over the first 50 gasoline samples is
deterministic, with no shuffle stream to reconcile against ours, and R `pls`
computes MSEP as `SSE/nobj`, the same divisor as our RMSECV.

Every case takes the same shape: compute a quantity, hand it to
`parity.check()` with the fixture entry it should match, and let the harness
decide the tolerance, the sign handling and the claim tier.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from chemometrics_workbench.datasets import load_corn, load_gasoline, load_tecator
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.preprocessing import (
    AutoscaleTransformer,
    BaselineCorrectTransformer,
    MeanCentreTransformer,
    MSCTransformer,
    NormaliseTransformer,
    SavitzkyGolayTransformer,
    SNVTransformer,
)
from chemometrics_workbench.regression import PLS, rmsecv_curve
from chemometrics_workbench.validation import (
    Fold,
    folds_from_indices,
    k_fold,
    leave_one_out,
    r2,
    rmse,
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

# The baseline references are computed on a wider, shallower block, because a
# baseline over eight variables is not a baseline. Repeated from the generator
# for the same reason as PREPROCESS_BLOCK: a mismatch would compare two
# different blocks and pass.
BASELINE_BLOCK = (slice(0, 3), slice(0, 120))
ASLS_LAM = 1e5
ASLS_P = 0.01
ASLS_ITERATIONS = 20
POLYNOMIAL_ORDER = 2


def _block(name: str) -> np.ndarray:
    return LOADERS[name]().spectra[PREPROCESS_BLOCK]


def _folds_from_entry(entry: dict[str, object]) -> list[Fold]:
    """The fixture's resolved indices, replayed as a stored `ResolvedSplit` is (§10)."""
    folds = entry["split"]["folds"]  # type: ignore[index]
    return folds_from_indices(
        [f["train_indices"] for f in folds], [f["test_indices"] for f in folds]
    )


def _baseline_block(name: str) -> np.ndarray:
    return LOADERS[name]().spectra[BASELINE_BLOCK]


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
# scatter correction and baselines, against chemotools
# --------------------------------------------------------------------------
#
# These five quantities had no external reference at all until #13 evaluated
# chemotools and #27 wired it in: neither scikit-learn nor SciPy implements
# scatter correction or baseline estimation. The identity tests in
# test_preprocessing.py stay exactly where they are — an SNV row has mean 0 and
# standard deviation 1 *by construction*, which is a stronger statement than
# agreement with anyone — and these add the second opinion that the identities
# cannot give: that our arithmetic matches somebody else's.


@pytest.mark.parametrize("dataset", DATASETS)
def test_snv_matches_the_reference_at_its_ddof(dataset: str) -> None:
    """`ddof=0` is passed explicitly, exactly as for autoscaling.

    chemotools uses the population standard deviation and offers no choice;
    our default is `ddof=1`, the sample convention used for eigenvalues and for
    SEC and SEP. Comparing our default here would fail on a convention rather
    than on a defect, so the comparison is made at the reference's convention —
    and at that convention the two are bit-identical.
    """
    ours = SNVTransformer(ddof=0).fit_transform(_block(dataset))
    result = parity.check(f"{dataset}.preprocess.snv.chemotools", ours)
    assert result.passed
    assert result.tier is parity.Tier.IDENTICAL


@pytest.mark.parametrize("dataset", DATASETS)
def test_msc_matches_the_reference(dataset: str) -> None:
    """The same estimator by a differently conditioned route.

    chemotools forms the normal equations for the two-column design
    `[reference, 1]` and inverts `A'A`; our kernel centres the reference and
    projects onto it. Same regression, and the normal-equation route squares
    the condition number — which is why this quantity has its own tolerance
    class and why the agreement is 3.5e-17 on gasoline and 2.9e-10 on tecator,
    whose absorbances are an order of magnitude larger and whose normal
    equations are correspondingly worse conditioned.
    """
    ours = MSCTransformer("mean").fit_transform(_block(dataset))
    assert parity.check(f"{dataset}.preprocess.msc.chemotools", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_asls_baseline_matches_the_reference(dataset: str) -> None:
    """Two independent implementations of Eilers and Boelens, and two solvers.

    chemotools factorises the penalised system as a banded Cholesky and runs a
    fixed iteration count with no convergence test; ours solves the sparse
    system and stops at an exact fixed point of the reweighting. The iteration
    caps are matched deliberately — and once our weights stop changing the
    remaining iterations change nothing, so the two agree whether ours stopped
    early or ran to the cap. That is asserted here rather than assumed.
    """
    kernel = BaselineCorrectTransformer("asls", lam=ASLS_LAM, p=ASLS_P, max_iter=ASLS_ITERATIONS)
    ours = kernel.fit_transform(_baseline_block(dataset))

    assert kernel.n_iterations_ is not None
    assert (kernel.n_iterations_ <= ASLS_ITERATIONS).all()
    assert parity.check(f"{dataset}.preprocess.baseline_asls.chemotools", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_rubberband_baseline_matches_the_reference(dataset: str) -> None:
    """Bit-identical, and it should be.

    The lower convex hull is decided by comparisons rather than by arithmetic,
    so two correct implementations cannot differ by rounding. Anything other
    than an exact match here is a different hull, not a different sum, and the
    tier is asserted for that reason.
    """
    ours = BaselineCorrectTransformer("rubberband").fit_transform(_baseline_block(dataset))
    result = parity.check(f"{dataset}.preprocess.baseline_rubberband.chemotools", ours)
    assert result.passed
    assert result.tier is parity.Tier.IDENTICAL
    assert result.max_abs_diff == 0.0


@pytest.mark.parametrize("dataset", DATASETS)
def test_polynomial_baseline_matches_the_reference(dataset: str) -> None:
    """The evidence that mapping the index onto [-1, 1] changes nothing.

    chemotools fits against the raw variable index; our kernel maps it onto
    [-1, 1] first, because a raw index over 700 variables to the fourth power
    spans 1e11 and the fit is then decided by the least-squares cutoff rather
    than by the data. `smoothing-and-baselines.md` claims that is exactly an
    affine change of variable. Two fits against differently scaled abscissae
    agreeing to the last bits is what that claim looks like when it is true.
    """
    ours = BaselineCorrectTransformer("polynomial", order=POLYNOMIAL_ORDER).fit_transform(
        _baseline_block(dataset)
    )
    assert parity.check(f"{dataset}.preprocess.baseline_polynomial.chemotools", ours).passed


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------


N_COMPONENTS = 5

# The confidence level the limit entries were generated at. chemotools takes a
# confidence where pca.md takes an alpha; 0.95 there is this.
LIMIT_ALPHA = 0.05

# The component count the RMSECV curve runs to. Both are repeated from the
# generator for the same reason as PREPROCESS_BLOCK above: a mismatch would
# compare a different model and pass.
MAX_PLS_COMPONENTS = 10

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


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_spe_limit_matches_the_reference(dataset: str, pca_models: dict[str, PCA]) -> None:
    """`pca.md` §8, and the claim #11 could not make when it landed.

    Every other PCA diagnostic in this file is *our* formula on scikit-learn's
    decomposition, because scikit-learn reports no `T^2` and no SPE. This one
    is different in kind: `chemotools` computes the Jackson-Mudholkar limit
    itself, so the comparison tests the formula and not only the decomposition
    it was fed. That is what the fourth verification step of #11 asked for and
    what nothing available could answer until #13.
    """
    result = parity.check(
        f"{dataset}.pca.spe_limit.chemotools", pca_models[dataset].spe_limit(LIMIT_ALPHA)
    )
    assert result.passed
    assert result.tier is parity.Tier.IDENTICAL


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_t2_limit_differs_from_the_reference_by_exactly_one_over_n(
    dataset: str, pca_models: dict[str, PCA], fixture_entries: dict[str, dict[str, object]]
) -> None:
    """A documented convention, and the factor is known exactly — so assert it.

    `chemotools` computes `a(n-1)/(n-a) F`; `pca.md` §7's new-sample form is
    `a(n²-1)/(n(n-a)) F`. The ratio is therefore exactly `(n+1)/n`, and our
    calibration limit — the beta form, which is the exact one for samples the
    model was fitted to — has no counterpart there at all.

    Recording this as a divergence rather than a failure is only honest if the
    difference really is the documented one, so the identity is checked to the
    last bits first. A drift in either formula would break the ratio and this
    case, where a bare `record_divergence()` would keep passing.
    """
    model = pca_models[dataset]
    n = model.n_samples_
    assert n is not None
    theirs = float(fixture_entries[f"{dataset}.pca.hotelling_t2_limit.chemotools"]["value"])  # type: ignore[arg-type]
    ours = model.hotelling_t2_limit(LIMIT_ALPHA, "new")

    assert ours / theirs == pytest.approx((n + 1) / n, rel=1e-12)
    assert model.hotelling_t2_limit(LIMIT_ALPHA, "calibration") < theirs

    result = parity.record_divergence(
        f"{dataset}.pca.hotelling_t2_limit.chemotools",
        reason=(
            "Both are F-distribution limits on the same scores and they differ by a "
            "convention, not by an error. chemotools computes a(n-1)/(n-a) F(a, n-a); "
            "pca.md §7 gives the new-sample form a(n^2-1)/(n(n-a)) F(a, n-a), which is "
            f"larger by exactly (n+1)/n - a factor of {(n + 1) / n:.4f} at n={n}. The "
            "two answer different questions: theirs is the limit for the calibration "
            "samples under an F approximation, ours is the exact limit for a sample "
            "the model has not seen, and pca.md §7 draws the beta form for calibration "
            "samples instead of approximating it. Which limit is drawn belongs in the "
            "plot legend, which is why we report both and name them."
        ),
    )
    assert result.tier is parity.Tier.DOCUMENTED_DIVERGENCE


# --------------------------------------------------------------------------
# the limits and SIMPLS coefficients, against R mdatools
#
# These were `unsourced` from #7 until #24 installed R. They matter more than
# their number suggests: chemotools is another Python implementation on the
# same NumPy, so agreeing with it says less than agreeing with a different
# language, a different author, and - for PLS - a different algorithm.
# --------------------------------------------------------------------------


#: Gasoline's residual eigenvalue spectrum decays slowly enough that
#: Jackson-Mudholkar's `h0` comes out negative. mdatools clamps it; we do not.
#: See the divergence recorded below, and #71 for our own handling.
_H0_CLAMPED_IN_MDATOOLS = "gasoline"


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_spe_limit_matches_r_mdatools(dataset: str, pca_models: dict[str, PCA]) -> None:
    """An implementation in another language, on the formula pca.md §8 names.

    For corn and tecator this is identical within float - two languages, two
    authors, the same number - which is the strongest statement the SPE limit
    has. Gasoline is a real difference and is recorded as one below rather than
    hidden inside a tolerance.
    """
    if dataset == _H0_CLAMPED_IN_MDATOOLS:
        pytest.skip("gasoline diverges by a documented convention; see the test below")

    result = parity.check(
        f"{dataset}.pca.spe_limit.r_mdatools", pca_models[dataset].spe_limit(LIMIT_ALPHA)
    )
    assert result.passed
    assert result.tier is parity.Tier.IDENTICAL


def test_the_gasoline_spe_limit_differs_because_mdatools_clamps_h0(
    pca_models: dict[str, PCA], fixture_entries: dict[str, dict[str, object]]
) -> None:
    """A divergence with a cause, proven before it is recorded.

    Jackson-Mudholkar assumes `h0 = 1 - 2θ₁θ₃/3θ₂²` is positive. Gasoline's
    residual spectrum decays slowly enough that it is not: `h0 = -0.0190`, and
    the limit is then a bracket raised to a large negative power. `mdatools`
    guards this by clamping `h0` to 0.001; our kernel uses it as computed.

    The test recomputes our own formula with their clamp and requires it to
    reproduce their number, so this is recorded as a convention only for as
    long as that is actually why the two differ. What our kernel *should* do
    about a negative `h0` is a separate question - #71.
    """
    dataset = _H0_CLAMPED_IN_MDATOOLS
    model = pca_models[dataset]
    theirs = float(fixture_entries[f"{dataset}.pca.spe_limit.r_mdatools"]["value"])  # type: ignore[arg-type]
    ours = model.spe_limit(LIMIT_ALPHA)

    eigenvalues = np.asarray(model.eigenvalues_, dtype=float)
    tail = eigenvalues[model.n_components :]
    theta = [float((tail**m).sum()) for m in (1, 2, 3)]
    h0 = 1.0 - (2.0 * theta[0] * theta[2]) / (3.0 * theta[1] ** 2)
    assert h0 < 0.0, "the divergence only exists because h0 is negative here"

    clamped = 0.001
    bracket = (
        float(norm.ppf(1.0 - LIMIT_ALPHA)) * math.sqrt(2.0 * theta[1] * clamped**2) / theta[0]
        + 1.0
        + theta[1] * clamped * (clamped - 1.0) / theta[0] ** 2
    )
    reconstructed = theta[0] * bracket ** (1.0 / clamped)
    assert reconstructed == pytest.approx(theirs, rel=1e-9), (
        "clamping h0 no longer reproduces mdatools, so the reason recorded below "
        "is not the reason any more"
    )
    assert ours != pytest.approx(theirs, rel=1e-3)

    result = parity.record_divergence(
        f"{dataset}.pca.spe_limit.r_mdatools",
        reason=(
            "Both compute Jackson-Mudholkar on the same residual eigenvalues, and "
            f"they differ because gasoline's h0 is negative ({h0:.4f}). mdatools "
            "clamps h0 to 0.001 before raising the bracket to the 1/h0 power; our "
            "kernel uses h0 as computed. Applying their clamp to our own formula "
            "reproduces their number to nine significant figures, so this is a "
            "guard against a degenerate spectrum rather than a different formula. "
            "Corn and tecator, whose h0 is positive, are identical within float."
        ),
    )
    assert result.tier is parity.Tier.DOCUMENTED_DIVERGENCE


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_t2_limit_differs_from_r_mdatools_by_exactly_one_over_n(
    dataset: str, pca_models: dict[str, PCA], fixture_entries: dict[str, dict[str, object]]
) -> None:
    """The same `(n+1)/n` convention as chemotools, from a different lineage.

    Two independent implementations differing from us by the identical factor
    is worth more than either alone: it says the convention is theirs and ours
    is deliberate, rather than one of them having a bug we happened to match.
    """
    model = pca_models[dataset]
    n = model.n_samples_
    assert n is not None
    theirs = float(fixture_entries[f"{dataset}.pca.hotelling_t2_limit.r_mdatools"]["value"])  # type: ignore[arg-type]
    ours = model.hotelling_t2_limit(LIMIT_ALPHA, "new")

    assert ours / theirs == pytest.approx((n + 1) / n, rel=1e-9)
    assert model.hotelling_t2_limit(LIMIT_ALPHA, "calibration") < theirs

    result = parity.record_divergence(
        f"{dataset}.pca.hotelling_t2_limit.r_mdatools",
        reason=(
            "mdatools computes the classic Hotelling limit a(n-1)/(n-a) F(a, n-a); "
            "pca.md §7's new-sample form is a(n^2-1)/(n(n-a)) F(a, n-a), larger by "
            f"exactly (n+1)/n - a factor of {(n + 1) / n:.4f} at n={n}. This is the "
            "same factor chemotools differs by, from an unrelated implementation in "
            "another language, which is what makes it a convention rather than "
            "either side's mistake."
        ),
    )
    assert result.tier is parity.Tier.DOCUMENTED_DIVERGENCE


# --------------------------------------------------------------------------
# PLS
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pls_models() -> dict[str, PLS]:
    """One fitted PLS per dataset, on the centred matrix and centred response.

    `pls-regression.md` §3: the kernel centres nothing, so both blocks are
    centred here through the preprocessing kernel — and the fixture was
    generated the same way, with `scale=False` passed to scikit-learn, which
    would otherwise have centred internally *and* scaled by default.
    """
    return {
        name: PLS(N_COMPONENTS).fit(_centred(name), _centred_response(name)) for name in DATASETS
    }


def _target(dataset: str) -> np.ndarray:
    values: np.ndarray = LOADERS[dataset]().targets[parity.load_fixture()["targets"][dataset]]
    return values


def _centred_response(dataset: str) -> np.ndarray:
    y = _target(dataset)
    return y - y.mean()


def _predictions(dataset: str, models: dict[str, PLS]) -> np.ndarray:
    """Calibration predictions in the response's original units.

    The kernel returns them on the scale it was fitted on; the calibration mean
    goes back on here, because `metrics-and-validation.md` §2 computes every
    metric in the original units of the reference method.
    """
    return models[dataset].predict(_centred(dataset)) + _target(dataset).mean()


@pytest.mark.parametrize("dataset", DATASETS)
def test_pls_coefficients_match_the_reference(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`pls-regression.md` §5: b = Rq, on the centred matrix so there is no intercept.

    Sign-invariant by construction — flipping a component negates `w`, `t`, `p`
    and `q` together and leaves `Rq` unchanged — so this is compared without
    alignment, and asserting that is part of the claim. It is also the case
    that would fail if y were not deflated (§4): the first component would
    still agree and the rest would not.
    """
    result = parity.check(f"{dataset}.pls.coefficients.sklearn", pls_models[dataset].coefficients_)
    assert result.passed
    assert not result.sign_aligned, "b = Rq is already sign-invariant (§5)"


@pytest.mark.parametrize("dataset", DATASETS)
def test_pls_coefficients_match_simpls_in_r(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`pls-regression.md` §2's claim, checked against a different algorithm.

    NIPALS and SIMPLS build different weights and different loadings, and the
    document claims they nonetheless coincide in coefficients and predictions
    for a single response. scikit-learn is NIPALS like ours, so it cannot test
    that claim. mdatools is SIMPLS, and it can: agreement here is the claim
    holding, not a tolerance being generous.

    Weights and loadings are deliberately not compared - the same document says
    they do not coincide, and comparing them would be asserting the opposite of
    what it states.
    """
    result = parity.check(
        f"{dataset}.pls.coefficients.r_mdatools", pls_models[dataset].coefficients_
    )
    assert result.passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_pls_predictions_match_the_reference(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`pls-regression.md` §5: y_hat = Xb, un-centred back to original units (§2)."""
    result = parity.check(f"{dataset}.pls.predictions.sklearn", _predictions(dataset, pls_models))
    assert result.passed
    assert not result.sign_aligned


@pytest.mark.parametrize("dataset", DATASETS)
def test_rmsec_follows_from_the_predictions(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`metrics-and-validation.md` §4: the divisor is n, not n - A - 1.

    A package dividing by n - A - 1 under this name is reporting our SEC, and
    on these datasets the difference is a few percent — comfortably outside
    the metric tolerance, so this case would catch it.
    """
    ours = rmse(_target(dataset), _predictions(dataset, pls_models))
    assert parity.check(f"{dataset}.pls.rmsec.sklearn", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_r2_is_the_residual_form(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`metrics-and-validation.md` §6: residual form, not squared correlation.

    The two coincide for a least-squares fit on the same data, which is what
    this case is, so passing here does not distinguish them. The distinction
    bites on a prediction set and is tested when one exists.
    """
    ours = r2(_target(dataset), _predictions(dataset, pls_models))
    assert parity.check(f"{dataset}.pls.r2.sklearn", ours).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_vip_matches_the_reference(dataset: str, pls_models: dict[str, PLS]) -> None:
    """`pls-regression.md` §8, Wold's form.

    scikit-learn reports no VIP, so the reference is computed in the generator
    from *its* weights, scores and y-loadings by the definition — our formula
    on an independent decomposition, which is exactly what the PCA T^2 and SPE
    claims are and must not be reported as more. The normalisation is checked
    against its own identity here as well, since a VIP that agrees with the
    reference and does not satisfy `sum VIP^2 = p` would mean both sides
    dropped the same factor.
    """
    vip = pls_models[dataset].vip()
    result = parity.check(f"{dataset}.pls.vip.sklearn", vip)
    assert result.passed
    assert not result.sign_aligned, "every weight is squared, so VIP carries no sign"
    assert float((vip**2).sum()) == pytest.approx(vip.size)


@pytest.mark.parametrize("dataset", DATASETS)
def test_rmsecv_curve_matches_the_reference(dataset: str) -> None:
    """The claim the whole cross-validation protocol rests on.

    **The fold indices are read out of the fixture**, never reseeded, because
    `metrics-and-validation.md` §8.2 makes ours `default_rng` (PCG64) and
    scikit-learn's a legacy `RandomState`: seeding both with 42 gives different
    folds, and a test written that way passes while comparing two different
    experiments. `test_our_splitter_reproduces_the_recorded_folds` below checks
    separately that our splitter would have produced these same indices.

    Centring is refitted inside each fold (§9), and the residuals are pooled
    across folds and rooted once (§7) rather than averaged as per-fold RMSEs.
    """
    entry = parity.entries_by_id()[f"{dataset}.pls.rmsecv_curve.sklearn"]
    folds = _folds_from_entry(entry)
    dataset_object = LOADERS[dataset]()

    curve = rmsecv_curve(dataset_object.spectra, _target(dataset), folds, MAX_PLS_COMPONENTS)
    assert parity.check(f"{dataset}.pls.rmsecv_curve.sklearn", curve).passed


@pytest.mark.parametrize("dataset", DATASETS)
def test_our_splitter_reproduces_the_recorded_folds(dataset: str) -> None:
    """The other half of the case above: the recorded indices are *ours*.

    Reading fold indices out of the fixture makes the RMSECV comparison valid,
    and it would stay valid even if our splitter drifted — so the splitter is
    checked against the same indices here. Together the two say that the curve
    agrees with scikit-learn *and* that the split it was computed on is the one
    `metrics-and-validation.md` §8.3 describes.
    """
    entry = parity.entries_by_id()[f"{dataset}.pls.rmsecv_curve.sklearn"]
    recorded = _folds_from_entry(entry)
    ours = k_fold(len(_target(dataset)), entry["split"]["n_folds"], seed=entry["split"]["seed"])

    for mine, theirs in zip(ours, recorded, strict=True):
        assert np.array_equal(mine.test, theirs.test)
        assert np.array_equal(mine.train, theirs.train)


# --------------------------------------------------------------------------
# the published R pls claims
# --------------------------------------------------------------------------


def test_leave_one_out_rmsecv_matches_the_r_pls_vignette() -> None:
    """The strongest claim in the fixture, and the only fully published one.

    Leave-one-out over the first 50 gasoline samples: deterministic, so there
    is no shuffle stream to reconcile (§8.4), and R `pls` computes MSEP as
    `SSE/nobj` — divisor `n`, the same as ours — so the curve compares with no
    definitional correction at all. The vignette prints four significant
    figures, which is why the harness gives transcribed values their own
    tolerance rather than the metric one.

    Key `0` is the intercept-only model: no components, so each held-out
    sample is predicted by its training fold's mean.
    """
    entry = parity.entries_by_id()["gasoline.pls.rmsecv_curve.r_pls_vignette"]
    calibration = entry["split"]["calibration_indices"]
    gasoline = load_gasoline()
    spectra = gasoline.spectra[calibration]
    y = gasoline.targets["octane"][calibration]

    folds = leave_one_out(len(calibration))
    intercept_only = np.array([float(y[fold.train].mean()) for fold in folds])
    curve = np.concatenate(
        [[rmse(y, intercept_only)], rmsecv_curve(spectra, y, folds, MAX_PLS_COMPONENTS)]
    )

    result = parity.check("gasoline.pls.rmsecv_curve.r_pls_vignette", curve)
    assert result.passed
    assert result.tier is parity.Tier.WITHIN_TOLERANCE, "a transcribed value is never identical"


def test_explained_variance_matches_the_r_pls_vignette() -> None:
    """The vignette's two-component `summary()`: 85.58% of X, 96.85% of octane.

    X's share is `||t_a||^2 ||p_a||^2` over the total sum of squares of the
    fitted matrix; the response's is `q_a^2 ||t_a||^2` over its own, whose
    running total is the model's R^2 because the X-scores are orthogonal (§4).
    Both denominators are the whole block — normalising over the retained
    components would put this at 100% and it would still look plausible.
    """
    gasoline = load_gasoline()
    calibration = parity.entries_by_id()["gasoline.pls.explained_variance.r_pls_vignette"]["split"][
        "calibration_indices"
    ]
    spectra = gasoline.spectra[calibration]
    y = gasoline.targets["octane"][calibration]

    model = PLS(2).fit(spectra - spectra.mean(axis=0), y - y.mean())
    ours = np.array(
        [
            model.cumulative_explained_variance("x")[-1],
            model.cumulative_explained_variance("y")[-1],
        ]
    )

    assert parity.check("gasoline.pls.explained_variance.r_pls_vignette", ours).passed


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
