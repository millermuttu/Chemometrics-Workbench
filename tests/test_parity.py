"""The parity suite. Every claim the report in #14 renders is made here.

Run it alone with `uv run pytest -m parity`; it also runs as part of the full
suite, and writes `parity-results.json` either way.

**These cases are the harness proving itself, not the kernels.** No kernel
exists yet — that is #9 through #12 — so what is compared here is arithmetic
the specifications define directly: mean centring, projection onto loadings,
prediction from coefficients, and the eigenvalue definition. Each is a real
comparison against a real fixture entry with a real tolerance, and each will
be replaced by the kernel that is supposed to produce it. When a kernel lands,
the case is rewritten to call the kernel; the entry id, the tolerance and the
tier stay as they are.

Every case takes the same shape: compute a quantity, hand it to
`parity.check()` with the fixture entry it should match, and let the harness
decide the tolerance, the sign handling and the claim tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_corn, load_gasoline, load_tecator
from tests import parity

pytestmark = pytest.mark.parity

DATASETS = ("corn", "gasoline", "tecator")

LOADERS = {"corn": load_corn, "gasoline": load_gasoline, "tecator": load_tecator}


@pytest.fixture(scope="module")
def fixture_entries() -> dict[str, dict[str, object]]:
    return parity.entries_by_id()


def _centred(name: str) -> np.ndarray:
    """Mean centring, the trivial case the harness is first proved against.

    `pca.md` §2: centring is an explicit step, never something the algorithm
    does for itself. Every comparison below depends on this being right, which
    is what makes it a useful thing to fail first.
    """
    spectra = LOADERS[name]().spectra
    return spectra - spectra.mean(axis=0)


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
def test_scores_are_the_centred_matrix_projected_onto_the_loadings(
    dataset: str, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """`pca.md` §4: T = XP, on the centred matrix.

    Mean centring plus one matrix product. If the centring convention or the
    loading orientation is wrong, this is where it shows.
    """
    loadings = parity.as_array(fixture_entries[f"{dataset}.pca.loadings.sklearn"]["value"])
    scores = _centred(dataset) @ loadings

    result = parity.check(f"{dataset}.pca.scores.sklearn", scores)
    assert result.passed
    assert result.sign_aligned, "scores are sign-invariant and must be aligned"


@pytest.mark.parametrize("dataset", DATASETS)
def test_eigenvalues_are_the_variance_of_the_scores(
    dataset: str, fixture_entries: dict[str, dict[str, object]]
) -> None:
    """`pca.md` §4: lambda_k = sigma_k^2/(n-1), which is var(t_k) with ddof=1.

    Sign-invariant by construction — squaring a column removes its sign — so
    this is also the case that would catch a sign convention leaking into a
    quantity that should not have one.
    """
    scores = parity.as_array(fixture_entries[f"{dataset}.pca.scores.sklearn"]["value"])
    n = scores.shape[0]
    eigenvalues = (scores**2).sum(axis=0) / (n - 1)

    result = parity.check(f"{dataset}.pca.eigenvalues.sklearn", eigenvalues)
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
