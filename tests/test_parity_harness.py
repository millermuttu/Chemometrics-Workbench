"""Tests of the harness itself, not of any scientific claim.

Kept apart from `test_parity.py` for one concrete reason: these cases
deliberately perturb values and provoke failures, and a run record that mixed
those with real parity claims would put fabricated numbers into the report.
The recorder is saved and restored around every case here, so nothing this
file does reaches `parity-results.json`.

What is being proved is the list in the issue: a sign flip must not fail a
comparison, a perturbation beyond tolerance must fail one, every result must
be tagged with a claim tier, and the run must write something the report
generator can read.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests import parity


@pytest.fixture(autouse=True)
def isolated_recorder() -> Iterator[None]:
    """Keep this file's deliberate failures out of the run record."""
    saved = list(parity.recorder.results)
    parity.recorder.clear()
    yield
    parity.recorder.results[:] = saved


@pytest.fixture(scope="module")
def loadings() -> np.ndarray:
    entry = parity.entries_by_id()["gasoline.pca.loadings.sklearn"]
    return parity.as_array(entry["value"])


# --------------------------------------------------------------------------
# sign invariance
# --------------------------------------------------------------------------


def test_a_flipped_component_still_passes(loadings: np.ndarray) -> None:
    """The issue's second verification step, done literally.

    Negate one whole component of the loadings and the comparison must still
    pass: the sign of a component is arbitrary (`pca.md` §5).
    """
    flipped = loadings.copy()
    flipped[:, 2] *= -1.0

    result = parity.check("gasoline.pca.loadings.sklearn", flipped)
    assert result.passed
    assert result.sign_aligned
    assert result.tier is parity.Tier.IDENTICAL


def test_every_component_flipped_still_passes(loadings: np.ndarray) -> None:
    result = parity.check("gasoline.pca.loadings.sklearn", -loadings)
    assert result.passed


def test_the_alignment_is_what_makes_it_pass(loadings: np.ndarray) -> None:
    """Turn alignment off and the same flipped input must fail.

    Without this, `test_a_flipped_component_still_passes` would also pass if
    the harness compared absolute values, or did nothing at all.
    """
    flipped = loadings.copy()
    flipped[:, 2] *= -1.0

    with pytest.raises(AssertionError, match="disagrees beyond"):
        parity.check("gasoline.pca.loadings.sklearn", flipped, sign_invariant=False)


def test_alignment_does_not_hide_an_internal_sign_disagreement() -> None:
    """Aligning is per component. Half a component flipped is a real error.

    This is why `align_signs` flips whole columns rather than comparing
    absolute values: `abs()` would pass a vector whose two halves disagree.
    """
    reference = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    corrupted = np.array([[-1.0, 2.0], [3.0, 4.0], [-5.0, 6.0]])

    aligned = parity.align_signs(corrupted, reference)
    assert not np.allclose(aligned, reference)


def test_align_signs_leaves_the_callers_array_alone() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    ours = np.array([-1.0, -2.0, -3.0])
    before = ours.copy()

    parity.align_signs(ours, reference)
    np.testing.assert_array_equal(ours, before)


def test_align_signs_handles_a_single_vector() -> None:
    reference = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(parity.align_signs(-reference, reference), reference)


# --------------------------------------------------------------------------
# tolerance
# --------------------------------------------------------------------------


def test_a_perturbed_value_fails(loadings: np.ndarray) -> None:
    """The issue's third verification step.

    The decomposition tolerance is rtol 1e-8; a relative perturbation of 1e-4
    is four orders of magnitude outside it.
    """
    perturbed = loadings.copy()
    perturbed[0, 0] += abs(perturbed[0, 0]) * 1e-4 + 1e-6

    with pytest.raises(AssertionError, match="disagrees beyond"):
        parity.check("gasoline.pca.loadings.sklearn", perturbed)


def test_a_failure_is_still_recorded(loadings: np.ndarray) -> None:
    """A failing claim must reach the report, not vanish with the test."""
    perturbed = loadings + 1.0
    with pytest.raises(AssertionError):
        parity.check("gasoline.pca.loadings.sklearn", perturbed)

    assert len(parity.recorder.results) == 1
    assert parity.recorder.results[0].passed is False


def test_the_failure_message_names_the_tolerance_and_refuses_to_suggest_widening(
    loadings: np.ndarray,
) -> None:
    with pytest.raises(AssertionError) as raised:
        parity.check("gasoline.pca.loadings.sklearn", loadings + 1.0)

    message = str(raised.value)
    assert "rtol=1e-08" in message
    assert "max absolute difference" in message
    assert "Widening the tolerance is not the fix" in message


def test_a_shape_mismatch_fails_before_any_comparison(loadings: np.ndarray) -> None:
    with pytest.raises(AssertionError, match="against reference"):
        parity.check("gasoline.pca.loadings.sklearn", loadings[:, :2])


def test_transcribed_values_get_the_looser_tolerance() -> None:
    """A value printed to four significant figures cannot be checked tighter."""
    entries = parity.entries_by_id()
    vignette = parity.tolerance_for(entries["gasoline.pls.rmsecv_curve.r_pls_vignette"])
    generated = parity.tolerance_for(entries["gasoline.pls.rmsecv_curve.sklearn"])

    assert vignette is parity.TOLERANCES["transcribed"]
    assert generated is parity.TOLERANCES["metrics"]
    assert vignette.rtol > generated.rtol


def test_a_quantity_with_no_agreed_tolerance_is_refused() -> None:
    """The harness never invents a tolerance nobody chose."""
    entry = dict(parity.entries_by_id()["gasoline.pca.loadings.sklearn"])
    entry["quantity"] = "something_nobody_has_classified"

    with pytest.raises(KeyError, match="no tolerance class"):
        parity.tolerance_for(entry)


def test_every_quantity_in_the_fixture_has_a_tolerance_class() -> None:
    """A fixture entry the harness could not compare is a gap, so fail early."""
    for entry in parity.load_fixture()["entries"]:
        assert entry["quantity"] in parity.QUANTITY_CLASS, entry["id"]


# --------------------------------------------------------------------------
# claim tiers
# --------------------------------------------------------------------------


def test_an_exact_match_is_tagged_identical(loadings: np.ndarray) -> None:
    assert parity.check("gasoline.pca.loadings.sklearn", loadings).tier is parity.Tier.IDENTICAL


def test_a_close_match_is_tagged_within_tolerance(loadings: np.ndarray) -> None:
    """Just outside float noise, comfortably inside the stated tolerance."""
    nudged = loadings * (1.0 + 1e-10)

    result = parity.check("gasoline.pca.loadings.sklearn", nudged)
    assert result.passed
    assert result.tier is parity.Tier.WITHIN_TOLERANCE


def test_a_divergence_is_tagged_and_needs_a_reason() -> None:
    result = parity.record_divergence("tecator.pls.sep.thodberg", reason="documented in §12")
    assert result.tier is parity.Tier.DOCUMENTED_DIVERGENCE
    assert result.passed
    assert result.reason == "documented in §12"

    with pytest.raises(ValueError, match="needs its reason recorded"):
        parity.record_divergence("tecator.pls.sep.thodberg", reason="   ")


def test_every_tier_is_reachable() -> None:
    assert {str(t) for t in parity.Tier} == {
        "identical_within_float",
        "agrees_within_tolerance",
        "differs_by_documented_convention",
    }


# --------------------------------------------------------------------------
# refusing to compare against a gap
# --------------------------------------------------------------------------


def test_an_unsourced_entry_cannot_be_checked() -> None:
    # The R mdatools entries used to stand here. #24 sourced them, so the
    # example is now the corn loading vector, which is the one gap left.
    with pytest.raises(ValueError, match="holds no value"):
        parity.check("corn.pca.loadings.literature", 1.0)


def test_an_unknown_entry_is_a_key_error() -> None:
    with pytest.raises(KeyError, match="no fixture entry"):
        parity.check("gasoline.pca.nonsense.sklearn", 1.0)


def test_comparable_entry_ids_excludes_gaps_and_context() -> None:
    ids = set(parity.comparable_entry_ids())
    assert "gasoline.pca.loadings.sklearn" in ids
    assert "tecator.pls.sep.thodberg" not in ids, "comparable=false"
    assert "corn.pca.loadings.literature" not in ids, "unsourced"


# --------------------------------------------------------------------------
# machine-readable output
# --------------------------------------------------------------------------


def test_the_run_record_is_written_and_readable(tmp_path: Path, loadings: np.ndarray) -> None:
    """The issue's fifth verification step: something #14 can consume."""
    parity.check("gasoline.pca.loadings.sklearn", loadings)
    parity.record_divergence("tecator.pls.sep.thodberg", reason="context only")

    path = tmp_path / "parity-results.json"
    parity.recorder.write(path)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert document["fixture_schema_version"] == 1
    assert document["totals"]["compared"] == 2
    assert document["totals"]["passed"] == 2
    assert document["totals"]["failed"] == 0
    assert document["totals"]["identical_within_float"] == 1
    assert document["totals"]["differs_by_documented_convention"] == 1

    first = document["results"][0]
    for key in (
        "entry_id",
        "dataset",
        "algorithm",
        "quantity",
        "tier",
        "passed",
        "software",
        "software_version",
        "citation",
        "rtol",
        "atol",
        "max_abs_diff",
        "sign_aligned",
        "reason",
    ):
        assert key in first, key


def test_the_run_record_names_what_was_never_compared(tmp_path: Path) -> None:
    """Coverage the report must show as a gap rather than silently omit."""
    parity.check(
        "gasoline.pca.loadings.sklearn",
        parity.as_array(parity.entries_by_id()["gasoline.pca.loadings.sklearn"]["value"]),
    )
    document = parity.recorder.write(tmp_path / "parity-results.json")

    not_compared = set(document["not_compared"])
    assert "gasoline.pca.loadings.sklearn" not in not_compared
    assert "corn.pls.coefficients.sklearn" in not_compared
    # Gaps and context are not coverage failures, so they are not listed here.
    assert "tecator.pls.sep.thodberg" not in not_compared
    assert "corn.pca.loadings.literature" not in not_compared
