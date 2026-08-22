"""The parity harness: how our numbers are compared against reference numbers.

Built before the kernels, deliberately, so that no kernel invents its own
comparison rules. A kernel test states what it computed and which fixture
entry it should match; everything about *how* the comparison is made — the
tolerance, the sign handling, the claim tier, the record kept for the report —
lives here and is decided once.

Read `tests/fixtures/reference_values.json` and `docs/algorithms/` first. This
module is the mechanics; those are the science.

## The three claim tiers

`PROPOSAL.md` §10.3 promises a report that says, quantity by quantity, how
strong the agreement is. That means a comparison cannot be a bare pass/fail —
it has to carry which of three things it is:

* **identical within floating point** — the same algorithm on the same data
  through a different code path. Anything worse than this is a real
  difference, not rounding.
* **agrees within stated tolerance** — the tolerance is chosen per quantity
  class in `TOLERANCES` below, with a reason, and is not adjusted to make a
  test pass.
* **differs by documented convention** — not compared numerically at all.
  The reason is recorded and shows up in the report. Every entry in the
  "known divergences" tables of the specification documents ends here.

The first two are decided by the numbers: a comparison whose worst element is
inside `identical_atol()` is tagged identical, otherwise it is tagged
within-tolerance. The third is declared by the caller, never inferred.

## Sign invariance

For scores, loadings and weights the sign of a whole component is arbitrary
(`pca.md` §5, `pls-regression.md` §6). Comparisons align by the inner product
with the reference and then compare signed values. **Comparing absolute
values instead would pass a result whose loading and score signs disagree with
each other, which is a real error.** Coefficients and predictions need no
alignment — `pls-regression.md` §5 shows `b = Rq` is already sign-invariant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FIXTURE = Path(__file__).parent / "fixtures" / "reference_values.json"

# Where the machine-readable run record is written. Regenerated on every
# parity run and consumed by the report generator in #14.
RESULTS = Path(__file__).parent.parent / "parity-results.json"


class Tier(StrEnum):
    """How strong a parity claim is. The report groups by this."""

    IDENTICAL = "identical_within_float"
    WITHIN_TOLERANCE = "agrees_within_tolerance"
    DOCUMENTED_DIVERGENCE = "differs_by_documented_convention"


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    reason: str = ""


# The strongest claim: the same computation reached by a different route,
# differing only in the last bits. It cannot be a fixed number, because "the
# last bits" depends on the magnitude of what is being compared — and it
# cannot be a relative tolerance either, because a score that happens to land
# near zero would then need to be exact, which no reordering of a sum can
# promise. So it is an absolute tolerance scaled to the data.
#
# Thirty-two units in the last place of the largest reference value. A matrix
# product over p terms accumulates roughly sqrt(p)*eps*scale, which is about
# 26 ulp at p = 700, the widest matrix in the fixture. The floor of 1.0 keeps
# quantities whose values are all tiny from being held to an absolute
# tolerance so small it is really a demand for bit-exactness.
IDENTICAL_ULPS = 32


def identical_atol(reference: NDArray[np.float64]) -> float:
    """Absolute tolerance for the identical-within-float claim, at this scale."""
    scale = float(np.max(np.abs(reference))) if reference.size else 1.0
    return IDENTICAL_ULPS * float(np.finfo(np.float64).eps) * max(scale, 1.0)


# Per quantity class, with the reason each was chosen. **These are not knobs.**
# A comparison that fails is a finding; widening the tolerance to make it pass
# converts a finding into a lie, and the parity report is the one artifact this
# project cannot afford to have lying in it.
TOLERANCES: dict[str, Tolerance] = {
    "decomposition": Tolerance(
        rtol=1e-8,
        atol=1e-10,
        reason=(
            "Scores, loadings, eigenvalues and explained variance. Two LAPACK "
            "drivers on the same matrix agree far better than this; the margin "
            "covers a different SVD path, not a different algorithm."
        ),
    ),
    "preprocessing": Tolerance(
        rtol=1e-12,
        atol=1e-14,
        reason=(
            "Scaling and scatter correction are a mean, a standard deviation and a "
            "division. There is no iteration and no decomposition to accumulate "
            "error, so anything beyond the last bits is a different formula rather "
            "than a different code path - a ddof convention, or a norm defined "
            "differently. Tight on purpose: this class exists to catch exactly that."
        ),
    ),
    "coefficients": Tolerance(
        rtol=1e-6,
        atol=1e-9,
        reason=(
            "Regression coefficients accumulate through per-component deflation "
            "(pls-regression.md §4), so error grows with the component count in a "
            "way a single decomposition's does not."
        ),
    ),
    "predictions": Tolerance(
        rtol=1e-6,
        atol=1e-9,
        reason="Predictions inherit the coefficient tolerance through one matrix product.",
    ),
    "metrics": Tolerance(
        rtol=1e-6,
        atol=1e-9,
        reason=(
            "RMSE, R2 and the RMSECV curve. A root of a mean of squares is well "
            "conditioned; this is the coefficient tolerance carried through."
        ),
    ),
    "transcribed": Tolerance(
        rtol=5e-3,
        atol=0.0,
        reason=(
            "Values copied out of a published document are only as precise as the "
            "document printed them. The R pls vignette prints four significant "
            "figures, so 0.2398 stands for anything in [0.23975, 0.23985) and a "
            "tolerance tighter than the printing is meaningless."
        ),
    ),
}

# Which class each fixture quantity belongs to. A quantity not listed here has
# no agreed tolerance, and the harness refuses to guess one.
QUANTITY_CLASS: dict[str, str] = {
    "scores": "decomposition",
    "loadings": "decomposition",
    "weights": "decomposition",
    "eigenvalues": "decomposition",
    "explained_variance_ratio": "decomposition",
    "cumulative_explained_variance_at_2_components": "decomposition",
    "spe_limit": "decomposition",
    "mean_centred": "preprocessing",
    "autoscaled": "preprocessing",
    "normalised_l1": "preprocessing",
    "normalised_l2": "preprocessing",
    "normalised_max": "preprocessing",
    "snv_corrected": "preprocessing",
    "msc_corrected": "preprocessing",
    "coefficients": "coefficients",
    "predictions": "predictions",
    "rmsec": "metrics",
    "rmsecv": "metrics",
    "rmsep": "metrics",
    "r2": "metrics",
    "sep": "metrics",
    "rmsecv_curve": "metrics",
}

# Quantities whose sign is arbitrary per component and must be aligned before
# comparing (pca.md §5, pls-regression.md §6).
SIGN_INVARIANT_QUANTITIES = frozenset({"scores", "loadings", "weights"})

# The exact prefix our fixture generator writes on everything it computes.
# Anything without it was transcribed out of a document by hand and is only as
# precise as the document printed it.
_GENERATED_PREFIX = "Generated by "


# --------------------------------------------------------------------------
# the fixture
# --------------------------------------------------------------------------


def load_fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def entries_by_id() -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in load_fixture()["entries"]}


def comparable_entry_ids() -> list[str]:
    """Entries a kernel is expected to be checked against.

    Excludes unsourced gaps, which hold no value, and entries flagged
    `comparable: false`, which are recorded for context rather than as parity
    targets — `tecator.pls.sep.thodberg` is the standing example.
    """
    return [
        e["id"] for e in load_fixture()["entries"] if e["status"] == "sourced" and e["comparable"]
    ]


def as_array(value: Any) -> NDArray[np.float64]:
    """Fixture values are floats, nested lists, or dicts keyed by a label."""
    if isinstance(value, dict):
        return np.asarray([value[k] for k in sorted(value, key=int)], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def tolerance_for(entry: dict[str, Any]) -> Tolerance:
    quantity = entry["quantity"]
    if quantity not in QUANTITY_CLASS:
        raise KeyError(
            f"{entry['id']}: quantity '{quantity}' has no tolerance class. Add one to "
            "TOLERANCES and QUANTITY_CLASS with a reason, rather than comparing it "
            "against a tolerance nobody chose."
        )
    if not entry["citation"].startswith(_GENERATED_PREFIX):
        return TOLERANCES["transcribed"]
    return TOLERANCES[QUANTITY_CLASS[quantity]]


# --------------------------------------------------------------------------
# sign alignment
# --------------------------------------------------------------------------


def align_signs(ours: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip whole components of `ours` to match the reference's orientation.

    Column `k` is negated when its inner product with the reference's column
    `k` is negative. Returns a new array; the caller's is never modified.

    A caller comparing scores *and* loadings must apply the same flip to both,
    which is why this returns the flip applied rather than absolute values —
    scores and loadings whose signs disagree with each other are wrong, and
    `abs()` would hide exactly that.
    """
    if ours.shape != reference.shape:
        raise ValueError(f"shape mismatch: ours {ours.shape}, reference {reference.shape}")
    if ours.ndim == 1:
        return -ours if float(np.dot(ours, reference)) < 0 else ours.copy()

    flips = np.where((ours * reference).sum(axis=0) < 0, -1.0, 1.0)
    return ours * flips


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class ParityResult:
    entry_id: str
    dataset: str
    algorithm: str
    quantity: str
    tier: Tier
    passed: bool
    software: str
    software_version: str
    citation: str
    rtol: float | None = None
    atol: float | None = None
    n_values: int | None = None
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    sign_aligned: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "quantity": self.quantity,
            "tier": str(self.tier),
            "passed": self.passed,
            "software": self.software,
            "software_version": self.software_version,
            "citation": self.citation,
            "rtol": self.rtol,
            "atol": self.atol,
            "n_values": self.n_values,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "sign_aligned": self.sign_aligned,
            "reason": self.reason,
        }


@dataclass
class Recorder:
    """Collects every comparison a run makes, for the report in #14."""

    results: list[ParityResult] = field(default_factory=list)

    def add(self, result: ParityResult) -> None:
        self.results.append(result)

    def clear(self) -> None:
        self.results.clear()

    def write(self, path: Path = RESULTS) -> dict[str, Any]:
        compared = {r.entry_id for r in self.results}
        document = {
            "schema_version": 1,
            "fixture_schema_version": load_fixture()["schema_version"],
            "totals": {
                "compared": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                **{str(tier): sum(1 for r in self.results if r.tier is tier) for tier in Tier},
            },
            # What the report must show as gaps rather than silently omit: a
            # fixture entry nothing was checked against is untested, and a
            # report that only lists what was tested overstates coverage.
            "not_compared": sorted(set(comparable_entry_ids()) - compared),
            "results": [r.as_dict() for r in self.results],
        }
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return document


recorder = Recorder()


# --------------------------------------------------------------------------
# the comparison itself
# --------------------------------------------------------------------------


def _result_for(entry: dict[str, Any], **kwargs: Any) -> ParityResult:
    return ParityResult(
        entry_id=entry["id"],
        dataset=entry["dataset"],
        algorithm=entry["algorithm"],
        quantity=entry["quantity"],
        software=entry["software"],
        software_version=entry["software_version"],
        citation=entry["citation"],
        **kwargs,
    )


def check(entry_id: str, ours: Any, *, sign_invariant: bool | None = None) -> ParityResult:
    """Compare `ours` against fixture entry `entry_id`, record it, and assert.

    `sign_invariant` defaults to whether the quantity is one whose component
    signs are arbitrary. Pass it explicitly only to override that, and say why
    in the test.

    Raises `AssertionError` when the values disagree beyond the tolerance for
    their class. The result is recorded either way, so a failing comparison
    still reaches the report rather than vanishing with the test.
    """
    entries = entries_by_id()
    if entry_id not in entries:
        raise KeyError(f"no fixture entry '{entry_id}'")
    entry = entries[entry_id]

    if entry["status"] != "sourced":
        raise ValueError(
            f"{entry_id} is {entry['status']} and holds no value. Nothing can be "
            "checked against it; fill the gap or leave it alone."
        )
    if not entry["comparable"]:
        raise ValueError(
            f"{entry_id} is recorded with comparable=false and is not a parity target. "
            f"Reason on the entry: {entry['notes'][:120]}"
        )

    reference = as_array(entry["value"])
    ours_array = np.asarray(ours, dtype=np.float64)
    if ours_array.shape != reference.shape:
        raise AssertionError(
            f"{entry_id}: shape {ours_array.shape} against reference {reference.shape}"
        )

    if sign_invariant is None:
        sign_invariant = entry["quantity"] in SIGN_INVARIANT_QUANTITIES
    if sign_invariant:
        ours_array = align_signs(ours_array, reference)

    difference = np.abs(ours_array - reference)
    max_abs = float(difference.max()) if difference.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(reference != 0, difference / np.abs(reference), 0.0)
    max_rel = float(np.max(relative)) if relative.size else 0.0

    tolerance = tolerance_for(entry)
    identical = max_abs <= identical_atol(reference)
    within = bool(np.allclose(ours_array, reference, rtol=tolerance.rtol, atol=tolerance.atol))

    result = _result_for(
        entry,
        tier=Tier.IDENTICAL if identical else Tier.WITHIN_TOLERANCE,
        passed=within,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        n_values=int(reference.size),
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        sign_aligned=sign_invariant,
        reason=tolerance.reason,
    )
    recorder.add(result)

    if not within:
        worst = int(np.argmax(difference))
        raise AssertionError(
            f"{entry_id} disagrees beyond rtol={tolerance.rtol:g} atol={tolerance.atol:g}.\n"
            f"  worst element {worst}: ours {ours_array.flat[worst]!r} "
            f"against reference {reference.flat[worst]!r}\n"
            f"  max absolute difference {max_abs:g}, max relative {max_rel:g}\n"
            f"  tolerance chosen because: {tolerance.reason}\n"
            "  Widening the tolerance is not the fix. Either the kernel is wrong or "
            "the difference is a convention, in which case record it with "
            "record_divergence() and say so in the specification."
        )
    return result


def record_divergence(entry_id: str, reason: str) -> ParityResult:
    """Record that a quantity differs from a reference by documented convention.

    No numbers are compared. Use this where the specification's "known
    divergences" table already says the two quantities are not the same thing —
    a different SPE limit formula, a different RMSEC denominator, a reference
    computed on inputs we do not produce. `reason` is rendered in the report,
    so write it for a reader who has not read the specification.
    """
    if not reason.strip():
        raise ValueError("a documented divergence needs its reason recorded")

    entries = entries_by_id()
    if entry_id not in entries:
        raise KeyError(f"no fixture entry '{entry_id}'")

    result = _result_for(
        entries[entry_id],
        tier=Tier.DOCUMENTED_DIVERGENCE,
        passed=True,
        reason=reason,
    )
    recorder.add(result)
    return result
