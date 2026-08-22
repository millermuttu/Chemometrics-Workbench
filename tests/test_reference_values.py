"""Tests for the parity fixture, `tests/fixtures/reference_values.json`.

These test the *fixture*, not the kernels — there are no kernels yet. What
they gate is that every entry is traceable and that no gap has been quietly
filled in with a plausible number. The comparison itself is the parity
harness, issue #8.

The one exception is the fold assignment: the split recorded in the fixture is
reproduced here from `metrics-and-validation.md` §8.3 by hand, so that a
change to the permutation, the seeding or the fold-size rule fails here rather
than silently invalidating every RMSECV in the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_corn, load_gasoline, load_tecator

FIXTURE = Path(__file__).parent / "fixtures" / "reference_values.json"

DATASETS = ("corn", "gasoline", "tecator")
ALGORITHMS = ("pca", "pls", "preprocess")

# Keys every entry must carry, whatever its status. This list is the promise
# the issue makes: a value nobody can trace is worse than no value.
REQUIRED_KEYS = frozenset(
    {
        "id",
        "dataset",
        "dataset_content_hash",
        "algorithm",
        "quantity",
        "status",
        "comparable",
        "preprocessing",
        "algorithm_variant",
        "split",
        "software",
        "software_version",
        "citation",
        "notes",
        "value",
    }
)


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def entries(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = fixture["entries"]
    return result


# --------------------------------------------------------------------------
# shape of the file
# --------------------------------------------------------------------------


def test_fixture_is_versioned(fixture: dict[str, Any]) -> None:
    assert fixture["schema_version"] == 1
    assert fixture["generator"] == "tests/fixtures/generate_reference_values.py"
    assert fixture["generated_at"]
    assert fixture["conventions"]


def test_entry_ids_are_unique(entries: list[dict[str, Any]]) -> None:
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_every_entry_carries_the_required_keys(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        assert set(e) == REQUIRED_KEYS, f"{e.get('id')}: {set(e) ^ REQUIRED_KEYS}"


def test_every_entry_records_its_provenance(entries: list[dict[str, Any]]) -> None:
    """Preprocessing chain, algorithm variant, software, version and citation."""
    for e in entries:
        where = e["id"]
        assert e["status"] in {"sourced", "unsourced"}, where
        assert e["dataset"] in DATASETS, where
        assert e["algorithm"] in ALGORITHMS, where
        assert e["preprocessing"], f"{where}: no preprocessing chain"
        assert e["algorithm_variant"], f"{where}: no algorithm variant"
        assert e["software"], f"{where}: no software named"
        assert e["software_version"], f"{where}: no software version"
        assert e["citation"], f"{where}: no citation"


# --------------------------------------------------------------------------
# sourced values are real, unsourced gaps are honest
# --------------------------------------------------------------------------


def test_sourced_entries_have_a_value(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        if e["status"] != "sourced":
            continue
        assert e["value"] is not None, f"{e['id']} is sourced but holds no value"
        flat = np.asarray(
            list(e["value"].values()) if isinstance(e["value"], dict) else e["value"],
            dtype=np.float64,
        )
        assert np.isfinite(flat).all(), f"{e['id']} holds a non-finite value"


def test_unsourced_entries_are_empty_and_explained(entries: list[dict[str, Any]]) -> None:
    """A gap that is written down is a task; one that is filled in is a lie."""
    for e in entries:
        if e["status"] != "unsourced":
            continue
        assert e["value"] is None, f"{e['id']} is unsourced but holds a value"
        assert not e["comparable"], e["id"]
        assert "not sourced" in e["citation"], e["id"]
        assert len(e["notes"]) > 40, f"{e['id']} gives no reason for the gap"


def test_a_value_that_is_not_a_parity_target_says_so(entries: list[dict[str, Any]]) -> None:
    """`comparable` is the field that stops #8 asserting against context."""
    by_id = {e["id"]: e for e in entries}
    thodberg = by_id["tecator.pls.sep.thodberg"]
    assert thodberg["status"] == "sourced"
    assert thodberg["comparable"] is False
    assert "NOT a parity target" in thodberg["notes"]


# --------------------------------------------------------------------------
# coverage: the issue's "done when"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_has_a_comparable_value_for_every_dataset(
    entries: list[dict[str, Any]], dataset: str, algorithm: str
) -> None:
    found = [
        e
        for e in entries
        if e["dataset"] == dataset
        and e["algorithm"] == algorithm
        and e["status"] == "sourced"
        and e["comparable"]
    ]
    assert found, f"no comparable reference value for {algorithm} on {dataset}"


def test_content_hashes_match_the_loaders_in_use() -> None:
    """A fixture generated against a different dataset is not a reference."""
    hashes = {
        "corn": load_corn().source.file_hash,
        "gasoline": load_gasoline().source.file_hash,
        "tecator": load_tecator().source.file_hash,
    }
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    checked = 0
    for e in data["entries"]:
        recorded = e["dataset_content_hash"]
        if recorded is None:
            continue
        assert recorded == hashes[e["dataset"]], e["id"]
        checked += 1
    assert checked, "no entry pins a dataset content hash"


# --------------------------------------------------------------------------
# the split, reproduced by hand from the specification
# --------------------------------------------------------------------------


def test_worked_example_from_the_specification() -> None:
    """metrics-and-validation.md §8.3, n=10, K=3, shuffle, seed 42."""
    perm = np.random.default_rng(42).permutation(10)
    assert perm.tolist() == [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]


def test_recorded_folds_partition_the_samples(entries: list[dict[str, Any]]) -> None:
    """Every sample predicted exactly once, per §7's disjoint-union assertion."""
    sizes = {"corn": 80, "gasoline": 60, "tecator": 240}
    checked = 0
    for e in entries:
        split = e["split"]
        if not split or "folds" not in split:
            continue
        n = sizes[e["dataset"]]
        held_out = sorted(i for fold in split["folds"] for i in fold["test_indices"])
        assert held_out == list(range(n)), e["id"]

        for fold in split["folds"]:
            train = set(fold["train_indices"])
            test = set(fold["test_indices"])
            assert not train & test, f"{e['id']}: a sample is in both halves of a fold"
            assert train | test == set(range(n)), e["id"]
        checked += 1
    assert checked, "no entry records resolved fold indices"


def test_fold_sizes_follow_the_specified_rule(entries: list[dict[str, Any]]) -> None:
    """First `n % K` folds are one larger — scikit-learn's rule, kept (§8.3)."""
    for e in entries:
        split = e["split"]
        if not split or "folds" not in split:
            continue
        n_folds = split["n_folds"]
        n = sum(len(fold["test_indices"]) for fold in split["folds"])
        quotient, remainder = divmod(n, n_folds)
        expected = [quotient + 1] * remainder + [quotient] * (n_folds - remainder)
        assert [len(f["test_indices"]) for f in split["folds"]] == expected, e["id"]


def test_cross_validated_entries_record_a_split(entries: list[dict[str, Any]]) -> None:
    """A cross-validated number without its split is not reproducible."""
    for e in entries:
        if e["status"] != "sourced" or "cv" not in e["quantity"]:
            continue
        assert e["split"], f"{e['id']} is cross-validated but records no split"
        assert e["split"]["strategy"], e["id"]


def test_generated_splits_use_the_default_seed(entries: list[dict[str, Any]]) -> None:
    for e in entries:
        split = e["split"]
        if not split or "seed" not in split:
            continue
        assert split["seed"] == 42, e["id"]
        assert split["shuffle"] is True, e["id"]
        assert "default_rng" in split["generator"], e["id"]


# --------------------------------------------------------------------------
# the published reference, which is the one worth the most
# --------------------------------------------------------------------------


def test_r_pls_vignette_entry_is_fully_specified(entries: list[dict[str, Any]]) -> None:
    by_id = {e["id"]: e for e in entries}
    e = by_id["gasoline.pls.rmsecv_curve.r_pls_vignette"]

    assert e["software_version"] == "2.8-5"
    assert e["split"]["strategy"] == "leave_one_out"
    assert e["split"]["calibration_indices"] == list(range(50))
    assert "kernelpls" in e["algorithm_variant"]
    assert "Kalivas" in e["citation"]

    curve = e["value"]
    assert curve["0"] == 1.545, "intercept-only model"
    assert curve["2"] == 0.2966, "the vignette's chosen two-component model"
    assert len(curve) == 11, "intercept plus ten component counts"


def test_our_gasoline_curve_agrees_with_r_where_the_splits_converge(
    entries: list[dict[str, Any]],
) -> None:
    """Different splits, so they need not match — but by five components they do.

    This is the closest thing to a real parity check the fixture can carry
    before the kernels exist, and it is what says the generator's centring,
    scaling and metric definitions line up with an independent implementation.
    """
    by_id = {e["id"]: e for e in entries}
    ours = by_id["gasoline.pls.rmsecv_curve.sklearn"]["value"]
    theirs = by_id["gasoline.pls.rmsecv_curve.r_pls_vignette"]["value"]

    for n_components in ("5", "6"):
        assert ours[n_components] == pytest.approx(float(theirs[n_components]), abs=0.02), (
            f"{n_components} components: ours {ours[n_components]}, R {theirs[n_components]}"
        )
