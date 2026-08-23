"""The committed contract fixtures agree with the schema and with each other.

## Why this is not a byte comparison

The obvious guard — regenerate and `git diff --exit-code` — is the gate #38
had to remove from the parity report, and for a reason that applies here
unchanged: the numbers come out of BLAS, their last bits depend on the build
NumPy was linked against, and a value sitting on a rounding boundary crosses
it on one runner and not on another. Rounding to six places hides most of that
and cannot be relied on to hide all of it, so a byte gate would be flaky by
construction rather than by accident.

What is machine-independent is *structure*: that every payload parses, that
the domain objects still satisfy their own model, that the shapes agree with
the dataset and the recipe, and that the pipeline hash the fixture publishes
is the hash `Pipeline.content_hash()` computes today. Those catch what
actually goes wrong — a renamed field, a schema change nobody regenerated
for, a payload quietly dropped — and they do not fail on a different BLAS.

Regenerate with:

    uv run python stub/generate_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from chemometrics_workbench.models import (
    DatasetVersion,
    Experiment,
    Pipeline,
    Project,
)

FIXTURES = Path(__file__).resolve().parents[1] / "stub" / "fixtures"

EXPECTED_FILES = {
    "project",
    "datasets",
    "pipeline",
    "pipeline_state",
    "experiment",
    "import_preview",
    "jobs",
    "error",
    "spectra",
    "pca",
}


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def test_every_expected_fixture_is_committed() -> None:
    """A payload the stub server needs and cannot find is a 404 at runtime."""
    present = {path.stem for path in FIXTURES.glob("*.json")}
    assert present == EXPECTED_FILES


@pytest.mark.parametrize(
    ("name", "model"),
    [("project", Project), ("pipeline", Pipeline), ("experiment", Experiment)],
)
def test_the_domain_payloads_still_satisfy_their_model(name: str, model: type[BaseModel]) -> None:
    """These are Pydantic dumps, so a schema change must invalidate them.

    Without this, a field renamed in `models.py` leaves the fixtures serving
    the old name and the UI is built against a shape that no longer exists.
    """
    model.model_validate(load(name))


def test_the_dataset_version_still_satisfies_its_model() -> None:
    payload = load("datasets")
    assert len(payload) == 1, "one dataset in the fixture project"
    for version in payload[0]["versions"]:
        DatasetVersion.model_validate(version)


def test_the_published_pipeline_hash_is_the_one_the_model_computes() -> None:
    """Lineage is keyed on this hash. If the recipe moved, the fixture is stale."""
    pipeline = Pipeline.model_validate(load("pipeline"))
    experiment = Experiment.model_validate(load("experiment"))
    assert experiment.pipeline_hash == pipeline.content_hash()


def test_the_experiment_runs_against_the_dataset_the_fixture_ships() -> None:
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    experiment = Experiment.model_validate(load("experiment"))
    assert experiment.dataset_version_id == version.version_id
    assert experiment.dataset_content_hash == version.content_hash


def test_every_pipeline_node_has_a_state_and_a_position() -> None:
    """The canvas draws every node or it draws a hole."""
    pipeline = Pipeline.model_validate(load("pipeline"))
    state = load("pipeline_state")
    ids = {node.id for node in pipeline.nodes}
    assert set(state["nodes"]) == ids
    assert set(state["layout"]) == ids


def test_the_canvas_can_show_every_run_state() -> None:
    """#46 and #49 need all five simultaneously, so the fixture must carry them."""
    states = {entry["state"] for entry in load("pipeline_state")["nodes"].values()}
    assert {"complete", "running", "queued", "stale", "not_run"} <= states


def test_node_layout_is_not_inside_the_pipeline_hash() -> None:
    """`design/data-model.md` is explicit: moving a node must not change the science."""
    pipeline = Pipeline.model_validate(load("pipeline"))
    assert "layout" not in load("pipeline")
    moved = pipeline.model_copy()
    assert moved.content_hash() == pipeline.content_hash()


def test_spectra_payloads_match_the_dataset_and_the_recipe() -> None:
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    pipeline = Pipeline.model_validate(load("pipeline"))
    spectra = load("spectra")

    drawable = {n.id for n in pipeline.nodes if n.type != "estimator"}
    assert set(spectra) == drawable, "one spectra payload per non-estimator node"

    for node_id, payload in spectra.items():
        axis = payload["axis"]["values"]
        assert payload["n_spectra"] == version.n_samples, node_id
        for trace in payload["traces"]:
            assert len(trace["y"]) == len(axis), f"{node_id}: trace off the axis"
            assert trace["sample_id"] == version.sample_ids[trace["index"]]


def test_large_sets_are_banded_rather_than_drawn_trace_by_trace() -> None:
    """PROPOSAL.md §13: a cap on drawn traces, the remainder as a density band."""
    for node_id, payload in load("spectra").items():
        assert payload["decimation"]["banded"] is True, node_id
        assert payload["decimation"]["traces_drawn"] < payload["n_spectra"], node_id
        band = payload["band"]
        axis_length = len(payload["axis"]["values"])
        assert len(band["y_lower"]) == axis_length
        assert len(band["y_median"]) == axis_length
        assert len(band["y_upper"]) == axis_length
        envelope = zip(band["y_lower"], band["y_median"], band["y_upper"], strict=True)
        assert all(lo <= mid <= hi for lo, mid, hi in envelope), node_id


def test_pca_payloads_match_their_estimator_nodes() -> None:
    pipeline = Pipeline.model_validate(load("pipeline"))
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    pca = load("pca")

    estimators = {n.id: n for n in pipeline.nodes if n.type == "estimator"}
    assert set(pca) == set(estimators)

    for node_id, payload in pca.items():
        components = estimators[node_id].spec.n_components
        assert payload["n_components"] == components
        assert payload["n_variables"] == version.n_variables
        assert len(payload["loadings"]["components"]) == components
        assert len(payload["loadings"]["axis"]["values"]) == version.n_variables
        for loading in payload["loadings"]["components"]:
            assert len(loading) == version.n_variables
        assert len(payload["scores"]) == payload["n_samples"]
        assert len(payload["samples"]) == payload["n_samples"]
        for row in payload["scores"]:
            assert len(row) == components
        assert len(payload["explained_variance_ratio"]) == components
        assert len(payload["cumulative_explained_variance"]) == components


def test_the_split_branch_is_fitted_on_its_training_rows_and_the_others_are_not() -> None:
    """The recipe says branch D sits under a split, so its PCA must not see every row."""
    experiment = Experiment.model_validate(load("experiment"))
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    pca = load("pca")

    (split,) = experiment.resolved_splits
    assert split.node_id == "split_d"
    assert len(split.train_indices) == len(split.test_indices) == 10

    fold_zero = len(split.train_indices[0])
    assert pca["pca_d"]["n_samples"] == fold_zero < version.n_samples
    for node_id in ("pca_a", "pca_b", "pca_c"):
        assert pca[node_id]["n_samples"] == version.n_samples


def test_every_sample_appears_in_exactly_one_validation_fold() -> None:
    """`validation.validate_partition`'s rule, asserted on what was published."""
    experiment = Experiment.model_validate(load("experiment"))
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    seen = [index for fold in experiment.resolved_splits[0].test_indices for index in fold]
    assert sorted(seen) == list(range(version.n_samples))


def test_the_diagnostics_carry_their_limits() -> None:
    """A scores plot with no T2 limit cannot draw its ellipse."""
    for node_id, payload in load("pca").items():
        diagnostics = payload["diagnostics"]
        assert diagnostics["hotelling_t2_limit"] > 0, node_id
        assert diagnostics["spe_limit"] > 0, node_id
        assert len(diagnostics["hotelling_t2"]) == payload["n_samples"]
        assert len(diagnostics["spe"]) == payload["n_samples"]


def test_every_job_sequence_starts_queued_and_reaches_a_terminal_state() -> None:
    """#53 replays these; a sequence that never settles hangs the status bar."""
    sequences = load("jobs")
    assert set(sequences) == {"succeeded", "failed", "cancelled"}
    for outcome, stages in sequences.items():
        assert stages[0]["status"] == "queued", outcome
        assert stages[-1]["status"] == outcome
        progress = [stage["progress"] for stage in stages]
        assert progress == sorted(progress), f"{outcome}: progress went backwards"
        assert all(stage["message"] for stage in stages), outcome
        assert len({stage["job_id"] for stage in stages}) == 1, outcome


def test_only_a_successful_job_reaches_full_progress() -> None:
    """A run that failed or was cancelled at 65% must not report itself complete."""
    sequences = load("jobs")
    assert sequences["succeeded"][-1]["progress"] == 1.0
    assert sequences["failed"][-1]["progress"] < 1.0
    assert sequences["cancelled"][-1]["progress"] < 1.0


def test_the_failed_job_names_a_cause_and_carries_no_traceback() -> None:
    """PROPOSAL.md §8.2: a failed experiment is a result. It has to say why."""
    final = load("jobs")["failed"][-1]
    assert "Traceback" not in final["message"]
    assert len(final["message"]) > 20, "a failure message that says nothing is a spinner"


def test_the_error_body_names_a_cause_and_carries_no_traceback() -> None:
    """PROPOSAL.md §6: a specific diagnostic, never a stack trace."""
    error = load("error")["error"]
    assert error["code"] and error["message"]
    assert "Traceback" not in error["message"]
    assert set(error) == {"code", "message", "detail"}


def test_the_import_preview_offers_alternatives_for_every_guessed_detection() -> None:
    """#44: a transposed layout or a decimal comma guessed wrongly is the common case."""
    detected = load("import_preview")["detected"]
    for field in ("delimiter", "decimal", "orientation"):
        assert detected[field]["alternatives"], field
        assert detected[field]["value"] not in detected[field]["alternatives"]


def test_the_import_preview_agrees_with_the_dataset_it_produces() -> None:
    version = DatasetVersion.model_validate(load("datasets")[0]["versions"][0])
    detected = load("import_preview")["detected"]
    assert detected["n_samples"] == version.n_samples
    assert detected["n_variables"] == version.n_variables
    assert detected["axis"]["kind"] == version.axis.kind.value
    assert detected["axis"]["start"] == pytest.approx(version.axis.values[0])
    assert detected["axis"]["end"] == pytest.approx(version.axis.values[-1])
