"""Tests for the pipeline executor.

The claim that matters most is the first one: the pipeline the Phase 1.1
fixture publishes executes end to end against the real dataset and reproduces
the arrays that fixture serves. Everything else here is about the executor's
own behaviour — what it recomputes, what it reuses, and what it refuses.

The one array the executor does *not* reproduce is `centre_d`, and that is a
finding rather than a failure: see
`test_a_node_below_a_split_is_refitted_per_fold_which_the_fixture_is_not`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from chemometrics_workbench import preprocessing, validation
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import (
    CACHE_FILE,
    ExecutorError,
    execute,
    node_keys,
)
from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    LeaveOneOut,
    MeanCentre,
    PCASpec,
    Pipeline,
    PreprocessNode,
    RangeSelect,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
    TrainTestSplit,
)
from chemometrics_workbench.project import create_project, write_array

FIXTURES = Path(__file__).resolve().parents[1] / "stub" / "fixtures"

#: Two effects put a floor under agreement with the fixture, and neither is a
#: divergence. `generate_fixtures._round` writes six decimal places, worth up
#: to 5e-7. And the executor works from the array store, which is float32 on
#: disk by #77's boundary, while the fixture computed in float64 straight out
#: of `load_tecator()`: computing the same chain in float64 here agrees with
#: the fixture to 5.0e-07, and through the store the worst node moves to
#: 1.3e-06. A real divergence is orders of magnitude larger - the one below is
#: 1.2e-03.
ROUNDING = 2e-6

#: Except at `autoscale_c`, where float32 is amplified rather than carried: a
#: first derivative is a difference of neighbours, most of the signal cancels,
#: and autoscaling then divides by a small standard deviation. Measured at
#: 2.8e-05 through the store against 5.0e-07 in float64, so the amplification
#: is the store and not the kernel. Worth its own number rather than one loose
#: tolerance over every node, which would stop the others saying anything.
AMPLIFIED_BY_FLOAT32 = 5e-5


# --------------------------------------------------------------------------
# a project with the real dataset in it
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tecator() -> Any:
    return load_tecator()


@pytest.fixture
def project(tmp_path: Path, tecator: Any) -> tuple[Path, DatasetVersion]:
    """A project directory holding Tecator, and the version that describes it."""
    directory = tmp_path / "project"
    create_project(directory, "executor tests")
    array_path, _ = write_array(directory, tecator.spectra)
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=tecator.source.file_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        array_path=array_path,
    )
    return directory, version


def _pipeline(version_id: UUID, *nodes: Any) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=version_id), *nodes],
    )


def fixture_pipeline(version_id: UUID) -> Pipeline:
    """The four branches `stub/generate_fixtures.py` publishes, rebuilt here.

    Rebuilt rather than imported because `stub/` is deleted in #89 and this
    test outlives it. The recipe is asserted against the fixture's own
    `pipeline.json` below, so the copy cannot drift while the fixture lasts.
    """
    savgol = SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
    return _pipeline(
        version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(id="centre_a", inputs=("snv",), step=MeanCentre()),
        EstimatorNode(id="pca_a", inputs=("centre_a",), spec=PCASpec(n_components=5)),
        PreprocessNode(id="msc", inputs=("source",), step=MSC()),
        PreprocessNode(id="centre_b", inputs=("msc",), step=MeanCentre()),
        EstimatorNode(id="pca_b", inputs=("centre_b",), spec=PCASpec(n_components=5)),
        PreprocessNode(id="savgol", inputs=("source",), step=savgol),
        PreprocessNode(id="autoscale_c", inputs=("savgol",), step=Autoscale()),
        EstimatorNode(id="pca_c", inputs=("autoscale_c",), spec=PCASpec(n_components=5)),
        PreprocessNode(id="snv_savgol", inputs=("snv",), step=savgol),
        SplitNode(id="split_d", inputs=("snv_savgol",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="centre_d", inputs=("split_d",), step=MeanCentre()),
        EstimatorNode(id="pca_d", inputs=("centre_d",), spec=PCASpec(n_components=5)),
    )


def _as_stored(values: np.ndarray) -> np.ndarray:
    """What the array store gives back: float32 on disk, float64 to a kernel.

    A hand-computed comparison has to pass through the same boundary, or it is
    comparing the executor's stored numbers against float64 ones and calling
    #77's documented narrowing a divergence.
    """
    return values.astype(np.float32).astype(np.float64)


def _fixture_rows(node_id: str) -> dict[int, np.ndarray]:
    """The sample rows `spectra.json` draws for one node, by sample index."""
    payload = json.loads((FIXTURES / "spectra.json").read_text(encoding="utf-8"))[node_id]
    return {trace["index"]: np.asarray(trace["y"], dtype=float) for trace in payload["traces"]}


# --------------------------------------------------------------------------
# the claim: the fixture's pipeline runs and reproduces the fixture's arrays
# --------------------------------------------------------------------------


def test_the_recipe_here_is_the_one_the_fixture_publishes(
    project: tuple[Path, DatasetVersion],
) -> None:
    """`fixture_pipeline` is a copy, so it is checked against the original."""
    _, version = project
    published = json.loads((FIXTURES / "pipeline.json").read_text(encoding="utf-8"))
    mine = json.loads(fixture_pipeline(version.version_id).model_dump_json())

    assert [n["id"] for n in mine["nodes"]] == [n["id"] for n in published["nodes"]]
    for node, original in zip(mine["nodes"], published["nodes"], strict=True):
        # Everything but the source's version id, which points at this project.
        assert {k: v for k, v in node.items() if k != "version_id"} == {
            k: v for k, v in original.items() if k != "version_id"
        }


def test_the_fixture_pipeline_executes_and_reproduces_the_fixture_arrays(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)

    # Every preprocessing node ran; the estimators are reported, not fitted.
    assert set(run.displays) == {
        "source",
        "snv",
        "centre_a",
        "msc",
        "centre_b",
        "savgol",
        "autoscale_c",
        "snv_savgol",
        "split_d",
        "centre_d",
    }
    assert run.pending_estimators == ["pca_a", "pca_b", "pca_c", "pca_d"]

    for node_id in ("source", "snv", "centre_a", "msc", "centre_b", "savgol", "autoscale_c"):
        computed = run.displays[node_id]
        tolerance = AMPLIFIED_BY_FLOAT32 if node_id == "autoscale_c" else ROUNDING
        for index, expected in _fixture_rows(node_id).items():
            np.testing.assert_allclose(
                computed[index],
                expected,
                atol=tolerance,
                rtol=0,
                err_msg=f"{node_id} row {index} does not match the fixture",
            )

    # Branch D's shared nodes too - the split passes values through unchanged.
    for node_id in ("snv_savgol", "split_d"):
        for index, expected in _fixture_rows(node_id).items():
            np.testing.assert_allclose(
                run.displays[node_id][index], expected, atol=ROUNDING, rtol=0
            )


def test_a_node_below_a_split_is_refitted_per_fold_which_the_fixture_is_not(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The one array the executor does not reproduce, and why it should not.

    `metrics-and-validation.md` §9: every node downstream of the split is
    refitted on the training fold alone. The fixture's `run_preprocessing`
    says in its own docstring that it does no fold handling, so its `centre_d`
    is a mean over all 240 samples. Reproducing that would mean fitting a
    node below a split on the rows it is supposed to be validated against.

    Both numbers are computed here so the divergence stays a measured fact.
    """
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)

    snv = preprocessing.SNVTransformer().fit_transform(tecator.spectra)
    savgol = preprocessing.SavitzkyGolayTransformer(11, 2, deriv=1).fit_transform(snv)
    fitted_on_everything = preprocessing.MeanCentreTransformer().fit_transform(savgol)

    rows = _fixture_rows("centre_d")
    index, expected = next(iter(rows.items()))

    # What the fixture holds is the mean over all samples...
    np.testing.assert_allclose(fitted_on_everything[index], expected, atol=ROUNDING)
    # ...and the executor deliberately differs from it, by far more than rounding.
    assert np.abs(run.displays["centre_d"][index] - expected).max() > 1e-4

    # What the executor holds is fold arithmetic, done here by hand off its own
    # stored input, and reproduced exactly rather than approximately.
    stored_savgol = run.displays["snv_savgol"]
    folds = validation.k_fold(version.n_samples, 10, seed=42)
    held_out = next(fold for fold in folds if index in fold.test)
    by_hand = preprocessing.MeanCentreTransformer().fit(stored_savgol[held_out.train])
    np.testing.assert_array_equal(
        run.displays["centre_d"][index], _as_stored(by_hand.transform(stored_savgol))[index]
    )


def test_every_sample_is_displayed_from_the_fold_that_held_it_out(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The assembled array is out of fold for every row, not only the first."""
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)

    savgol = run.displays["snv_savgol"]
    folds = validation.k_fold(version.n_samples, 10, seed=42)
    expected = np.empty_like(savgol)
    for fold in folds:
        centred = preprocessing.MeanCentreTransformer().fit(savgol[fold.train])
        expected[fold.test] = _as_stored(centred.transform(savgol[fold.test]))

    np.testing.assert_array_equal(run.displays["centre_d"], expected)


def test_the_split_resolves_its_folds_and_records_them(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)

    assert [split.node_id for split in run.resolved_splits] == ["split_d"]
    resolved = run.resolved_splits[0]
    assert len(resolved.train_indices) == 10

    folds = validation.k_fold(version.n_samples, 10, seed=42)
    assert resolved.test_indices == [fold.test.tolist() for fold in folds]
    assert sorted(i for fold in resolved.test_indices for i in fold) == list(
        range(version.n_samples)
    )


# --------------------------------------------------------------------------
# the cache, and what invalidates it
# --------------------------------------------------------------------------


def test_a_second_run_recomputes_nothing(project: tuple[Path, DatasetVersion]) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)

    first = execute(directory, pipeline, version)
    assert first.reused == []

    second = execute(directory, pipeline, version)
    assert second.computed == []
    np.testing.assert_array_equal(first.displays["centre_d"], second.displays["centre_d"])


def test_editing_one_node_recomputes_it_and_its_descendants_and_nothing_else(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    execute(directory, fixture_pipeline(version.version_id), version)

    # Branch C's Savitzky-Golay window changes: that node, and what is below it.
    edited = fixture_pipeline(version.version_id)
    nodes = [
        node.model_copy(update={"step": SavitzkyGolay(window_length=9, polyorder=2, deriv=1)})
        if node.id == "savgol"
        else node
        for node in edited.nodes
    ]
    run = execute(directory, edited.model_copy(update={"nodes": nodes}), version)

    assert sorted(run.computed) == ["autoscale_c", "savgol"]
    assert "snv" in run.reused and "centre_a" in run.reused


def test_an_edit_below_a_split_does_not_disturb_the_branch_above_it(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    execute(directory, fixture_pipeline(version.version_id), version)

    edited = fixture_pipeline(version.version_id)
    nodes = [
        node.model_copy(update={"step": Autoscale()}) if node.id == "centre_d" else node
        for node in edited.nodes
    ]
    run = execute(directory, edited.model_copy(update={"nodes": nodes}), version)

    assert run.computed == ["centre_d"]
    assert "split_d" in run.reused and "snv_savgol" in run.reused


def test_the_cache_key_ignores_everything_that_is_not_the_recipe(
    project: tuple[Path, DatasetVersion],
) -> None:
    """Moving a node on the canvas must not invalidate a result.

    Layout coordinates live in `pipeline_state.json`, outside the model, so the
    strongest statement available here is the one that makes that safe: the key
    is a function of the node's own JSON and its inputs' keys, and of nothing
    else the pipeline carries. A pipeline with a different id, name and
    creation time hashes every node identically.
    """
    directory, version = project
    original = fixture_pipeline(version.version_id)
    renamed = original.model_copy(
        update={"pipeline_id": uuid4(), "name": "moved about on the canvas"}
    )

    assert node_keys(renamed, version) == node_keys(original, version)

    execute(directory, original, version)
    assert execute(directory, renamed, version).computed == []


def test_a_different_dataset_is_a_different_key(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The source is keyed on the version, so the same recipe over new data reruns."""
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    execute(directory, pipeline, version)

    shifted = tecator.spectra + 1.0
    array_path, _ = write_array(directory, shifted)
    other = version.model_copy(
        update={"version_id": uuid4(), "version": 2, "array_path": array_path}
    )

    run = execute(directory, fixture_pipeline(other.version_id), other)
    assert run.reused == []


def test_a_pruned_array_is_recomputed_rather_than_refused(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The index is a hint about what exists, not a promise."""
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    (directory / first.outputs["centre_a"].array_path).unlink()
    run = execute(directory, pipeline, version)

    assert "centre_a" in run.computed
    np.testing.assert_allclose(run.displays["centre_a"], first.displays["centre_a"])


def test_a_corrupt_index_costs_a_recomputation_and_nothing_more(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    (directory / CACHE_FILE).write_text("{ not json", encoding="utf-8")
    run = execute(directory, pipeline, version)

    assert run.reused == []
    np.testing.assert_allclose(run.displays["centre_a"], first.displays["centre_a"])


def test_use_cache_false_neither_reads_nor_writes_the_index(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version, use_cache=False)

    assert run.reused == []
    assert not (directory / CACHE_FILE).exists()


# --------------------------------------------------------------------------
# what the executor takes from the dataset rather than the recipe
# --------------------------------------------------------------------------


def test_range_select_reads_its_axis_off_the_dataset_version(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The bounds are in nanometres, and only the version knows what those are."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=900.0, end=1000.0)),
    )
    run = execute(directory, pipeline, version)

    axis = np.asarray(version.axis.values)
    kept = int(((axis >= 900.0) & (axis <= 1000.0)).sum())
    assert run.displays["window"].shape == (version.n_samples, kept)
    assert run.outputs["window"].n_variables == kept
    np.testing.assert_allclose(
        run.displays["window"], tecator.spectra[:, (axis >= 900.0) & (axis <= 1000.0)]
    )


def test_a_version_whose_array_is_not_the_shape_it_records_is_refused(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    directory, version = project
    array_path, _ = write_array(directory, tecator.spectra[:, :10])
    lying = version.model_copy(update={"array_path": array_path})

    with pytest.raises(ExecutorError, match="240x10 array where the version records 240x100"):
        execute(directory, _pipeline(lying.version_id), lying)


def test_a_missing_array_names_the_node_rather_than_raising_a_project_error(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    absent = version.model_copy(update={"array_path": "arrays/not-here.npy"})

    with pytest.raises(ExecutorError, match="node 'source' could not read the dataset"):
        execute(directory, _pipeline(absent.version_id), absent)


# --------------------------------------------------------------------------
# failure names the node
# --------------------------------------------------------------------------


def test_a_step_that_fails_names_the_node_it_failed_at(
    project: tuple[Path, DatasetVersion],
) -> None:
    """A one-variable window is legal; SNV over it is not.

    The kernel's own sentence is kept - it says what is wrong with the data -
    and the node id and step kind go in front of it, because the caller is
    looking at a canvas and needs to know where to click.
    """
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=850.1)),
        PreprocessNode(id="scatter", inputs=("window",), step=SNV()),
    )
    with pytest.raises(ExecutorError) as raised:
        execute(directory, pipeline, version)

    assert "node 'scatter' (snv) failed" in str(raised.value)
    assert "needs more than 1 variables" in str(raised.value)


def test_a_failure_below_a_split_says_which_training_fold_it_was_fitting(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=850.1)),
        SplitNode(id="split", inputs=("window",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="scatter", inputs=("split",), step=SNV()),
    )
    with pytest.raises(ExecutorError, match="on a training fold of 216 samples"):
        execute(directory, pipeline, version)


def test_a_range_select_below_another_one_is_refused_by_the_axis_it_was_given(
    project: tuple[Path, DatasetVersion],
) -> None:
    """A real limit of taking the axis from the dataset, surfaced rather than guessed.

    `RangeSelect` drops variables, so a second one downstream is being asked to
    read bounds against an axis that no longer describes its input. The kernel
    refuses on the shape, and the node it happened at is named. Threading a
    per-node axis through the walk would make the recipe's meaning depend on
    where a node sits, which is a schema question and not one to settle here.
    """
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=852.1)),
        PreprocessNode(id="narrower", inputs=("window",), step=RangeSelect(start=850.0, end=851.0)),
    )
    with pytest.raises(ExecutorError, match=r"node 'narrower' \(range_select\) failed"):
        execute(directory, pipeline, version)


def test_a_split_below_a_split_is_refused_by_name(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="outer", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        SplitNode(id="inner", inputs=("outer",), spec=KFoldSplit(n_splits=3, seed=42)),
    )
    with pytest.raises(ExecutorError, match="node 'inner' is a split below another split"):
        execute(directory, pipeline, version)


def test_a_split_with_no_splitter_yet_says_so(project: tuple[Path, DatasetVersion]) -> None:
    """Three of the five split specs have no kernel. That is said, not guessed at."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="holdout", inputs=("source",), spec=TrainTestSplit(test_size=0.3)),
    )
    with pytest.raises(ExecutorError, match="'train_test' split, which has no splitter yet"):
        execute(directory, pipeline, version)


def test_leave_one_out_is_the_other_splitter_that_does_work(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """240 folds, each fitting on 239 samples - slow enough to test small."""
    directory, version = project
    array_path, _ = write_array(directory, tecator.spectra[:8])
    small = version.model_copy(
        update={
            "version_id": uuid4(),
            "n_samples": 8,
            "sample_ids": list(tecator.sample_ids[:8]),
            "array_path": array_path,
        }
    )
    pipeline = _pipeline(
        small.version_id,
        SplitNode(id="loo", inputs=("source",), spec=LeaveOneOut()),
        PreprocessNode(id="centre", inputs=("loo",), step=MeanCentre()),
    )
    run = execute(directory, pipeline, small)

    assert run.outputs["centre"].n_folds == 8
    stored = _as_stored(tecator.spectra[:8])
    for i in range(8):
        others = np.delete(np.arange(8), i)
        expected = stored[i] - stored[others].mean(axis=0)
        np.testing.assert_array_equal(run.displays["centre"][i], _as_stored(expected))
