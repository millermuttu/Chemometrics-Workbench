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

import dataclasses
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from chemometrics_workbench import executor as executor_module
from chemometrics_workbench import preprocessing, validation
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.executor import (
    ExecutorError,
    execute,
    experiment_for,
    node_keys,
    permutation_test_for,
    result_path,
)
from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    KNNSpec,
    LDASpec,
    LeaveOneOut,
    MeanCentre,
    PCASpec,
    PCRSpec,
    Pipeline,
    PLSDASpec,
    PLSRegressionSpec,
    PreprocessNode,
    RangeSelect,
    RepeatedKFoldSplit,
    SavitzkyGolay,
    SIMCASpec,
    SourceNode,
    SplitNode,
    TrainTestSplit,
)
from chemometrics_workbench.preprocessing import SNVTransformer
from chemometrics_workbench.project import (
    create_project,
    read_array,
    read_cache_index,
    write_array,
    write_cache_index,
)
from chemometrics_workbench.regression import PLS
from chemometrics_workbench.validation import (
    bias,
    by_group,
    k_fold,
    leave_one_out,
    r2,
    rmse,
    sec,
    stratified_k_fold,
    stratified_train_test,
    train_test,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "contract"

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
        # The reference values, so a regression node has something to model.
        # A decomposition ignores them and every test above was written before
        # they were here.
        targets={name: [float(v) for v in values] for name, values in tecator.targets.items()},
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
    """The four branches the Phase 1.1 fixture generator publishes, rebuilt here.

    Rebuilt rather than imported because `stub/` was deleted in #89 and this
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
    # Every estimator in this pipeline is a PCA, so none of them is pending.
    assert run.pending_estimators == []
    assert sorted(run.results) == ["pca_a", "pca_b", "pca_c", "pca_d"]

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


def test_a_run_holds_no_arrays_and_reads_its_displays_from_the_store(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#176: `Run.displays` is a mapping over the store, not a dict of arrays,
    and every node's display array is indexed beside its fold arrays - the
    same file above a split, an assembled one below it."""
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    run = execute(directory, pipeline, version)

    assert not isinstance(run.displays, dict)
    assert not any(isinstance(value, np.ndarray) for value in vars(run).values())
    assert set(run.displays) == {n.id for n in pipeline.nodes if n.type != "estimator"}
    assert run.displays["centre_d"].shape == (version.n_samples, version.n_variables)

    index = read_cache_index(directory)
    keys = node_keys(pipeline, version)
    for node in pipeline.nodes:
        if node.type == "estimator":
            continue
        display = index[f"{keys[node.id]}#display"]
        assert len(display) == 1
        if node.id == "centre_d":
            # Below the split: the out-of-fold assembly is its own file.
            assert display[0] not in index[keys[node.id]]
        else:
            assert display == index[keys[node.id]][:1]


def test_stored_display_reads_the_stored_assembly_and_assembles_for_an_old_index(
    project: tuple[Path, DatasetVersion], monkeypatch: pytest.MonkeyPatch
) -> None:
    from chemometrics_workbench.executor import stored_display

    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    run = execute(directory, pipeline, version)
    expected = run.displays["centre_d"]

    fresh = stored_display(directory, pipeline, version, "centre_d")
    assert fresh is not None
    np.testing.assert_array_equal(fresh, expected)

    # A project whose index predates the display entries (#176) still serves
    # the same array, assembled from its fold arrays.
    monkeypatch.setattr(
        executor_module,
        "read_cache_index",
        lambda path: {k: v for k, v in read_cache_index(path).items() if "#display" not in k},
    )
    old = stored_display(directory, pipeline, version, "centre_d")
    assert old is not None
    np.testing.assert_array_equal(old, expected)


def test_a_second_run_recomputes_nothing(project: tuple[Path, DatasetVersion]) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)

    first = execute(directory, pipeline, version)
    assert first.reused == []

    second = execute(directory, pipeline, version)
    assert second.computed == []
    np.testing.assert_array_equal(first.displays["centre_d"], second.displays["centre_d"])


def test_a_cached_run_writes_no_array_and_reports_the_same_outputs(
    project: tuple[Path, DatasetVersion], monkeypatch: pytest.MonkeyPatch
) -> None:
    """#174: a cache hit used to re-serialise and re-hash every stored array."""
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    def refuse(*_: object) -> None:
        raise AssertionError("a cached node wrote an array")

    monkeypatch.setattr("chemometrics_workbench.executor.write_array", refuse)
    second = execute(directory, pipeline, version)

    assert second.outputs == {
        node_id: dataclasses.replace(output, from_cache=True)
        for node_id, output in first.outputs.items()
    }


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

    Layout coordinates live in their own table, outside the model, so the
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


def test_an_index_entry_naming_a_missing_array_costs_a_recomputation(
    project: tuple[Path, DatasetVersion],
) -> None:
    """What a corrupt `cache.json` used to cost, in the form the table has.

    The index cannot be malformed any more - it is rows, and a row that does
    not parse as a list is skipped - so the failure worth testing is the one
    that can still happen: an entry that points at an array nobody can read.
    """
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    index = read_cache_index(directory)
    write_cache_index(directory, {key: ["arrays/gone.npy"] for key in index})
    run = execute(directory, pipeline, version)

    assert run.reused == []
    np.testing.assert_allclose(run.displays["centre_a"], first.displays["centre_a"])


def test_use_cache_false_neither_reads_nor_writes_the_index(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version, use_cache=False)

    assert run.reused == []
    assert read_cache_index(directory) == {}


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


def test_a_range_select_below_another_one_reads_its_bounds_on_the_narrowed_axis(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#312. This used to be refused: the executor handed every step the
    dataset's axis, and a second selection's axis no longer described its
    input. A bound is a wavelength, and the wavelengths a first selection keeps
    are the same wavelengths, so the second reads its bounds on what the first
    left - the composition `api.node_axis` always drew, and
    `test_node_axis.py::test_two_selections_compose_in_order` holds.
    """
    directory, version = project
    axis = np.asarray(version.axis.values)
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=852.1)),
        PreprocessNode(id="narrower", inputs=("window",), step=RangeSelect(start=850.0, end=851.0)),
    )
    run = execute(directory, pipeline, version)
    kept = (axis >= 850.0) & (axis <= 851.0)
    assert run.outputs["narrower"].n_variables == int(kept.sum())


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
    """Two of the five split specs have no kernel. That is said, not guessed at."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(
            id="repeated",
            inputs=("source",),
            spec=RepeatedKFoldSplit(n_splits=3, n_repeats=2),
        ),
    )
    with pytest.raises(ExecutorError, match="'repeated_kfold' split, which has no splitter yet"):
        execute(directory, pipeline, version)


def test_a_train_test_split_holds_out_once_and_reports_p_metrics(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#183, `metrics-and-validation.md` §8.6: one fold, metrics with the P
    suffix and none with CV, and the node below it displays the one array its
    training rows' parameters produced."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="holdout", inputs=("source",), spec=TrainTestSplit(test_size=0.25, seed=42)),
        PreprocessNode(id="centre", inputs=("holdout",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)

    [resolved] = run.resolved_splits
    [fold] = validation.train_test(version.n_samples, 0.25, seed=42)
    assert resolved.test_indices == [fold.test.tolist()]
    assert len(fold.test) == 60

    result = run.results["pls"]
    assert result.fold == 0
    assert result.held_out == fold.test.tolist()
    assert {"rmsec", "r2", "rmsep", "sep"} <= set(result.metrics)
    for absent in ("rmsecv", "q2", "rmsecv_std", "rmsecv_a1"):
        assert absent not in result.metrics
    assert len(result.held_out_predicted) == 60

    # One array, centred by the training rows' mean: those rows average to
    # zero and the held-out ones, pushed through the same mean, do not.
    display = run.displays["centre"]
    assert display.shape == (version.n_samples, version.n_variables)
    np.testing.assert_allclose(display[fold.train].mean(axis=0), 0.0, atol=1e-4)
    assert abs(display[fold.test].mean()) > 1e-4


def _grades(version: DatasetVersion) -> DatasetVersion:
    """Tecator with an unbalanced three-level column: 40 'a', 80 'b', 120 'c'."""
    labels = ["a"] * 40 + ["b"] * 80 + ["c"] * 120
    return version.model_copy(update={"metadata_columns": {"grade": labels}})


@pytest.mark.parametrize(
    "spec",
    [
        TrainTestSplit(test_size=0.25, stratify_by="grade"),
        KFoldSplit(n_splits=4, stratify_by="grade"),
    ],
)
def test_a_stratified_split_resolves_to_the_kernel_s_folds(
    project: tuple[Path, DatasetVersion], spec: Any
) -> None:
    """§8.7: the run's folds are the stratified kernel's on the named column."""
    directory, version = project
    version = _grades(version)
    pipeline = _pipeline(version.version_id, SplitNode(id="s", inputs=("source",), spec=spec))
    run = execute(directory, pipeline, version)
    [resolved] = run.resolved_splits
    labels = version.metadata_columns["grade"]
    expected = (
        stratified_train_test(labels, 0.25)
        if isinstance(spec, TrainTestSplit)
        else stratified_k_fold(labels, 4)
    )
    assert resolved.test_indices == [fold.test.tolist() for fold in expected]
    for test in resolved.test_indices:
        held = [labels[i] for i in test]
        share = len(test) / len(labels)
        for level, count in (("a", 40), ("b", 80), ("c", 120)):
            assert abs(held.count(level) - share * count) <= 1


def test_stratifying_by_a_column_the_dataset_lacks_is_refused_by_name(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(
            id="holdout", inputs=("source",), spec=KFoldSplit(n_splits=5, stratify_by="batch")
        ),
    )
    with pytest.raises(ExecutorError, match="splits by 'batch', which this dataset does not"):
        execute(directory, pipeline, version)


def test_a_level_too_small_to_stratify_is_refused_naming_the_split_and_the_level(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    labels = ["rare"] + ["common"] * (version.n_samples - 1)
    version = version.model_copy(update={"metadata_columns": {"grade": labels}})
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="s", inputs=("source",), spec=KFoldSplit(n_splits=5, stratify_by="grade")),
    )
    with pytest.raises(
        ExecutorError, match=r"kfold stratified by 'grade'\) failed: the level 'rare'"
    ):
        execute(directory, pipeline, version)


def test_an_unstratified_kfold_keeps_the_cache_key_it_had_before_stratification() -> None:
    """#268: the field is left out of the dump when unset, so no stored run is orphaned."""
    assert KFoldSplit(n_splits=10).model_dump_json() == (
        '{"kind":"kfold","n_splits":10,"shuffle":true,"seed":42}'
    )


def _pairs(version: DatasetVersion) -> DatasetVersion:
    """Tecator as if every sample had been scanned twice: rows 2i and 2i+1 are one sample."""
    return version.model_copy(
        update={"metadata_columns": {"sample": [f"s{i // 2}" for i in range(version.n_samples)]}}
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (KFoldSplit(n_splits=5, group_by="sample"), lambda n: k_fold(n, 5)),
        (TrainTestSplit(test_size=0.25, group_by="sample"), lambda n: train_test(n, 0.25)),
        (LeaveOneOut(group_by="sample"), leave_one_out),
    ],
)
def test_a_grouped_split_resolves_to_the_kernel_s_folds_and_keeps_pairs_together(
    project: tuple[Path, DatasetVersion], spec: Any, expected: Any
) -> None:
    """§8.8 (#329): replicates never straddle training and held-out rows."""
    directory, version = project
    version = _pairs(version)
    pipeline = _pipeline(version.version_id, SplitNode(id="s", inputs=("source",), spec=spec))
    run = execute(directory, pipeline, version)
    [resolved] = run.resolved_splits
    groups = version.metadata_columns["sample"]
    assert resolved.test_indices == [f.test.tolist() for f in by_group(groups, expected)]
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        assert not {groups[i] for i in train} & {groups[i] for i in test}
        assert all(i ^ 1 in test for i in test)


def test_too_few_groups_is_refused_naming_the_split_and_the_column(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    version = version.model_copy(
        update={"metadata_columns": {"batch": ["x", "y"] * (version.n_samples // 2)}}
    )
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="s", inputs=("source",), spec=KFoldSplit(n_splits=5, group_by="batch")),
    )
    with pytest.raises(
        ExecutorError, match=r"kfold grouped by 'batch'\) failed: over the 2 groups: 5 folds"
    ):
        execute(directory, pipeline, version)


def test_an_ungrouped_split_keeps_the_cache_key_it_had_before_grouping() -> None:
    """#329: `group_by` is left out of the dump when unset, so no stored run is orphaned."""
    assert LeaveOneOut().model_dump_json() == '{"kind":"loo"}'
    assert TrainTestSplit(test_size=0.2).model_dump_json() == (
        '{"kind":"train_test","test_size":0.2,"seed":42,"stratify_by":null}'
    )


def test_a_split_cannot_be_both_grouped_and_stratified() -> None:
    with pytest.raises(ValueError, match="either grouped or stratified, not both"):
        KFoldSplit(n_splits=5, group_by="sample", stratify_by="grade")


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


# --------------------------------------------------------------------------
# estimators (#87)
# --------------------------------------------------------------------------


def _fixture_pca(node_id: str) -> dict[str, Any]:
    published: dict[str, Any] = json.loads((FIXTURES / "pca.json").read_text(encoding="utf-8"))
    return published[node_id]  # type: ignore[no-any-return]


def test_the_pca_branches_reproduce_the_fixtures_numbers(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The three branches with no split above them, to the fixture's rounding.

    `explained_variance_ratio` and the eigenvalues are written to eight
    decimals; the diagnostics are vectors over 240 samples and carry the
    float32 of the store through the whole chain, so they are compared
    relatively. The limits are not rounded at all in the fixture - a limit the
    interface shows should be the limit the kernel produced - and are asserted
    tightly.
    """
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)

    for node_id in ("pca_a", "pca_b", "pca_c"):
        result = run.results[node_id]
        expected = _fixture_pca(node_id)

        assert (result.n_samples, result.n_variables) == (
            expected["n_samples"],
            expected["n_variables"],
        )
        assert result.fold is None and result.held_out == []

        np.testing.assert_allclose(
            result.explained_variance_ratio,
            expected["explained_variance_ratio"],
            rtol=1e-4,
            err_msg=f"{node_id} explained variance",
        )
        np.testing.assert_allclose(
            result.hotelling_t2_limit, expected["diagnostics"]["hotelling_t2_limit"], rtol=1e-12
        )
        np.testing.assert_allclose(
            result.spe_limit, expected["diagnostics"]["spe_limit"], rtol=1e-3
        )
        np.testing.assert_allclose(
            result.hotelling_t2, expected["diagnostics"]["hotelling_t2"], rtol=1e-3
        )


def test_a_pca_below_a_split_is_fitted_on_every_sample_with_fold_zeros_held_out_view(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#330: the model is the all-sample one; fold zero's held-out rows stay as its view."""
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)
    result = run.results["pca_d"]

    folds = validation.k_fold(version.n_samples, 10, seed=42)
    assert result.all_samples and result.fold == 0
    assert result.rows == list(range(240))
    assert result.n_samples == len(result.scores) == 240
    assert result.held_out == folds[0].test.tolist()
    assert len(result.held_out_scores) == len(folds[0].test) == 24
    assert len(result.held_out_hotelling_t2) == len(result.held_out_spe) == 24

    # A fit on every row, through a centring fitted on every row.
    savgol = run.displays["snv_savgol"]
    centred = _as_stored(preprocessing.MeanCentreTransformer().fit_transform(savgol))
    reference = PCA(5, data_eps=PCA.STORED_EPS).fit(centred)
    np.testing.assert_allclose(
        result.explained_variance_ratio, reference.explained_variance_ratio(), rtol=1e-9
    )
    np.testing.assert_allclose(
        result.hotelling_t2_limit, reference.hotelling_t2_limit(executor_module.ALPHA), rtol=1e-12
    )


def test_pca_d_diverges_from_the_fixture_for_the_reason_centre_d_does(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The second casualty of #97, measured rather than asserted away.

    `pca_d` is fitted on `centre_d`, and `centre_d` is the array the fixture
    fits on all 240 samples where §9 requires the training fold alone. The
    model below inherits the difference. The fixture's own convention is
    reconstructed here and shown to match it, so what diverges is the input and
    not the kernel.
    """
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)
    expected = _fixture_pca("pca_d")
    folds = validation.k_fold(version.n_samples, 10, seed=42)

    savgol = run.displays["snv_savgol"]
    fitted_on_everything = preprocessing.MeanCentreTransformer().fit_transform(savgol)
    theirs = PCA(5).fit(fitted_on_everything[folds[0].train])

    # Their convention reproduces their number...
    np.testing.assert_allclose(
        theirs.explained_variance_ratio(), expected["explained_variance_ratio"], rtol=1e-4
    )
    # ...and ours does not, by far more than the store's float32.
    ours = np.asarray(run.results["pca_d"].explained_variance_ratio)
    assert np.abs(ours - np.asarray(expected["explained_variance_ratio"])).max() > 1e-6

    # The fixture's model is fold zero's; ours is fitted on every sample (#330),
    # so even the limits, which depend only on n and a, now differ.
    assert run.results["pca_d"].n_samples == 240 != len(folds[0].train)


def test_a_result_is_stored_keyed_the_way_its_node_is(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    run = execute(directory, pipeline, version)

    keys = node_keys(pipeline, version)
    stored = result_path(directory, keys["pca_a"])
    assert stored.exists()
    assert json.loads(stored.read_text(encoding="utf-8"))["node_id"] == "pca_a"
    assert run.results["pca_a"].key == keys["pca_a"]


def test_editing_a_node_above_an_estimator_gives_the_estimator_a_new_result(
    project: tuple[Path, DatasetVersion],
) -> None:
    """Staleness reaches the results too: a result is keyed like the node it is."""
    directory, version = project
    first = execute(directory, fixture_pipeline(version.version_id), version)

    edited = fixture_pipeline(version.version_id)
    nodes = [
        node.model_copy(update={"spec": PCASpec(n_components=3)}) if node.id == "pca_a" else node
        for node in edited.nodes
    ]
    run = execute(directory, edited.model_copy(update={"nodes": nodes}), version)

    assert run.results["pca_a"].key != first.results["pca_a"].key
    assert run.results["pca_a"].n_components == 3
    assert run.results["pca_b"].key == first.results["pca_b"].key
    # The arrays below the untouched branches were not recomputed either.
    assert "centre_a" in run.reused


def test_a_stored_result_is_read_back_rather_than_refitted(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    stored = result_path(directory, first.results["pca_a"].key)
    stored.write_text(
        json.dumps({**json.loads(stored.read_text(encoding="utf-8")), "rank": 4}), encoding="utf-8"
    )
    again = execute(directory, pipeline, version)

    assert again.results["pca_a"].rank == 4, "the stored result was refitted instead of read"


def test_an_unreadable_result_is_refitted_rather_than_refused(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    first = execute(directory, pipeline, version)

    result_path(directory, first.results["pca_a"].key).write_text("{ not json", encoding="utf-8")
    again = execute(directory, pipeline, version)

    assert again.results["pca_a"].rank == first.results["pca_a"].rank


def test_an_estimator_without_a_kernel_is_named_rather_than_skipped(
    project: tuple[Path, DatasetVersion], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spec has a kernel since #185, so PLS-DA's is taken away through the
    one tuple `has_kernel` reads: the pending path has to keep working for the
    next spec that lands before its kernel does."""
    monkeypatch.setattr(executor_module, "_FITTED", (PCASpec, PLSRegressionSpec))
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="plsda", inputs=("centre",), spec=PLSDASpec(n_components=4, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)

    assert run.pending_estimators == ["plsda"]
    assert "plsda" not in run.results


def test_a_pca_that_cannot_be_fitted_names_its_node(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=850.1)),
        EstimatorNode(id="pca", inputs=("window",), spec=PCASpec(n_components=5)),
    )
    with pytest.raises(ExecutorError, match=r"node 'pca' \(pca\) failed"):
        execute(directory, pipeline, version)


def test_the_reported_rank_is_the_fixtures_now_that_the_tolerance_knows_the_precision(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#101, fixed: the rank tolerance is quoted for the precision the data has.

    Mean centring removes a degree of freedom, so a centred 240x100 matrix has
    rank 99 and the fixture says so. Reading the centred array back out of the
    store rounds it to float32, its columns no longer sum to exactly zero, and
    the SVD finds a hundredth singular value - which a tolerance stated in
    float64 terms admits, because it describes an arithmetic the numbers did
    not go through.

    The executor now tells `PCA` what precision the data arrived in, and the
    two error terms are added rather than multiplied: `max(n, p)` scales the
    rounding the decomposition accumulates, while the data term is a
    perturbation of the matrix itself, which by Weyl moves each singular value
    by at most its norm - no dimension factor. Scaling the data term by
    `max(n, p)` too was tried and measured: it discards 33 genuine components
    and reports rank 66.
    """
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)
    expected = _fixture_pca("pca_a")

    assert expected["rank"] == 99
    assert run.results["pca_a"].rank == 99, "the store's precision is accounted for"

    # In float64, with no store in the way, the same chain gives the same - the
    # default tolerance is unmoved to within one part in max(n, p), so nothing
    # computed in float64 throughout shifted underneath the parity suite.
    snv = preprocessing.SNVTransformer().fit_transform(tecator.spectra)
    centred = preprocessing.MeanCentreTransformer().fit_transform(snv)
    assert PCA(5).fit(centred).rank_ == 99

    # The margin, measured rather than hoped for: the spurious singular value
    # and the smallest genuine one are more than two orders apart, so the
    # threshold is not balanced on a knife edge between them.
    stored = _as_stored(centred)
    values = np.linalg.svd(stored, full_matrices=False, compute_uv=False)
    assert values[99] < 1e-6 < 1e-5 < values[98], (values[98], values[99])
    assert values[98] / values[99] > 100

    # And the numbers that are not integers are unmoved by any of it.
    np.testing.assert_allclose(
        run.results["pca_a"].spe_limit, expected["diagnostics"]["spe_limit"], rtol=1e-6
    )


# --------------------------------------------------------------------------
# PLS, #142
# --------------------------------------------------------------------------


def test_a_pls_node_is_fitted_and_its_result_is_a_regression(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=6, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)

    assert run.pending_estimators == []
    result = run.results["pls"]
    assert result.task == "regression"
    assert result.target == "fat"
    assert result.n_components == 6
    assert len(result.observed) == len(result.predicted) == version.n_samples
    assert len(result.coefficients) == version.n_variables
    assert len(result.vip) == version.n_variables
    # One per component, because the T-squared ellipse is drawn from these and
    # `analysis.ts` indexes them directly. #142 published an empty list and the
    # ellipse came out with NaN radii; nothing caught it because no fixture or
    # demo had a PLS node to draw (#146).
    assert len(result.eigenvalues) == result.n_components
    assert all(value > 0 for value in result.eigenvalues)


def test_the_executor_computes_nothing_the_kernels_do_not(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The executor orchestrates. Every number it reports is reproduced here by
    calling `regression.py` and `validation.py` on the same array."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=5, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["pls"]

    matrix = read_array(directory, run.outputs["centre"].array_path)
    y = np.asarray(version.targets["fat"], dtype=np.float64)
    x_mean, y_mean = matrix.mean(axis=0), float(y.mean())
    model = PLS(5).fit(matrix - x_mean, y - y_mean)
    predicted = model.predict(matrix - x_mean) + y_mean

    assert result.predicted == pytest.approx([float(v) for v in predicted])
    assert model.coefficients_ is not None
    assert result.coefficients == pytest.approx([float(v) for v in model.coefficients_])
    assert result.vip == pytest.approx([float(v) for v in model.vip()])
    assert result.metrics["rmsec"] == pytest.approx(rmse(y, predicted))
    assert result.metrics["r2"] == pytest.approx(r2(y, predicted))
    assert result.metrics["bias"] == pytest.approx(bias(y, predicted))
    assert result.metrics["sec"] == pytest.approx(sec(y, predicted, n_components=5))


def test_below_a_split_the_curve_is_every_folds_and_the_model_is_every_samples(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#142's design call, revised by #330: the curve is every fold's, and the
    model is refitted on every sample through a centring fitted on every sample."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["pls"]

    assert result.all_samples and result.fold == 0
    folds = k_fold(version.n_samples, 5, seed=42)
    assert result.rows == list(range(version.n_samples))
    assert result.held_out == [int(row) for row in folds[0].test]

    # The all-sample model equals a fit on all rows.
    spectra = _as_stored(tecator.spectra)
    centred = _as_stored(spectra - spectra.mean(axis=0))
    y = np.asarray(tecator.targets["fat"], dtype=np.float64)
    x_mean, y_mean = centred.mean(axis=0), float(y.mean())
    reference = PLS(4).fit(centred - x_mean, y - y_mean)
    assert reference.coefficients_ is not None
    np.testing.assert_allclose(
        np.asarray(result.coefficients), reference.coefficients_, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        result.predicted, reference.predict(centred - x_mean) + y_mean, rtol=1e-9
    )

    # The held-out view is fold zero's model's, which never saw those rows.
    train = folds[0].train
    view_x = _as_stored(spectra - spectra[train].mean(axis=0))
    view_mean = view_x[train].mean(axis=0)
    view = PLS(4).fit(view_x[train] - view_mean, y[train] - y[train].mean())
    np.testing.assert_allclose(
        result.held_out_predicted,
        view.predict(view_x[folds[0].test] - view_mean) + y[train].mean(),
        rtol=1e-9,
    )
    assert result.metrics["rmsep"] == pytest.approx(
        rmse(y[folds[0].test], np.asarray(result.held_out_predicted))
    )

    # The curve is every fold's, one entry per component count, and the
    # reported RMSECV is the curve's last point rather than its minimum.
    curve = [result.metrics[f"rmsecv_a{a}"] for a in range(1, 5)]
    assert len(curve) == 4
    assert result.metrics["rmsecv"] == pytest.approx(curve[-1])
    assert all(f"rmsecv_fold_{k}" in result.metrics for k in range(5))
    assert "q2" in result.metrics and "rmsecv_std" in result.metrics


def test_the_experiment_carries_a_regressions_named_metrics(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#188: `Metrics` had fields for RMSEC, RMSECV, R2 and Q2 and every
    experiment left them `None`. A run whose last estimator is a regression
    fills them from that node's own result; the rest of its table is in
    `extra`; and a metric the node does not carry stays `None` (§11)."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    metrics = experiment_for(pipeline, version, run).metrics
    assert metrics is not None
    own = run.results["pls"].metrics

    assert metrics.rmsec == own["rmsec"]
    assert metrics.rmsecv == own["rmsecv"]
    assert metrics.r2 == own["r2"]
    assert metrics.q2 == own["q2"]
    assert metrics.bias == own["bias"]
    # Fold zero's held-out rows give RMSEP too; a regression above any split
    # would leave it None, which the decomposition test below covers for r2.
    assert metrics.rmsep == own["rmsep"]
    assert metrics.extra["sec"] == own["sec"]
    assert metrics.extra["rmsecv_a1"] == own["rmsecv_a1"]
    assert "spe_limit" in metrics.extra


def test_a_decompositions_experiment_leaves_the_regression_fields_absent(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = fixture_pipeline(version.version_id)
    run = execute(directory, pipeline, version)
    metrics = experiment_for(pipeline, version, run).metrics
    assert metrics is not None
    assert metrics.explained_variance and len(metrics.explained_variance) == 5
    assert (metrics.rmsec, metrics.rmsecv, metrics.r2, metrics.q2) == (None, None, None, None)


def _with_classes(
    version: DatasetVersion, tecator: Any, column: str = "fat_class"
) -> DatasetVersion:
    """Tecator with a two-valued metadata column derived from `fat` (pls-da.md §9)."""
    fat = np.asarray(tecator.targets["fat"])
    labels = ["high" if value > np.median(fat) else "low" for value in fat]
    return version.model_copy(update={"metadata_columns": {column: labels}})


def test_a_plsda_node_is_the_regression_on_a_dummy_response_and_tallies_it(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """`pls-da.md` §2 to §6: the model quantities are PLS1's on the {0, 1}
    response, and what is added is the coding, the assignments and the
    confusion matrix. Below a k-fold the `_cv` tally comes from the pooled
    held-out predictions and the `_p` one from fold zero's rows (§7)."""
    directory, version = project
    version = _with_classes(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="plsda",
            inputs=("centre",),
            spec=PLSDASpec(n_components=3, class_column="fat_class"),
        ),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["plsda"]

    assert result.task == "classification"
    assert result.classes == ["high", "low"]
    assert result.target == "fat_class"
    assert set(result.observed) == {0.0, 1.0}
    # The class coded 1 is the sorted second label, and the coding matches the column.
    labels = version.metadata_columns["fat_class"]
    assert result.observed == [1.0 if labels[row] == "low" else 0.0 for row in result.rows]
    assert result.predicted_class == [1 if value >= 0.5 else 0 for value in result.predicted]

    (tn, fp), (fn, tp) = result.confusion["calibration"]
    assert tn + fp + fn + tp == len(result.rows)
    assert result.metrics["accuracy"] == (tp + tn) / len(result.rows)
    assert result.metrics["sensitivity"] == tp / (tp + fn)
    assert result.metrics["specificity"] == tn / (tn + fp)
    assert result.metrics["accuracy"] > 0.8, "fat above its median is separable from NIR"

    # Below a k-fold: a cross-validated tally over every sample, and fold
    # zero's held-out tally, each with its suffix.
    cv = result.confusion["cross_validation"]
    assert sum(sum(row) for row in cv) == version.n_samples
    assert "accuracy_cv" in result.metrics and "accuracy_p" in result.metrics
    assert sum(sum(row) for row in result.confusion["held_out"]) == len(result.held_out)
    assert len(result.cross_validated_predicted) == version.n_samples

    # The regression half is filled in exactly as for a regression: the same
    # kernel on the dummy response, and the same curve.
    assert len(result.coefficients) == version.n_variables
    assert "rmsecv" in result.metrics and len(result.vip) == version.n_variables
    # And the sibling regression on `fat` itself is untouched by any of it.
    assert run.results["pls"].task == "regression"
    assert "accuracy" not in run.results["pls"].metrics


def test_a_class_column_the_dataset_does_not_carry_or_with_one_value_is_refused(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    directory, version = project
    two = _with_classes(version, tecator)
    missing = _pipeline(
        two.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="plsda", inputs=("centre",), spec=PLSDASpec(n_components=3, class_column="grade")
        ),
    )
    with pytest.raises(ExecutorError, match="'grade', which this dataset does not carry"):
        execute(directory, missing, two)

    one = two.model_copy(update={"metadata_columns": {"grade": ["a"] * version.n_samples}})
    pipeline = _pipeline(
        one.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="plsda", inputs=("centre",), spec=PLSDASpec(n_components=3, class_column="grade")
        ),
    )
    with pytest.raises(ExecutorError, match="has one value"):
        execute(directory, pipeline, one)


def _terciles(version: DatasetVersion, tecator: Any) -> DatasetVersion:
    """Tecator with three classes, its fat in thirds: lean, mid, rich."""
    fat = np.asarray(tecator.targets["fat"])
    low, high = np.quantile(fat, [1 / 3, 2 / 3])
    labels = ["lean" if f < low else "mid" if f < high else "rich" for f in fat]
    return version.model_copy(update={"metadata_columns": {"grade": labels}})


def test_three_classes_are_pls2_on_a_one_hot_response_assigned_by_argmax(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#274, pls-da.md sections 2, 3, 5 and 7. Every assignment is checked
    against scikit-learn's PLSRegression fitted to the same one-hot matrix
    to its fixed point, on the run's own folds, each preprocessed by its own
    training rows."""
    import warnings

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.exceptions import ConvergenceWarning

    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, stratify_by="grade")),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="plsda", inputs=("centre",), spec=PLSDASpec(n_components=6, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["plsda"]
    assert result.task == "classification"
    assert result.classes == ["lean", "mid", "rich"]
    assert np.asarray(result.coefficient_matrix).shape == (version.n_variables, 3)
    assert len(result.y_means) == 3 and result.coefficients == []
    assert [len(row) for row in result.confusion["cross_validation"]] == [3, 3, 3]
    assert sum(map(sum, result.confusion["cross_validation"])) == version.n_samples
    assert "accuracy_cv" in result.metrics and "sensitivity" not in result.metrics

    labels = version.metadata_columns["grade"]
    codes = np.asarray([result.classes.index(label) for label in labels])
    onehot = np.eye(3)[codes]
    spectra = _as_stored(tecator.spectra)
    [resolved] = run.resolved_splits
    assigned = np.empty(version.n_samples, dtype=int)
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        mean = spectra[train].mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            reference = PLSRegression(6, scale=False, tol=0.0, max_iter=2000).fit(
                spectra[train] - mean, onehot[train]
            )
        assigned[test] = reference.predict(spectra[test] - mean).argmax(axis=1)
    # #330: the calibration assignments are the all-sample model's.
    mean = spectra.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final = PLSRegression(6, scale=False, tol=0.0, max_iter=2000).fit(spectra - mean, onehot)
    assert result.predicted_class == final.predict(spectra - mean).argmax(axis=1).tolist()
    expected = [[int(np.sum((codes == j) & (assigned == k))) for k in range(3)] for j in range(3)]
    assert result.confusion["cross_validation"] == expected


def test_a_pls_node_above_a_split_reports_no_cross_validated_metric(
    project: tuple[Path, DatasetVersion],
) -> None:
    """§11: a metric that could not be computed is absent, never zero."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    result = execute(directory, pipeline, version).results["pls"]

    assert "rmsec" in result.metrics
    for absent in ("rmsecv", "q2", "rmsep", "sep", "rmsecv_std"):
        assert absent not in result.metrics


def test_a_target_the_dataset_does_not_carry_names_itself(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="octane")
        ),
    )
    with pytest.raises(ExecutorError, match="does not carry"):
        execute(directory, pipeline, version)


def test_sep_and_rmsep_satisfy_the_identity_the_specification_names(
    project: tuple[Path, DatasetVersion],
) -> None:
    """`metrics-and-validation.md` §5 offers this as a cheap unit test, so take it.

    `RMSEP^2 = bias^2 + (n_p - 1)/n_p * SEP^2` ties the two exactly on a
    prediction set. It holds only if both were computed on the same residuals
    with the denominators §5 specifies, so it catches a SEP wired to the wrong
    rows or divided by the wrong `n` — which no amount of "the number looks
    plausible" would.
    """
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    result = execute(directory, pipeline, version).results["pls"]

    observed = np.asarray(result.held_out_observed)
    predicted = np.asarray(result.held_out_predicted)
    n_p = observed.size
    held_out_bias = bias(observed, predicted)

    left = result.metrics["rmsep"] ** 2
    right = held_out_bias**2 + ((n_p - 1) / n_p) * result.metrics["sep"] ** 2
    assert left == pytest.approx(right)


def _held_out_predictions(arrays: list[np.ndarray], y: np.ndarray, folds: Any, a: int) -> Any:
    """Each fold's held-out rows predicted from the array given for that fold."""
    predicted = np.empty_like(y)
    for fold, values in zip(folds, arrays, strict=True):
        x_mean = values[fold.train].mean(axis=0)
        y_mean = y[fold.train].mean()
        model = PLS(a).fit(values[fold.train] - x_mean, y[fold.train] - y_mean)
        predicted[fold.test] = model.predict(values[fold.test] - x_mean) + y_mean
    return predicted


def test_each_fold_is_cross_validated_through_its_own_preprocessing(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#173: fold `i`'s held-out rows go through fold `i`'s fitted preprocessing.

    Autoscale rather than MeanCentre, because the kernel re-centres every fold
    and would hide the leak: a mean fitted on the wrong rows is removed again,
    a scale is not. The fold-zero computation is asserted to *differ*, so the
    test says which of the two the executor matched rather than that it moved.
    """
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="scale", inputs=("split",), step=Autoscale()),
        EstimatorNode(
            id="pls", inputs=("scale",), spec=PLSRegressionSpec(n_components=6, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    metrics = run.results["pls"].metrics

    arrays = [read_array(directory, path) for path in run.outputs["scale"].array_paths]
    y = np.asarray(version.targets["fat"], dtype=np.float64)
    folds = k_fold(version.n_samples, 5, seed=42)

    own = rmse(y, _held_out_predictions(arrays, y, folds, 6))
    leaked = rmse(y, _held_out_predictions([arrays[0]] * 5, y, folds, 6))

    assert own != pytest.approx(leaked, abs=1e-6)
    assert metrics["rmsecv"] == pytest.approx(own, rel=1e-9)


def test_a_stored_step_below_an_estimator_fails_the_run_by_name(
    project: tuple[Path, DatasetVersion],
) -> None:
    """#296: a pipeline saved before the write refused it still loads, and the
    run says what is wrong rather than raising a bare KeyError."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        EstimatorNode(id="pca", inputs=("source",), spec=PCASpec(n_components=2)),
        PreprocessNode(id="snv", inputs=("pca",), step=SNV()),
    )
    with pytest.raises(ExecutorError, match="node 'snv' takes its input from the estimator 'pca'"):
        execute(directory, pipeline, version)


def test_a_pcr_node_fits_and_cross_validates_like_a_regression(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#272, `pcr.md`: a regression result with PCR's numbers and no VIP. The
    RMSECV is checked against scikit-learn's PCA then least squares on the
    run's own resolved folds, each fold preprocessed by its own training rows."""
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.linear_model import LinearRegression

    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(id="pcr", inputs=("centre",), spec=PCRSpec(n_components=4, target="fat")),
    )
    run = execute(directory, pipeline, version)
    result = run.results["pcr"]

    assert result.task == "regression"
    assert result.vip == []
    assert len(result.coefficients) == version.n_variables
    assert [f"rmsecv_a{a}" in result.metrics for a in range(1, 5)] == [True] * 4

    spectra = _as_stored(tecator.spectra)
    fat = np.asarray(tecator.targets["fat"])
    [resolved] = run.resolved_splits
    held = np.empty_like(fat)
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        mean = spectra[train].mean(axis=0)
        pca = SkPCA(n_components=4, svd_solver="full").fit(spectra[train] - mean)
        model = LinearRegression().fit(pca.transform(spectra[train] - mean), fat[train])
        held[test] = model.predict(pca.transform(spectra[test] - mean))
    assert result.metrics["rmsecv"] == pytest.approx(rmse(fat, held), rel=1e-6)


def test_a_simca_node_decides_every_set_with_its_folds_own_models(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#275, simca.md sections 5 and 6: the cross-validated table pools every
    fold's held-out decisions, each made by models fitted on that fold alone."""
    from chemometrics_workbench.classification import SIMCA, acceptance_table

    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=4, stratify_by="grade")),
        PreprocessNode(id="snv", inputs=("split",), step=SNV()),
        EstimatorNode(
            id="simca", inputs=("snv",), spec=SIMCASpec(n_components=3, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["simca"]
    assert result.task == "classification" and result.method == "simca"
    assert result.classes == ["lean", "mid", "rich"]
    assert set(result.simca["sets"]) == {"calibration", "held_out", "cross_validation"}
    assert [model["class"] for model in result.simca["models"]] == result.classes

    labels = version.metadata_columns["grade"]
    codes = np.asarray([result.classes.index(label) for label in labels])
    # Through the store twice, as the run does: the source, and the SNV output.
    corrected = _as_stored(SNVTransformer().fit_transform(_as_stored(tecator.spectra)))
    [resolved] = run.resolved_splits
    distances = np.zeros((version.n_samples, 3))
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        model = SIMCA(3).fit(corrected[train], codes[train], 3)
        distances[test] = model.distances(corrected[test])
    cv = result.simca["sets"]["cross_validation"]
    np.testing.assert_allclose(np.asarray(cv["distances"]), distances, rtol=1e-9)
    table, none = acceptance_table(codes, distances <= 1.0)
    assert (cv["table"], cv["none"]) == (table, none)
    assert {"sensitivity_cv", "specificity_cv", "sensitivity_p"} <= set(result.metrics)


def test_a_simca_class_too_small_is_refused_by_name(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    labels = ["rare"] * 3 + ["common"] * (version.n_samples - 3)
    version = version.model_copy(update={"metadata_columns": {"grade": labels}})
    pipeline = _pipeline(
        version.version_id,
        EstimatorNode(
            id="simca", inputs=("source",), spec=SIMCASpec(n_components=4, class_column="grade")
        ),
    )
    with pytest.raises(ExecutorError, match="class 'rare' has 3 calibration samples"):
        execute(directory, pipeline, version)


def test_an_lda_node_classifies_by_its_folds_own_models(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#276, lda.md: every cross-validated assignment equals scikit-learn's
    PCA then LinearDiscriminantAnalysis on the run's own resolved folds."""
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, stratify_by="grade")),
        PreprocessNode(id="snv", inputs=("split",), step=SNV()),
        EstimatorNode(
            id="lda", inputs=("snv",), spec=LDASpec(n_components=6, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["lda"]
    assert (result.task, result.method) == ("classification", "lda")
    assert np.asarray(result.coefficient_matrix).shape == (version.n_variables, 3)
    assert len(result.scores[0]) == 6

    codes = np.asarray([result.classes.index(label) for label in version.metadata_columns["grade"]])
    corrected = _as_stored(SNVTransformer().fit_transform(_as_stored(tecator.spectra)))
    [resolved] = run.resolved_splits
    assigned = np.empty(version.n_samples, dtype=int)
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        mean = corrected[train].mean(axis=0)
        pca = SkPCA(6, svd_solver="full").fit(corrected[train] - mean)
        lda = LinearDiscriminantAnalysis().fit(pca.transform(corrected[train] - mean), codes[train])
        assigned[test] = lda.predict(pca.transform(corrected[test] - mean))
    expected = [[int(np.sum((codes == j) & (assigned == k))) for k in range(3)] for j in range(3)]
    assert result.confusion["cross_validation"] == expected


def test_a_knn_node_classifies_by_its_folds_own_neighbours(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """#277, knn.md: every cross-validated assignment equals scikit-learn's
    PCA then KNeighborsClassifier on the run's own resolved folds."""
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.neighbors import KNeighborsClassifier

    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, stratify_by="grade")),
        PreprocessNode(id="snv", inputs=("split",), step=SNV()),
        EstimatorNode(
            id="knn", inputs=("snv",), spec=KNNSpec(k=3, n_components=6, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)
    result = run.results["knn"]
    assert (result.task, result.method, result.k) == ("classification", "knn", 3)
    assert len(result.training_classes) == len(result.rows)

    codes = np.asarray([result.classes.index(label) for label in version.metadata_columns["grade"]])
    corrected = _as_stored(SNVTransformer().fit_transform(_as_stored(tecator.spectra)))
    [resolved] = run.resolved_splits
    assigned = np.empty(version.n_samples, dtype=int)
    for train, test in zip(resolved.train_indices, resolved.test_indices, strict=True):
        mean = corrected[train].mean(axis=0)
        pca = SkPCA(6, svd_solver="full").fit(corrected[train] - mean)
        knn = KNeighborsClassifier(3).fit(pca.transform(corrected[train] - mean), codes[train])
        assigned[test] = knn.predict(pca.transform(corrected[test] - mean))
    expected = [[int(np.sum((codes == j) & (assigned == k))) for k in range(3)] for j in range(3)]
    assert result.confusion["cross_validation"] == expected


def test_a_knn_asked_for_more_neighbours_than_samples_is_refused(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        EstimatorNode(
            id="knn", inputs=("source",), spec=KNNSpec(k=500, n_components=3, class_column="grade")
        ),
    )
    with pytest.raises(ExecutorError, match="k = 500 neighbours were asked of 240"):
        execute(directory, pipeline, version)


# --------------------------------------------------------------------------
# the permutation test (#333)
# --------------------------------------------------------------------------


def test_a_permutation_test_reruns_the_stored_cross_validation(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """metrics-and-validation.md section 14: the observed score is the run's own
    CV score, the null follows from the seed, and a real model beats it."""
    directory, version = project
    version = _terciles(version, tecator)
    pipeline = _pipeline(
        version.version_id,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=4, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
        EstimatorNode(
            id="knn", inputs=("centre",), spec=KNNSpec(k=3, n_components=4, class_column="grade")
        ),
    )
    run = execute(directory, pipeline, version)

    pls = permutation_test_for(directory, pipeline, version, "pls", 10, seed=1)
    assert pls.observed == pytest.approx(run.results["pls"].metrics["rmsecv"], rel=1e-12)
    assert not pls.greater_is_better and min(pls.null) > pls.observed
    assert pls.p_value == pytest.approx(1 / 11)
    assert pls == permutation_test_for(directory, pipeline, version, "pls", 10, seed=1)

    knn = permutation_test_for(directory, pipeline, version, "knn", 5, seed=1)
    assert knn.observed == pytest.approx(run.results["knn"].metrics["accuracy_cv"], rel=1e-12)
    assert knn.greater_is_better and max(knn.null) < knn.observed


def test_a_permutation_test_without_a_split_is_refused_by_name(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        EstimatorNode(
            id="pls", inputs=("source",), spec=PLSRegressionSpec(n_components=2, target="fat")
        ),
    )
    execute(directory, pipeline, version)
    with pytest.raises(ExecutorError, match="needs a K-fold or leave-one-out split"):
        permutation_test_for(directory, pipeline, version, "pls", 5)
