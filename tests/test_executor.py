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
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.executor import (
    ExecutorError,
    execute,
    node_keys,
    result_path,
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
    PLSDASpec,
    PLSRegressionSpec,
    PreprocessNode,
    RangeSelect,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
    TrainTestSplit,
)
from chemometrics_workbench.project import (
    create_project,
    read_array,
    read_cache_index,
    write_array,
    write_cache_index,
)
from chemometrics_workbench.regression import PLS
from chemometrics_workbench.validation import bias, k_fold, r2, rmse, sec

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


def test_a_pca_below_a_split_is_fitted_on_fold_zeros_training_rows(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    run = execute(directory, fixture_pipeline(version.version_id), version)
    result = run.results["pca_d"]

    folds = validation.k_fold(version.n_samples, 10, seed=42)
    assert result.fold == 0
    assert result.rows == folds[0].train.tolist()
    assert result.held_out == folds[0].test.tolist()
    assert result.n_samples == len(folds[0].train) == 216
    assert len(result.scores) == 216
    assert len(result.held_out_scores) == len(folds[0].test) == 24
    assert len(result.held_out_hotelling_t2) == len(result.held_out_spe) == 24


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

    # The limits depend only on n and a, so they agree whatever the input was.
    np.testing.assert_allclose(
        run.results["pca_d"].hotelling_t2_limit,
        expected["diagnostics"]["hotelling_t2_limit"],
        rtol=1e-12,
    )


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


def test_pls_da_has_no_kernel_here_and_is_named_rather_than_skipped(
    project: tuple[Path, DatasetVersion],
) -> None:
    """PLS itself is fitted since #142. PLS-DA still needs a second result shape."""
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


def test_below_a_split_the_curve_is_every_folds_and_the_model_is_fold_zeros(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The one design call in #142, asserted rather than left in a docstring."""
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

    # The model is fold zero's: its rows are that fold's training rows.
    assert result.fold == 0
    folds = k_fold(version.n_samples, 5, seed=42)
    assert result.rows == [int(row) for row in folds[0].train]
    assert result.held_out == [int(row) for row in folds[0].test]

    # The curve is every fold's, one entry per component count, and the
    # reported RMSECV is the curve's last point rather than its minimum.
    curve = [result.metrics[f"rmsecv_a{a}"] for a in range(1, 5)]
    assert len(curve) == 4
    assert result.metrics["rmsecv"] == pytest.approx(curve[-1])
    assert all(f"rmsecv_fold_{k}" in result.metrics for k in range(5))
    assert "q2" in result.metrics and "rmsecv_std" in result.metrics


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
