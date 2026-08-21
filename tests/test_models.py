"""Invariants the core schema must enforce.

These started life as a self-check inside the schema module. They live here now
so they run on every commit rather than only when someone remembers to execute
the file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from chemometrics_workbench.models import (
    MSC,
    SNV,
    AxisKind,
    Dataset,
    DatasetVersion,
    Environment,
    EstimatorNode,
    Experiment,
    ExperimentStatus,
    KFoldSplit,
    Metrics,
    Model,
    Pipeline,
    PLSRegressionSpec,
    PreprocessNode,
    Project,
    RangeSelect,
    ResolvedSplit,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
    TaskKind,
    TrainTestSplit,
    VariableAxis,
)

HASH = "sha256:" + "a" * 64


@pytest.fixture
def project() -> Project:
    return Project(name="Corn NIR study", directory="/home/lab/corn")


@pytest.fixture
def version(project: Project) -> DatasetVersion:
    dataset = Dataset(project_id=project.project_id, name="corn_raw")
    return DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash=HASH,
        n_samples=3,
        n_variables=4,
        axis=VariableAxis(
            kind=AxisKind.WAVELENGTH_NM,
            values=[1100.0, 1102.0, 1104.0, 1106.0],
            unit="nm",
        ),
        sample_ids=["c1", "c2", "c3"],
        targets={"moisture": [10.1, 9.4, 11.2]},
        array_path="datasets/corn_raw/v1.npy",
    )


@pytest.fixture
def pipeline(project: Project, version: DatasetVersion) -> Pipeline:
    return Pipeline(
        project_id=project.project_id,
        name="SNV + SG + PLS",
        nodes=[
            SourceNode(id="src", version_id=version.version_id),
            PreprocessNode(id="snv", inputs=("src",), step=SNV()),
            PreprocessNode(
                id="sg",
                inputs=("snv",),
                step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
            ),
            SplitNode(id="cv", inputs=("sg",), spec=KFoldSplit(n_splits=10, seed=42)),
            EstimatorNode(
                id="pls",
                inputs=("cv",),
                spec=PLSRegressionSpec(n_components=6, target="moisture"),
            ),
        ],
    )


@pytest.fixture
def experiment(project: Project, version: DatasetVersion, pipeline: Pipeline) -> Experiment:
    return Experiment(
        project_id=project.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=HASH,
        status=ExperimentStatus.SUCCEEDED,
        resolved_splits=[ResolvedSplit(node_id="cv", train_indices=[[0, 1]], test_indices=[[2]])],
        metrics=Metrics(rmsecv=0.389, r2=0.981, bias=-0.004),
        environment=Environment(
            app_version="0.1.0",
            python_version="3.13.9",
            platform="linux",
            packages={"numpy": "2.1.0"},
        ),
    )


def _swap_snv_for_msc(pipeline: Pipeline) -> Pipeline:
    nodes = [
        n if n.id != "snv" else PreprocessNode(id="snv", inputs=("src",), step=MSC())
        for n in pipeline.nodes
    ]
    return pipeline.model_copy(update={"nodes": nodes})


def test_terminal_node_is_the_estimator(pipeline: Pipeline) -> None:
    assert [n.id for n in pipeline.terminal_nodes()] == ["pls"]


def test_content_hash_ignores_identity_and_timestamps(pipeline: Pipeline) -> None:
    """A re-created pipeline that computes the same thing must hash the same."""
    twin = pipeline.model_copy(update={"pipeline_id": uuid4()})
    assert twin.content_hash() == pipeline.content_hash()


def test_content_hash_tracks_every_parameter(pipeline: Pipeline) -> None:
    """One changed step changes the hash. This is what powers model comparison."""
    assert _swap_snv_for_msc(pipeline).content_hash() != pipeline.content_hash()


def test_pipeline_round_trips_through_json(pipeline: Pipeline) -> None:
    restored = Pipeline.model_validate_json(pipeline.model_dump_json())
    assert restored.content_hash() == pipeline.content_hash()
    node = restored.nodes[2]
    assert isinstance(node, PreprocessNode)
    assert isinstance(node.step, SavitzkyGolay)
    assert node.step.window_length == 11


def test_pipeline_is_json_serialisable(pipeline: Pipeline) -> None:
    json.loads(pipeline.model_dump_json())


def test_experiment_reports_the_hash_of_its_snapshot(
    experiment: Experiment, pipeline: Pipeline
) -> None:
    assert experiment.pipeline_hash == pipeline.content_hash()


def test_editing_a_pipeline_does_not_touch_an_existing_snapshot(
    experiment: Experiment, pipeline: Pipeline
) -> None:
    """The reason Experiment stores a snapshot rather than a reference."""
    edited = _swap_snv_for_msc(pipeline)
    assert experiment.pipeline_snapshot.content_hash() != edited.content_hash()
    assert experiment.pipeline_snapshot.content_hash() == pipeline.content_hash()


def test_model_carries_its_experiment_metrics(project: Project, experiment: Experiment) -> None:
    assert experiment.metrics is not None
    model = Model(
        project_id=project.project_id,
        experiment_id=experiment.experiment_id,
        name="Model C",
        task=TaskKind.REGRESSION,
        node_id="pls",
        artifact_path="models/model_c.json",
        artifact_hash=HASH,
        metrics=experiment.metrics,
    )
    assert model.metrics.rmsecv == 0.389


@pytest.mark.parametrize(
    ("case", "build"),
    [
        ("savgol even window", lambda: SavitzkyGolay(window_length=10, polyorder=2)),
        ("savgol polyorder >= window", lambda: SavitzkyGolay(window_length=5, polyorder=5)),
        (
            "savgol deriv > polyorder",
            lambda: SavitzkyGolay(window_length=11, polyorder=1, deriv=2),
        ),
        ("range reversed", lambda: RangeSelect(start=2500, end=1100)),
        ("test_size out of range", lambda: TrainTestSplit(test_size=1.5)),
        ("too few folds", lambda: KFoldSplit(n_splits=1)),
        (
            "non-monotonic axis",
            lambda: VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=[1100.0, 1104.0, 1102.0]),
        ),
    ],
)
def test_rejects_invalid_step_parameters(case: str, build: Callable[[], Any]) -> None:
    with pytest.raises(ValidationError):
        build()


def test_rejects_axis_that_disagrees_with_variable_count(version: DatasetVersion) -> None:
    with pytest.raises(ValidationError):
        DatasetVersion(
            dataset_id=version.dataset_id,
            version=1,
            content_hash=HASH,
            n_samples=3,
            n_variables=9,
            axis=version.axis,
            array_path="x.npy",
        )


def test_rejects_failed_experiment_without_an_error(
    project: Project, version: DatasetVersion, pipeline: Pipeline
) -> None:
    with pytest.raises(ValidationError):
        Experiment(
            project_id=project.project_id,
            pipeline_snapshot=pipeline,
            dataset_version_id=version.version_id,
            dataset_content_hash=HASH,
            status=ExperimentStatus.FAILED,
        )


@pytest.mark.parametrize(
    ("case", "nodes"),
    [
        (
            "cycle",
            lambda vid: [
                SourceNode(id="src", version_id=vid),
                PreprocessNode(id="a", inputs=("b",), step=SNV()),
                PreprocessNode(id="b", inputs=("a",), step=SNV()),
            ],
        ),
        ("dangling input", lambda vid: [PreprocessNode(id="a", inputs=("nope",), step=SNV())]),
        (
            "duplicate ids",
            lambda vid: [
                SourceNode(id="src", version_id=vid),
                PreprocessNode(id="src", inputs=("src",), step=SNV()),
            ],
        ),
    ],
)
def test_rejects_invalid_graphs(
    case: str, nodes: Callable[[Any], list[Any]], project: Project, version: DatasetVersion
) -> None:
    with pytest.raises(ValidationError):
        Pipeline(project_id=project.project_id, name=case, nodes=nodes(version.version_id))
