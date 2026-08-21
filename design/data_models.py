"""Core data models for the Chemometrics Workbench.

These are the schema for the reproducibility model described in PROPOSAL.md
section 8: a pipeline is a serialisable DAG of typed steps, datasets are
identified by content hash rather than filename, and an experiment captures
everything needed to reproduce its own result.

Design notes worth knowing before changing anything here:

* An Experiment stores a *snapshot* of the pipeline it ran, not a reference to
  it. Pipelines are edited constantly; without the snapshot, editing one would
  silently invalidate the provenance of every experiment that used it.
* A SplitSpec holds strategy and seed. The resolved index sets live on the
  Experiment, because they are a fact about one run rather than part of the
  recipe. Storing them means a split survives a change in library version.
* Node parameters use discriminated unions rather than free-form dicts, so an
  invalid pipeline fails at parse time instead of halfway through a ten-minute
  cross-validation.

Run this file directly to execute its self-check.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(BaseModel):
    """Mutable entity base."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Frozen(BaseModel):
    """Value objects that must not change once recorded."""

    model_config = ConfigDict(extra="forbid", frozen=True)


NodeId = str
ContentHash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class AxisKind(StrEnum):
    WAVELENGTH_NM = "wavelength_nm"
    WAVENUMBER_CM1 = "wavenumber_cm-1"
    RAMAN_SHIFT_CM1 = "raman_shift_cm-1"
    INDEX = "index"


class TaskKind(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    DECOMPOSITION = "decomposition"


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


class Environment(Frozen):
    """Captured at execution time so a result can be explained years later."""

    app_version: str
    python_version: str
    platform: str
    packages: dict[str, str] = Field(
        default_factory=dict,
        description="Version of every package that can move a number: numpy, scipy, "
        "scikit-learn, and any algorithm-providing dependency.",
    )
    recorded_at: datetime = Field(default_factory=_now)


class SourceFile(Frozen):
    """Where a dataset came from, and what read it."""

    filename: str
    file_hash: ContentHash
    reader: str = Field(description="Reader module, e.g. 'jcamp_dx' or 'bruker_opus'.")
    reader_version: str
    imported_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


class VariableAxis(Frozen):
    """The x-axis shared by every spectrum in a dataset."""

    kind: AxisKind
    values: list[float] = Field(min_length=1)
    unit: str | None = None

    @model_validator(mode="after")
    def _monotonic(self) -> Self:
        v = self.values
        if not (all(a < b for a, b in zip(v, v[1:])) or all(a > b for a, b in zip(v, v[1:]))):
            raise ValueError("variable axis must be strictly monotonic")
        return self


class DatasetVersion(Frozen):
    """An immutable snapshot of a 2-D dataset: n_samples x n_variables.

    Identified by the hash of its contents, never by filename. Renaming a file
    must not break lineage; silently editing one must not go unnoticed.
    """

    version_id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    version: int = Field(ge=1)
    content_hash: ContentHash
    n_samples: int = Field(gt=0)
    n_variables: int = Field(gt=0)
    axis: VariableAxis
    sample_ids: list[str] = Field(default_factory=list)
    targets: dict[str, list[float]] = Field(
        default_factory=dict, description="Reference values by property name, e.g. {'moisture': [...]}."
    )
    metadata_columns: dict[str, list[str]] = Field(default_factory=dict)
    excluded_samples: list[int] = Field(default_factory=list)
    excluded_variables: list[int] = Field(default_factory=list)
    source: SourceFile | None = None
    derived_from: UUID | None = Field(
        default=None, description="Parent version, when this one came from an edit rather than an import."
    )
    array_path: str = Field(description="Path within the project directory. The database never stores contents.")
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _shapes_agree(self) -> Self:
        if len(self.axis.values) != self.n_variables:
            raise ValueError(f"axis has {len(self.axis.values)} values but n_variables is {self.n_variables}")
        if self.sample_ids and len(self.sample_ids) != self.n_samples:
            raise ValueError(f"{len(self.sample_ids)} sample ids for {self.n_samples} samples")
        for name, values in self.targets.items():
            if len(values) != self.n_samples:
                raise ValueError(f"target {name!r} has {len(values)} values for {self.n_samples} samples")
        return self


class Dataset(Base):
    """Named container for an ordered series of versions."""

    dataset_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    name: str = Field(min_length=1)
    description: str = ""
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# pipeline: preprocessing steps
# --------------------------------------------------------------------------


class SNV(Frozen):
    kind: Literal["snv"] = "snv"


class MSC(Frozen):
    kind: Literal["msc"] = "msc"
    reference: Literal["mean", "median", "supplied"] = "mean"


class SavitzkyGolay(Frozen):
    kind: Literal["savgol"] = "savgol"
    window_length: int = Field(gt=2)
    polyorder: int = Field(ge=0)
    deriv: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def _valid_window(self) -> Self:
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length")
        if self.deriv > self.polyorder:
            raise ValueError("deriv cannot exceed polyorder")
        return self


class MeanCentre(Frozen):
    kind: Literal["mean_centre"] = "mean_centre"


class Autoscale(Frozen):
    kind: Literal["autoscale"] = "autoscale"
    ddof: int = Field(default=1, ge=0, le=1, description="Denominator convention for the standard deviation.")


class Normalise(Frozen):
    kind: Literal["normalise"] = "normalise"
    norm: Literal["l1", "l2", "max", "area"] = "l2"


class BaselineCorrect(Frozen):
    kind: Literal["baseline"] = "baseline"
    method: Literal["asls", "rubberband", "polynomial"] = "asls"
    order: int | None = Field(default=None, ge=0, description="Polynomial order, when method is polynomial.")
    lam: float | None = Field(default=None, gt=0, description="Smoothness, when method is asls.")
    p: float | None = Field(default=None, gt=0, lt=1, description="Asymmetry, when method is asls.")


class RangeSelect(Frozen):
    kind: Literal["range_select"] = "range_select"
    start: float
    end: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start >= self.end:
            raise ValueError("start must be less than end")
        return self


PreprocessStep = Annotated[
    SNV | MSC | SavitzkyGolay | MeanCentre | Autoscale | Normalise | BaselineCorrect | RangeSelect,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------
# pipeline: splits and estimators
# --------------------------------------------------------------------------


class TrainTestSplit(Frozen):
    kind: Literal["train_test"] = "train_test"
    test_size: float = Field(gt=0, lt=1)
    seed: int = 42
    stratify_by: str | None = None


class KFoldSplit(Frozen):
    kind: Literal["kfold"] = "kfold"
    n_splits: int = Field(ge=2)
    shuffle: bool = True
    seed: int = 42


class RepeatedKFoldSplit(Frozen):
    kind: Literal["repeated_kfold"] = "repeated_kfold"
    n_splits: int = Field(ge=2)
    n_repeats: int = Field(ge=2)
    seed: int = 42


class LeaveOneOut(Frozen):
    kind: Literal["loo"] = "loo"


class ExternalSet(Frozen):
    kind: Literal["external"] = "external"
    validation_version_id: UUID


SplitSpec = Annotated[
    TrainTestSplit | KFoldSplit | RepeatedKFoldSplit | LeaveOneOut | ExternalSet,
    Field(discriminator="kind"),
]


class PLSAlgorithm(StrEnum):
    NIPALS = "nipals"
    SIMPLS = "simpls"


class PCASpec(Frozen):
    kind: Literal["pca"] = "pca"
    n_components: int = Field(ge=1)


class PLSRegressionSpec(Frozen):
    kind: Literal["pls"] = "pls"
    n_components: int = Field(ge=1, description="Latent variables.")
    algorithm: PLSAlgorithm = PLSAlgorithm.NIPALS
    target: str = Field(description="Which target column in the dataset is being modelled.")


class PLSDASpec(Frozen):
    kind: Literal["plsda"] = "plsda"
    n_components: int = Field(ge=1)
    algorithm: PLSAlgorithm = PLSAlgorithm.NIPALS
    class_column: str


EstimatorSpec = Annotated[PCASpec | PLSRegressionSpec | PLSDASpec, Field(discriminator="kind")]


# --------------------------------------------------------------------------
# pipeline: nodes and graph
# --------------------------------------------------------------------------


class SourceNode(Frozen):
    type: Literal["source"] = "source"
    id: NodeId
    inputs: tuple[()] = ()
    version_id: UUID


class PreprocessNode(Frozen):
    type: Literal["preprocess"] = "preprocess"
    id: NodeId
    inputs: tuple[NodeId]
    step: PreprocessStep


class SplitNode(Frozen):
    type: Literal["split"] = "split"
    id: NodeId
    inputs: tuple[NodeId]
    spec: SplitSpec


class EstimatorNode(Frozen):
    type: Literal["estimator"] = "estimator"
    id: NodeId
    inputs: tuple[NodeId]
    spec: EstimatorSpec


PipelineNode = Annotated[
    SourceNode | PreprocessNode | SplitNode | EstimatorNode, Field(discriminator="type")
]


class Pipeline(Frozen):
    """A serialisable DAG. Executing one is the only path from data to result.

    Frozen because experiments embed a snapshot: editing means producing a new
    Pipeline, which keeps every prior experiment's provenance intact.
    """

    pipeline_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    name: str = Field(min_length=1)
    nodes: list[PipelineNode] = Field(min_length=1)
    created_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def _valid_dag(self) -> Self:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        known = set(ids)
        for node in self.nodes:
            for parent in node.inputs:
                if parent not in known:
                    raise ValueError(f"node {node.id!r} references unknown input {parent!r}")
        if not any(n.type == "source" for n in self.nodes):
            raise ValueError("pipeline needs at least one source node")

        by_id = {n.id: n for n in self.nodes}
        state: dict[NodeId, int] = {}  # 0 = visiting, 1 = done

        def visit(nid: NodeId) -> None:
            mark = state.get(nid)
            if mark == 1:
                return
            if mark == 0:
                raise ValueError(f"pipeline contains a cycle through node {nid!r}")
            state[nid] = 0
            for parent in by_id[nid].inputs:
                visit(parent)
            state[nid] = 1

        for nid in ids:
            visit(nid)
        return self

    def content_hash(self) -> str:
        """Stable hash of the recipe, ignoring identity and timestamps.

        Two pipelines that would compute the same thing hash the same, which is
        what makes "these two models differ only in preprocessing" answerable.
        """
        payload = self.model_dump_json(include={"nodes"})
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    def terminal_nodes(self) -> list[PipelineNode]:
        consumed = {p for n in self.nodes for p in n.inputs}
        return [n for n in self.nodes if n.id not in consumed]


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


class ResolvedSplit(Frozen):
    """The index sets a split actually produced, stored so the run can be repeated."""

    node_id: NodeId
    train_indices: list[list[int]] = Field(description="One list per fold; a single entry for a plain split.")
    test_indices: list[list[int]]

    @model_validator(mode="after")
    def _folds_match(self) -> Self:
        if len(self.train_indices) != len(self.test_indices):
            raise ValueError("train and test index sets must have the same number of folds")
        return self


class Metrics(Frozen):
    """Named explicitly, because metric definitions vary between packages.

    The definition used for each is documented per algorithm; see PROPOSAL.md
    section 10. `extra` carries anything algorithm-specific.
    """

    rmsec: float | None = None
    rmsecv: float | None = None
    rmsep: float | None = None
    r2: float | None = None
    q2: float | None = None
    bias: float | None = None
    explained_variance: list[float] | None = None
    accuracy: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class Experiment(Base):
    """One execution of a pipeline against one dataset version."""

    experiment_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    pipeline_snapshot: Pipeline = Field(
        description="Frozen copy of the pipeline as it was when run. Not a reference: pipelines get edited."
    )
    dataset_version_id: UUID
    dataset_content_hash: ContentHash
    status: ExperimentStatus = ExperimentStatus.PENDING
    resolved_splits: list[ResolvedSplit] = Field(default_factory=list)
    metrics: Metrics | None = None
    environment: Environment | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = Field(default=None, description="A failed experiment is a result. Keep it.")

    @model_validator(mode="after")
    def _status_consistent(self) -> Self:
        if self.status == ExperimentStatus.FAILED and not self.error:
            raise ValueError("a failed experiment must record its error")
        if self.status == ExperimentStatus.SUCCEEDED and self.environment is None:
            raise ValueError("a succeeded experiment must record its environment")
        return self

    @property
    def pipeline_hash(self) -> str:
        return self.pipeline_snapshot.content_hash()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class Model(Base):
    """A fitted model produced by an experiment, saved for prediction and export."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())

    model_id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    experiment_id: UUID
    name: str = Field(min_length=1)
    task: TaskKind
    node_id: NodeId = Field(description="Which estimator node in the pipeline produced this.")
    artifact_path: str = Field(description="Fitted parameters on disk; the database stores the reference only.")
    artifact_hash: ContentHash
    metrics: Metrics
    created_at: datetime = Field(default_factory=_now)


class Project(Base):
    project_id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1)
    description: str = ""
    directory: str = Field(description="Project directory on the user's disk. Datasets and artifacts live here.")
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _demo() -> None:
    import json

    from pydantic import ValidationError

    h = "sha256:" + "a" * 64
    project = Project(name="Corn NIR study", directory="/home/lab/corn")

    dataset = Dataset(project_id=project.project_id, name="corn_raw")
    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash=h,
        n_samples=3,
        n_variables=4,
        axis=VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=[1100.0, 1102.0, 1104.0, 1106.0], unit="nm"),
        sample_ids=["c1", "c2", "c3"],
        targets={"moisture": [10.1, 9.4, 11.2]},
        array_path="datasets/corn_raw/v1.npy",
    )

    pipeline = Pipeline(
        project_id=project.project_id,
        name="SNV + SG + PLS",
        nodes=[
            SourceNode(id="src", version_id=version.version_id),
            PreprocessNode(id="snv", inputs=("src",), step=SNV()),
            PreprocessNode(id="sg", inputs=("snv",), step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1)),
            SplitNode(id="cv", inputs=("sg",), spec=KFoldSplit(n_splits=10, seed=42)),
            EstimatorNode(id="pls", inputs=("cv",), spec=PLSRegressionSpec(n_components=6, target="moisture")),
        ],
    )
    assert [n.id for n in pipeline.terminal_nodes()] == ["pls"]

    # The recipe hashes independently of identity, so a re-created pipeline matches.
    twin = pipeline.model_copy(update={"pipeline_id": uuid4(), "created_at": _now()})
    assert twin.content_hash() == pipeline.content_hash()

    # Changing one parameter changes the hash: this is what powers model comparison.
    swapped = pipeline.model_copy(
        update={"nodes": [n if n.id != "snv" else PreprocessNode(id="snv", inputs=("src",), step=MSC()) for n in pipeline.nodes]}
    )
    assert swapped.content_hash() != pipeline.content_hash()

    # A pipeline round-trips through JSON with its typed parameters intact.
    restored = Pipeline.model_validate_json(pipeline.model_dump_json())
    assert restored.content_hash() == pipeline.content_hash()
    assert isinstance(restored.nodes[2].step, SavitzkyGolay)
    assert restored.nodes[2].step.window_length == 11

    experiment = Experiment(
        project_id=project.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=h,
        status=ExperimentStatus.SUCCEEDED,
        resolved_splits=[ResolvedSplit(node_id="cv", train_indices=[[0, 1]], test_indices=[[2]])],
        metrics=Metrics(rmsecv=0.389, r2=0.981, bias=-0.004),
        environment=Environment(
            app_version="0.1.0", python_version="3.13.9", platform="linux", packages={"numpy": "2.1.0"}
        ),
    )
    assert experiment.pipeline_hash == pipeline.content_hash()

    model = Model(
        project_id=project.project_id,
        experiment_id=experiment.experiment_id,
        name="Model C",
        task=TaskKind.REGRESSION,
        node_id="pls",
        artifact_path="models/model_c.json",
        artifact_hash=h,
        metrics=experiment.metrics,
    )
    assert model.metrics.rmsecv == 0.389

    # Editing the pipeline afterwards must not touch the experiment's snapshot.
    assert experiment.pipeline_snapshot.content_hash() != swapped.content_hash()

    def rejects(fn) -> None:
        try:
            fn()
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    rejects(lambda: SavitzkyGolay(window_length=10, polyorder=2))  # even window
    rejects(lambda: SavitzkyGolay(window_length=5, polyorder=5))  # polyorder >= window
    rejects(lambda: SavitzkyGolay(window_length=11, polyorder=1, deriv=2))  # deriv > polyorder
    rejects(lambda: RangeSelect(start=2500, end=1100))  # reversed
    rejects(lambda: TrainTestSplit(test_size=1.5))  # out of range
    rejects(lambda: KFoldSplit(n_splits=1))  # too few folds
    rejects(
        lambda: VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=[1100.0, 1104.0, 1102.0])
    )  # non-monotonic axis
    rejects(
        lambda: DatasetVersion(
            dataset_id=dataset.dataset_id,
            version=1,
            content_hash=h,
            n_samples=3,
            n_variables=9,  # disagrees with the axis
            axis=version.axis,
            array_path="x.npy",
        )
    )
    rejects(
        lambda: Experiment(
            project_id=project.project_id,
            pipeline_snapshot=pipeline,
            dataset_version_id=version.version_id,
            dataset_content_hash=h,
            status=ExperimentStatus.FAILED,  # no error recorded
        )
    )
    rejects(
        lambda: Pipeline(
            project_id=project.project_id,
            name="cycle",
            nodes=[
                SourceNode(id="src", version_id=version.version_id),
                PreprocessNode(id="a", inputs=("b",), step=SNV()),
                PreprocessNode(id="b", inputs=("a",), step=SNV()),
            ],
        )
    )
    rejects(
        lambda: Pipeline(
            project_id=project.project_id,
            name="dangling",
            nodes=[PreprocessNode(id="a", inputs=("nope",), step=SNV())],
        )
    )
    rejects(
        lambda: Pipeline(
            project_id=project.project_id,
            name="duplicate ids",
            nodes=[
                SourceNode(id="src", version_id=version.version_id),
                PreprocessNode(id="src", inputs=("src",), step=SNV()),
            ],
        )
    )

    json.loads(pipeline.model_dump_json())  # serialisable end to end
    print("all checks passed")


if __name__ == "__main__":
    _demo()
