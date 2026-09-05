"""The pipeline executor: the only path from a dataset to a result.

`PROPOSAL.md` §11 puts one rule above the rest — the kernels are pure
functions over arrays with no knowledge of the application, and the executor
holds the orchestration. So nothing here reaches into `preprocessing.py`, and
nothing about caching, folds or jobs is added to it. `from_spec` is the seam,
and after #82 it needs exactly one thing the recipe does not carry: the
variable axis for `RangeSelect`, which belongs to the `DatasetVersion`.

## What a node's output is

One array per node, `n_samples x n_variables`, stored through #77's array
store — float32 on disk, float64 at the kernel boundary, converted at that
boundary and nowhere else.

Every node's output is read back out of the store before the nodes below it
see it. A node therefore computes from what is on disk, not from the float64
it would have held in memory, and a run that hit the cache agrees with a run
that recomputed to the last bit. The alternative is a cache that changes an
answer, which is worse than a slow one.

A node *below a split* has one array per fold instead, because
`metrics-and-validation.md` §9 says every node downstream of the split is
refitted on the training fold alone. Fold `i`'s array holds every sample
transformed with fold `i`'s fitted parameters; the training rows are the ones
that fitted them and the held-out rows are the ones pushed through, which is
the same array indexed two ways rather than two arrays to keep in step.

The single array such a node *displays* is assembled out of fold: each sample
takes the row from the fold that held it out. Every sample appears exactly
once — that is what `validate_partition` guarantees — so the assembled array
is the same shape as an unsplit node's, and every row in it was produced by
parameters that never saw that row.

## Caching, and what invalidates it

Each node has a key: the SHA-256 of its own JSON together with the keys of its
inputs, with the source node keyed on the dataset version instead. A merkle
chain, so editing a node changes that node's key and every descendant's, and
nothing else's — which is the staleness rule stated as arithmetic rather than
maintained as a separate flag.

The key is derived from the same node JSON `Pipeline.content_hash` uses, and
canvas coordinates live outside the model entirely, in their own table.
A node cannot be moved into a cache miss.

The index from key to stored path is a table in the project's database - it is
a map of references, and `PROPOSAL.md` §11 puts those there. The arrays
themselves are content-addressed files, so two nodes that compute the same
values share one.

## Estimators

A `PCASpec` node is fitted and its result stored as JSON at
`results/<key>.json` — the same key the arrays use, so a result goes stale
exactly when the node above it does. The file is not content-addressed the way
arrays are: a key names one result, and the path is derived from it rather than
looked up in an index.

A node below a split is fitted on the training rows of **fold zero**, which is
what the Phase 1.1 fixture does and is deliberately not an aggregation. There
is no single model over ten folds, and inventing one — an average of loadings,
say — would be arithmetic no document specifies. The fold is recorded in the
result, and the held-out rows are projected through the fitted model and stored
beside the calibration ones, because §9's rule is that they are pushed through
the training fold's parameters rather than left out of the picture.

A `PLSRegressionSpec` node is fitted too, since #142. **The model is fold
zero's and the cross-validated numbers are every fold's**, which is not a
contradiction: §13's reported quantities belong to one fitted model, and
RMSECV is a property of the *split* rather than of any model — which is why §7
pools residuals across folds instead of averaging per-fold errors. Taking the
curve from fold zero alone would be a worse estimate from the same work.

The response is centred by the estimator rather than by a node, because `y` is
not on the canvas and no `MeanCentre` can reach it. Predictions come back in
the response's original units.

`PLSDASpec` is still not fitted and is reported in `Run.pending_estimators`. It
needs a class column and a confusion matrix, which is a second result shape.

## What is not here

**Jobs.** `execute` runs to completion in the calling thread. It reports
progress and asks whether it has been cancelled, but it owns no thread and no
job table — `jobs.py` (#85) wraps it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench import preprocessing, validation
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.models import (
    DatasetVersion,
    Environment,
    EstimatorSpec,
    Experiment,
    ExperimentStatus,
    KFoldSplit,
    LeaveOneOut,
    Metrics,
    NodeId,
    PCASpec,
    Pipeline,
    PipelineNode,
    PLSRegressionSpec,
    ResolvedSplit,
)
from chemometrics_workbench.project import (
    ProjectError,
    read_array,
    read_cache_index,
    write_array,
    write_cache_index,
    write_json,
)
from chemometrics_workbench.regression import (
    PLS,
    cross_validated_predictions,
    rmsecv_curve,
)
from chemometrics_workbench.validation import Fold, k_fold, leave_one_out, validate_partition

__all__ = [
    "ALPHA",
    "RESULTS_DIR",
    "EstimatorResult",
    "ExecutorError",
    "NodeOutput",
    "Progress",
    "Run",
    "RunCancelled",
    "capture_environment",
    "execute",
    "experiment_for",
    "governing_folds",
    "has_kernel",
    "node_keys",
    "node_label",
    "result_path",
    "stored",
    "stored_display",
    "stored_result",
]


def has_kernel(spec: EstimatorSpec) -> bool:
    """Whether this build can actually fit that estimator.

    **The one place that knows.** The routing below and the `estimator_not_fitted`
    warning both ask here, so #142 adds `PLSRegressionSpec` to the tuple once
    rather than in two files that have to be remembered together — which is the
    shape #131 was about.
    """
    return isinstance(spec, _FITTED)


#: What `_estimator` can fit. `PLSDASpec` is absent: it needs a class column
#: and a confusion matrix, which is a second result shape.
_FITTED: tuple[type, ...] = (PCASpec, PLSRegressionSpec)


RESULTS_DIR = "results"

#: The confidence level every limit is quoted at. One number, in one place: a
#: T-squared limit at 0.05 next to an SPE limit at 0.01 is two pictures of the
#: same model that cannot be read together.
ALPHA = 0.05


class RunCancelled(Exception):
    """The caller asked for the run to stop, and it did.

    Deliberately not an `ExecutorError`: a cancelled run is not a failed one.
    Nothing went wrong, there is no cause to report to the user, and a job
    table that turned this into a failure would put a red screen in front of
    someone who pressed Cancel.
    """


@dataclass(frozen=True)
class Progress:
    """Where a run has got to, reported as the walk advances.

    `completed` counts nodes actually finished, so the fraction moves when work
    finishes rather than on a timer. `node_id` and `label` say what has just
    been done, which is what the job's message carries.
    """

    completed: int
    total: int
    node_id: NodeId
    label: str

    @property
    def fraction(self) -> float:
        return self.completed / self.total if self.total else 1.0


class ExecutorError(Exception):
    """A pipeline could not be executed, naming the node it stopped at.

    One exception rather than a hierarchy, for the reason `ProjectError` is
    one: the only caller that distinguishes cases is the HTTP layer, and it
    turns all of them into the same error body. What the caller needs is the
    node id, so it is carried as a field as well as in the sentence — a canvas
    that wants to mark the node red cannot parse it back out of English.
    """

    def __init__(self, message: str, node_id: NodeId | None = None) -> None:
        super().__init__(message)
        self.node_id = node_id


@dataclass(frozen=True)
class NodeOutput:
    """Where one node's result is stored, and whether it had to be computed."""

    node_id: NodeId
    key: str
    array_paths: tuple[str, ...]
    content_hashes: tuple[str, ...]
    n_samples: int
    n_variables: int
    from_cache: bool

    @property
    def array_path(self) -> str:
        """The one array for a node above a split; fold zero's below one.

        Callers that mean "the array to draw" want `Run.display`, which
        assembles the out-of-fold rows. This is the stored path, and for a
        split branch there is more than one.
        """
        return self.array_paths[0]

    @property
    def n_folds(self) -> int:
        return len(self.array_paths)


@dataclass(frozen=True)
class EstimatorResult:
    """One fitted estimator, as `results/<key>.json` holds it.

    Every number here is the kernel's own, unrounded. Turning it into the
    payload the analysis screen draws — adding the sample ids, the variable
    axis and the node's label — is the HTTP layer's job, because those come
    from the `DatasetVersion` rather than from the model.

    `rows` are the samples the model was fitted on. Below a split those are one
    fold's training rows, and `held_out` are the rows it is validated against,
    projected through the same fitted model.
    """

    node_id: NodeId
    key: str
    task: str
    n_components: int
    n_samples: int
    n_variables: int
    rank: int
    fold: int | None
    rows: list[int]
    scores: list[list[float]]
    loadings: list[list[float]]
    eigenvalues: list[float]
    explained_variance_ratio: list[float]
    cumulative_explained_variance: list[float]
    hotelling_t2: list[float]
    hotelling_t2_limit: float
    spe: list[float]
    spe_limit: float
    alpha: float = ALPHA
    held_out: list[int] = field(default_factory=list)
    held_out_scores: list[list[float]] = field(default_factory=list)
    held_out_hotelling_t2: list[float] = field(default_factory=list)
    held_out_spe: list[float] = field(default_factory=list)

    # --- The regression half (#142) ---------------------------------------
    #
    # Additive, and absent on a decomposition. This dataclass was shaped for
    # PCA and the obvious move was a second one; but the two share `task`,
    # `rows`, `fold`, the scores, the loadings, the x-variances and both
    # diagnostics, which is most of it. A second type would restate all of that
    # so the two could differ in the last third, and every reader would grow a
    # branch to tell them apart. `task` already distinguishes them.

    target: str | None = None
    """Which target column was modelled. `None` on a decomposition."""

    observed: list[float] = field(default_factory=list)
    """The reference values for `rows`, so a predicted-versus-actual plot needs
    this record alone and not the dataset beside it."""

    predicted: list[float] = field(default_factory=list)
    """Calibration predictions, in the response's original units."""

    held_out_observed: list[float] = field(default_factory=list)
    held_out_predicted: list[float] = field(default_factory=list)

    coefficients: list[float] = field(default_factory=list)
    """`b` on the matrix the model was fitted on — the node's own axis, not the
    dataset's. Folding the preprocessing back out needs the fitted chain, which
    the executor does not keep; that is #144."""

    y_loadings: list[float] = field(default_factory=list)
    vip: list[float] = field(default_factory=list)

    y_explained_variance_ratio: list[float] = field(default_factory=list)
    """`pls-regression.md` §8's YVar. The x-block's stays in
    `explained_variance_ratio`, shared with PCA, because a screen plotting
    "variance captured" wants both blocks."""

    metrics: dict[str, float] = field(default_factory=dict)
    """`metrics-and-validation.md` §11's table, flattened.

    **A metric that could not be computed is absent**, never `0.0` and never
    `NaN` — §11 is explicit, and the UI renders absence as an em dash. So
    RMSECV and Q² are missing entirely when the node is not under a split, and
    SEC is missing when `n - A - 1 <= 0` rather than falling back to a
    denominator that would silently produce something that is not SEC."""

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> EstimatorResult:
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in document.items() if key in known})


@dataclass(frozen=True)
class Run:
    """What one execution produced.

    `displays` holds the arrays in memory because every caller in 1.2 wants
    them immediately — the spectra endpoint to decimate, the tests to compare.
    They are on disk as well, at the paths in `outputs`.
    """

    pipeline_id: str
    outputs: dict[NodeId, NodeOutput]
    displays: dict[NodeId, NDArray[np.float64]]
    resolved_splits: list[ResolvedSplit]
    results: dict[NodeId, EstimatorResult]
    pending_estimators: list[NodeId]
    """Estimator nodes this build has no kernel for: PLS and PLS-DA, in #142."""

    @property
    def computed(self) -> list[NodeId]:
        return [nid for nid, out in self.outputs.items() if not out.from_cache]

    @property
    def reused(self) -> list[NodeId]:
        return [nid for nid, out in self.outputs.items() if out.from_cache]


@dataclass
class _State:
    """A node's arrays as the walk carries them.

    `arrays` is one array for a node above a split and one per fold below one.
    `folds` is the split governing the node, inherited from its input, so a
    node knows how it must be fitted without looking back up the graph.
    """

    arrays: list[NDArray[np.float64]]
    folds: list[Fold] | None

    @property
    def display(self) -> NDArray[np.float64]:
        if self.folds is None:
            return self.arrays[0]
        assembled = np.empty_like(self.arrays[0])
        for fold, values in zip(self.folds, self.arrays, strict=True):
            assembled[fold.test] = values[fold.test]
        return assembled


def execute(
    directory: str | Path,
    pipeline: Pipeline,
    version: DatasetVersion,
    *,
    use_cache: bool = True,
    on_progress: Callable[[Progress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Run:
    """Run every preprocessing node in `pipeline` against `version`'s array.

    The dataset is read from the project directory, not passed in: the array
    store is the only place a dataset's values live, and a caller handing in
    its own matrix could quietly execute a recipe against something other than
    the version the experiment records.

    `on_progress` is called after each node with what has just finished, and
    `is_cancelled` is asked before each one. Both are optional and neither
    brings a thread with it: this still runs to completion in the calling
    thread, and #85's job table is what puts it on another one.

    A cancelled run raises `RunCancelled` after saving the index for the nodes
    that did finish. Those arrays are complete and correct — the store writes
    through a temporary file and renames, and a node's key is a function of its
    recipe and its data, not of what ran after it — so keeping them means a
    resumed run does not repeat work, and discarding them would be throwing
    away valid results to look tidy.
    """
    path = Path(directory)
    axis = np.asarray(version.axis.values, dtype=np.float64)
    keys = node_keys(pipeline, version)
    index = read_cache_index(path) if use_cache else {}

    states: dict[NodeId, _State] = {}
    outputs: dict[NodeId, NodeOutput] = {}
    splits: list[ResolvedSplit] = []
    results: dict[NodeId, EstimatorResult] = {}
    pending: list[NodeId] = []
    index_changed = False

    ordered = _topological(pipeline)
    completed = 0

    def announce(node: PipelineNode) -> None:
        nonlocal completed
        completed += 1
        if on_progress is not None:
            on_progress(Progress(completed, len(ordered), node.id, node_label(node)))

    def check_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            if index_changed:
                write_cache_index(path, index)
            raise RunCancelled(f"the run was cancelled before node {node.id!r}")

    for node in ordered:
        check_cancelled()
        if node.type == "estimator":
            if not has_kernel(node.spec):
                pending.append(node.id)
                announce(node)
                continue
            results[node.id] = _estimator(
                node, states[node.inputs[0]], keys[node.id], path, use_cache, version
            )
            announce(node)
            continue

        key = keys[node.id]
        parent = states[node.inputs[0]] if node.inputs else None
        folds = _folds_for(node, parent, version.n_samples)

        cached = _from_cache(path, index.get(key), folds) if use_cache else None
        state = cached or _compute(node, parent, folds, path, version, axis)

        stored: list[str] = []
        hashes: list[str] = []
        for values in state.arrays:
            array_path, content_hash = write_array(path, values)
            stored.append(array_path)
            hashes.append(content_hash)

        if cached is None:
            # Read back what was written, so a node's successors are fed the
            # stored values rather than the float64 they were computed in.
            # Otherwise a run that hit the cache and a run that recomputed
            # would disagree in the last few digits, and a cache would be
            # something that changes an answer. The narrowing itself stays
            # where #77 put it, at the store.
            state = _State(arrays=[read_array(path, p) for p in stored], folds=folds)

        states[node.id] = state
        if node.type == "split":
            splits.append(_resolved(node.id, folds))

        if use_cache and index.get(key) != stored:
            index[key] = stored
            index_changed = True

        outputs[node.id] = NodeOutput(
            node_id=node.id,
            key=key,
            array_paths=tuple(stored),
            content_hashes=tuple(hashes),
            n_samples=int(state.arrays[0].shape[0]),
            n_variables=int(state.arrays[0].shape[1]),
            from_cache=cached is not None,
        )
        announce(node)

    if index_changed:
        write_cache_index(path, index)

    return Run(
        pipeline_id=str(pipeline.pipeline_id),
        outputs=outputs,
        displays={nid: state.display for nid, state in states.items()},
        resolved_splits=splits,
        results=results,
        pending_estimators=pending,
    )


def capture_environment() -> Environment:
    """What was installed when a run happened, so a number can be explained later.

    Only the packages that can move a number are recorded — `models.py` says so
    on the field. `scikit-learn` is deliberately absent: it is a development
    dependency and never runs here.
    """
    import platform

    import scipy

    import chemometrics_workbench as cw

    return Environment(
        app_version=cw.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages={"numpy": np.__version__, "scipy": scipy.__version__},
    )


def experiment_for(
    pipeline: Pipeline,
    version: DatasetVersion,
    run: Run | None = None,
    *,
    status: ExperimentStatus = ExperimentStatus.SUCCEEDED,
    started_at: datetime | None = None,
    error: str | None = None,
) -> Experiment:
    """The record one execution leaves behind.

    Built here rather than in `jobs.py` because the metrics come out of the
    `Run`, and a caller that runs the executor directly — the Playwright seed
    does — needs the same record the endpoint writes. The pipeline is snapshot
    by value: `Experiment` says so, and a pipeline gets edited.

    The headline metrics are the *last* estimator's in topological order. One
    experiment carries one set, which is Phase 1.2's simplification and not a
    claim that a four-branch pipeline has a single explained variance; #87's
    per-node results are where each branch's own numbers live.
    """
    metrics: Metrics | None = None
    if run is not None and run.results:
        last = list(run.results.values())[-1]
        metrics = Metrics(
            explained_variance=[float(value) for value in last.explained_variance_ratio],
            extra={
                "hotelling_t2_limit": float(last.hotelling_t2_limit),
                "spe_limit": float(last.spe_limit),
            },
        )
    return Experiment(
        project_id=pipeline.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=version.content_hash,
        status=status,
        resolved_splits=list(run.resolved_splits) if run is not None else [],
        metrics=metrics,
        # A succeeded experiment must record its environment; the model refuses
        # one that does not, so this is not optional for the success path.
        environment=capture_environment() if status == ExperimentStatus.SUCCEEDED else None,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        error=error,
    )


def node_keys(pipeline: Pipeline, version: DatasetVersion) -> dict[NodeId, str]:
    """The cache key of every node: its own content, chained through its inputs.

    Exposed because staleness is a question the HTTP surface has to answer
    without running anything — #85 and #87 both need to say "this node's result
    is out of date" — and re-deriving the rule in two places is how the two
    answers drift apart.
    """
    by_id = {node.id: node for node in pipeline.nodes}
    keys: dict[NodeId, str] = {}

    def key_of(node_id: NodeId) -> str:
        if node_id in keys:
            return keys[node_id]
        node = by_id[node_id]
        # The node's own JSON, which is what `Pipeline.content_hash` hashes and
        # therefore excludes identity, timestamps and anything the canvas
        # stores about where the node sits.
        parts = [node.model_dump_json()]
        if node.type == "source":
            parts.append(str(version.version_id))
            parts.append(version.content_hash)
        parts.extend(key_of(parent) for parent in node.inputs)
        digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
        keys[node_id] = digest
        return digest

    for node in pipeline.nodes:
        key_of(node.id)
    return keys


# --- the walk -------------------------------------------------------------


def node_label(node: PipelineNode) -> str:
    """What a node is, in words, for a progress message.

    Short and derived from the recipe rather than from a table of pretty names:
    a table drifts from the schema the moment a step is added, and a progress
    message that names the wrong step is worse than one that names the kind.
    """
    if node.type == "source":
        return "Reading the dataset"
    if node.type == "preprocess":
        return f"Preprocessing: {node.step.kind}"
    if node.type == "split":
        return f"Splitting: {node.spec.kind}"
    return f"Fitting: {node.spec.kind}"


def _topological(pipeline: Pipeline) -> list[PipelineNode]:
    """Inputs before the nodes that consume them.

    `Pipeline` has already refused a cycle and an unknown input, so this only
    has to order what is known to be a DAG.
    """
    by_id = {node.id: node for node in pipeline.nodes}
    ordered: list[PipelineNode] = []
    seen: set[NodeId] = set()

    def visit(node_id: NodeId) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        for parent in by_id[node_id].inputs:
            visit(parent)
        ordered.append(by_id[node_id])

    for node in pipeline.nodes:
        visit(node.id)
    return ordered


def _folds_for(node: PipelineNode, parent: _State | None, n_samples: int) -> list[Fold] | None:
    """The split governing a node: its own if it is one, else its input's."""
    if node.type != "split":
        return parent.folds if parent is not None else None

    if parent is not None and parent.folds is not None:
        raise ExecutorError(
            f"node {node.id!r} is a split below another split. Nested resampling is "
            "not modelled: the inner folds would have no defined relationship to "
            "the outer ones, and the experiment record has one ResolvedSplit per "
            "split node with no way to say which outer fold it belonged to.",
            node.id,
        )

    spec = node.spec
    if isinstance(spec, KFoldSplit):
        folds = k_fold(n_samples, spec.n_splits, shuffle=spec.shuffle, seed=spec.seed)
    elif isinstance(spec, LeaveOneOut):
        folds = leave_one_out(n_samples)
    else:
        raise ExecutorError(
            f"node {node.id!r} asks for the {spec.kind!r} split, which has no splitter "
            "yet. K-fold and leave-one-out are implemented; train/test, repeated "
            "K-fold and an external set are not.",
            node.id,
        )

    validate_partition(folds, n_samples)
    return folds


def _compute(
    node: PipelineNode,
    parent: _State | None,
    folds: list[Fold] | None,
    directory: Path,
    version: DatasetVersion,
    axis: NDArray[np.float64],
) -> _State:
    """One node's arrays, computed from its input's."""
    if node.type == "source":
        try:
            values = read_array(directory, version.array_path)
        except ProjectError as error:
            raise ExecutorError(
                f"node {node.id!r} could not read the dataset: {error}", node.id
            ) from error
        if values.shape != (version.n_samples, version.n_variables):
            raise ExecutorError(
                f"node {node.id!r} read a {values.shape[0]}x{values.shape[1]} array where "
                f"the version records {version.n_samples}x{version.n_variables}.",
                node.id,
            )
        return _State(arrays=[values], folds=None)

    assert parent is not None, "only a source node has no input, and it returned above"

    if node.type == "split":
        # A split changes which rows fit what below it, never the values. Every
        # fold starts from the same input array - the same object, and one file
        # in the content-addressed store - and the nodes below diverge from
        # there.
        assert folds is not None, "a split node always resolves its folds"
        return _State(arrays=[parent.arrays[0]] * len(folds), folds=folds)

    if node.type != "preprocess":
        raise ExecutorError(
            f"node {node.id!r} has type {node.type!r}, which is not executable.", node.id
        )

    if folds is None:
        return _State(arrays=[_transform(node, parent.arrays[0], None, axis)], folds=None)

    # §9: refitted on the training fold alone, and the held-out rows pushed
    # through those parameters. Fitting on `values[fold.train]` and then
    # transforming every row gives both in one call, because a fitted
    # transformer treats each row independently of the others.
    return _State(
        arrays=[
            _transform(node, values, fold, axis)
            for values, fold in zip(parent.arrays, folds, strict=True)
        ],
        folds=folds,
    )


def _transform(
    node: PipelineNode,
    values: NDArray[np.float64],
    fold: Fold | None,
    axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the step's transformer, fit it, apply it — naming the node if it fails."""
    assert node.type == "preprocess"
    try:
        transformer = preprocessing.from_spec(node.step, axis=axis)
        transformer.fit(values if fold is None else values[fold.train])
        return transformer.transform(values)
    except (ValueError, RuntimeError, NotImplementedError) as error:
        where = "" if fold is None else f" on a training fold of {fold.train.size} samples"
        raise ExecutorError(
            f"node {node.id!r} ({node.step.kind}) failed{where}: {error}", node.id
        ) from error


def result_path(directory: str | Path, key: str) -> Path:
    """Where one estimator's result is stored.

    Derived from the key rather than looked up, because a key names exactly one
    result. Arrays need an index because they are content-addressed and two
    nodes can share a file; a result belongs to its node.
    """
    return Path(directory) / RESULTS_DIR / f"{key}.json"


def _estimator(
    node: PipelineNode,
    parent: _State,
    key: str,
    directory: Path,
    use_cache: bool,
    version: DatasetVersion,
) -> EstimatorResult:
    """Fit one estimator node, or read back the result of having done so.

    `version` is here for the response: a regression needs a `y`, and the
    reference values live on the `DatasetVersion` rather than in any array the
    pipeline produced. A decomposition ignores it.
    """
    assert node.type == "estimator"
    stored = result_path(directory, key)
    if use_cache and stored.exists():
        try:
            return EstimatorResult.from_json(json.loads(stored.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            # A result that cannot be read is recomputed, for the reason a
            # pruned array is: the cache is a saving, never an authority.
            pass

    fold = None if parent.folds is None else 0
    if parent.folds is None:
        matrix = parent.arrays[0]
        rows = np.arange(matrix.shape[0], dtype=np.intp)
        held_out = np.array([], dtype=np.intp)
    else:
        # Fold zero, as the Phase 1.1 fixture fits it. Not an aggregation:
        # there is no single model over ten folds, and averaging loadings
        # across them is arithmetic no document specifies.
        matrix = parent.arrays[0]
        rows = parent.folds[0].train
        held_out = parent.folds[0].test

    if isinstance(node.spec, PLSRegressionSpec):
        result = _pls(node, node.spec, parent, key, matrix, rows, held_out, fold, version)
        stored.parent.mkdir(parents=True, exist_ok=True)
        write_json(stored, result.as_json())
        return result

    assert isinstance(node.spec, PCASpec)
    try:
        # Every array reaches here through the store, which is float32 on disk
        # (#83), so the rank tolerance is quoted for the precision the numbers
        # actually have rather than the one they are computed in. Without this
        # a centred matrix reports one rank too many: the round trip leaves its
        # columns summing to near zero rather than zero, and the SVD finds a
        # singular value sixteen orders down that a float64 tolerance admits.
        model = PCA(node.spec.n_components, data_eps=PCA.STORED_EPS).fit(matrix[rows])
    except (ValueError, RuntimeError) as error:
        raise ExecutorError(f"node {node.id!r} (pca) failed: {error}", node.id) from error

    calibration = matrix[rows]
    result = EstimatorResult(
        node_id=node.id,
        key=key,
        task="decomposition",
        n_components=model.n_components,
        n_samples=int(model.n_samples_ or 0),
        n_variables=int(model.n_variables_ or 0),
        rank=int(model.rank_ or 0),
        fold=fold,
        rows=[int(row) for row in rows],
        scores=_rows(model.scores_),
        loadings=_rows(np.asarray(model.loadings_).T),
        eigenvalues=_values(np.asarray(model.eigenvalues_)[: model.n_components]),
        explained_variance_ratio=_values(model.explained_variance_ratio()),
        cumulative_explained_variance=_values(model.cumulative_explained_variance()),
        hotelling_t2=_values(model.hotelling_t2()),
        hotelling_t2_limit=float(model.hotelling_t2_limit(ALPHA)),
        spe=_values(model.spe(calibration)),
        spe_limit=float(model.spe_limit(ALPHA)),
        held_out=[int(row) for row in held_out],
        # §9: the held-out rows are pushed through the training fold's
        # parameters, exactly as new samples are at prediction time. They are
        # kept beside the calibration rows rather than mixed into them - a
        # diagnostic on a row the model was fitted on and one on a row it was
        # not are different claims.
        held_out_scores=_rows(model.transform(matrix[held_out])) if held_out.size else [],
        held_out_hotelling_t2=(
            _values(model.hotelling_t2(matrix[held_out])) if held_out.size else []
        ),
        held_out_spe=_values(model.spe(matrix[held_out])) if held_out.size else [],
    )

    stored.parent.mkdir(parents=True, exist_ok=True)
    write_json(stored, result.as_json())
    return result


def _response(version: DatasetVersion, node: PipelineNode, name: str) -> NDArray[np.float64]:
    """The reference values a regression is fitted against.

    Refused here rather than at the kernel, with the columns the dataset does
    have, because "target 'moisture' not found" a screen can act on beats a
    `KeyError` from three frames down.
    """
    if name not in version.targets:
        available = ", ".join(sorted(version.targets)) or "none"
        raise ExecutorError(
            f"node {node.id!r} (pls) models target {name!r}, which this dataset does not "
            f"carry. It has: {available}.",
            node.id,
        )
    return np.asarray(version.targets[name], dtype=np.float64)


def _pls(
    node: PipelineNode,
    spec: PLSRegressionSpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """Fit one PLS node and measure it, per `metrics-and-validation.md` §4-§9.

    **The model is fold zero's; the cross-validated numbers are every fold's.**
    Those are not in tension. §13's reported quantities — weights, loadings,
    coefficients, VIP — belong to one fitted model, and fold zero is the one
    PCA already uses, so a regression and a decomposition below the same split
    describe the same samples. RMSECV is not a property of a model at all but
    of the split, which is exactly why §7 pools residuals across every fold
    rather than averaging per-fold errors. Taking the curve from fold zero
    alone would be a worse estimate calculated from the same work.

    **The response is centred here, not by a pipeline node.** `y` is not on the
    canvas, so no `MeanCentre` can reach it; PLS fits what it is given and
    centres nothing of its own (`pls-regression.md` §3), and predictions are
    returned in the response's original units by adding the training mean back.
    `checks.py` warns separately when `X` has no centring above it.
    """
    response = _response(version, node, spec.target)
    if response.size != matrix.shape[0]:
        raise ExecutorError(
            f"node {node.id!r} (pls) has {matrix.shape[0]} samples and target "
            f"{spec.target!r} has {response.size} values.",
            node.id,
        )

    train_x, train_y = matrix[rows], response[rows]
    x_mean = train_x.mean(axis=0)
    y_mean = float(train_y.mean())

    try:
        model = PLS(spec.n_components).fit(train_x - x_mean, train_y - y_mean)
    except (ValueError, RuntimeError) as error:
        raise ExecutorError(f"node {node.id!r} (pls) failed: {error}", node.id) from error

    predicted = model.predict(train_x - x_mean) + y_mean
    a = model.n_components_ or spec.n_components

    metrics: dict[str, float] = {
        "rmsec": validation.rmse(train_y, predicted),
        "r2": validation.r2(train_y, predicted),
        "bias": validation.bias(train_y, predicted),
        "r2_pearson": float(np.corrcoef(train_y, predicted)[0, 1] ** 2),
    }
    # §5: `n - A - 1 <= 0` makes SEC undefined. Absent, and never a fallback
    # denominator - that would be a number which is not SEC.
    if train_y.size - a - 1 > 0:
        metrics["sec"] = validation.sec(train_y, predicted, n_components=a)

    held_x = matrix[held_out]
    held_y = response[held_out]
    held_predicted = (
        model.predict(held_x - x_mean) + y_mean if held_out.size else np.array([], dtype=np.float64)
    )
    if held_out.size:
        metrics["rmsep"] = validation.rmse(held_y, held_predicted)
        metrics["sep"] = validation.sep(held_y, held_predicted)

    # §9: one split, one pass, one curve. The whole fold assignment, not fold
    # zero's, and `A` is never re-selected inside a fold - every fold model is
    # fitted with the same `A` and the curve is a property of the split.
    if parent.folds is not None:
        folds = parent.folds
        curve = rmsecv_curve(matrix, response, folds, a)
        for index, value in enumerate(curve, start=1):
            metrics[f"rmsecv_a{index}"] = float(value)
        metrics["rmsecv"] = float(curve[-1])

        cross_validated = cross_validated_predictions(matrix, response, folds, a)
        # §6: PRESS over the whole calibration set, against the full
        # calibration mean. Never a per-fold mean - packages differ on this and
        # it is what keeps Q2 and R2 on a common denominator.
        metrics["q2"] = validation.q2(response, cross_validated)
        for index, one in enumerate(folds):
            metrics[f"rmsecv_fold_{index}"] = validation.rmse(
                response[one.test], cross_validated[one.test]
            )
        # §8.5: the spread across folds, so a curve's minimum can be read
        # against how much it moves.
        per_fold = [metrics[f"rmsecv_fold_{i}"] for i in range(len(folds))]
        metrics["rmsecv_std"] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0

    return EstimatorResult(
        node_id=node.id,
        key=key,
        task="regression",
        n_components=a,
        n_samples=int(train_x.shape[0]),
        n_variables=int(train_x.shape[1]),
        # A PLS model reports no rank of its own; `A` is the user's parameter
        # and `stopped_early_` is how the kernel says the response ran out.
        rank=a,
        fold=fold,
        rows=[int(row) for row in rows],
        scores=_rows(model.x_scores_),
        loadings=_rows(np.asarray(model.x_loadings_).T),
        # The score variances, not a decomposition's spectrum of them - but the
        # same quantity `PCA.eigenvalues_` carries and the same one the T2
        # ellipse is drawn from. #142 published an empty list here on the
        # reasoning that "a PLS model reports no rank of its own", which is
        # true of `rank` and irrelevant to these; the ellipse came out NaN.
        eigenvalues=_values(model.score_eigenvalues()),
        explained_variance_ratio=_values(model.explained_variance_ratio("x")),
        cumulative_explained_variance=_values(model.cumulative_explained_variance("x")),
        y_explained_variance_ratio=_values(model.explained_variance_ratio("y")),
        hotelling_t2=_values(model.hotelling_t2()),
        hotelling_t2_limit=float(model.hotelling_t2_limit(ALPHA)),
        spe=_values(model.spe(train_x - x_mean)),
        spe_limit=float(model.spe_limit(ALPHA)),
        target=spec.target,
        observed=_values(train_y),
        predicted=_values(predicted),
        coefficients=_values(model.coefficients_),
        y_loadings=_values(model.y_loadings_),
        vip=_values(model.vip()),
        metrics=metrics,
        held_out=[int(row) for row in held_out],
        held_out_observed=_values(held_y) if held_out.size else [],
        held_out_predicted=_values(held_predicted) if held_out.size else [],
        held_out_scores=_rows(model.transform(held_x - x_mean)) if held_out.size else [],
        held_out_hotelling_t2=(
            _values(model.hotelling_t2(held_x - x_mean)) if held_out.size else []
        ),
        held_out_spe=_values(model.spe(held_x - x_mean)) if held_out.size else [],
    )


def _values(array: object) -> list[float]:
    return [float(value) for value in np.asarray(array, dtype=np.float64).ravel()]


def _rows(array: object) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(array, dtype=np.float64)]


def _resolved(node_id: NodeId, folds: list[Fold] | None) -> ResolvedSplit:
    """The index sets a split produced, stored so the run can be repeated (§10)."""
    assert folds is not None, "a split node always resolves its folds"
    return ResolvedSplit(
        node_id=node_id,
        train_indices=[fold.train.tolist() for fold in folds],
        test_indices=[fold.test.tolist() for fold in folds],
    )


# --- reading back what a run stored ---------------------------------------


def stored(directory: str | Path, pipeline: Pipeline, version: DatasetVersion) -> dict[NodeId, str]:
    """Which nodes have a result on disk, keyed by node id, valued by key.

    Cheap: it reads the index and the results directory and computes nothing.
    This is how the HTTP surface answers "is this node's result current?"
    without running anything, and how a canvas knows what to dim.
    """
    path = Path(directory)
    keys = node_keys(pipeline, version)
    index = read_cache_index(path)
    by_id = {node.id: node for node in pipeline.nodes}

    present: dict[NodeId, str] = {}
    for node_id, key in keys.items():
        if by_id[node_id].type == "estimator":
            if result_path(path, key).exists():
                present[node_id] = key
        elif index.get(key):
            present[node_id] = key
    return present


def stored_display(
    directory: str | Path, pipeline: Pipeline, version: DatasetVersion, node_id: NodeId
) -> NDArray[np.float64] | None:
    """One node's array as it would be drawn, read back rather than recomputed.

    `None` when the node has not been run, which the caller reports as such
    rather than serving something out of date.
    """
    path = Path(directory)
    keys = node_keys(pipeline, version)
    if node_id not in keys:
        return None
    paths = read_cache_index(path).get(keys[node_id])
    if not paths:
        return None

    by_id = {node.id: node for node in pipeline.nodes}
    folds = governing_folds(node_id, by_id, version.n_samples)
    state = _from_cache(path, paths, folds)
    return None if state is None else state.display


def stored_result(
    directory: str | Path, pipeline: Pipeline, version: DatasetVersion, node_id: NodeId
) -> EstimatorResult | None:
    """One estimator's stored result, or `None` if it has not been fitted."""
    keys = node_keys(pipeline, version)
    if node_id not in keys:
        return None
    file = result_path(Path(directory), keys[node_id])
    if not file.exists():
        return None
    try:
        return EstimatorResult.from_json(json.loads(file.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def governing_folds(
    node_id: NodeId, by_id: dict[NodeId, PipelineNode], n_samples: int
) -> list[Fold] | None:
    """The split above a node, resolved again from its spec.

    Recomputed rather than stored: a `SplitSpec` and `n` determine the folds
    entirely (`metrics-and-validation.md` §8), so deriving them is cheaper than
    keeping a second copy that can disagree with the recipe.
    """
    current = by_id[node_id]
    while True:
        if current.type == "split":
            return _folds_for(current, None, n_samples)
        if not current.inputs:
            return None
        current = by_id[current.inputs[0]]


# --- the cache index ------------------------------------------------------


def _from_cache(
    directory: Path, stored: list[str] | None, folds: list[Fold] | None
) -> _State | None:
    """Read a node's arrays back, or `None` if they cannot be served.

    A missing file is a miss rather than an error: the index is a hint about
    what has been computed, and a project whose `arrays/` has been pruned
    should recompute rather than refuse to run.
    """
    if not stored:
        return None
    expected = 1 if folds is None else len(folds)
    if len(stored) != expected:
        return None
    try:
        arrays = [read_array(directory, path) for path in stored]
    except ProjectError:
        return None
    return _State(arrays=arrays, folds=folds)
