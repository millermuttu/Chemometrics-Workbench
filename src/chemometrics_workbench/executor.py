"""The pipeline executor: the only path from a dataset to a result.

`PROPOSAL.md` §11 puts one rule above the rest — the kernels are pure
functions over arrays with no knowledge of the application, and the executor
holds the orchestration. So nothing here reaches into `preprocessing.py`, and
nothing about caching, folds or jobs is added to it. `from_spec` is the seam,
and after #82 it needs exactly one thing the recipe does not carry: the
variable axis for `RangeSelect`, which belongs to the `DatasetVersion`. Each
step is given the axis of its own input - the dataset's, narrowed by every
selection above it (`node_axis`, #312).

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

A **train/test split** (#183) is one fold, and not a partition: its training
rows are never held out. A node below it has one array, every row transformed
with the training rows' parameters — the calibration rows fitted them and the
held-out rows were pushed through — and that array is what it displays. An
estimator below it reports RMSEP and SEP on the held-out rows and nothing with
a CV suffix, because one hold-out is not a cross-validation
(`metrics-and-validation.md` §8.6).

## What a run holds in memory

Nothing it has finished with (#176). Every node's arrays are released the
moment its last consumer has run - a count of pending consumers per node,
decremented as the walk goes - so a run holds the arrays of the frontier, not
of the graph. The display array is written to the store when its node
completes, under `<key>#display` in the index, so the spectra endpoint reads
one array rather than assembling k fold arrays on every request; and
`Run.displays` reads from the store on access rather than holding anything,
which is what lets a finished job sit in the job table costing nothing.

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

A node below a split is **refitted on every sample** (#330): cross-validation
estimates the error of the recipe, and the model kept is fitted once more on
all of it. To make that possible every node below a split carries, beside its
fold arrays, an all-sample array (`_State.full`, stored under `<key>#all`):
every row through parameters fitted on every row. The estimator is fitted on
its input's. There is no single model over ten folds and no average of them -
that would be arithmetic no document specifies - but there is one over every
sample, and it is the one a user saves and exports.

Fold zero's model survives as the **held-out view**: fitted on fold zero's
training rows of fold zero's array, it predicts that fold's held-out rows, and
those predictions, diagnostics and `_p` metrics are stored beside the
all-sample model's (`_with_held_out`), because §9's rule is that a held-out row
is pushed through parameters that never saw it.

A `PLSRegressionSpec` node is fitted too, since #142. **The model is every
sample's and the cross-validated numbers are every fold's**, which is not a
contradiction: §13's reported quantities belong to one fitted model, and
RMSECV is a property of the *split* rather than of any model — which is why §7
pools residuals across folds instead of averaging per-fold errors.

The response is centred by the estimator rather than by a node, because `y` is
not on the canvas and no `MeanCentre` can reach it. Predictions come back in
the response's original units.

A `PLSDASpec` node is fitted since #185, as `pls-da.md` specifies: two classes
from a metadata column, coded {0, 1} in Unicode order of the labels, PLS1 on
that dummy response through the same `_fit_pls1` a regression uses, and a class
assigned at 0.5. What is added to the result is the coding, the assignments and
the confusion matrices; every model quantity is the regression's.

## What is not here

**Jobs.** `execute` runs to completion in the calling thread. It reports
progress and asks whether it has been cancelled, but it owns no thread and no
job table — `jobs.py` (#85) wraps it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench import preprocessing, validation
from chemometrics_workbench.classification import (
    KNN,
    LDA,
    SIMCA,
    acceptance_table,
    simca_metrics,
)
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.models import (
    DatasetVersion,
    Environment,
    EstimatorSpec,
    Experiment,
    ExperimentStatus,
    KFoldSplit,
    KNNSpec,
    LDASpec,
    LeaveOneOut,
    Metrics,
    NodeId,
    PCASpec,
    PCRSpec,
    Pipeline,
    PipelineNode,
    PLSDASpec,
    PLSRegressionSpec,
    RangeSelect,
    ResolvedSplit,
    SelectVariables,
    SIMCASpec,
    TrainTestSplit,
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
    PCR,
    PLS,
    PLS2,
    cross_validated_predictions,
    rmsecv_curve,
)
from chemometrics_workbench.validation import (
    Fold,
    PermutationResult,
    by_group,
    k_fold,
    leave_one_out,
    permutation_test,
    stratified_k_fold,
    stratified_train_test,
    train_test,
    validate_partition,
)

__all__ = [
    "ALPHA",
    "RESULTS_DIR",
    "EstimatorResult",
    "ExecutorError",
    "NodeOutput",
    "Progress",
    "Run",
    "RunCancelled",
    "StoredDisplays",
    "assign_classes",
    "capture_environment",
    "class_metrics",
    "classification_metrics",
    "confusion_matrix",
    "execute",
    "experiment_for",
    "governing_folds",
    "governing_split",
    "has_kernel",
    "metrics_for",
    "node_keys",
    "node_label",
    "permutation_test_for",
    "result_path",
    "stored",
    "stored_display",
    "stored_fitted_matrix",
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


#: What `_estimator` can fit. All three since #185; the tuple stays because
#: `has_kernel` is the one place the answer lives, and the next estimator will
#: not have a kernel on the day its spec lands either.
_FITTED: tuple[type, ...] = (
    PCASpec,
    PLSRegressionSpec,
    PCRSpec,
    PLSDASpec,
    SIMCASpec,
    LDASpec,
    KNNSpec,
)


#: Part of every estimator's key. Changed when what a stored result means
#: changes under an unchanged recipe, so every older result is a cache miss.
RESULT_FORMAT = "final-model"

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
        """The one array for a node above a split; fold zero's below one (its
        all-sample array is under `<key>#all`, #330).

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

    `rows` are the samples the model was fitted on: every sample, below a split
    as above one (#330). Below a split `held_out` are fold zero's held-out rows,
    projected through fold zero's model rather than this one, which has seen them.
    """

    node_id: NodeId
    key: str
    task: str
    n_components: int
    n_samples: int
    n_variables: int
    rank: int
    fold: int | None
    """Which fold's held-out rows `held_out` are: 0 below a split, `None` above one.
    Before #330 it was also the fold the model was fitted on; see `all_samples`."""
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
    spe_limit_caveat: str | None = None
    """Why `spe_limit` is outside its approximation's domain (#71): the kernel's
    own sentence when Jackson-Mudholkar's `h0` is not positive, else `None`. A
    regression's chi-squared limit has no such domain and leaves it `None`."""
    held_out: list[int] = field(default_factory=list)
    held_out_scores: list[list[float]] = field(default_factory=list)
    held_out_hotelling_t2: list[float] = field(default_factory=list)
    held_out_spe: list[float] = field(default_factory=list)
    all_samples: bool = False
    """Below a split, the model is fitted on every sample (#330) and `held_out*`
    come from fold zero's model, which never saw those rows. `False` above a
    split, where every sample is all there is. A result stored before #330 is
    under an older key (`RESULT_FORMAT`) and is never served."""

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

    method: str = ""
    """The estimator kind the regression half came from - `pls`, `pcr` or
    `plsda` (#272) - so a screen can name it. Empty on a decomposition and on
    a result stored before it."""

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

    x_mean: list[float] = field(default_factory=list)
    """The column means the estimator subtracted before fitting, and adds back
    to nothing - predictions come back in the response's units through
    `y_mean`. Not a pipeline node's centring: `y` is not on the canvas and no
    `MeanCentre` reaches it (`pls-regression.md` §3), so the estimator centres
    `X` by its fit rows too and this is that. Empty on a decomposition. Kept
    since #211, because a model artifact cannot carry a fitted model without
    it and `folded_coefficients` had to refit the chain to recover it."""

    y_mean: float | None = None
    """The response mean the estimator subtracted, added back to every
    prediction. `None` on a decomposition."""

    rotations: list[list[float]] = field(default_factory=list)
    """`a x p`, like `loadings`: what a row is multiplied by to get its scores.
    PCA's are its loadings; PLS's are `R = W(P'W)^-1` (`pls-regression.md`
    §5). Kept since #186 so a contribution plot can be computed from the
    stored result and the stored input row without refitting anything. Empty
    on a result stored before then, which the endpoint says."""

    y_explained_variance_ratio: list[float] = field(default_factory=list)
    """`pls-regression.md` §8's YVar. The x-block's stays in
    `explained_variance_ratio`, shared with PCA, because a screen plotting
    "variance captured" wants both blocks."""

    cross_validated_predicted: list[float] = field(default_factory=list)
    """One held-out prediction per sample, pooled over the folds
    (`metrics-and-validation.md` §7). Empty above a split and below a single
    hold-out. Kept since #185 because a classification tallies its
    cross-validated confusion matrix from it; a regression could draw a
    cross-validated predicted-versus-measured from the same list."""

    # --- The classification half (#185) ----------------------------------
    #
    # Additive again, and for the same reason as the regression half: a
    # two-class PLS-DA *is* the regression above on a dummy response
    # (`pls-da.md` §2), so everything up to here is filled in the same way and
    # `task` says "classification". These are what a classification adds.

    classes: list[str] = field(default_factory=list)
    """`[C_0, C_1]` in Unicode order; `C_1` is the class coded 1 (§3)."""

    predicted_class: list[int] = field(default_factory=list)
    """Calibration assignments as indices into `classes` (§5)."""

    held_out_predicted_class: list[int] = field(default_factory=list)

    coefficient_matrix: list[list[float]] = field(default_factory=list)
    """`p x N`, one column per class, assigned by the largest: a PLS-DA of
    three or more classes (#274) and every LDA (#276). Empty otherwise."""

    y_means: list[float] = field(default_factory=list)
    """Added to every row of `X @ coefficient_matrix`: the one-hot response's
    column means for a PLS-DA, the discriminant intercepts for an LDA (#276)."""

    training_classes: list[int] = field(default_factory=list)
    """A kNN's calibration rows' classes, as indices into `classes`, beside
    `scores`: the neighbours every later sample is measured against (#277)."""

    k: int | None = None
    """A kNN's neighbour count; `None` for every other estimator."""

    simca: dict[str, Any] = field(default_factory=dict)
    """A SIMCA's class models and its decisions per set (`simca.md` §5, #275).
    Empty for every other estimator. A SIMCA has no single X model, so the
    shared scores, loadings and limits above are empty and zero for it."""

    confusion: dict[str, list[list[int]]] = field(default_factory=dict)
    """`calibration`, and below a split `cross_validation` and `held_out`: rows
    observed, columns assigned, in `classes` order (§6)."""

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


class StoredDisplays(Mapping[NodeId, NDArray[np.float64]]):
    """`Run.displays`: each node's display array, read from the store on access.

    A mapping rather than a dict of arrays (#176): a `Run` sits in the job
    table for the life of the process, and one that held every node's display
    would keep a whole run's worth of float64 resident after it finished.
    Reading on access costs one `np.load` per look and holds nothing.
    """

    def __init__(self, directory: Path, paths: dict[NodeId, str]) -> None:
        self._directory = directory
        self._paths = dict(paths)

    def __getitem__(self, node_id: NodeId) -> NDArray[np.float64]:
        return read_array(self._directory, self._paths[node_id])

    def __iter__(self) -> Iterator[NodeId]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


@dataclass(frozen=True)
class Run:
    """What one execution produced.

    `displays` reads from the store on access; the arrays are on disk at the
    paths in `outputs`, and nothing here holds one (#176).
    """

    pipeline_id: str
    outputs: dict[NodeId, NodeOutput]
    displays: Mapping[NodeId, NDArray[np.float64]]
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
    full: NDArray[np.float64] | None = None
    """Below a split, every row through parameters fitted on every row (#330):
    the matrix the final model is fitted on. `None` above a split, where
    `arrays[0]` already is that."""

    @property
    def display(self) -> NDArray[np.float64]:
        # One fold is a hold-out, not a partition: its training rows are never
        # held out, so there is nothing to assemble from. The one array has
        # every row through the training rows' parameters, which is the
        # picture §9 asks for.
        if self.folds is None or len(self.folds) == 1:
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
    display_paths: dict[NodeId, str] = {}
    splits: list[ResolvedSplit] = []
    results: dict[NodeId, EstimatorResult] = {}
    pending: list[NodeId] = []
    index_changed = False

    # How many nodes still need each node's arrays. When it reaches zero the
    # arrays are let go (#176): a run holds its frontier, not its history.
    consumers = Counter(parent for node in pipeline.nodes for parent in node.inputs)

    def release(node: PipelineNode) -> None:
        for parent in node.inputs:
            consumers[parent] -= 1
            if consumers[parent] <= 0:
                states.pop(parent, None)

    if below := pipeline.estimator_inputs():
        node_id, estimator = below[0]
        # #296: a pipeline saved before the write refused this still loads,
        # and would otherwise fail below with a bare KeyError.
        raise ExecutorError(
            f"node {node_id!r} takes its input from the estimator {estimator!r}, which "
            "produces a model, not spectra. Connect it to the step above instead.",
            node_id,
        )
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
            release(node)
            announce(node)
            continue

        key = keys[node.id]
        parent = states[node.inputs[0]] if node.inputs else None
        folds = _folds_for(node, parent, version)

        cached = _from_cache(path, index.get(key), folds) if use_cache else None

        stored: list[str] = []
        hashes: list[str] = []
        if cached is not None:
            # Already on disk under these paths. Writing them again cost a
            # float32 copy, a serialisation and a SHA-256 per array, per fold,
            # on every run that recomputed nothing (#174). The store is
            # content-addressed, so the hash is the file's name.
            state = cached
            stored = list(index[key])
            hashes = [f"sha256:{Path(p).stem}" for p in stored]
        else:
            # One fold at a time (#176): computed, written, and read back
            # before the next is computed, so a node below a k-fold holds its
            # k stored arrays and one transient, never k computed and k read
            # back at once. Read back rather than kept, so a node's successors
            # are fed the stored values rather than the float64 they were
            # computed in: otherwise a run that hit the cache and a run that
            # recomputed would disagree in the last few digits, and a cache
            # would be something that changes an answer. The narrowing itself
            # stays where #77 put it, at the store. A split's folds are one
            # array k times, and one file, and are read back once.
            arrays: list[NDArray[np.float64]] = []
            read_back: dict[str, NDArray[np.float64]] = {}
            # #312: the axis this step's input is on, which a selection above
            # it has narrowed.
            given = (
                node_axis(pipeline, node.inputs[0], version) if node.type == "preprocess" else axis
            )
            for values in _computed(node, parent, folds, path, version, given):
                array_path, content_hash = write_array(path, values)
                del values
                stored.append(array_path)
                hashes.append(content_hash)
                # A split's k folds are one array and one file, so the read
                # back happens once and every fold shares it.
                if array_path not in read_back:
                    read_back[array_path] = read_array(path, array_path)
                arrays.append(read_back[array_path])
            del read_back
            state = _State(arrays=arrays, folds=folds)

        if folds is not None:
            assert parent is not None, "a node below a split has an input"
            state.full, full_path = _all_samples(
                node,
                parent,
                path,
                index.get(f"{key}#all") if use_cache else None,
                pipeline,
                version,
            )
            if use_cache and index.get(f"{key}#all") != [full_path]:
                index[f"{key}#all"] = [full_path]
                index_changed = True

        states[node.id] = state
        if node.type == "split":
            splits.append(_resolved(node.id, folds))

        if use_cache and index.get(key) != stored:
            index[key] = stored
            index_changed = True

        # The display array, stored once at completion (#176). Above a split it
        # is the node's one array and costs nothing - the same content hash is
        # the same file; below one it is the out-of-fold assembly, written so
        # `stored_display` reads one array rather than k. A cache hit whose
        # index already names it writes nothing (#174).
        display_key = f"{key}#display"
        known = index.get(display_key) if use_cache and cached is not None else None
        if known and (path / known[0]).is_file():
            display_paths[node.id] = known[0]
        else:
            display_path = (
                stored[0] if len(state.arrays) == 1 else write_array(path, state.display)[0]
            )
            display_paths[node.id] = display_path
            if use_cache and index.get(display_key) != [display_path]:
                index[display_key] = [display_path]
                index_changed = True
        release(node)

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
        displays=StoredDisplays(path, display_paths),
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


#: The `Metrics` fields a regression fills by name (#188). Everything else in
#: a result's metrics table travels in `extra`.
_NAMED_METRICS = ("rmsec", "rmsecv", "rmsep", "r2", "q2", "bias", "accuracy")


def metrics_for(result: EstimatorResult) -> Metrics:
    """One estimator's flat metrics table as the schema's `Metrics`.

    The named fields a regression fills are filled by name (#188); everything
    else the result carries - SEC, SEP, the RMSECV curve, the per-fold errors -
    goes into `extra` beside the two limits. A metric the result does not carry
    stays `None`, which is `metrics-and-validation.md` §11's absence and not
    zero.

    Shared by the experiment record and the model registry (#219), because a
    saved model's metrics and its run's are the same numbers and two mappings
    would be two chances to disagree about them.
    """
    named = {name: result.metrics.get(name) for name in _NAMED_METRICS}
    return Metrics(
        **named,
        explained_variance=[float(value) for value in result.explained_variance_ratio],
        extra={
            "hotelling_t2_limit": float(result.hotelling_t2_limit),
            "spe_limit": float(result.spe_limit),
            **{
                key: float(value)
                for key, value in result.metrics.items()
                if key not in _NAMED_METRICS
            },
        },
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

    A regression's named metrics - RMSEC, RMSECV, RMSEP, R2, Q2, bias - fill
    the `Metrics` fields that were written for them and stayed `None` until
    #188; the rest of `EstimatorResult.metrics` (SEC, SEP, the RMSECV curve
    and per-fold errors) goes into `extra` beside the two limits. A metric the
    result does not carry stays `None`, which is §11's absence and not zero.
    """
    metrics: Metrics | None = None
    if run is not None and run.results:
        metrics = metrics_for(list(run.results.values())[-1])
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
        if node.type == "estimator":
            # #330 changed what an estimator below a split stores - the model
            # refitted on every sample, not fold zero's - under the same recipe.
            # A new key is how every stored result made before it goes stale.
            parts.append(RESULT_FORMAT)
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


def _folds_for(
    node: PipelineNode, parent: _State | None, version: DatasetVersion
) -> list[Fold] | None:
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

    n_samples = version.n_samples
    spec = node.spec
    stratify_by = spec.stratify_by if isinstance(spec, TrainTestSplit | KFoldSplit) else None
    group_by: str | None = getattr(spec, "group_by", None)
    labels = None if stratify_by is None else _metadata_column(version, node, stratify_by)
    groups = None if group_by is None else _metadata_column(version, node, group_by)

    def grouped(split: Callable[[int], list[Fold]]) -> list[Fold]:
        # §8.8: the splitter runs over the groups, not the rows.
        return split(n_samples) if groups is None else by_group(groups, split)

    try:
        if isinstance(spec, TrainTestSplit):
            # A hold-out, not a partition: `validate_partition` is §7's rule
            # for pooling residuals across folds and this has one.
            if labels is not None:
                return stratified_train_test(labels, spec.test_size, seed=spec.seed)
            return grouped(lambda n: train_test(n, spec.test_size, seed=spec.seed))
        if isinstance(spec, KFoldSplit):
            shuffle, seed = spec.shuffle, spec.seed
            folds = (
                stratified_k_fold(labels, spec.n_splits, shuffle=shuffle, seed=seed)
                if labels is not None
                else grouped(lambda n: k_fold(n, spec.n_splits, shuffle=shuffle, seed=seed))
            )
        elif isinstance(spec, LeaveOneOut):
            folds = grouped(leave_one_out)
        else:
            raise ExecutorError(
                f"node {node.id!r} asks for the {spec.kind!r} split, which has no splitter "
                "yet. K-fold, leave-one-out and train/test are implemented; repeated "
                "K-fold and an external set are not.",
                node.id,
            )
    except ValueError as error:
        by = (
            f" stratified by {stratify_by!r}"
            if stratify_by is not None
            else f" grouped by {group_by!r}"
            if group_by is not None
            else ""
        )
        raise ExecutorError(f"node {node.id!r} ({spec.kind}{by}) failed: {error}", node.id) from (
            error
        )

    validate_partition(folds, n_samples)
    return folds


def _metadata_column(version: DatasetVersion, node: PipelineNode, name: str) -> list[str]:
    """The column a split stratifies or groups by, refused by name when absent (§8.7, §8.8)."""
    labels = version.metadata_columns.get(name)
    if labels is None:
        available = ", ".join(sorted(version.metadata_columns)) or "none"
        raise ExecutorError(
            f"node {node.id!r} splits by {name!r}, which this dataset does not carry as "
            f"a metadata column. It has: {available}.",
            node.id,
        )
    return labels


def _all_samples(
    node: PipelineNode,
    parent: _State,
    directory: Path,
    stored: list[str] | None,
    pipeline: Pipeline,
    version: DatasetVersion,
) -> tuple[NDArray[np.float64], str]:
    """A node's all-sample array below a split, and where it is stored (#330).

    The split passes its input through; a step below it is fitted on every row
    of its input's all-sample array. Stored under `<key>#all`, written and read
    back like every other array so a cached run and a fresh one agree. An index
    written before #330 has no such entry and computes it here.
    """
    if stored:
        try:
            return read_array(directory, stored[0]), stored[0]
        except ProjectError:
            pass
    if node.type == "split":
        values = parent.full if parent.full is not None else parent.arrays[0]
    else:
        assert node.type == "preprocess" and parent.full is not None
        values = _transform(node, parent.full, None, node_axis(pipeline, node.inputs[0], version))
    array_path, _ = write_array(directory, values)
    return read_array(directory, array_path), array_path


def _computed(
    node: PipelineNode,
    parent: _State | None,
    folds: list[Fold] | None,
    directory: Path,
    version: DatasetVersion,
    axis: NDArray[np.float64],
) -> Iterator[NDArray[np.float64]]:
    """One node's arrays, computed from its input's, one at a time.

    A generator so the walk can store each fold before the next exists
    (#176): below a ten-fold split the alternative held ten float64 arrays it
    was about to narrow and discard.
    """
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
        yield values
        return

    assert parent is not None, "only a source node has no input, and it returned above"

    if node.type == "split":
        # A split changes which rows fit what below it, never the values. Every
        # fold starts from the same input array - the same object, and one file
        # in the content-addressed store - and the nodes below diverge from
        # there.
        assert folds is not None, "a split node always resolves its folds"
        for _ in folds:
            yield parent.arrays[0]
        return

    if node.type != "preprocess":
        raise ExecutorError(
            f"node {node.id!r} has type {node.type!r}, which is not executable.", node.id
        )

    if folds is None:
        yield _transform(node, parent.arrays[0], None, axis)
        return

    # §9: refitted on the training fold alone, and the held-out rows pushed
    # through those parameters. Fitting on `values[fold.train]` and then
    # transforming every row gives both in one call, because a fitted
    # transformer treats each row independently of the others.
    for values, fold in zip(parent.arrays, folds, strict=True):
        yield _transform(node, values, fold, axis)


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

    _refuse_one_class_folds(node, parent.folds, version)
    if parent.folds is not None and parent.full is not None:
        # #330: below a split the model is refitted on every sample, through
        # every step above it refitted on every sample too. Cross-validation
        # estimates its error - the CV numbers are every fold's, as before -
        # and fold zero's model, fitted on its training rows, supplies the
        # held-out view: rows that model never saw.
        everyone = np.arange(parent.full.shape[0], dtype=np.intp)
        final = _fitted(node, parent, key, parent.full, everyone, _NONE, None, version)
        fold = parent.folds[0]
        view = _fitted(
            node,
            _State(arrays=[parent.arrays[0]], folds=None),
            key,
            parent.arrays[0],
            fold.train,
            fold.test,
            0,
            version,
        )
        result = _with_held_out(final, view)
    else:
        matrix = parent.arrays[0]
        rows = np.arange(matrix.shape[0], dtype=np.intp)
        result = _fitted(node, parent, key, matrix, rows, _NONE, None, version)
    stored.parent.mkdir(parents=True, exist_ok=True)
    write_json(stored, result.as_json())
    return result


_NONE = np.array([], dtype=np.intp)


def _refuse_one_class_folds(
    node: PipelineNode, folds: list[Fold] | None, version: DatasetVersion
) -> None:
    """A classifier cannot be fitted on a training fold that holds one class (#343).

    Checked once here for every classifier - anything whose spec names a
    `class_column` - because the kernels fail it each in their own words, and
    PLS-DA's ("X and y have no covariance") is true without saying why. A
    grouped or small split puts every member of a class in one validation set
    this way; the column being missing is `_class_labels`'s to refuse.
    """
    spec = getattr(node, "spec", None)
    column = getattr(spec, "class_column", None)
    labels = version.metadata_columns.get(column) if column is not None else None
    if spec is None or folds is None or labels is None:
        return
    for number, fold in enumerate(folds, start=1):
        present = sorted({str(labels[row]) for row in fold.train})
        if len(present) < 2:
            raise ExecutorError(
                f"node {node.id!r} ({spec.kind}) cannot be fitted: training fold "
                f"{number} of {len(folds)} holds only {', '.join(map(repr, present))} "
                f"in {column!r}, and a classifier needs two classes to separate. Stratify "
                f"the split by {column!r}, or group it by a column other than the class.",
                node.id,
            )


#: Metrics measured on the held-out rows (`metrics-and-validation.md` §11):
#: RMSEP and SEP, and the `_p` classification metrics.
_HELD_OUT_METRICS = ("rmsep", "sep")


def _with_held_out(final: EstimatorResult, view: EstimatorResult) -> EstimatorResult:
    """The all-sample model, with fold zero's held-out rows as its held-out view (#330).

    Everything that describes the model - scores, loadings, coefficients, the
    calibration predictions and metrics, the cross-validated numbers - is the
    final model's. What describes the held-out rows is fold zero's model's,
    because the final model has seen them.
    """
    held = {f.name: getattr(view, f.name) for f in fields(view) if f.name.startswith("held_out")}
    metrics = dict(final.metrics)
    metrics.update(
        {k: v for k, v in view.metrics.items() if k in _HELD_OUT_METRICS or k.endswith("_p")}
    )
    confusion = dict(final.confusion)
    if "held_out" in view.confusion:
        confusion["held_out"] = view.confusion["held_out"]
    simca = final.simca
    if "held_out" in view.simca.get("sets", {}):
        simca = {
            **final.simca,
            "sets": {**final.simca["sets"], "held_out": view.simca["sets"]["held_out"]},
        }
    return replace(
        final,
        **held,
        fold=view.fold,
        all_samples=True,
        metrics=metrics,
        confusion=confusion,
        simca=simca,
    )


def _fitted(
    node: PipelineNode,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """Fit one estimator on `matrix[rows]`, validated on `held_out` and, when
    `parent` carries more than one fold, cross-validated on its fold arrays."""
    assert node.type == "estimator"
    if isinstance(node.spec, PLSRegressionSpec):
        return _pls(node, node.spec, parent, key, matrix, rows, held_out, fold, version)

    if isinstance(node.spec, PCRSpec):
        return _fit_regression(
            node,
            "pcr",
            node.spec.n_components,
            node.spec.target,
            _response(version, node, node.spec.target),
            parent,
            key,
            matrix,
            rows,
            held_out,
            fold,
            estimator=PCR,
        )

    if isinstance(node.spec, KNNSpec):
        return _knn(node, node.spec, parent, key, matrix, rows, held_out, fold, version)

    if isinstance(node.spec, LDASpec):
        return _lda(node, node.spec, parent, key, matrix, rows, held_out, fold, version)

    if isinstance(node.spec, SIMCASpec):
        return _simca(node, node.spec, parent, key, matrix, rows, held_out, fold, version)

    if isinstance(node.spec, PLSDASpec):
        return _plsda(node, node.spec, parent, key, matrix, rows, held_out, fold, version)

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
        rotations=_rows(np.asarray(model.loadings_).T),
        eigenvalues=_values(np.asarray(model.eigenvalues_)[: model.n_components]),
        explained_variance_ratio=_values(model.explained_variance_ratio()),
        cumulative_explained_variance=_values(model.cumulative_explained_variance()),
        hotelling_t2=_values(model.hotelling_t2()),
        hotelling_t2_limit=float(model.hotelling_t2_limit(ALPHA)),
        spe=_values(model.spe(calibration)),
        spe_limit=float(model.spe_limit(ALPHA)),
        spe_limit_caveat=model.spe_limit_caveat(),
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

    **The model is the one `matrix[rows]` gives; the cross-validated numbers
    are every fold's.** Those are not in tension. §13's reported quantities —
    weights, loadings, coefficients, VIP — belong to one fitted model, which
    below a split `_estimator` asks for twice: on every sample, and on fold
    zero's training rows for the held-out view (#330). RMSECV is not a property
    of a model at all but of the split, which is exactly why §7 pools residuals
    across every fold rather than averaging per-fold errors.

    **The response is centred here, not by a pipeline node.** `y` is not on the
    canvas, so no `MeanCentre` can reach it; PLS fits what it is given and
    centres nothing of its own (`pls-regression.md` §3), and predictions are
    returned in the response's original units by adding the training mean back.
    `checks.py` warns separately when `X` has no centring above it.
    """
    response = _response(version, node, spec.target)
    return _fit_regression(
        node,
        "pls",
        spec.n_components,
        spec.target,
        response,
        parent,
        key,
        matrix,
        rows,
        held_out,
        fold,
    )


def _fit_regression(
    node: PipelineNode,
    kind: str,
    n_components: int,
    target: str,
    response: NDArray[np.float64],
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    estimator: type[PLS] | type[PCR] = PLS,
) -> EstimatorResult:
    """PLS1 or PCR on `response`, with every quantity `pls-regression.md` §13 names.

    Shared by a PLS regression, a two-class PLS-DA, which is this on a dummy
    response (`pls-da.md` §2), and a PCR (`pcr.md`), which has the same
    interface and no VIP. `kind` is only for the sentences.
    """
    if response.size != matrix.shape[0]:
        raise ExecutorError(
            f"node {node.id!r} ({kind}) has {matrix.shape[0]} samples and target "
            f"{target!r} has {response.size} values.",
            node.id,
        )
    spec_components = n_components

    train_x, train_y = matrix[rows], response[rows]
    x_mean = train_x.mean(axis=0)
    y_mean = float(train_y.mean())

    try:
        model = estimator(spec_components).fit(train_x - x_mean, train_y - y_mean)
    except (ValueError, RuntimeError) as error:
        raise ExecutorError(f"node {node.id!r} ({kind}) failed: {error}", node.id) from error

    predicted = model.predict(train_x - x_mean) + y_mean
    a = model.n_components_ or spec_components

    metrics: dict[str, float] = {
        "rmsec": validation.rmse(train_y, predicted),
        "r2": validation.r2(train_y, predicted),
        "bias": validation.bias(train_y, predicted),
    }
    # §11: absent, never NaN. A constant prediction has no correlation to square.
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson = float(np.corrcoef(train_y, predicted)[0, 1] ** 2)
    if np.isfinite(pearson):
        metrics["r2_pearson"] = pearson
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
    #
    # Each fold is evaluated on its own array, preprocessed with that fold's
    # training rows. Fold zero's array fitted its preprocessing on every other
    # fold's test rows, so using it for all of them leaked (#173).
    #
    # One fold is a train/test hold-out (#183), and one hold-out is not a
    # cross-validation: its training rows are never predicted, so there is no
    # RMSECV and no Q2 - §11 says absent, and RMSEP above is its number.
    cross_validated = np.array([], dtype=np.float64)
    if parent.folds is not None and len(parent.folds) > 1:
        folds = parent.folds
        curve = rmsecv_curve(parent.arrays, response, folds, a, estimator)
        for index, value in enumerate(curve, start=1):
            metrics[f"rmsecv_a{index}"] = float(value)
        metrics["rmsecv"] = float(curve[-1])

        cross_validated = cross_validated_predictions(parent.arrays, response, folds, a, estimator)
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
        rotations=_rows(np.asarray(model.rotations_).T),
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
        spe_limit_caveat=model.spe_limit_caveat() if isinstance(model, PCR) else None,
        target=target,
        method=kind,
        observed=_values(train_y),
        predicted=_values(predicted),
        coefficients=_values(model.coefficients_),
        x_mean=_values(x_mean),
        y_mean=y_mean,
        y_loadings=_values(model.y_loadings_),
        # VIP is a PLS quantity: PCR's components are chosen without y (pcr.md §6).
        vip=_values(model.vip()) if isinstance(model, PLS) else [],
        cross_validated_predicted=_values(cross_validated),
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


def _class_labels(
    version: DatasetVersion, node: PipelineNode, name: str, kind: str = "plsda"
) -> tuple[list[str], NDArray[np.intp]]:
    """The classes in Unicode order and each sample's index into them (`pls-da.md` §3).

    Refused here, by name, when the column is not in the dataset or holds a
    single value, which has nothing to separate.
    """
    labels = version.metadata_columns.get(name)
    if labels is None:
        available = ", ".join(sorted(version.metadata_columns)) or "none"
        raise ExecutorError(
            f"node {node.id!r} ({kind}) classifies by {name!r}, which this dataset does not "
            f"carry as a metadata column. It has: {available}.",
            node.id,
        )
    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        raise ExecutorError(
            f"node {node.id!r} ({kind}) classifies by {name!r}, which has one value "
            f"({classes[0]!r} on every sample): there is nothing to separate.",
            node.id,
        )
    index = {label: position for position, label in enumerate(classes)}
    return classes, np.asarray([index[str(label)] for label in labels], dtype=np.intp)


def assign_classes(predicted: object) -> NDArray[np.intp]:
    """`pls-da.md` §5: at or above 0.5 is the class coded 1."""
    return (np.asarray(predicted, dtype=np.float64) >= 0.5).astype(np.intp)


def confusion_matrix(observed: object, assigned: object, n_classes: int = 2) -> list[list[int]]:
    """`classification.md` §2: rows observed, columns assigned, in `classes`
    order. Two classes give `pls-da.md` §6's `[[TN, FP], [FN, TP]]`."""
    truth = np.asarray(observed, dtype=np.intp)
    guess = np.asarray(assigned, dtype=np.intp)
    return [
        [int(np.count_nonzero((truth == j) & (guess == k))) for k in range(n_classes)]
        for j in range(n_classes)
    ]


def classification_metrics(confusion: list[list[int]], suffix: str = "") -> dict[str, float]:
    """`classification.md` §3's flat metrics, absent rather than NaN.

    Accuracy for any number of classes; two classes also keep `pls-da.md` §6's
    sensitivity and specificity of the class coded 1.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    metrics: dict[str, float] = {}
    if matrix.sum():
        metrics[f"accuracy{suffix}"] = float(np.trace(matrix) / matrix.sum())
    if matrix.shape == (2, 2):
        (tn, fp), (fn, tp) = confusion
        if tp + fn:
            metrics[f"sensitivity{suffix}"] = tp / (tp + fn)
        if tn + fp:
            metrics[f"specificity{suffix}"] = tn / (tn + fp)
    return metrics


def class_metrics(confusion: list[list[int]]) -> list[dict[str, float]]:
    """`classification.md` §3's per-class table, one class against the rest.

    A metric whose denominator is zero is left out of its class's entry.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    total = int(matrix.sum())
    table: list[dict[str, float]] = []
    for j in range(matrix.shape[0]):
        observed = int(matrix[j].sum())
        assigned = int(matrix[:, j].sum())
        others = total - observed
        rejected = total - observed - assigned + int(matrix[j, j])
        entry: dict[str, float] = {"n": float(observed)}
        if observed:
            entry["sensitivity"] = int(matrix[j, j]) / observed
        if others:
            entry["specificity"] = rejected / others
        if assigned:
            entry["precision"] = int(matrix[j, j]) / assigned
        table.append(entry)
    return table


def _plsda(
    node: PipelineNode,
    spec: PLSDASpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """PLS-DA: two classes are PLS1 on a {0, 1} dummy (#185), three or more are
    PLS2 on a one-hot response assigned by its largest column (#274)."""
    classes, codes = _class_labels(version, node, spec.class_column)
    if len(classes) > 2:
        return _plsda_multiclass(
            node, spec, parent, key, matrix, rows, held_out, fold, classes, codes
        )
    response = codes.astype(np.float64)
    fitted = _fit_regression(
        node,
        "plsda",
        spec.n_components,
        spec.class_column,
        response,
        parent,
        key,
        matrix,
        rows,
        held_out,
        fold,
    )

    observed = assign_classes(fitted.observed)
    predicted_class = assign_classes(fitted.predicted)
    confusion = {"calibration": confusion_matrix(observed, predicted_class)}
    metrics = {**fitted.metrics, **classification_metrics(confusion["calibration"])}

    held_out_class = (
        assign_classes(fitted.held_out_predicted) if held_out.size else np.array([], dtype=np.intp)
    )
    if held_out.size:
        confusion["held_out"] = confusion_matrix(
            assign_classes(fitted.held_out_observed), held_out_class
        )
        metrics.update(classification_metrics(confusion["held_out"], "_p"))
    if fitted.cross_validated_predicted:
        confusion["cross_validation"] = confusion_matrix(
            assign_classes(response), assign_classes(fitted.cross_validated_predicted)
        )
        metrics.update(classification_metrics(confusion["cross_validation"], "_cv"))

    return replace(
        fitted,
        task="classification",
        classes=classes,
        predicted_class=[int(value) for value in predicted_class],
        held_out_predicted_class=[int(value) for value in held_out_class],
        confusion=confusion,
        metrics=metrics,
    )


def _pooled_rmse(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    """RMSE over every element of a one-hot response (`pls-da.md` §7)."""
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def _plsda_multiclass(
    node: PipelineNode,
    spec: PLSDASpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    classes: list[str],
    codes: NDArray[np.intp],
) -> EstimatorResult:
    """`pls-da.md` §3 to §7 for N > 2: PLS2 on the one-hot response, each
    prediction assigned to its largest column, tallied by `classification.md`.

    The fitted model is the one `matrix[rows]` gives, as everywhere; the cross-validated
    assignments and the dummy RMSECV curve are every fold's, each fold fitted
    on its own preprocessed array (#173).
    """
    n_classes = len(classes)
    response = np.eye(n_classes)[codes]
    a = spec.n_components

    def fit(values: NDArray[np.float64], train: NDArray[np.intp]) -> tuple[PLS2, Any, Any]:
        x_mean = values[train].mean(axis=0)
        y_mean = response[train].mean(axis=0)
        try:
            model = PLS2(a).fit(values[train] - x_mean, response[train] - y_mean)
        except (ValueError, RuntimeError) as error:
            raise ExecutorError(f"node {node.id!r} (plsda) failed: {error}", node.id) from error
        return model, x_mean, y_mean

    model, x_mean, y_mean = fit(matrix, rows)
    fitted_a = model.n_components_ or a
    centred = matrix[rows] - x_mean
    predicted = model.predict(centred) + y_mean
    assigned = predicted.argmax(axis=1)
    confusion = {"calibration": confusion_matrix(codes[rows], assigned, n_classes)}
    metrics = {**classification_metrics(confusion["calibration"])}
    metrics["rmsec"] = _pooled_rmse(response[rows], predicted)

    held_x = matrix[held_out] - x_mean
    held_class = np.array([], dtype=np.intp)
    if held_out.size:
        held_predicted = model.predict(held_x) + y_mean
        held_class = held_predicted.argmax(axis=1)
        confusion["held_out"] = confusion_matrix(codes[held_out], held_class, n_classes)
        metrics.update(classification_metrics(confusion["held_out"], "_p"))
        metrics["rmsep"] = _pooled_rmse(response[held_out], held_predicted)

    if parent.folds is not None and len(parent.folds) > 1:
        # One fit per fold and A, as for PLS1 (#174): the first `k` components
        # of an A-component NIPALS fit are the k-component fit.
        curve = np.zeros((fitted_a, *response.shape))
        for one, values in zip(parent.folds, parent.arrays, strict=True):
            fold_model, fold_x, fold_y = fit(values, one.train)
            scores = (values[one.test] - fold_x) @ fold_model._fitted("rotations_")
            loadings = fold_model._fitted("y_loadings_")
            for k in range(fitted_a):
                width = min(k + 1, scores.shape[1])
                curve[k][one.test] = scores[:, :width] @ loadings[:, :width].T + fold_y
        for k in range(fitted_a):
            metrics[f"rmsecv_a{k + 1}"] = _pooled_rmse(response, curve[k])
        metrics["rmsecv"] = metrics[f"rmsecv_a{fitted_a}"]
        confusion["cross_validation"] = confusion_matrix(codes, curve[-1].argmax(axis=1), n_classes)
        metrics.update(classification_metrics(confusion["cross_validation"], "_cv"))

    coefficients = model._fitted("coefficients_")
    return EstimatorResult(
        node_id=node.id,
        key=key,
        task="classification",
        n_components=fitted_a,
        n_samples=int(rows.size),
        n_variables=int(matrix.shape[1]),
        rank=fitted_a,
        fold=fold,
        rows=[int(row) for row in rows],
        scores=_rows(model.x_scores_),
        loadings=_rows(np.asarray(model.x_loadings_).T),
        rotations=_rows(np.asarray(model.rotations_).T),
        eigenvalues=_values(model.score_eigenvalues()),
        explained_variance_ratio=_values(model.explained_variance_ratio("x")),
        cumulative_explained_variance=_values(model.cumulative_explained_variance("x")),
        y_explained_variance_ratio=_values(model.explained_variance_ratio("y")),
        hotelling_t2=_values(model.hotelling_t2()),
        hotelling_t2_limit=float(model.hotelling_t2_limit(ALPHA)),
        spe=_values(model.spe(centred)),
        spe_limit=float(model.spe_limit(ALPHA)),
        target=spec.class_column,
        method="plsda",
        x_mean=_values(x_mean),
        vip=_values(model.vip()),
        coefficient_matrix=_rows(coefficients),
        y_means=_values(y_mean),
        classes=classes,
        predicted_class=[int(value) for value in assigned],
        held_out_predicted_class=[int(value) for value in held_class],
        confusion=confusion,
        metrics=metrics,
        held_out=[int(row) for row in held_out],
        held_out_scores=_rows(model.transform(held_x)) if held_out.size else [],
        held_out_hotelling_t2=_values(model.hotelling_t2(held_x)) if held_out.size else [],
        held_out_spe=_values(model.spe(held_x)) if held_out.size else [],
    )


def _knn(
    node: PipelineNode,
    spec: KNNSpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """`knn.md`: PCA-kNN, tallied by `classification.md`."""
    classes, codes = _class_labels(version, node, spec.class_column, kind="knn")
    n_classes = len(classes)

    def fit(values: NDArray[np.float64], train: NDArray[np.intp]) -> KNN:
        try:
            return KNN(spec.k, spec.n_components).fit(values[train], codes[train], n_classes)
        except ValueError as error:
            raise ExecutorError(f"node {node.id!r} (knn) failed: {error}", node.id) from error

    model = fit(matrix, rows)
    assigned = model.predict(matrix[rows])
    confusion = {"calibration": confusion_matrix(codes[rows], assigned, n_classes)}
    metrics = classification_metrics(confusion["calibration"])
    held_class = np.array([], dtype=np.intp)
    if held_out.size:
        held_class = model.predict(matrix[held_out])
        confusion["held_out"] = confusion_matrix(codes[held_out], held_class, n_classes)
        metrics.update(classification_metrics(confusion["held_out"], "_p"))
    if parent.folds is not None and len(parent.folds) > 1:
        cv = np.empty(matrix.shape[0], dtype=np.intp)
        for one, values in zip(parent.folds, parent.arrays, strict=True):
            cv[one.test] = fit(values, one.train).predict(values[one.test])
        confusion["cross_validation"] = confusion_matrix(codes, cv, n_classes)
        metrics.update(classification_metrics(confusion["cross_validation"], "_cv"))

    pca = model.pca_
    x_mean = np.asarray(model.x_mean_)
    centred = matrix[rows] - x_mean
    held_x = matrix[held_out] - x_mean
    return EstimatorResult(
        node_id=node.id,
        key=key,
        task="classification",
        n_components=spec.n_components,
        n_samples=int(rows.size),
        n_variables=int(matrix.shape[1]),
        rank=spec.n_components,
        fold=fold,
        rows=[int(row) for row in rows],
        # The PCA front's (knn.md section 4); the calibration scores are also
        # the neighbours, with their classes in `training_classes`.
        scores=_rows(pca.transform(centred)),
        loadings=_rows(np.asarray(pca.loadings_).T),
        rotations=_rows(np.asarray(pca.loadings_).T),
        eigenvalues=_values(np.asarray(pca.eigenvalues_)[: spec.n_components]),
        explained_variance_ratio=_values(pca.explained_variance_ratio()),
        cumulative_explained_variance=_values(pca.cumulative_explained_variance()),
        hotelling_t2=_values(pca.hotelling_t2(centred)),
        hotelling_t2_limit=float(pca.hotelling_t2_limit(ALPHA)),
        spe=_values(pca.spe(centred)),
        spe_limit=float(pca.spe_limit(ALPHA)),
        spe_limit_caveat=pca.spe_limit_caveat(),
        target=spec.class_column,
        method="knn",
        x_mean=_values(x_mean),
        classes=classes,
        training_classes=[int(value) for value in codes[rows]],
        k=spec.k,
        predicted_class=[int(value) for value in assigned],
        held_out_predicted_class=[int(value) for value in held_class],
        confusion=confusion,
        metrics=metrics,
        held_out=[int(row) for row in held_out],
        held_out_scores=_rows(pca.transform(held_x)) if held_out.size else [],
        held_out_hotelling_t2=_values(pca.hotelling_t2(held_x)) if held_out.size else [],
        held_out_spe=_values(pca.spe(held_x)) if held_out.size else [],
    )


def _lda(
    node: PipelineNode,
    spec: LDASpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """`lda.md`: PCA-LDA, tallied by `classification.md`."""
    classes, codes = _class_labels(version, node, spec.class_column, kind="lda")
    n_classes = len(classes)

    def fit(values: NDArray[np.float64], train: NDArray[np.intp]) -> LDA:
        try:
            return LDA(spec.n_components).fit(values[train], codes[train], n_classes)
        except ValueError as error:
            named = str(error)
            for k, name in enumerate(classes):
                named = named.replace(f"class {k} ", f"class {name!r} ")
            raise ExecutorError(f"node {node.id!r} (lda) failed: {named}", node.id) from error

    model = fit(matrix, rows)
    assigned = model.predict(matrix[rows])
    confusion = {"calibration": confusion_matrix(codes[rows], assigned, n_classes)}
    metrics = classification_metrics(confusion["calibration"])
    held_class = np.array([], dtype=np.intp)
    if held_out.size:
        held_class = model.predict(matrix[held_out])
        confusion["held_out"] = confusion_matrix(codes[held_out], held_class, n_classes)
        metrics.update(classification_metrics(confusion["held_out"], "_p"))
    if parent.folds is not None and len(parent.folds) > 1:
        cv = np.empty(matrix.shape[0], dtype=np.intp)
        for one, values in zip(parent.folds, parent.arrays, strict=True):
            cv[one.test] = fit(values, one.train).predict(values[one.test])
        confusion["cross_validation"] = confusion_matrix(codes, cv, n_classes)
        metrics.update(classification_metrics(confusion["cross_validation"], "_cv"))

    pca = model.pca_
    x_mean = np.asarray(model.x_mean_)
    centred = matrix[rows] - x_mean
    held_x = matrix[held_out] - x_mean
    eigenvalues = np.asarray(pca.eigenvalues_)[: spec.n_components]
    return EstimatorResult(
        node_id=node.id,
        key=key,
        task="classification",
        n_components=spec.n_components,
        n_samples=int(rows.size),
        n_variables=int(matrix.shape[1]),
        rank=spec.n_components,
        fold=fold,
        rows=[int(row) for row in rows],
        # The PCA front's (lda.md section 5): what the scores and diagnostics
        # panels draw. The discriminant itself is the coefficient matrix.
        scores=_rows(pca.transform(centred)),
        loadings=_rows(np.asarray(pca.loadings_).T),
        rotations=_rows(np.asarray(pca.loadings_).T),
        eigenvalues=_values(eigenvalues),
        explained_variance_ratio=_values(pca.explained_variance_ratio()),
        cumulative_explained_variance=_values(pca.cumulative_explained_variance()),
        hotelling_t2=_values(pca.hotelling_t2(centred)),
        hotelling_t2_limit=float(pca.hotelling_t2_limit(ALPHA)),
        spe=_values(pca.spe(centred)),
        spe_limit=float(pca.spe_limit(ALPHA)),
        spe_limit_caveat=pca.spe_limit_caveat(),
        target=spec.class_column,
        method="lda",
        x_mean=_values(x_mean),
        coefficient_matrix=_rows(np.asarray(model.coefficients_)),
        y_means=_values(np.asarray(model.intercepts_)),
        classes=classes,
        predicted_class=[int(value) for value in assigned],
        held_out_predicted_class=[int(value) for value in held_class],
        confusion=confusion,
        metrics=metrics,
        held_out=[int(row) for row in held_out],
        held_out_scores=_rows(pca.transform(held_x)) if held_out.size else [],
        held_out_hotelling_t2=_values(pca.hotelling_t2(held_x)) if held_out.size else [],
        held_out_spe=_values(pca.spe(held_x)) if held_out.size else [],
    )


def _simca(
    node: PipelineNode,
    spec: SIMCASpec,
    parent: _State,
    key: str,
    matrix: NDArray[np.float64],
    rows: NDArray[np.intp],
    held_out: NDArray[np.intp],
    fold: int | None,
    version: DatasetVersion,
) -> EstimatorResult:
    """`simca.md`: one PCA per class, decisions per set."""
    classes, codes = _class_labels(version, node, spec.class_column, kind="simca")
    n_classes = len(classes)

    def fit(values: NDArray[np.float64], train: NDArray[np.intp]) -> SIMCA:
        try:
            return SIMCA(spec.n_components, ALPHA).fit(values[train], codes[train], n_classes)
        except ValueError as error:
            named = str(error)
            for k, name in enumerate(classes):
                named = named.replace(f"class {k} ", f"class {name!r} ")
            raise ExecutorError(f"node {node.id!r} (simca) failed: {named}", node.id) from error

    def decided(
        model: SIMCA, values: NDArray[np.float64], picked: NDArray[np.intp]
    ) -> dict[str, Any]:
        distances = model.distances(values[picked])
        table, none = acceptance_table(codes[picked], distances <= 1.0)
        sizes = [int(np.count_nonzero(codes[picked] == k)) for k in range(n_classes)]
        return {
            "rows": [int(row) for row in picked],
            "distances": _rows(distances),
            "table": table,
            "none": none,
            "sizes": sizes,
        }

    model = fit(matrix, rows)
    sets = {"calibration": decided(model, matrix, rows)}
    metrics = simca_metrics(sets["calibration"]["table"], sets["calibration"]["sizes"])
    if held_out.size:
        sets["held_out"] = decided(model, matrix, held_out)
        metrics.update(simca_metrics(sets["held_out"]["table"], sets["held_out"]["sizes"], "_p"))
    if parent.folds is not None and len(parent.folds) > 1:
        distances = np.zeros((matrix.shape[0], n_classes))
        for one, values in zip(parent.folds, parent.arrays, strict=True):
            distances[one.test] = fit(values, one.train).distances(values[one.test])
        everyone = np.arange(matrix.shape[0], dtype=np.intp)
        table, none = acceptance_table(codes, distances <= 1.0)
        sizes = [int(np.count_nonzero(codes == k)) for k in range(n_classes)]
        sets["cross_validation"] = {
            "rows": [int(row) for row in everyone],
            "distances": _rows(distances),
            "table": table,
            "none": none,
            "sizes": sizes,
        }
        metrics.update(simca_metrics(table, sizes, "_cv"))

    fitted = model.models_ or []
    return EstimatorResult(
        node_id=node.id,
        key=key,
        task="classification",
        n_components=spec.n_components,
        n_samples=int(rows.size),
        n_variables=int(matrix.shape[1]),
        rank=spec.n_components,
        fold=fold,
        rows=[int(row) for row in rows],
        scores=[],
        loadings=[],
        eigenvalues=[],
        explained_variance_ratio=[],
        cumulative_explained_variance=[],
        hotelling_t2=[],
        hotelling_t2_limit=0.0,
        spe=[],
        spe_limit=0.0,
        target=spec.class_column,
        method="simca",
        classes=classes,
        held_out=[int(row) for row in held_out],
        metrics=metrics,
        simca={
            "models": [
                {
                    "class": name,
                    "n_samples": one.n_samples,
                    "t2_limit": one.t2_limit,
                    "q_limit": one.q_limit,
                    "spe_limit_caveat": one.pca.spe_limit_caveat(),
                    "mean": _values(one.mean),
                    "loadings": _rows(np.asarray(one.pca.loadings_).T),
                    "eigenvalues": _values(np.asarray(one.pca.eigenvalues_)[: spec.n_components]),
                }
                for name, one in zip(classes, fitted, strict=True)
            ],
            "sets": sets,
        },
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
    index = read_cache_index(path)
    paths = index.get(keys[node_id])
    if not paths:
        return None

    # The assembly stored at run time (#176), one read. An index written
    # before it was kept falls back to assembling from the fold arrays, which
    # is the same array by construction.
    stored = index.get(f"{keys[node_id]}#display")
    if stored:
        try:
            return read_array(path, stored[0])
        except ProjectError:
            pass
    by_id = {node.id: node for node in pipeline.nodes}
    folds = governing_folds(node_id, by_id, version)
    state = _from_cache(path, paths, folds)
    return None if state is None else state.display


def stored_fitted_matrix(
    directory: str | Path, pipeline: Pipeline, version: DatasetVersion, node_id: NodeId
) -> NDArray[np.float64] | None:
    """The array an estimator was fitted from, read back.

    Above a split, its input's one array. Below one, its input's all-sample
    array (#330): every row through parameters fitted on every row, which is
    what `_estimator` fitted the final model on. A contribution plot (#186)
    needs the sample's row from *this* array, not from the display array a
    spectra plot draws, whose rows below a split come from whichever fold held
    each one out. `None` when the node is not an estimator or its input has
    not been run.
    """
    path = Path(directory)
    by_id = {node.id: node for node in pipeline.nodes}
    node = by_id.get(node_id)
    if node is None or node.type != "estimator":
        return None
    parent = node.inputs[0]
    key = node_keys(pipeline, version)[parent]
    index = read_cache_index(path)
    folds = governing_folds(parent, by_id, version)
    paths = index.get(f"{key}#all") if folds is not None else index.get(key)
    if not paths:
        return None
    try:
        return read_array(path, paths[0])
    except ProjectError:
        return None


def node_axis(pipeline: Pipeline, node_id: NodeId, version: DatasetVersion) -> NDArray[np.float64]:
    """The axis a node's output is on, which is not always the dataset's.

    `RangeSelect` and `SelectVariables` (#280) change the variable count, so a node
    under one is on a shorter axis than the `DatasetVersion` records and every
    payload that pairs the two has to know it, and so does the executor: each
    step is built with its input's axis, not the dataset's (#312). No per-node
    axis is stored - a second thing beside the cached arrays would have to
    stay consistent with them - so this derives it instead, from the recipe.

    That derivation is free of the executor's guarantees precisely because it
    is a pure function of the pipeline: it reads no array, writes nothing, and
    cannot move a content hash or invalidate a cache entry.

    Every non-source node holds exactly one input, so the ancestry is a chain
    rather than a tree and the selections apply in order down it. The mask is
    taken from the step's `Selection` transformer rather than restated here, so the
    interval's meaning — inclusive bounds, either axis direction, an empty
    selection refused — is stated once.
    """
    by_id = {node.id: node for node in pipeline.nodes}
    chain: list[PipelineNode] = []
    current = node_id
    while True:
        node = by_id[current]
        chain.append(node)
        if not node.inputs:
            break
        current = node.inputs[0]

    axis = np.asarray(version.axis.values, dtype=np.float64)
    for node in reversed(chain):
        if node.type != "preprocess" or not isinstance(node.step, RangeSelect | SelectVariables):
            continue
        transformer = preprocessing.from_spec(node.step, axis=axis)
        assert isinstance(transformer, preprocessing.Selection)
        # Fitting a range selection needs the axis and the variable count, not
        # the data: `_fit` reads `X.shape[1]` and nothing else. One empty row
        # supplies the width without loading an array this function has no
        # reason to read.
        transformer.fit(np.zeros((1, axis.size)))
        axis = transformer.selected_axis()
    return axis


def stored_fold_matrices(
    directory: str | Path, pipeline: Pipeline, version: DatasetVersion, node_id: NodeId
) -> tuple[list[NDArray[np.float64]], list[Fold]] | None:
    """An estimator's input as every fold saw it, and the folds (#282).

    Below a split each preprocessing node is refitted per training fold, so a
    method that cross-validates on the estimator's input - iPLS - needs fold
    `i`'s own matrix for fold `i`. `None` when the node is not an estimator,
    its input has not been run, or there is no split above it.
    """
    path = Path(directory)
    by_id = {node.id: node for node in pipeline.nodes}
    node = by_id.get(node_id)
    if node is None or node.type != "estimator":
        return None
    parent = node.inputs[0]
    folds = governing_folds(parent, by_id, version)
    paths = read_cache_index(path).get(node_keys(pipeline, version)[parent])
    if not paths or folds is None:
        return None
    state = _from_cache(path, paths, folds)
    return None if state is None else (list(state.arrays), folds)


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


def permutation_test_for(
    directory: str | Path,
    pipeline: Pipeline,
    version: DatasetVersion,
    node_id: NodeId,
    n_permutations: int,
    *,
    seed: int = 0,
    on_progress: Callable[[int], None] | None = None,
) -> PermutationResult:
    """`metrics-and-validation.md` §14 (#333): one estimator's cross-validation,
    rerun with its response or class labels permuted.

    The folds and the per-fold matrices are the stored ones, so every
    permutation is scored on exactly the split and the preprocessing the
    observed model was. The score is RMSECV for a PLS or PCR and the
    cross-validated accuracy for a PLS-DA, LDA or kNN.
    """
    path = Path(directory)
    by_id = {node.id: node for node in pipeline.nodes}
    node = by_id.get(node_id)
    if node is None or node.type != "estimator" or not isinstance(node.spec, _PERMUTABLE):
        raise ExecutorError(
            f"node {node_id!r} is not a PLS, PCR, PLS-DA, LDA or kNN, so it has no "
            "cross-validated score to permute.",
            node_id,
        )
    stored = stored_fold_matrices(path, pipeline, version, node_id)
    if stored is None or len(stored[1]) < 2:
        raise ExecutorError(
            f"node {node_id!r} has no stored cross-validation to rerun: it needs a K-fold or "
            "leave-one-out split above it with at least two folds, and a completed run.",
            node_id,
        )
    arrays, folds = stored
    parent = _State(arrays=arrays, folds=folds)
    key = node_keys(pipeline, version)[node_id]
    regression = isinstance(node.spec, PLSRegressionSpec | PCRSpec)
    column = node.spec.target if regression else node.spec.class_column  # type: ignore[union-attr]
    values = list(
        version.targets[column] if regression else version.metadata_columns.get(column, [])
    )
    if len(values) != version.n_samples:
        raise ExecutorError(f"node {node_id!r} has no column {column!r} to permute.", node_id)
    metric = "rmsecv" if regression else "accuracy_cv"

    def score(order: NDArray[np.intp]) -> float:
        permuted = [values[i] for i in order]
        field_name = "targets" if regression else "metadata_columns"
        columns = {**getattr(version, field_name), column: permuted}
        shuffled = version.model_copy(update={field_name: columns})
        # The model fitted here is fold zero's and discarded: only the CV
        # score, every fold's, is read.
        fitted = _fitted(node, parent, key, arrays[0], folds[0].train, _NONE, None, shuffled)
        return fitted.metrics[metric]

    return permutation_test(
        score,
        version.n_samples,
        n_permutations,
        seed=seed,
        greater_is_better=not regression,
        on_progress=on_progress,
    )


#: The estimators §14 permutes: those with a cross-validated score.
_PERMUTABLE = (PLSRegressionSpec, PCRSpec, PLSDASpec, LDASpec, KNNSpec)


def governing_split(node_id: NodeId, by_id: dict[NodeId, PipelineNode]) -> PipelineNode | None:
    """The split node above this one, or `None` if it sits above every split."""
    current = by_id[node_id]
    while True:
        if current.type == "split":
            return current
        if not current.inputs:
            return None
        current = by_id[current.inputs[0]]


def governing_folds(
    node_id: NodeId, by_id: dict[NodeId, PipelineNode], version: DatasetVersion
) -> list[Fold] | None:
    """The split above a node, resolved again from its spec.

    Recomputed rather than stored: a `SplitSpec` and the dataset version - its
    `n`, and the column a stratified split reads - determine the folds entirely
    (`metrics-and-validation.md` §8), so deriving them is cheaper than keeping a
    second copy that can disagree with the recipe.
    """
    split = governing_split(node_id, by_id)
    return None if split is None else _folds_for(split, None, version)


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
