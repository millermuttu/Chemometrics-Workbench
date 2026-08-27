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
canvas coordinates live in `pipeline_state.json` outside the model entirely.
A node cannot be moved into a cache miss.

The index from key to stored path is `cache.json` in the project directory.
The arrays themselves are content-addressed by the store, so two nodes that
compute the same values share one file.

## What is not here

**Estimators.** An `EstimatorNode` is walked and reported in
`Run.pending_estimators`, not fitted: what a fitted estimator stores, and with
which diagnostics and limits, is #87's subject, and guessing at it here would
be the invented contract Phase 1.1 existed to avoid.

**Jobs.** `execute` runs to completion in the calling thread. Progress,
cancellation and failure reporting are #85, which wraps this.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench import preprocessing
from chemometrics_workbench.models import (
    DatasetVersion,
    KFoldSplit,
    LeaveOneOut,
    NodeId,
    Pipeline,
    PipelineNode,
    ResolvedSplit,
)
from chemometrics_workbench.project import ProjectError, read_array, write_array, write_json
from chemometrics_workbench.validation import Fold, k_fold, leave_one_out, validate_partition

__all__ = [
    "CACHE_FILE",
    "ExecutorError",
    "NodeOutput",
    "Run",
    "execute",
    "node_keys",
]

CACHE_FILE = "cache.json"


class ExecutorError(Exception):
    """A pipeline could not be executed, naming the node it stopped at.

    One exception rather than a hierarchy, for the reason `ProjectError` is
    one: the only caller that distinguishes cases is the HTTP layer, and it
    turns all of them into the same error body. What the caller needs is the
    node id, and every message carries it.
    """


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
    pending_estimators: list[NodeId]

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
) -> Run:
    """Run every preprocessing node in `pipeline` against `version`'s array.

    The dataset is read from the project directory, not passed in: the array
    store is the only place a dataset's values live, and a caller handing in
    its own matrix could quietly execute a recipe against something other than
    the version the experiment records.
    """
    path = Path(directory)
    axis = np.asarray(version.axis.values, dtype=np.float64)
    keys = node_keys(pipeline, version)
    index = _load_index(path) if use_cache else {}

    states: dict[NodeId, _State] = {}
    outputs: dict[NodeId, NodeOutput] = {}
    splits: list[ResolvedSplit] = []
    pending: list[NodeId] = []
    index_changed = False

    for node in _topological(pipeline):
        if node.type == "estimator":
            pending.append(node.id)
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

    if index_changed:
        _save_index(path, index)

    return Run(
        pipeline_id=str(pipeline.pipeline_id),
        outputs=outputs,
        displays={nid: state.display for nid, state in states.items()},
        resolved_splits=splits,
        pending_estimators=pending,
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
            "split node with no way to say which outer fold it belonged to."
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
            "K-fold and an external set are not."
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
            raise ExecutorError(f"node {node.id!r} could not read the dataset: {error}") from error
        if values.shape != (version.n_samples, version.n_variables):
            raise ExecutorError(
                f"node {node.id!r} read a {values.shape[0]}x{values.shape[1]} array where "
                f"the version records {version.n_samples}x{version.n_variables}."
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
        raise ExecutorError(f"node {node.id!r} has type {node.type!r}, which is not executable.")

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
            f"node {node.id!r} ({node.step.kind}) failed{where}: {error}"
        ) from error


def _resolved(node_id: NodeId, folds: list[Fold] | None) -> ResolvedSplit:
    """The index sets a split produced, stored so the run can be repeated (§10)."""
    assert folds is not None, "a split node always resolves its folds"
    return ResolvedSplit(
        node_id=node_id,
        train_indices=[fold.train.tolist() for fold in folds],
        test_indices=[fold.test.tolist() for fold in folds],
    )


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


def _load_index(directory: Path) -> dict[str, list[str]]:
    """The key-to-paths index, or an empty one if it is absent or unreadable.

    A corrupt index costs a recomputation, which is the cheapest possible
    failure here and the reason it is not an error.
    """
    try:
        document = json.loads((directory / CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    return {
        key: [str(path) for path in value]
        for key, value in document.items()
        if isinstance(key, str) and isinstance(value, list)
    }


def _save_index(directory: Path, index: dict[str, list[str]]) -> None:
    # ponytail: the index only grows - an edited-away node's entry stays, and
    # so does its array. Pruning wants a mark-and-sweep against the pipelines
    # in the project, which is #89's business once there is a list of them.
    write_json(directory / CACHE_FILE, index)
