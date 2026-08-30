"""The HTTP surface: the real handlers, growing one issue at a time.

Phase 1.1 built the frontend against a stub server so that 1.2 could replace
handlers behind unchanged URLs rather than integrate in one moment. This module
is where the replacements live, and in #89 it becomes the whole server.

**Not one URL changes.** That was the point of building the frontend against
these paths from its first commit.

#99 recorded why these could not be swapped in one at a time: the project the
frontend lists, the dataset it opens and the pipeline it runs are one chain, so
the swap is one cut. This is that cut.

## What is here now

The import endpoints (#81) and the project and dataset reads they need to be
reachable at all: a preview cannot be confirmed if the dataset it produces has
nowhere to appear.

- `GET  /api/projects`, `GET /api/projects/{id}` — the open project
- `GET  /api/projects/{id}/datasets` — read from `datasets.json` on disk
- `POST /api/import/preview` — the reader's detection, nothing committed
- `POST /api/import` — commits with the user's corrections applied, and starts
  a pipeline on the dataset if the project has none
- `GET  /api/pipelines/{id}` and `/state`, `POST /api/pipelines/{id}/validate`
- `GET  /api/experiments/{id}`, `POST /api/experiments/{id}/run`
- `GET  /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`
- `GET  /api/spectra/{node_id}`, `GET /api/results/{node_id}`
- `GET  /api/schema/steps`, `POST /api/steps/validate`

`current` is a real id here: the frontend has asked for `pipelines/current` and
`experiments/current` since its first commit, and a project holds one of each
until there is a database to hold more.

`results_payload` (#87) renders an estimator result for `results/{node_id}`,
`spectra_payload` (#86) renders `spectra/{node_id}`, and `validation_payload`
(#84) renders `pipelines/{id}/validate`. Neither
endpoint is served here yet: both take a pipeline, and there is nowhere to keep
one until #89's pipeline store — the same cut #99 describes. The stub calls
`validation_payload` for the one pipeline it has, so that response is computed
rather than constant.

## The open project

There is no project browser yet and no database to list projects from, so the
server opens exactly one: `CHEMOMETRICS_PROJECT` if it is set, else
`<config dir>/projects/default`, created on first use. `known_projects()` from
#77 keeps the registry up to date, which is what a project browser will read
when #89 or 1.3 adds one.

## Uploads

A file arrives as a multipart upload and is written to a temporary file, whose
suffix is the original's because `reader_for` chooses by suffix. The readers
take a path, so the temporary file is what they are given, and it is deleted
whether or not the read succeeded.

`MAX_UPLOAD_BYTES` bounds it. §4.3 calls localhost a trust boundary, not a
private room, and an unbounded upload is a way to fill the user's disk from a
page in their own browser.

The file is uploaded once to preview and once to commit. On a loopback socket
that is a memory copy, and staging the first upload to serve the second would
mean a lifetime to manage — when it expires, what happens on a restart, what
happens when the user previews ten files and imports none.

## Errors

Every failure has a body: `{"error": {"code", "message", "detail"}}`, which is
what `stub/fixtures/error.json` documents and every screen renders. A
`ReaderError` or a `ProjectError` becomes one, with its own sentence intact —
§6's rule that an unreadable file produces a specific diagnostic rather than a
stack trace.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from numpy.typing import NDArray
from pydantic import BaseModel, TypeAdapter, ValidationError

from chemometrics_workbench import readers
from chemometrics_workbench.checks import check_pipeline
from chemometrics_workbench.executor import EstimatorResult, stored
from chemometrics_workbench.executor import stored_display as _stored_display
from chemometrics_workbench.executor import stored_result as _stored_result
from chemometrics_workbench.jobs import Job, Jobs, submit_run
from chemometrics_workbench.models import (
    Dataset,
    DatasetVersion,
    NodeId,
    Pipeline,
    PipelineNode,
    PreprocessStep,
    Project,
    SourceNode,
)
from chemometrics_workbench.project import (
    DatasetEntry,
    ProjectError,
    add_dataset,
    config_dir,
    create_project,
    is_project,
    open_project,
    read_datasets,
    read_experiment,
    read_layout,
    read_pipeline,
    write_array,
    write_layout,
    write_pipeline,
)

__all__ = [
    "MAX_POINTS",
    "MAX_TRACES",
    "MAX_UPLOAD_BYTES",
    "open_project_directory",
    "results_payload",
    "router",
    "spectra_payload",
    "validation_payload",
]

#: How many individually drawn traces a plot carries before the remainder
#: becomes a density band. §13: 20,000 spectra is far past what Plotly draws,
#: and 60 is what the artboards were designed against.
MAX_TRACES = 60

#: How many x positions a decimated trace carries. §13 again: 4,000 points per
#: trace times 20,000 traces is 80 million, so the axis is bucketed before
#: anything is drawn. Kept even, because min/max decimation emits two points
#: per bucket.
MAX_POINTS = 1000

#: An upload past this is refused before it is written. Above §13's envelope —
#: 20,000 x 4,000 float32 is about 320 MB — because a text file holding that
#: many values is several times larger than the array it becomes, and refusing
#: a file the application could have read would be the wrong failure.
MAX_UPLOAD_BYTES = 2_000_000_000

#: Read in blocks, so a refused upload is refused without being held in memory.
_BLOCK = 1 << 20

router = APIRouter()

#: The one job table this process has. Jobs do not survive a restart, which is
#: Phase 1.3's; see `jobs.py`.
JOBS = Jobs()

#: Where the canvas puts a node it has never seen. Left to right by depth, in
#: the artboards' spacing, so a generated graph reads like a drawn one until
#: the user moves something.
_LAYOUT_STEP_X = 170
_LAYOUT_STEP_Y = 130


#: Serialises the check-then-create in `open_project_directory`; see there.
_CREATE_LOCK = threading.Lock()


def open_project_directory() -> Path:
    """The one project this server has open, created on first use.

    Returning a path rather than a `Project` because most callers want to read
    or write inside the directory, and the record is one `open_project` away.
    """
    configured = os.environ.get("CHEMOMETRICS_PROJECT")
    directory = Path(configured) if configured else config_dir() / "projects" / "default"
    # Check-then-act, under a lock. A page load asks six questions at once and
    # every one of them opens the project, so on a directory that is not a
    # project yet every one of them tries to create it. `create_project` makes
    # `arrays/` before it writes `project.json`, so the losers found a
    # directory that was neither empty nor yet a project and refused - turning
    # the first load of a new project into a 500, intermittently.
    #
    # A lock rather than a retry because the two cases a retry has to tell
    # apart - a winner halfway through and a directory that really is not a
    # project - look identical from outside, and only waiting distinguishes
    # them. This lock is about *this process*: it stops six requests on one
    # page load racing each other. Two *processes* over one directory is the
    # database's to arbitrate and it does - WAL lets a reader read while a
    # write is in flight, writers take turns, and one that cannot get its turn
    # inside the busy timeout is told which project is busy (#123).
    with _CREATE_LOCK:
        if not is_project(directory):
            create_project(directory, name=directory.name, description="")
    return directory


def _fail(status: int, code: str, message: str, **detail: Any) -> HTTPException:
    """The documented error body, as an exception the router can raise."""
    return HTTPException(
        status_code=status, detail={"code": code, "message": message, "detail": detail}
    )


def _project() -> tuple[Path, Project]:
    directory = open_project_directory()
    try:
        return directory, open_project(directory)
    except ProjectError as error:
        raise _fail(500, "project_unavailable", str(error), directory=str(directory)) from error


def _entry_json(entry: DatasetEntry) -> Any:
    return json.loads(entry.model_dump_json())


# --- Projects and datasets ------------------------------------------------
#
# ## Pagination is deferred, and this is the reason (#89)
#
# Phase 1.1 marked pagination a GUESS: the list endpoints return a bare JSON
# array, with no envelope to hang `next` or `total` off. #89 keeps that, and
# the decision is recorded here rather than left to be rediscovered.
#
# There is nothing to page. A project holds one pipeline and, until SQLite
# arrives in Phase 1.3, its datasets are a JSON file read whole - paging a list
# that is already entirely in memory adds a cursor the client must thread
# through and buys nothing. `GET /projects` returns the single open project for
# the same reason: the server has one.
#
# What would change the answer is Phase 1.3's database and more than one
# project per server, and by then the store can page properly instead of
# slicing a list it just parsed. Adding the envelope now would fix the shape of
# an answer before knowing the question - which is what Phase 1.1 existed to
# avoid, and why these endpoints were built against a published contract.


@router.get("/projects")
def list_projects() -> Any:
    _, project = _project()
    return [json.loads(project.model_dump_json())]


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> Any:
    _, project = _project()
    if str(project.project_id) != project_id:
        raise _fail(404, "not_found", f"no project {project_id} is open.", project_id=project_id)
    return json.loads(project.model_dump_json())


@router.get("/projects/{project_id}/datasets")
def list_datasets(project_id: str) -> Any:
    directory, project = _project()
    if str(project.project_id) != project_id:
        raise _fail(404, "not_found", f"no project {project_id} is open.", project_id=project_id)
    try:
        return [_entry_json(entry) for entry in read_datasets(directory)]
    except ProjectError as error:
        raise _fail(500, "project_unavailable", str(error)) from error


# --- Import ---------------------------------------------------------------


@router.post("/import/preview")
async def import_preview(file: Annotated[UploadFile, File()]) -> Any:
    """What the reader found, with alternatives. Nothing is committed."""
    async with _uploaded(file) as path:
        try:
            return readers.preview(path)
        except readers.ReaderError as error:
            raise _reader_failed(file, error) from error


@router.post("/import")
async def import_dataset(
    file: Annotated[UploadFile, File()],
    corrections: Annotated[str, Form()] = "{}",
    name: Annotated[str | None, Form()] = None,
) -> Any:
    """Read the file with the user's corrections applied and record what it became.

    The array is written through #77's store, so it is float32 on disk and the
    `content_hash` falls out of storing it. The `DatasetVersion` references
    that path; the index never holds values.
    """
    directory, project = _project()
    corrected = _corrections(corrections)

    async with _uploaded(file) as path:
        try:
            imported = readers.read(path, corrected)
        except readers.ReaderError as error:
            raise _reader_failed(file, error) from error

        try:
            array_path, content_hash = write_array(directory, imported.values)
        except ProjectError as error:
            raise _fail(500, "project_unavailable", str(error)) from error

    dataset = Dataset(
        project_id=project.project_id,
        name=name or Path(imported.source.filename).stem,
        description="",
    )
    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash=content_hash,
        n_samples=int(imported.values.shape[0]),
        n_variables=int(imported.values.shape[1]),
        axis=imported.axis,
        sample_ids=list(imported.sample_ids),
        targets={key: list(values) for key, values in imported.targets.items()},
        metadata_columns={key: list(values) for key, values in imported.metadata_columns.items()},
        source=imported.source,
        array_path=array_path,
    )

    try:
        entry = add_dataset(directory, dataset, version)
        if read_pipeline(directory) is None:
            start_pipeline(directory, project, version)
    except ProjectError as error:
        raise _fail(500, "project_unavailable", str(error)) from error
    return _entry_json(entry)


def _corrections(raw: str) -> dict[str, str]:
    """The corrections field, which arrives as JSON in a form part."""
    try:
        parsed = json.loads(raw or "{}")
    except ValueError as error:
        raise _fail(422, "bad_request", f"corrections is not JSON: {error}") from error
    if not isinstance(parsed, dict) or not all(isinstance(v, str) for v in parsed.values()):
        raise _fail(422, "bad_request", "corrections must be an object of strings.")
    return {str(key): str(value) for key, value in parsed.items()}


def _reader_failed(file: UploadFile, error: readers.ReaderError) -> HTTPException:
    """A file the reader refuses is a 422 with the reader's own sentence.

    Not a 500: nothing went wrong with the server. The message is the one the
    reader wrote, because it names what is wrong with the file and what to do
    about it, and replacing it with "could not read file" would throw that away.
    """
    return _fail(422, "reader_failed", str(error), file=file.filename or "upload")


class _uploaded:
    """An upload as a path on disk, removed afterwards whichever way it ends.

    The readers take a path — they read files in blocks, seek inside workbooks
    and reopen with a different encoding — so an upload has to become a real
    file before one can look at it.

    It keeps the user's own filename inside a temporary directory, rather than
    a temporary name with the right suffix, because that name is what the
    reader records in `SourceFile` and what the import screen shows. A dataset
    whose provenance says `tmpw3k1p8.csv` is a dataset with no provenance.

    The name is taken apart first: only its final component is used, so an
    upload calling itself `../../project.json` writes a file called
    `project.json` in a temporary directory and nothing else. §4.3 calls
    localhost a trust boundary.
    """

    def __init__(self, file: UploadFile) -> None:
        self._file = file
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> Path:
        name = Path(self._file.filename or "").name
        if not Path(name).suffix:
            raise _fail(
                422,
                "bad_request",
                f"{name or 'the upload'} has no file extension, so no reader claims it. "
                "The readers choose by suffix rather than by guessing at content.",
                file=name,
            )

        self._directory = tempfile.TemporaryDirectory(prefix="chemometrics-import-")
        path = Path(self._directory.name) / name
        written = 0
        try:
            with path.open("wb") as handle:
                while block := await self._file.read(_BLOCK):
                    written += len(block)
                    if written > MAX_UPLOAD_BYTES:
                        raise _fail(
                            413,
                            "upload_too_large",
                            f"{name} is larger than the "
                            f"{MAX_UPLOAD_BYTES // 1_000_000} MB an import accepts.",
                            file=name,
                            limit_bytes=MAX_UPLOAD_BYTES,
                        )
                    handle.write(block)
        except BaseException:
            self._cleanup()
            raise
        return path

    async def __aexit__(self, *_: object) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None


# --- Results --------------------------------------------------------------


def results_payload(result: EstimatorResult, version: DatasetVersion) -> dict[str, Any]:
    """What `results/{node_id}` serves: a fitted estimator, ready to draw.

    The kernel's numbers come from the executor unrounded; the sample ids and
    the variable axis come from the `DatasetVersion`, because a model does not
    know what its rows and columns were called. The shape is the one
    `stub/fixtures/pca.json` publishes and the analysis screen already renders.

    A split branch adds `validation`: the held-out rows of the fitted fold,
    projected through the model that never saw them. It is an addition rather
    than a change — the calibration arrays keep the lengths the fixture has, so
    a screen that ignores the new key renders exactly what it rendered before.
    """
    payload: dict[str, Any] = {
        "node_id": result.node_id,
        "task": result.task,
        "n_components": result.n_components,
        "n_samples": result.n_samples,
        "n_variables": result.n_variables,
        "rank": result.rank,
        "samples": _samples(result.rows, version),
        "scores": result.scores,
        "loadings": {
            "axis": {
                "kind": version.axis.kind,
                "unit": version.axis.unit,
                "values": list(version.axis.values),
            },
            "components": result.loadings,
        },
        "eigenvalues": result.eigenvalues,
        "explained_variance_ratio": result.explained_variance_ratio,
        "cumulative_explained_variance": result.cumulative_explained_variance,
        "diagnostics": {
            "hotelling_t2": result.hotelling_t2,
            "hotelling_t2_limit": result.hotelling_t2_limit,
            "spe": result.spe,
            "spe_limit": result.spe_limit,
            "alpha": result.alpha,
        },
    }
    if result.fold is not None:
        payload["validation"] = {
            "fold": result.fold,
            "samples": _samples(result.held_out, version),
            "scores": result.held_out_scores,
            "hotelling_t2": result.held_out_hotelling_t2,
            "spe": result.held_out_spe,
        }
    return payload


def _samples(rows: list[int], version: DatasetVersion) -> list[dict[str, Any]]:
    """Row indices with the ids the dataset gave them, or none if it gave none."""
    ids = version.sample_ids
    return [
        {"index": row, "sample_id": ids[row] if row < len(ids) else f"row {row}"} for row in rows
    ]


# --- Validation -----------------------------------------------------------


def validation_payload(pipeline: Pipeline) -> dict[str, Any]:
    """What `pipelines/{id}/validate` answers, computed rather than constant.

    The Phase 1.1 envelope was published as a GUESS — `{"valid", "problems"}`,
    with `problems` a list of sentences — and the screen renders `problems`
    joined together when `valid` is false. Both are kept exactly, so no screen
    changes, and the structured form is added alongside under `warnings`, where
    a canvas that wants to point at the node can find its id.

    `valid` therefore means "there is nothing to tell you" rather than "this
    will run". Everything here runs: `checks.py` warns and never blocks, and
    reporting `valid: true` while holding a warning would mean the screen said
    "valid" and dropped the sentence.
    """
    warnings = check_pipeline(pipeline)
    return {
        "pipeline_id": str(pipeline.pipeline_id),
        "valid": not warnings,
        "problems": [warning.message for warning in warnings],
        "warnings": [
            {
                "code": warning.code,
                "node_id": warning.node_id,
                "related": list(warning.related),
                "severity": warning.severity,
                "message": warning.message,
            }
            for warning in warnings
        ],
    }


# --- Spectra --------------------------------------------------------------


def decimate(
    values: NDArray[np.float64], axis: NDArray[np.float64], *, max_points: int = MAX_POINTS
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Which variable indices survive, per bucket, preserving peaks.

    **Min and max per bucket, never a plain stride.** A stride keeps every
    `k`-th point and drops whatever falls between, so a peak one or two
    channels wide appears at one zoom level and vanishes at the next — which is
    the flicker §13's "preserves the visible shape" is about. Taking each
    bucket's extremes keeps the envelope of the signal at every zoom, so a
    peak's *height* survives even when its exact position is rounded to the
    bucket.

    Two indices come out of each bucket, in the order they occur in the data,
    so the drawn line goes up and down as the real one does. A bucket whose
    extremes are the same sample yields that sample twice, which keeps every
    trace the same length as the axis and costs one duplicated point.

    Returns the indices and the axis values at them. Below the budget the axis
    is returned whole and nothing is decided about it.
    """
    n_variables = values.shape[1]
    if n_variables <= max_points:
        kept = np.arange(n_variables, dtype=np.intp)
        return kept, axis[kept]

    # Two points per bucket, so the bucket count is half the budget.
    buckets = np.array_split(np.arange(n_variables, dtype=np.intp), max_points // 2)
    # The extremes are taken over the mean spectrum rather than per trace: the
    # axis is shared by every trace in the payload, and a per-trace axis would
    # mean the payload carried one x array per spectrum.
    profile = values.mean(axis=0)

    kept_list: list[int] = []
    for bucket in buckets:
        if bucket.size == 0:
            continue
        window = profile[bucket]
        low, high = int(bucket[int(window.argmin())]), int(bucket[int(window.argmax())])
        kept_list.extend(sorted((low, high)))
    kept = np.asarray(kept_list, dtype=np.intp)
    return kept, axis[kept]


def spectra_payload(
    node_id: str,
    values: NDArray[np.float64],
    version: DatasetVersion,
    *,
    label: str | None = None,
    ordinate: str = "Absorbance",
    highlight: Sequence[int] = (),
    max_traces: int = MAX_TRACES,
    max_points: int = MAX_POINTS,
) -> dict[str, Any]:
    """One spectra plot's worth of data, decimated for the wire.

    The shape is the one `stub/fixtures/spectra.json` publishes and the plot
    screen already renders: a shared axis, individually drawn traces, and a
    band when there are more spectra than the cap. `highlighted` is added
    beside them for §13's "selected or highlighted spectra are drawn at full
    resolution", and carries its own full axis because that is what full
    resolution means. A screen that ignores the key renders what it rendered
    before.

    The band is taken over **every** spectrum rather than over the undrawn
    remainder: it describes the distribution, and leaving out the drawn ones
    would make it describe a subset nobody asked about.
    """
    axis = np.asarray(version.axis.values, dtype=np.float64)
    n_spectra, n_variables = values.shape
    if axis.size != n_variables:
        raise _fail(
            500,
            "shape_mismatch",
            f"node {node_id!r} produced {n_variables} variables and the dataset's axis has "
            f"{axis.size}. A range selection changes the axis and the payload cannot guess it.",
            node_id=node_id,
        )

    kept, kept_axis = decimate(values, axis, max_points=max_points)
    banded = n_spectra > max_traces
    # An evenly spaced subset, so the drawn traces span the set rather than
    # showing its first sixty samples.
    drawn = (
        np.linspace(0, n_spectra - 1, max_traces).round().astype(int)
        if banded
        else np.arange(n_spectra)
    )

    payload: dict[str, Any] = {
        "node_id": node_id,
        "label": label or node_id,
        "axis": {
            "kind": version.axis.kind,
            "unit": version.axis.unit,
            "values": [float(value) for value in kept_axis],
        },
        "ordinate": {"label": ordinate},
        "n_spectra": int(n_spectra),
        "decimation": {
            "variables_total": int(n_variables),
            "variables_kept": int(kept.size),
            "traces_total": int(n_spectra),
            "traces_drawn": int(drawn.size),
            "banded": banded,
        },
        "traces": [_trace(int(i), values[i, kept], version) for i in drawn],
    }
    if banded:
        window = values[:, kept]
        lower, median, upper = np.percentile(window, (5, 50, 95), axis=0)
        payload["band"] = {
            "n_spectra": int(n_spectra),
            "y_lower": [float(value) for value in lower],
            "y_median": [float(value) for value in median],
            "y_upper": [float(value) for value in upper],
        }
    if highlight:
        rows = sorted({int(row) for row in highlight})
        unknown = [row for row in rows if not 0 <= row < n_spectra]
        if unknown:
            raise _fail(
                404,
                "not_found",
                f"node {node_id!r} has {n_spectra} spectra and was asked for {unknown}.",
                node_id=node_id,
            )
        payload["highlighted"] = {
            "axis": {"values": [float(value) for value in axis]},
            "traces": [_trace(row, values[row], version) for row in rows],
        }
    return payload


def _trace(index: int, y: NDArray[np.float64], version: DatasetVersion) -> dict[str, Any]:
    ids = version.sample_ids
    return {
        "index": index,
        "sample_id": ids[index] if index < len(ids) else f"row {index}",
        "y": [float(value) for value in y],
    }


# --- Pipelines ------------------------------------------------------------


def _current_pipeline(directory: Path) -> Pipeline:
    pipeline = read_pipeline(directory)
    if pipeline is None:
        raise _fail(
            404,
            "not_found",
            "this project has no pipeline yet. Import a dataset and one is started for it.",
        )
    return pipeline


@router.get("/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str) -> Any:
    """The project's pipeline. `current` is the id the frontend has always used."""
    directory, _ = _project()
    pipeline = _current_pipeline(directory)
    if pipeline_id not in ("current", str(pipeline.pipeline_id)):
        raise _fail(404, "not_found", f"no pipeline {pipeline_id}.", pipeline_id=pipeline_id)
    return json.loads(pipeline.model_dump_json())


@router.get("/pipelines/{pipeline_id}/state")
def get_pipeline_state(pipeline_id: str) -> Any:
    """Which nodes have a current result, which are running, and where they sit.

    Derived, never stored: a node is `complete` because its output is on disk
    under its own key, and `not_run` because it is not. That is #83's staleness
    rule read back rather than a second flag kept beside the graph — a flag can
    disagree with the arrays, and this cannot.

    The layout *is* stored, in its own file outside `Pipeline.content_hash()`,
    because where a node sits is not something the arrays can imply.
    """
    directory, _ = _project()
    pipeline = _current_pipeline(directory)
    version = _current_version(directory, pipeline)

    present = stored(directory, pipeline, version) if version is not None else {}
    running = _running_job(str(pipeline.pipeline_id))
    failed = _failed_job(str(pipeline.pipeline_id))
    states: dict[str, dict[str, Any]] = {}
    for node in pipeline.nodes:
        states[node.id] = _node_state(node, present, running, failed)

    return {
        "pipeline_id": str(pipeline.pipeline_id),
        "nodes": states,
        "layout": _layout(directory, pipeline),
    }


class PipelineWrite(BaseModel):
    """What a client may change about a pipeline: its recipe, and its name.

    Identity is not in the body. `pipeline_id`, `project_id` and `created_at`
    belong to the server, and a client that could send them could rename one
    pipeline into another - so they are taken from the stored record and the
    body carries only what the canvas can actually edit.

    Layout is not here either, and deliberately: where a node sits lives in its
    own file outside `Pipeline.content_hash()`, so that moving a node cannot
    invalidate an executor cache entry. Folding coordinates into this body
    would undo that in one line.
    """

    nodes: list[PipelineNode]
    name: str | None = None


@router.put("/pipelines/{pipeline_id}")
def put_pipeline(pipeline_id: str, body: PipelineWrite) -> Any:
    """Replace the pipeline's node list with the one sent.

    **The whole list, not a patch.** One project holds one pipeline and one
    user edits it, so last-write-wins needs no conflict rules, and the canvas
    already holds the entire graph it is drawing - sending a diff would mean
    inventing an operation language for a problem nobody has yet.

    **Nothing here writes staleness.** A node's cache key is its own JSON
    chained through its inputs' keys, so editing a node changes its key and its
    descendants' and nothing else's; the arrays that no longer match simply
    stop being found, and `pipelines/{id}/state` reports those nodes as
    `not_run` because they are. A stale flag written beside the graph could
    disagree with the arrays. This cannot.
    """
    directory, _ = _project()
    existing = _current_pipeline(directory)
    if pipeline_id not in ("current", str(existing.pipeline_id)):
        raise _fail(404, "not_found", f"no pipeline {pipeline_id}.", pipeline_id=pipeline_id)

    # Constructed rather than `model_copy(update=...)`, which does not re-run
    # validators: the DAG rules - unique ids, known inputs, at least one
    # source, no cycles - are on `Pipeline` itself, and a copy would skip them.
    try:
        updated = Pipeline(
            pipeline_id=existing.pipeline_id,
            project_id=existing.project_id,
            name=body.name or existing.name,
            nodes=body.nodes,
            created_at=existing.created_at,
        )
    except ValidationError as error:
        first = error.errors()[0]
        raise _fail(
            422,
            "invalid_pipeline",
            str(first.get("msg", "the pipeline is not valid")),
            field=".".join(str(part) for part in first["loc"]),
        ) from error

    # A source node naming a dataset this project does not hold would be
    # accepted by the schema and then fail at run time with nothing to point
    # at. §4.3 makes this a trust boundary; refuse it where it is written.
    if _current_version(directory, updated) is None:
        raise _fail(
            422,
            "invalid_pipeline",
            "the pipeline's source node names a dataset version this project does not hold.",
        )

    try:
        write_pipeline(directory, updated)
    except ProjectError as error:
        raise _fail(500, "project_unavailable", str(error)) from error
    return json.loads(updated.model_dump_json())


@router.post("/pipelines/{pipeline_id}/validate")
def validate_pipeline(pipeline_id: str) -> Any:
    directory, _ = _project()
    return validation_payload(_current_pipeline(directory))


def _node_state(
    node: PipelineNode, present: dict[NodeId, str], running: Job | None, failed: Job | None = None
) -> dict[str, Any]:
    if running is not None:
        if running.node_id == node.id:
            return {"state": "running", "progress": running.progress}
        if node.id not in present:
            return {"state": "queued"}
    # A run that failed names the node it failed on, and that node carries the
    # cause. Without this the canvas showed a failed run as a graph of nodes
    # that merely never ran, and the artboard's `failed` encoding - the left
    # stripe and the footer - was a state only the fixture could produce.
    if failed is not None and failed.node_id == node.id:
        return {"state": "failed", "message": failed.message}
    if node.id not in present:
        return {"state": "not_run" if running is None else "queued"}
    return {"state": "complete"}


def _layout(directory: Path, pipeline: Pipeline) -> dict[str, dict[str, float]]:
    """Where each node sits, generated for any node the canvas has not placed.

    Generated rather than left empty because a graph with every node at the
    origin is unreadable, and the canvas has no way to send a layout back yet.
    Anything already stored wins, so a node the user has moved stays moved.
    """
    stored_layout = read_layout(directory)
    by_id = {node.id: node for node in pipeline.nodes}

    depth: dict[str, int] = {}

    def depth_of(node_id: str) -> int:
        if node_id in depth:
            return depth[node_id]
        inputs = by_id[node_id].inputs
        depth[node_id] = 0 if not inputs else 1 + max(depth_of(parent) for parent in inputs)
        return depth[node_id]

    rows: dict[int, int] = {}
    layout: dict[str, dict[str, float]] = {}
    for node in pipeline.nodes:
        if node.id in stored_layout:
            layout[node.id] = stored_layout[node.id]
            continue
        column = depth_of(node.id)
        row = rows.get(column, 0)
        rows[column] = row + 1
        layout[node.id] = {
            "x": float(40 + column * _LAYOUT_STEP_X),
            "y": float(40 + row * _LAYOUT_STEP_Y),
        }
    return layout


# --- Experiments and jobs -------------------------------------------------


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> Any:
    directory, _ = _project()
    experiment = read_experiment(directory)
    if experiment is None:
        raise _fail(404, "not_found", "nothing has been run in this project yet.")
    if experiment_id not in ("current", str(experiment.experiment_id)):
        raise _fail(404, "not_found", f"no experiment {experiment_id}.")
    return json.loads(experiment.model_dump_json())


@router.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str) -> Any:
    """Submit the pipeline and answer at once, before any work has happened.

    §11: experiments are submitted as jobs, not waited on. The body is the
    job's, which is what the screen polls.
    """
    directory, _ = _project()
    pipeline = _current_pipeline(directory)
    version = _current_version(directory, pipeline)
    if version is None:
        raise _fail(
            409,
            "no_dataset",
            "the pipeline's source node points at a dataset version this project does not "
            "hold. Import the file again, or start a pipeline on a dataset it does have.",
        )
    job = submit_run(JOBS, str(pipeline.pipeline_id), directory, pipeline, version)
    return job.payload()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Any:
    job = JOBS.get(job_id)
    if job is None:
        raise _fail(404, "not_found", f"no job {job_id}.", job_id=job_id)
    return job.payload()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Any:
    job = JOBS.cancel(job_id)
    if job is None:
        raise _fail(404, "not_found", f"no job {job_id}.", job_id=job_id)
    return job.payload()


def _running_job(experiment_id: str) -> Job | None:
    return JOBS.running_for(experiment_id)


def _failed_job(experiment_id: str) -> Job | None:
    """The most recent failed run for this pipeline, if the last one failed.

    Only the latest, and only while it is still the latest: a node that failed
    an hour ago and has since been re-run is not failed, and the cheapest way
    to say so is to let the next run's job replace this one.
    """
    return JOBS.latest_failed_for(experiment_id)


# --- Node outputs ---------------------------------------------------------


@router.get("/spectra/{node_id}")
def get_spectra(node_id: str, highlight: Annotated[str | None, Query()] = None) -> Any:
    directory, pipeline, version = _runnable()
    values = _stored_display(directory, pipeline, version, node_id)
    if values is None:
        raise _fail(
            404,
            "not_found",
            f"node {node_id!r} has no result yet. Run the pipeline, or pick a node that has.",
            node_id=node_id,
        )
    return spectra_payload(node_id, values, version, highlight=_indices(highlight))


@router.get("/results/{node_id}")
def get_results(node_id: str) -> Any:
    directory, pipeline, version = _runnable()
    result = _stored_result(directory, pipeline, version, node_id)
    if result is None:
        raise _fail(
            404,
            "not_found",
            f"node {node_id!r} has no fitted result yet.",
            node_id=node_id,
        )
    return results_payload(result, version)


def _indices(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise _fail(422, "bad_request", f"highlight must be whole numbers: {error}") from error


# --- The step schema ------------------------------------------------------


@router.get("/schema/steps")
def step_schema() -> Any:
    """The preprocessing steps' JSON Schema, from the live models.

    Served from `models.py` rather than from a file, which is a change of
    source and not of shape: the inspector builds its parameter forms from
    this, so a field's bounds come from the same place the backend enforces
    them.
    """
    return TypeAdapter(PreprocessStep).json_schema()


@router.post("/steps/validate")
def validate_step(step: dict[str, Any]) -> Any:
    """Validate one step against the model that will enforce it.

    The cross-field rules — an odd Savitzky-Golay window, `polyorder` below it,
    `start` below `end` — live in `model_validator` and have no JSON Schema
    equivalent, so a form checking them itself would be restating `models.py`
    in TypeScript and drifting from it.
    """
    try:
        TypeAdapter(PreprocessStep).validate_python(step)
    except ValidationError as error:
        return {
            "valid": False,
            "errors": [
                {
                    "field": ".".join(str(part) for part in problem["loc"]) or "step",
                    "message": problem["msg"].removeprefix("Value error, "),
                }
                for problem in error.errors()
            ],
        }
    return {"valid": True, "errors": []}


# --- What the handlers above share ----------------------------------------


def _current_version(directory: Path, pipeline: Pipeline) -> DatasetVersion | None:
    """The dataset version the pipeline's source node names."""
    sources = [node for node in pipeline.nodes if isinstance(node, SourceNode)]
    wanted = {str(node.version_id) for node in sources}
    for entry in read_datasets(directory):
        for version in entry.versions:
            if str(version.version_id) in wanted:
                return version
    return None


def _runnable() -> tuple[Path, Pipeline, DatasetVersion]:
    directory, _ = _project()
    pipeline = _current_pipeline(directory)
    version = _current_version(directory, pipeline)
    if version is None:
        raise _fail(404, "not_found", "the pipeline's dataset is not in this project.")
    return directory, pipeline, version


def start_pipeline(directory: Path, project: Project, version: DatasetVersion) -> Pipeline:
    """The pipeline an import leaves behind: one source node, nothing else.

    A project with a dataset and no recipe has nowhere for the canvas to start,
    and the frontend has no way to create a pipeline — it has asked for
    `pipelines/current` since its first commit and never posted one. So an
    import starts the recipe it is obviously the beginning of, and the user
    builds from there.
    """
    pipeline = Pipeline(
        project_id=project.project_id,
        name=f"{version.source.filename if version.source else 'dataset'} pipeline",
        nodes=[SourceNode(id="source", version_id=version.version_id)],
    )
    write_pipeline(directory, pipeline)
    write_layout(directory, {"source": {"x": 40.0, "y": 40.0}})
    return pipeline
