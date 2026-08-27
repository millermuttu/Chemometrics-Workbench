"""The HTTP surface: the real handlers, growing one issue at a time.

Phase 1.1 built the frontend against a stub server so that 1.2 could replace
handlers behind unchanged URLs rather than integrate in one moment. This module
is where the replacements live, and in #89 it becomes the whole server.

**Not one URL changes.** That was the point of building the frontend against
these paths from its first commit.

**The stub does not include this router yet, and that is a finding rather than
an oversight** — see #99. Swapping one handler at a time works only where the
handlers are independent, and the import flow is not: the project the frontend
lists, the dataset it opens and the pipeline it runs are one chain, and a real
import produces a dataset the fixture pipeline knows nothing about. On top of
that the 1.1 import screen sends no file at all — its `<input type="file">`
discards what the user picked and posts an empty body — so these endpoints
could not be reached from it even if the rest lined up. Both are #89's to
settle, with the walkthrough rewritten in #90. Until then this router is
exercised by `tests/test_api.py` and the stub keeps serving the 1.1 flow.

## What is here now

The import endpoints (#81) and the project and dataset reads they need to be
reachable at all: a preview cannot be confirmed if the dataset it produces has
nowhere to appear.

- `GET  /api/projects`, `GET /api/projects/{id}` — the open project
- `GET  /api/projects/{id}/datasets` — read from `datasets.json` on disk
- `POST /api/import/preview` — the reader's detection, nothing committed
- `POST /api/import` — commits with the user's corrections applied

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
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from numpy.typing import NDArray

from chemometrics_workbench import readers
from chemometrics_workbench.checks import check_pipeline
from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import Dataset, DatasetVersion, Pipeline, Project
from chemometrics_workbench.project import (
    DatasetEntry,
    ProjectError,
    add_dataset,
    config_dir,
    create_project,
    open_project,
    read_datasets,
    write_array,
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


def open_project_directory() -> Path:
    """The one project this server has open, created on first use.

    Returning a path rather than a `Project` because most callers want to read
    or write inside the directory, and the record is one `open_project` away.
    """
    configured = os.environ.get("CHEMOMETRICS_PROJECT")
    directory = Path(configured) if configured else config_dir() / "projects" / "default"
    if not (directory / "project.json").exists():
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
