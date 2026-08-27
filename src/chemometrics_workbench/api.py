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

`results_payload` (#87) renders an estimator result for `results/{node_id}`.
The endpoint itself waits for the pipeline store, because a result is a node in
a pipeline and there is nowhere yet to keep one — #89 again.

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
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from chemometrics_workbench import readers
from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import Dataset, DatasetVersion, Project
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
    "MAX_UPLOAD_BYTES",
    "open_project_directory",
    "results_payload",
    "router",
]

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
