"""The project directory: a database of references, beside the files it names.

`PROPOSAL.md` §11 splits storage in two — the database holds metadata and
lineage, the project directory holds datasets, processed arrays, model
artifacts and reports, and the database stores references to files rather than
their contents. Both halves live in the same directory, which is what lets it
be zipped and sent to a colleague.

The index is `project.db` (`db.py`). It replaced five JSON files —
`project.json`, `datasets.json`, `pipeline.json`, `pipeline_layout.json` and
`experiment.json` — and every function here kept its signature when it did, so
nothing above this module changed. What did not move: arrays and estimator
results are still files, because they are contents rather than references, and
the registry of directories the user has opened still lives in their config
directory, because it is the one thing a project directory cannot know.

**Arrays are float32 on disk and float64 at the kernel boundary.** §13's
envelope — 20,000 spectra by 4,000 variables, about 320 MB — is a float32
budget, and `pca.md` §10 requires float64 computation whatever the caller
stores. Both are true at once because the conversion happens here, at the store
boundary, and nowhere else: `write_array` takes float64 and narrows it,
`read_array` widens it back. A kernel never sees a float32 array and a disk
file is never float64.

Arrays are content-addressed — `arrays/<sha256>.npy` — which makes writing the
same array twice a no-op and makes the `content_hash` a `DatasetVersion` needs
fall out of storing it rather than being computed separately and drifting.

Every path recorded inside a project directory is relative to it. An absolute
path would survive zipping and then point at a stranger's home directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from chemometrics_workbench import db
from chemometrics_workbench.arrays import as_float64
from chemometrics_workbench.models import (
    Dataset,
    DatasetVersion,
    Experiment,
    Frozen,
    NodeId,
    Pipeline,
    Project,
)

__all__ = [
    "ARRAYS_DIR",
    "DATASETS_FILE",
    "EXPERIMENT_FILE",
    "LAYOUT_FILE",
    "LAYOUT_VERSION",
    "PIPELINE_FILE",
    "PROJECT_FILE",
    "DatasetEntry",
    "ProjectError",
    "add_dataset",
    "config_dir",
    "create_project",
    "forget_project",
    "is_project",
    "known_projects",
    "open_project",
    "read_array",
    "read_datasets",
    "read_experiment",
    "read_layout",
    "read_pipeline",
    "write_array",
    "write_experiment",
    "write_json",
    "write_layout",
    "write_pipeline",
]

#: The five files the database replaced. They are still named here because a
#: directory written before it exists has them, and #121 reads them once.
PROJECT_FILE = "project.json"
DATASETS_FILE = "datasets.json"
PIPELINE_FILE = "pipeline.json"
LAYOUT_FILE = "pipeline_layout.json"
EXPERIMENT_FILE = "experiment.json"

ARRAYS_DIR = "arrays"
REGISTRY_FILE = "projects.json"

#: The version those five files were written at. The database carries its own
#: guard now — `db.SCHEMA_VERSION` — and this remains for reading a directory
#: written before there was one.
LAYOUT_VERSION = 1


class ProjectError(Exception):
    """A project directory could not be created, opened or read.

    Every message names what is wrong with which path. The reason this is one
    exception with good messages rather than a hierarchy is that the only
    caller that distinguishes cases is the HTTP layer, and it turns all of them
    into the same error body: §6's rule that an unreadable file produces a
    specific diagnostic, never a stack trace.
    """


# --- The project directory ------------------------------------------------


def create_project(directory: str | os.PathLike[str], name: str, description: str = "") -> Project:
    """Create a project directory and return the `Project` it holds.

    The directory may exist as long as it is empty; that is the normal case
    when the user picks a folder they have just made in a file dialog. It may
    not already be a project, because silently adopting one would lose whatever
    it recorded.
    """
    path = Path(directory)
    if db.database_path(path).exists() or (path / PROJECT_FILE).exists():
        raise ProjectError(f"{path} is already a project. Open it instead of creating one over it.")
    if path.exists() and any(path.iterdir()):
        raise ProjectError(
            f"{path} is not empty and is not a project. Choose an empty directory, or a new one."
        )
    try:
        (path / ARRAYS_DIR).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProjectError(f"cannot create a project at {path}: {error.strerror}") from error

    project = Project(name=name, description=description, directory=str(path.resolve()))
    with _session(path, create=True) as session:
        session.add(
            db.ProjectRow(
                project_id=str(project.project_id),
                name=project.name,
                created_at=project.created_at.isoformat(),
                document=_document(project),
            )
        )
        session.commit()
    _remember(path, project)
    return project


def open_project(directory: str | os.PathLike[str]) -> Project:
    """Open an existing project directory.

    `directory` wins over whatever the file says about where it lives: a
    project that has been zipped, moved or restored from a backup is still that
    project, and the path it was created at is history rather than truth.
    """
    path = Path(directory)
    if not path.exists():
        raise ProjectError(f"there is no directory at {path}.")
    if not path.is_dir():
        raise ProjectError(f"{path} is a file, not a project directory.")

    if not db.database_path(path).exists():
        if not (path / PROJECT_FILE).exists():
            raise ProjectError(f"{path} is not a project directory: it has no {db.DATABASE_FILE}.")
        _import_json_project(path)

    with _session(path) as session:
        record = session.scalars(select(db.ProjectRow)).first()
    if record is None:
        raise ProjectError(f"{db.database_path(path)} holds no project record.")

    try:
        project = Project.model_validate_json(record.document)
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "project"
        raise ProjectError(
            f"{db.database_path(path)} does not hold a valid project: {field} {first['msg']}."
        ) from error
    # The recorded directory is history; the one being opened is the truth.
    project.directory = str(path)

    # An empty directory does not survive every zip tool, so a project whose
    # arrays are all still to come is repaired rather than rejected.
    (path / ARRAYS_DIR).mkdir(exist_ok=True)

    _remember(path, project)
    return project


def _import_json_project(path: Path) -> None:
    """Read a project written before the database existed, once.

    Phase 1.2 kept the index in five JSON files. A directory holding those and
    no `project.db` is read into a fresh database here, in one transaction, and
    **the files are left where they are and never read again** - they are not a
    second copy to keep in step, they are what this directory used to be.

    Not a refusal, because the alternative was telling anyone with a project on
    disk to import it again, and the seeded end-to-end projects are real users
    of the old shape. Deleted when nothing can plausibly still be on 1.2.
    """
    record = _json_document(path / PROJECT_FILE)
    if not isinstance(record, dict):
        raise ProjectError(f"{path / PROJECT_FILE} should hold an object.")

    written = record.get("layout_version")
    if written != LAYOUT_VERSION:
        raise ProjectError(
            f"{path} was written with layout version {written!r} and this build reads "
            f"version {LAYOUT_VERSION}. Upgrade the application rather than editing the file."
        )

    try:
        project = Project.model_validate({**record.get("project", {}), "directory": str(path)})
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "project"
        raise ProjectError(
            f"{path / PROJECT_FILE} is not a valid project: {field} {first['msg']}."
        ) from error

    datasets = _json_document(path / DATASETS_FILE) or []
    pipeline_record = _json_document(path / PIPELINE_FILE)
    layout_record = _json_document(path / LAYOUT_FILE)
    experiment_record = _json_document(path / EXPERIMENT_FILE)

    try:
        entries = [DatasetEntry.model_validate(entry) for entry in datasets]
        pipeline = Pipeline.model_validate(pipeline_record) if pipeline_record else None
        experiment = Experiment.model_validate(experiment_record) if experiment_record else None
    except ValidationError as error:
        raise ProjectError(f"the project at {path} could not be read: {error}") from error

    # One transaction: a directory either has a database holding everything the
    # files held, or it still has only the files and is read again next time.
    with _session(path, create=True) as session:
        session.add(
            db.ProjectRow(
                project_id=str(project.project_id),
                name=project.name,
                created_at=project.created_at.isoformat(),
                document=_document(project),
            )
        )
        for entry in entries:
            session.add(
                db.DatasetRow(
                    dataset_id=str(entry.dataset.dataset_id),
                    name=entry.dataset.name,
                    created_at=entry.dataset.created_at.isoformat(),
                    document=entry.dataset.model_dump_json(),
                )
            )
            for version in entry.versions:
                session.add(
                    db.DatasetVersionRow(
                        version_id=str(version.version_id),
                        dataset_id=str(version.dataset_id),
                        version=version.version,
                        content_hash=version.content_hash,
                        n_samples=version.n_samples,
                        n_variables=version.n_variables,
                        array_path=version.array_path,
                        created_at=version.created_at.isoformat(),
                        document=version.model_dump_json(),
                    )
                )
        if pipeline is not None:
            session.add(
                db.PipelineRow(
                    pipeline_id=str(pipeline.pipeline_id),
                    name=pipeline.name,
                    content_hash=pipeline.content_hash(),
                    created_at=pipeline.created_at.isoformat(),
                    document=pipeline.model_dump_json(),
                )
            )
            # Flushed before the layout that points at it: without a declared
            # relationship between the two mappers, the unit of work is free to
            # attempt the child insert first, and the foreign key refuses it.
            session.flush()
            # Layout is keyed on the pipeline, so a layout without one has
            # nothing to hang from and is dropped rather than invented.
            if isinstance(layout_record, dict):
                session.add(
                    db.PipelineLayoutRow(
                        pipeline_id=str(pipeline.pipeline_id),
                        document=json.dumps(layout_record),
                    )
                )
        if experiment is not None:
            session.add(
                db.ExperimentRow(
                    experiment_id=str(experiment.experiment_id),
                    dataset_version_id=str(experiment.dataset_version_id),
                    status=experiment.status.value,
                    started_at=(
                        experiment.started_at.isoformat() if experiment.started_at else None
                    ),
                    finished_at=(
                        experiment.finished_at.isoformat() if experiment.finished_at else None
                    ),
                    document=experiment.model_dump_json(),
                )
            )
        session.commit()


def _json_document(path: Path) -> Any:
    """One of the pre-database files, or `None` if it is not there.

    A file that exists and cannot be parsed is an error rather than an absence:
    a project whose pipeline is unreadable has lost work, and quietly starting
    again from nothing is the wrong answer to that.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProjectError(f"{path} could not be read: {error}") from error


def _document(project: Project) -> str:
    """The project as JSON, deliberately without the directory it lives in.

    `Project.directory` is absolute and belongs to this machine. Recording it
    would make a zipped project carry a path that is wrong everywhere else, so
    it is dropped here and supplied by `open_project` from the directory the
    caller actually opened. `Project` requires the field, so it is emptied
    rather than omitted.
    """
    return project.model_copy(update={"directory": ""}).model_dump_json()


def _session(path: Path, *, create: bool = False) -> Session:
    """A session on this project's database, with database errors named.

    Every failure below arrives at the HTTP layer as `ProjectError`, which §6
    turns into a diagnostic rather than a stack trace. `db.DatabaseError` and
    SQLAlchemy's own exceptions mean the same thing to a caller - this
    directory cannot be read as a project - so they are one type here.
    """
    if not create and not db.database_path(path).exists():
        raise ProjectError(f"{path} is not a project directory: it has no {db.DATABASE_FILE}.")
    try:
        return db.open_session(path, create=create)
    except db.DatabaseError as error:
        raise ProjectError(str(error)) from error
    except SQLAlchemyError as error:
        raise ProjectError(f"cannot open the database at {db.database_path(path)}: {error}") from (
            error
        )


def write_json(path: Path, document: Any) -> None:
    """Write JSON through a temporary file, so an interrupted write loses nothing.

    A half-written `project.json` is a project the user can no longer open. The
    rename is atomic on every platform we ship to. Public because every small
    document written inside a project directory wants the same guarantee - the
    executor's cache index is the second.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ProjectError(f"cannot write {path}: {error.strerror}") from error


# --- The dataset index ----------------------------------------------------


class DatasetEntry(Frozen):
    """One dataset and its versions, as `datasets.json` holds them.

    The file is the project's index of what has been imported: SQLite arrives
    in Phase 1.3 and takes this over, and until then a restart must not lose
    the dataset list the way it currently loses the project list. The shape is
    the one the Phase 1.1 frontend already reads, so #81 serves it unchanged.

    Contents still live in files. This holds a `DatasetVersion`, which holds an
    `array_path` - never an array.
    """

    dataset: Dataset
    versions: list[DatasetVersion] = Field(min_length=1)


def read_datasets(directory: str | os.PathLike[str]) -> list[DatasetEntry]:
    """Every dataset imported into this project, oldest first.

    A project with nothing imported holds no dataset rows and returns an empty
    list, which is the empty-project state arrived at honestly rather than by a
    query parameter.
    """
    path = Path(directory)
    with _session(path) as session:
        datasets = session.scalars(
            select(db.DatasetRow).order_by(db.DatasetRow.created_at, db.DatasetRow.dataset_id)
        ).all()
        versions = session.scalars(
            select(db.DatasetVersionRow).order_by(
                db.DatasetVersionRow.dataset_id, db.DatasetVersionRow.version
            )
        ).all()

    by_dataset: dict[str, list[DatasetVersion]] = {}
    try:
        for row in versions:
            by_dataset.setdefault(row.dataset_id, []).append(
                DatasetVersion.model_validate_json(row.document)
            )
        return [
            DatasetEntry(
                dataset=Dataset.model_validate_json(row.document),
                versions=by_dataset[row.dataset_id],
            )
            for row in datasets
            if row.dataset_id in by_dataset
        ]
    except ValidationError as error:
        raise ProjectError(
            f"the dataset index in {db.database_path(path)} is not valid: {error}"
        ) from error


def add_dataset(
    directory: str | os.PathLike[str], dataset: Dataset, version: DatasetVersion
) -> DatasetEntry:
    """Record an imported dataset, or a new version of one already recorded.

    A version is appended to its dataset rather than replacing it, because a
    `DatasetVersion` is an immutable snapshot and lineage is the whole point of
    keeping them. Which dataset it belongs to is `version.dataset_id`, not the
    name: two files can be given the same name and remain two datasets.
    """
    path = Path(directory)

    if version.dataset_id != dataset.dataset_id:
        raise ProjectError(
            f"version {version.version_id} says it belongs to dataset "
            f"{version.dataset_id}, which is not {dataset.dataset_id}."
        )

    # One insert in one transaction. The JSON index this replaced was a
    # read-modify-write of the whole list with nothing holding the two halves
    # together, so two imports at once could lose one.
    with _session(path) as session:
        existing = session.get(db.DatasetRow, str(dataset.dataset_id))
        if existing is None:
            session.add(
                db.DatasetRow(
                    dataset_id=str(dataset.dataset_id),
                    name=dataset.name,
                    created_at=dataset.created_at.isoformat(),
                    document=dataset.model_dump_json(),
                )
            )
        session.add(
            db.DatasetVersionRow(
                version_id=str(version.version_id),
                dataset_id=str(version.dataset_id),
                version=version.version,
                content_hash=version.content_hash,
                n_samples=version.n_samples,
                n_variables=version.n_variables,
                array_path=version.array_path,
                created_at=version.created_at.isoformat(),
                document=version.model_dump_json(),
            )
        )
        session.commit()

    for entry in read_datasets(path):
        if entry.dataset.dataset_id == dataset.dataset_id:
            return entry
    raise ProjectError(
        f"dataset {dataset.dataset_id} was not recorded in {db.database_path(path)}."
    )


# --- The pipeline store ---------------------------------------------------
#
# One pipeline per project, until there is a database to hold more. The
# frontend has asked for `pipelines/current` since its first commit, which is
# the shape this matches: a project is a dataset and the recipe being built on
# it. A second pipeline is a schema question and a screen that does not exist.


def read_pipeline(directory: str | os.PathLike[str]) -> Pipeline | None:
    """The project's pipeline, or `None` if nothing has been built yet.

    Newest first, so a table that somehow holds two answers the same way the
    single file it replaced did: the last write wins.
    """
    path = Path(directory)
    with _session(path) as session:
        row = session.scalars(
            select(db.PipelineRow).order_by(
                db.PipelineRow.created_at.desc(), db.PipelineRow.pipeline_id
            )
        ).first()
    if row is None:
        return None
    try:
        return Pipeline.model_validate_json(row.document)
    except ValidationError as error:
        raise ProjectError(
            f"the pipeline in {db.database_path(path)} is not valid: {error}"
        ) from error


def write_pipeline(directory: str | os.PathLike[str], pipeline: Pipeline) -> None:
    path = Path(directory)
    with _session(path) as session:
        session.merge(
            db.PipelineRow(
                pipeline_id=str(pipeline.pipeline_id),
                name=pipeline.name,
                content_hash=pipeline.content_hash(),
                created_at=pipeline.created_at.isoformat(),
                document=pipeline.model_dump_json(),
            )
        )
        session.commit()


def read_layout(directory: str | os.PathLike[str]) -> dict[NodeId, dict[str, float]]:
    """Where the canvas put each node.

    Kept in its own table, deliberately outside `Pipeline` and therefore
    outside `Pipeline.content_hash()`: moving a node must not change the
    science, and must not invalidate an executor cache entry either.
    `design/data-model.md` says so and #83's keys depend on it. A separate
    table rather than a column on `pipeline` is the same argument in schema
    form - writing a position touches a different row than the recipe.
    """
    path = Path(directory)
    with _session(path) as session:
        row = session.scalars(select(db.PipelineLayoutRow)).first()
        document = json.loads(row.document) if row is not None else None
    if not isinstance(document, dict):
        return {}
    return {
        str(node): {"x": float(place.get("x", 0.0)), "y": float(place.get("y", 0.0))}
        for node, place in document.items()
        if isinstance(place, dict)
    }


def write_layout(directory: str | os.PathLike[str], layout: dict[NodeId, dict[str, float]]) -> None:
    """Record where the canvas put each node, against the project's pipeline.

    A layout with no pipeline to belong to is refused rather than stored: the
    row is keyed on the pipeline, and a position for a graph that does not
    exist is a coordinate nobody can draw.
    """
    path = Path(directory)
    pipeline = read_pipeline(path)
    if pipeline is None:
        raise ProjectError(f"{path} has no pipeline for a layout to belong to.")
    with _session(path) as session:
        session.merge(
            db.PipelineLayoutRow(
                pipeline_id=str(pipeline.pipeline_id),
                document=json.dumps(layout),
            )
        )
        session.commit()


def read_experiment(directory: str | os.PathLike[str]) -> Experiment | None:
    """The last experiment recorded, or `None` if nothing has been run.

    The file this replaced held exactly one; the table keeps every one it is
    given and this returns the most recently started. An experiment *history*
    is a screen that does not exist yet, so nothing else reads the rest.
    """
    path = Path(directory)
    with _session(path) as session:
        row = session.scalars(
            select(db.ExperimentRow).order_by(
                db.ExperimentRow.started_at.desc(), db.ExperimentRow.experiment_id
            )
        ).first()
    if row is None:
        return None
    try:
        return Experiment.model_validate_json(row.document)
    except ValidationError as error:
        raise ProjectError(
            f"the experiment in {db.database_path(path)} is not valid: {error}"
        ) from error


def write_experiment(directory: str | os.PathLike[str], experiment: Experiment) -> None:
    path = Path(directory)
    with _session(path) as session:
        session.merge(
            db.ExperimentRow(
                experiment_id=str(experiment.experiment_id),
                dataset_version_id=str(experiment.dataset_version_id),
                status=experiment.status.value,
                started_at=(experiment.started_at.isoformat() if experiment.started_at else None),
                finished_at=(
                    experiment.finished_at.isoformat() if experiment.finished_at else None
                ),
                document=experiment.model_dump_json(),
            )
        )
        session.commit()


# --- The array store ------------------------------------------------------


def write_array(directory: str | os.PathLike[str], values: object) -> tuple[str, str]:
    """Store one matrix and return its path within the project and its content hash.

    The path is relative — `arrays/<sha256>.npy` — because it is recorded in a
    `DatasetVersion` that travels with the directory. The hash is the same
    string a `DatasetVersion.content_hash` takes, `sha256:` and sixty-four hex
    digits, and it is the hash of the stored float32 bytes rather than of the
    float64 the caller passed: it identifies what is on disk, which is what a
    later read can actually be checked against.

    `as_float64` is the same contract every kernel enforces, so a NaN or a 1-D
    array is refused here with the message it would have been refused with
    three steps later.
    """
    path = Path(directory)
    _require_project(path)
    stored = as_float64(values, "array").astype(np.float32)

    blob = _npy_bytes(stored)
    digest = hashlib.sha256(blob).hexdigest()
    relative = f"{ARRAYS_DIR}/{digest}.npy"
    target = path / ARRAYS_DIR / f"{digest}.npy"

    # Content-addressed, so an identical array is already stored and rewriting
    # it would only risk truncating a file something else is reading.
    if not target.exists():
        temporary = target.with_name(target.name + ".tmp")
        try:
            temporary.write_bytes(blob)
            temporary.replace(target)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise ProjectError(f"cannot write {target}: {error.strerror}") from error

    return relative, f"sha256:{digest}"


def read_array(directory: str | os.PathLike[str], array_path: str) -> NDArray[np.float64]:
    """Read a stored matrix back as float64, ready for a kernel.

    `array_path` is the relative path a `DatasetVersion` recorded. It is
    resolved inside the project directory and refused if it points outside it:
    §4.3 calls localhost a trust boundary, and this path arrives from a client
    over HTTP one issue later.
    """
    path = Path(directory)
    target = _resolve_inside(path, array_path)
    if not target.exists():
        raise ProjectError(f"{array_path} is recorded but missing from {path}.")

    try:
        stored = np.load(target, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ProjectError(f"cannot read the array at {array_path}: {error}") from error
    return as_float64(stored, array_path)


def _npy_bytes(values: NDArray[np.float32]) -> bytes:
    """Serialise to `.npy` in memory, so the bytes can be hashed before they land.

    Hashing the file after writing it would name a file by a digest taken from
    a different read, and would mean writing a file whose name is not yet known.
    """
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def is_project(directory: str | os.PathLike[str]) -> bool:
    """Whether this directory is a project - the database is the marker.

    Public because `api.py` decides whether to create or open, and the answer
    is a fact about the directory layout, which is this module's to know. It
    asked for `project.json` by name until the database replaced it.
    """
    directory = Path(directory)
    # A directory written before the database counts: `open_project` reads it
    # into one on the way past, and answering "no" here would send `api.py` to
    # `create_project`, which refuses over an existing project.
    return db.database_path(directory).exists() or (directory / PROJECT_FILE).exists()


def _require_project(path: Path) -> None:
    """The database is the marker: a directory with one is a project."""
    if not db.database_path(path).exists():
        raise ProjectError(f"{path} is not a project directory: it has no {db.DATABASE_FILE}.")


def _resolve_inside(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing anything that escapes it."""
    if Path(relative).is_absolute():
        raise ProjectError(
            f"{relative} is an absolute path. Paths recorded in a project are relative to it."
        )
    root_resolved = root.resolve()
    target = (root_resolved / relative).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ProjectError(f"{relative} points outside the project directory.")
    return target


# --- The registry of known projects ---------------------------------------


def config_dir() -> Path:
    """Where the list of projects the user has opened is kept.

    Mirrors `datasets.cache_dir`: an explicit override first, then the XDG
    variable, then the conventional home directory. This is the one piece of
    state that lives outside every project, because it is the one thing a
    project directory cannot know — that the user has it.
    """
    override = os.environ.get("CHEMOMETRICS_CONFIG_HOME")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / "chemometrics-workbench"


def known_projects() -> list[dict[str, str]]:
    """The projects this user has created or opened, most recently opened first.

    Entries whose directory has since been moved, unmounted or deleted are
    still listed. Pruning them silently is how a project on an external drive
    disappears while the drive is unplugged; whether an entry still resolves is
    the caller's question and `Path.exists` answers it.
    """
    registry = config_dir() / REGISTRY_FILE
    if not registry.exists():
        return []
    try:
        entries = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # The registry is a convenience, not a record of anything. A corrupt
        # one costs the user a re-open; refusing to start would cost more.
        return []
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and "directory" in entry]


def forget_project(directory: str | os.PathLike[str]) -> None:
    """Drop a directory from the registry. The project itself is untouched."""
    wanted = str(Path(directory).resolve())
    remaining = [entry for entry in known_projects() if entry.get("directory") != wanted]
    _write_registry(remaining)


def _remember(path: Path, project: Project) -> None:
    """Record that this project exists, newest first, without duplicates.

    Two things this deliberately does not do, both because `open_project` is on
    the path of every HTTP request that touches a project:

    **It does not rewrite the registry when nothing would change.** A server
    answering six queries on one page load opened the project six times, and
    each open rewrote a file shared by every project on the machine. The
    registry says which projects exist and which was opened last; re-recording
    the same directory as the newest entry says nothing new, so it is skipped.

    **It never fails the open.** The registry is a convenience - `known_projects`
    already says so by returning `[]` for a corrupt one - and a project that
    cannot be listed is still a project that can be used. Before this, two
    concurrent requests racing on the write turned a `GET` into a 500.
    """
    directory = str(path.resolve())
    entries = known_projects()
    if entries and entries[0].get("directory") == directory:
        return

    entry = {
        "directory": directory,
        "project_id": str(project.project_id),
        "name": project.name,
        "last_opened": datetime.now(UTC).isoformat(),
    }
    others = [item for item in entries if item.get("directory") != directory]
    with contextlib.suppress(ProjectError, OSError):
        _write_registry([entry, *others])


def _write_registry(entries: list[dict[str, str]]) -> None:
    directory = config_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProjectError(f"cannot write the project registry: {error.strerror}") from error
    write_json(directory / REGISTRY_FILE, entries)
