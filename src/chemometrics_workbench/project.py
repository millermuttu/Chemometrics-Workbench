"""The project directory: what a project *is* before there is a database.

`PROPOSAL.md` §11 splits storage in two — the database holds metadata and
lineage, the project directory holds datasets, processed arrays, model
artifacts and reports, and the database stores references to files rather than
their contents. That split is what lets a project directory be zipped and sent
to a colleague, so it holds from the first commit rather than being retrofitted
when SQLite arrives in Phase 1.3.

Until then there is no database at all, which has one visible consequence and
one invisible one. The visible one: a restart loses the project *list*, not the
data, so a registry of directories the user has opened lives in their config
directory and stands in for the missing table. The invisible one: everything
here is written as if the database already existed. Nothing that belongs in a
file gets inlined into JSON because there is currently nowhere else to put it.

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

from chemometrics_workbench.arrays import as_float64
from chemometrics_workbench.models import Dataset, DatasetVersion, Frozen, Project

__all__ = [
    "ARRAYS_DIR",
    "DATASETS_FILE",
    "LAYOUT_VERSION",
    "PROJECT_FILE",
    "DatasetEntry",
    "ProjectError",
    "add_dataset",
    "config_dir",
    "create_project",
    "forget_project",
    "known_projects",
    "open_project",
    "read_array",
    "read_datasets",
    "write_array",
    "write_json",
]

PROJECT_FILE = "project.json"
DATASETS_FILE = "datasets.json"
ARRAYS_DIR = "arrays"
REGISTRY_FILE = "projects.json"

#: Bumped when the on-disk layout changes in a way an older build cannot read.
#: A directory written by a newer layout is refused by name rather than
#: half-read, because the second failure mode is the one that loses data.
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
    if (path / PROJECT_FILE).exists():
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
    _write_project_file(path, project)
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

    project_file = path / PROJECT_FILE
    if not project_file.exists():
        raise ProjectError(f"{path} is not a project directory: it has no {PROJECT_FILE}.")

    try:
        record = json.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ProjectError(f"cannot read {project_file}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProjectError(
            f"{project_file} is not valid JSON: {error.msg} at line {error.lineno}."
        ) from error
    if not isinstance(record, dict):
        raise ProjectError(f"{project_file} should hold an object, not {type(record).__name__}.")

    layout = record.get("layout_version")
    if layout != LAYOUT_VERSION:
        raise ProjectError(
            f"{path} was written with layout version {layout!r} and this build reads "
            f"version {LAYOUT_VERSION}. Upgrade the application rather than editing the file."
        )

    try:
        project = Project.model_validate({**record.get("project", {}), "directory": str(path)})
    except ValidationError as error:
        first = error.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "project"
        raise ProjectError(f"{project_file} is not a valid project: {field} {first['msg']}.") from (
            error
        )

    # An empty directory does not survive every zip tool, so a project whose
    # arrays are all still to come is repaired rather than rejected.
    (path / ARRAYS_DIR).mkdir(exist_ok=True)

    _remember(path, project)
    return project


def _write_project_file(path: Path, project: Project) -> None:
    """Write `project.json`, deliberately without the directory it lives in.

    `Project.directory` is absolute and belongs to this machine. Recording it
    would make a zipped project carry a path that is wrong everywhere else, so
    it is dropped here and supplied by `open_project` from the directory the
    caller actually opened.
    """
    record = project.model_dump(mode="json", exclude={"directory"})
    document = {"layout_version": LAYOUT_VERSION, "project": record}
    write_json(path / PROJECT_FILE, document)


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

    A project with nothing imported has no `datasets.json` and returns an empty
    list, which is the empty-project state arrived at honestly rather than by a
    query parameter.
    """
    path = Path(directory)
    _require_project(path)
    index = path / DATASETS_FILE
    if not index.exists():
        return []

    try:
        document = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProjectError(f"the dataset index at {index} is not readable JSON: {error}") from error
    if not isinstance(document, list):
        raise ProjectError(f"the dataset index at {index} is not a list of datasets.")

    try:
        return [DatasetEntry.model_validate(entry) for entry in document]
    except ValidationError as error:
        raise ProjectError(f"the dataset index at {index} is not valid: {error}") from error


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
    entries = read_datasets(path)

    if version.dataset_id != dataset.dataset_id:
        raise ProjectError(
            f"version {version.version_id} says it belongs to dataset "
            f"{version.dataset_id}, which is not {dataset.dataset_id}."
        )

    updated: DatasetEntry | None = None
    for position, entry in enumerate(entries):
        if entry.dataset.dataset_id == dataset.dataset_id:
            updated = DatasetEntry(dataset=entry.dataset, versions=[*entry.versions, version])
            entries[position] = updated
            break
    if updated is None:
        updated = DatasetEntry(dataset=dataset, versions=[version])
        entries.append(updated)

    write_json(path / DATASETS_FILE, [json.loads(entry.model_dump_json()) for entry in entries])
    return updated


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


def _require_project(path: Path) -> None:
    if not (path / PROJECT_FILE).exists():
        raise ProjectError(f"{path} is not a project directory: it has no {PROJECT_FILE}.")


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
    """Record that this project exists, newest first, without duplicates."""
    directory = str(path.resolve())
    entry = {
        "directory": directory,
        "project_id": str(project.project_id),
        "name": project.name,
        "last_opened": datetime.now(UTC).isoformat(),
    }
    others = [item for item in known_projects() if item.get("directory") != directory]
    _write_registry([entry, *others])


def _write_registry(entries: list[dict[str, str]]) -> None:
    directory = config_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProjectError(f"cannot write the project registry: {error.strerror}") from error
    write_json(directory / REGISTRY_FILE, entries)
