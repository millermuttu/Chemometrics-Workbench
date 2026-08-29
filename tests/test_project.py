"""The project directory and the array store (#77).

The verification steps in `feature_list.json` are the test names: a project
round-trips through a new process, arrays are float32 on disk and float64 at
the kernel boundary, the registry lists what was opened, a directory that is
not a project says so, and the whole thing survives being zipped and opened
somewhere else.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import select

from chemometrics_workbench import db
from chemometrics_workbench.project import (
    ARRAYS_DIR,
    ProjectError,
    config_dir,
    create_project,
    forget_project,
    known_projects,
    open_project,
    read_array,
    write_array,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the developer's real registry while testing."""
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture
def spectra() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=(12, 40))


# --- The directory --------------------------------------------------------


def test_a_new_project_has_the_documented_layout(tmp_path: Path) -> None:
    project = create_project(tmp_path / "corn", name="Corn")

    root = tmp_path / "corn"
    # The database is the index and the marker; arrays are the files it names.
    assert (root / db.DATABASE_FILE).is_file()
    assert (root / ARRAYS_DIR).is_dir()
    assert project.name == "Corn"
    assert Path(project.directory) == root.resolve()


def test_a_project_is_not_created_over_an_existing_one(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")

    with pytest.raises(ProjectError, match="already a project"):
        create_project(tmp_path / "corn", name="Corn again")


def test_a_project_is_not_created_in_a_directory_holding_other_files(tmp_path: Path) -> None:
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(ProjectError, match="not empty"):
        create_project(busy, name="Corn")


def test_reopening_returns_the_same_project(tmp_path: Path) -> None:
    created = create_project(tmp_path / "corn", name="Corn", description="NIR")
    reopened = open_project(tmp_path / "corn")

    assert reopened.project_id == created.project_id
    assert reopened.description == "NIR"


def test_no_absolute_path_is_recorded_inside_the_project(tmp_path: Path) -> None:
    """What makes a project directory portable. `directory` is deliberately emptied.

    Asserted against the database file's own bytes rather than against the
    record read back through the models: what gets zipped and sent is the file,
    and a path that reached it any other way would still be wrong everywhere
    else.
    """
    create_project(tmp_path / "corn", name="Corn")

    with db.open_session(tmp_path / "corn") as session:
        record = json.loads(session.scalars(select(db.ProjectRow)).one().document)
    assert record["directory"] == ""
    assert str(tmp_path) not in (tmp_path / "corn" / db.DATABASE_FILE).read_bytes().decode(
        "utf-8", "ignore"
    )


# --- Diagnostics, not stack traces ----------------------------------------


def test_a_missing_directory_is_named(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="no directory at"):
        open_project(tmp_path / "nowhere")


def test_a_directory_that_is_not_a_project_is_named(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(ProjectError, match=f"no {db.DATABASE_FILE}"):
        open_project(plain)


def test_a_file_is_not_a_project_directory(tmp_path: Path) -> None:
    lonely = tmp_path / "spectra.csv"
    lonely.write_text("1,2,3", encoding="utf-8")

    with pytest.raises(ProjectError, match="is a file"):
        open_project(lonely)


def test_a_database_that_is_not_a_database_names_the_problem(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    db.dispose_all()
    (tmp_path / "corn" / db.DATABASE_FILE).write_bytes(b"not a database")

    # A diagnostic, never a stack trace - PROPOSAL.md section 6.
    with pytest.raises(ProjectError):
        open_project(tmp_path / "corn")


def test_a_newer_schema_is_refused_by_name(tmp_path: Path) -> None:
    """The guard `layout_version` used to give the five JSON files."""
    create_project(tmp_path / "corn", name="Corn")
    db.dispose_all()
    with sqlite3.connect(tmp_path / "corn" / db.DATABASE_FILE) as connection:
        connection.execute(f"PRAGMA user_version={db.SCHEMA_VERSION + 1}")

    with pytest.raises(ProjectError, match="newer version"):
        open_project(tmp_path / "corn")


def test_an_invalid_project_record_names_its_field(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    with db.open_session(tmp_path / "corn") as session:
        row = session.scalars(select(db.ProjectRow)).one()
        record = json.loads(row.document)
        record["name"] = ""
        row.document = json.dumps(record)
        session.commit()
    db.dispose_all()

    with pytest.raises(ProjectError, match="name"):
        open_project(tmp_path / "corn")


def test_a_missing_arrays_directory_is_repaired(tmp_path: Path) -> None:
    """Zip tools drop empty directories. A project with no arrays yet is still a project."""
    create_project(tmp_path / "corn", name="Corn")
    (tmp_path / "corn" / ARRAYS_DIR).rmdir()

    open_project(tmp_path / "corn")
    assert (tmp_path / "corn" / ARRAYS_DIR).is_dir()


# --- The array store ------------------------------------------------------


def test_an_array_is_float32_on_disk_and_float64_at_the_kernel_boundary(
    tmp_path: Path, spectra: np.ndarray
) -> None:
    create_project(tmp_path / "corn", name="Corn")
    relative, _ = write_array(tmp_path / "corn", spectra)

    on_disk = np.load(tmp_path / "corn" / relative)
    assert on_disk.dtype == np.float32

    back = read_array(tmp_path / "corn", relative)
    assert back.dtype == np.float64
    assert np.array_equal(back, spectra.astype(np.float32))


def test_the_content_hash_has_the_shape_the_schema_requires(
    tmp_path: Path, spectra: np.ndarray
) -> None:
    from chemometrics_workbench.models import DatasetVersion, VariableAxis

    create_project(tmp_path / "corn", name="Corn")
    relative, content_hash = write_array(tmp_path / "corn", spectra)

    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=content_hash,
        n_samples=spectra.shape[0],
        n_variables=spectra.shape[1],
        axis=VariableAxis(kind="index", values=list(range(spectra.shape[1]))),
        array_path=relative,
    )
    assert version.array_path == relative
    assert not Path(version.array_path).is_absolute()


def test_the_same_array_is_stored_once(tmp_path: Path, spectra: np.ndarray) -> None:
    create_project(tmp_path / "corn", name="Corn")
    first, first_hash = write_array(tmp_path / "corn", spectra)
    second, second_hash = write_array(tmp_path / "corn", spectra.copy())

    assert (first, first_hash) == (second, second_hash)
    assert len(list((tmp_path / "corn" / ARRAYS_DIR).iterdir())) == 1


def test_a_different_array_is_stored_separately(tmp_path: Path, spectra: np.ndarray) -> None:
    create_project(tmp_path / "corn", name="Corn")
    write_array(tmp_path / "corn", spectra)
    write_array(tmp_path / "corn", spectra + 1.0)

    assert len(list((tmp_path / "corn" / ARRAYS_DIR).iterdir())) == 2


def test_the_kernel_array_contract_is_enforced_at_the_store(tmp_path: Path) -> None:
    """A NaN is refused here with the message it would be refused with later."""
    create_project(tmp_path / "corn", name="Corn")

    with pytest.raises(ValueError, match="non-finite"):
        write_array(tmp_path / "corn", [[1.0, np.nan], [3.0, 4.0]])
    with pytest.raises(ValueError, match="must be 2-D"):
        write_array(tmp_path / "corn", [1.0, 2.0, 3.0])


def test_writing_outside_a_project_is_refused(tmp_path: Path, spectra: np.ndarray) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    with pytest.raises(ProjectError, match="not a project directory"):
        write_array(plain, spectra)


def test_a_recorded_array_that_is_missing_is_named(tmp_path: Path, spectra: np.ndarray) -> None:
    create_project(tmp_path / "corn", name="Corn")
    relative, _ = write_array(tmp_path / "corn", spectra)
    (tmp_path / "corn" / relative).unlink()

    with pytest.raises(ProjectError, match="missing from"):
        read_array(tmp_path / "corn", relative)


@pytest.mark.parametrize("escape", ["../secrets.npy", "arrays/../../secrets.npy"])
def test_a_path_that_escapes_the_project_is_refused(tmp_path: Path, escape: str) -> None:
    """§4.3: this path arrives from a client over HTTP one issue later."""
    create_project(tmp_path / "corn", name="Corn")

    with pytest.raises(ProjectError, match="outside the project"):
        read_array(tmp_path / "corn", escape)


def test_an_absolute_array_path_is_refused(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")

    with pytest.raises(ProjectError, match="absolute path"):
        read_array(tmp_path / "corn", str(tmp_path / "secrets.npy"))


# --- Across processes, and across machines --------------------------------


def test_a_project_survives_a_new_process(tmp_path: Path, spectra: np.ndarray) -> None:
    """The verification step, taken literally: a *new process* reads it back."""
    create_project(tmp_path / "corn", name="Corn")
    relative, content_hash = write_array(tmp_path / "corn", spectra)

    script = (
        "import hashlib, json\n"
        "from chemometrics_workbench.project import open_project, read_array\n"
        f"project = open_project({str(tmp_path / 'corn')!r})\n"
        f"values = read_array({str(tmp_path / 'corn')!r}, {relative!r})\n"
        "print(json.dumps({'name': project.name, 'dtype': str(values.dtype),"
        " 'digest': hashlib.sha256(values.tobytes()).hexdigest(),"
        " 'shape': list(values.shape)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    reported = json.loads(completed.stdout)

    assert reported["name"] == "Corn"
    assert reported["dtype"] == "float64"
    assert reported["shape"] == list(spectra.shape)
    # Bit-identical, not merely close: the bytes the new process read are the
    # bytes float32 storage can carry.
    expected = spectra.astype(np.float32).astype(np.float64).tobytes()
    assert reported["digest"] == hashlib.sha256(expected).hexdigest()
    assert content_hash.startswith("sha256:")


def test_a_project_can_be_zipped_and_opened_elsewhere(tmp_path: Path, spectra: np.ndarray) -> None:
    """The point of the storage split: a project directory travels."""
    source = tmp_path / "corn"
    create_project(source, name="Corn")
    relative, _ = write_array(source, spectra)

    archive = tmp_path / "corn.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for item in source.rglob("*"):
            bundle.write(item, item.relative_to(source))

    elsewhere = tmp_path / "elsewhere"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(elsewhere)
    shutil.rmtree(source)

    project = open_project(elsewhere)
    assert project.name == "Corn"
    assert Path(project.directory) == elsewhere
    assert np.array_equal(read_array(elsewhere, relative), spectra.astype(np.float32))


# --- The registry ---------------------------------------------------------


def test_the_registry_lists_a_project_opened_earlier(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    create_project(tmp_path / "wheat", name="Wheat")

    directories = [entry["directory"] for entry in known_projects()]
    assert str((tmp_path / "corn").resolve()) in directories
    assert str((tmp_path / "wheat").resolve()) in directories


def test_the_most_recently_opened_project_is_first(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    create_project(tmp_path / "wheat", name="Wheat")
    open_project(tmp_path / "corn")

    assert known_projects()[0]["directory"] == str((tmp_path / "corn").resolve())
    assert len(known_projects()) == 2


def test_a_project_is_listed_once_however_often_it_is_opened(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    open_project(tmp_path / "corn")
    open_project(tmp_path / "corn")

    assert len(known_projects()) == 1


def test_forgetting_a_project_leaves_the_project_alone(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    forget_project(tmp_path / "corn")

    assert known_projects() == []
    assert (tmp_path / "corn" / db.DATABASE_FILE).is_file()
    assert open_project(tmp_path / "corn").name == "Corn"


def test_a_registry_still_lists_a_directory_that_has_gone(tmp_path: Path) -> None:
    """An unplugged external drive must not silently delete the user's project list."""
    create_project(tmp_path / "corn", name="Corn")
    shutil.rmtree(tmp_path / "corn")

    assert [entry["name"] for entry in known_projects()] == ["Corn"]


def test_a_corrupt_registry_costs_a_reopen_and_nothing_more(tmp_path: Path) -> None:
    create_project(tmp_path / "corn", name="Corn")
    (config_dir() / "projects.json").write_text("{ not json", encoding="utf-8")

    assert known_projects() == []
    assert open_project(tmp_path / "corn").name == "Corn"
    assert len(known_projects()) == 1
