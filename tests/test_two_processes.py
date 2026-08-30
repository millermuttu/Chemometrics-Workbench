"""Two application instances over one project directory.

Until the database there was no answer to this. `api.py` held one in-process
lock and said so: "two processes over one directory is a database question, not
this one". These are the answers.

WAL is what makes them defined. A reader sees the last committed state while a
writer is mid-transaction, writers take turns, and a writer that cannot get its
turn inside `busy_timeout` fails with a sentence naming the project rather than
`database is locked`.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from chemometrics_workbench import db
from chemometrics_workbench.models import Dataset, DatasetVersion, VariableAxis
from chemometrics_workbench.project import (
    ProjectError,
    add_dataset,
    create_project,
    open_project,
    read_datasets,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))
    yield
    db.dispose_all()


def a_dataset(project_id: str, name: str) -> tuple[Dataset, DatasetVersion]:
    dataset = Dataset(project_id=project_id, name=name)
    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash="sha256:" + f"{abs(hash(name)):064x}"[:64],
        n_samples=2,
        n_variables=2,
        axis=VariableAxis(kind="index", values=[1.0, 2.0]),
        array_path=f"arrays/{uuid4().hex}.npy",
    )
    return dataset, version


def test_two_writers_on_one_directory_both_land(tmp_path: Path) -> None:
    """Writers take turns rather than overwriting each other."""
    project = create_project(tmp_path / "corn", name="Corn")
    root = tmp_path / "corn"

    for name in ("first", "second"):
        # Disposing between writes makes each one a fresh connection, which is
        # what a second instance of the application is.
        dataset, version = a_dataset(str(project.project_id), name)
        add_dataset(root, dataset, version)
        db.dispose_all()

    assert sorted(entry.dataset.name for entry in read_datasets(root)) == ["first", "second"]


def test_a_reader_is_not_blocked_by_an_open_write(tmp_path: Path) -> None:
    """The whole reason for WAL: a page load during a write still renders."""
    project = create_project(tmp_path / "corn", name="Corn")
    root = tmp_path / "corn"
    dataset, version = a_dataset(str(project.project_id), "already here")
    add_dataset(root, dataset, version)
    db.dispose_all()

    holder = sqlite3.connect(root / db.DATABASE_FILE, timeout=1)
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO cache_entry (key, document) VALUES ('held', '[]')")

        # Uncommitted, so it is not visible - but the read itself succeeds,
        # which is the claim.
        assert [entry.dataset.name for entry in read_datasets(root)] == ["already here"]
        assert open_project(root).project_id == project.project_id
    finally:
        holder.rollback()
        holder.close()


def test_a_writer_that_cannot_get_its_turn_says_which_project_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`database is locked` names neither the project nor the reason."""
    project = create_project(tmp_path / "corn", name="Corn")
    root = tmp_path / "corn"
    db.dispose_all()
    # A tenth of a second rather than five, so the test waits as briefly as the
    # claim allows. The pragma is set per connection, hence the dispose above.
    monkeypatch.setattr(db, "BUSY_TIMEOUT_MS", 100)

    holder = sqlite3.connect(root / db.DATABASE_FILE, timeout=1)
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO cache_entry (key, document) VALUES ('held', '[]')")

        dataset, version = a_dataset(str(project.project_id), "blocked")
        with pytest.raises(ProjectError) as refusal:
            add_dataset(root, dataset, version)
    finally:
        holder.rollback()
        holder.close()

    assert "is busy" in str(refusal.value)
    assert str(root) in str(refusal.value)
    assert "database is locked" not in str(refusal.value)


def test_a_second_process_writes_and_this_one_sees_it(tmp_path: Path) -> None:
    """Not two connections in one process: two processes, which is the case
    `api.py` said it could not answer."""
    project = create_project(tmp_path / "corn", name="Corn")
    root = tmp_path / "corn"
    db.dispose_all()

    written = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                from uuid import uuid4
                from chemometrics_workbench.models import Dataset, DatasetVersion, VariableAxis
                from chemometrics_workbench.project import add_dataset

                dataset = Dataset(project_id="{project.project_id}", name="from another process")
                version = DatasetVersion(
                    dataset_id=dataset.dataset_id,
                    version=1,
                    content_hash="sha256:" + "7" * 64,
                    n_samples=2,
                    n_variables=2,
                    axis=VariableAxis(kind="index", values=[1.0, 2.0]),
                    array_path="arrays/" + uuid4().hex + ".npy",
                )
                add_dataset(r"{root}", dataset, version)
                print("written")
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "written" in written.stdout
    assert [entry.dataset.name for entry in read_datasets(root)] == ["from another process"]
