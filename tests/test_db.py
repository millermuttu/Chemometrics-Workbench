"""The project's database: creating one, opening one, and refusing one.

The tables themselves are exercised by the code that uses them - nothing does
yet, which is deliberate (#119 is the module, #120 is the swap). What is worth
testing here is the part that has no other test: the file appears where a
zipped project would carry it, the pragmas that make two processes defined are
actually set, and a database written by a newer application is refused rather
than half-read.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from chemometrics_workbench import db


@pytest.fixture(autouse=True)
def _dispose_engines() -> Iterator[None]:
    """Engines are cached by path; tmp_path reuses none, but a leaked handle on
    Windows keeps the file open and the next test cannot delete it."""
    yield
    db.dispose_all()


def test_the_database_is_created_inside_the_project_directory(tmp_path: Path) -> None:
    # PROPOSAL.md section 11: a project directory is zipped and sent to a
    # colleague. A database anywhere else would stay behind.
    db.engine_for(tmp_path, create=True)

    assert (tmp_path / "project.db").is_file()
    assert db.database_path(tmp_path) == tmp_path / "project.db"


def test_a_new_database_is_stamped_with_the_schema_version(tmp_path: Path) -> None:
    engine = db.engine_for(tmp_path, create=True)

    assert db.schema_version(engine) == db.SCHEMA_VERSION == 1


def test_every_table_is_written_when_the_database_is_created(tmp_path: Path) -> None:
    engine = db.engine_for(tmp_path, create=True)

    assert set(inspect(engine).get_table_names()) == {
        "project",
        "dataset",
        "dataset_version",
        "pipeline",
        "pipeline_layout",
        "experiment",
        "cache_entry",
    }


def test_a_database_written_by_a_newer_application_is_refused_by_name(tmp_path: Path) -> None:
    db.engine_for(tmp_path, create=True)
    db.dispose_all()
    with sqlite3.connect(tmp_path / "project.db") as connection:
        connection.execute(f"PRAGMA user_version={db.SCHEMA_VERSION + 1}")

    # Refused rather than misread: a newer version may have written columns
    # this one would silently read as absent.
    with pytest.raises(db.DatabaseError) as refusal:
        db.engine_for(tmp_path)

    assert str(db.SCHEMA_VERSION + 1) in str(refusal.value)
    assert str(db.SCHEMA_VERSION) in str(refusal.value)
    assert "project.db" in str(refusal.value)


def test_a_directory_with_no_database_is_an_error_rather_than_an_empty_project(
    tmp_path: Path,
) -> None:
    with pytest.raises(db.DatabaseError) as refusal:
        db.engine_for(tmp_path)

    assert "does not exist" in str(refusal.value)


def test_the_connection_pragmas_that_make_two_processes_defined_are_set(tmp_path: Path) -> None:
    engine = db.engine_for(tmp_path, create=True)

    with engine.connect() as connection:
        journal = connection.execute(text("PRAGMA journal_mode")).scalar_one()
        keys = connection.execute(text("PRAGMA foreign_keys")).scalar_one()
        timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

    # WAL is what lets a reader read while a write is in flight; the timeout is
    # the difference between waiting a turn and failing with "database is
    # locked" at once.
    assert journal.lower() == "wal"
    assert keys == 1
    assert timeout == db.BUSY_TIMEOUT_MS


def test_one_engine_serves_a_directory_however_it_is_spelled(tmp_path: Path) -> None:
    first = db.engine_for(tmp_path, create=True)
    second = db.engine_for(tmp_path / "." / ".." / tmp_path.name)

    # Two engines over one SQLite file would each hold their own pool.
    assert first is second


def test_opening_an_existing_database_does_not_rewrite_it(tmp_path: Path) -> None:
    engine = db.engine_for(tmp_path, create=True)
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO cache_entry (key, document) VALUES ('k', '[\"a.npy\"]')")
        )
    db.dispose_all()

    reopened = db.engine_for(tmp_path)

    with reopened.connect() as connection:
        assert connection.execute(text("SELECT document FROM cache_entry")).scalar_one() == (
            '["a.npy"]'
        )


def test_a_session_writes_and_reads_a_row_back(tmp_path: Path) -> None:
    with db.open_session(tmp_path, create=True) as session:
        session.add(db.CacheEntryRow(key="sha256:abc", document='["arrays/one.npy"]'))
        session.commit()

    with db.open_session(tmp_path) as session:
        stored = session.get(db.CacheEntryRow, "sha256:abc")
        assert stored is not None
        assert stored.document == '["arrays/one.npy"]'
