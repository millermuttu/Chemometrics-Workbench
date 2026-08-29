"""The project's index: one SQLite database, inside the project directory.

`PROPOSAL.md` §11 splits storage in two — the database holds metadata,
pipeline definitions, experiments, metrics and lineage; the project directory
holds datasets, processed arrays, model artifacts and reports; and **the
database stores references to files, never file contents.** That is why the
database lives at `<project_dir>/project.db` rather than somewhere central: a
project directory has to be zippable and sendable to a colleague, and metadata
in an application-level database would stay behind.

Three decisions here are `docs/decisions/0002-phase-1-shape.md`'s, not this
module's, and are not re-argued:

**Tables hold identity, the columns that are actually queried, and the Pydantic
model as JSON.** `models.py` is the schema of record and its invariants are
tested there; mirroring twenty Pydantic classes into columns would create two
sources of truth that drift. So a row carries a `document` — the model's own
JSON — and beside it only what a query needs to filter or order on. Nothing is
promoted to a column speculatively: every column below is one `api.py` reads
today.

**No Alembic in Phase 1.** `create_all` writes the tables and stamps
`PRAGMA user_version`; opening a database stamped *newer* than this code is
refused by name. Migrations arrive when the first schema change ships to
someone with real projects on disk.

**One engine per project directory**, cached by resolved path, because SQLite
holds its own connection pool and two engines over one file would each keep
their own. Sessions are short-lived and never shared between threads — the
executor runs in a thread, and it opens its own.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Engine,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

__all__ = [
    "DATABASE_FILE",
    "SCHEMA_VERSION",
    "CacheEntryRow",
    "DatabaseError",
    "DatasetRow",
    "DatasetVersionRow",
    "ExperimentRow",
    "PipelineLayoutRow",
    "PipelineRow",
    "ProjectRow",
    "database_path",
    "dispose_all",
    "engine_for",
    "open_session",
    "schema_version",
]

DATABASE_FILE = "project.db"

#: Bumped when a table changes shape. A database stamped higher than this is
#: refused rather than misread - the same rule `project.py`'s LAYOUT_VERSION
#: gives the files it still owns.
SCHEMA_VERSION = 1

#: How long a writer waits for another writer before giving up, in
#: milliseconds. Readers never wait: WAL lets them read the last committed
#: state while a write is in flight.
BUSY_TIMEOUT_MS = 5_000


class DatabaseError(Exception):
    """Raised when a project's database cannot be opened as this code expects.

    Separate from `project.ProjectError` because `project.py` imports this
    module and the reverse would be a cycle. Callers there are free to let it
    through: both mean "this directory is not a project this version can read",
    and both name the path.
    """


class Base(DeclarativeBase):
    """Every table below. Types are the portable ones on purpose.

    `PROPOSAL.md` §12 keeps one commitment about the schema — that it stays
    engine-portable — so identifiers are stored as strings rather than as
    SQLite's rowid tricks, and timestamps as ISO-8601 text, which sorts
    correctly because everything `models.py` writes is UTC.
    """


class ProjectRow(Base):
    __tablename__ = "project"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class DatasetRow(Base):
    __tablename__ = "dataset"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class DatasetVersionRow(Base):
    """One imported array's record. `array_path` is a path, never an array."""

    __tablename__ = "dataset_version"

    version_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("dataset.dataset_id"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    n_variables: Mapped[int] = mapped_column(Integer, nullable=False)
    array_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class PipelineRow(Base):
    """`content_hash` is a column because "which pipelines compute the same
    thing" is the query `design/data-model.md` says the model exists to make
    answerable."""

    __tablename__ = "pipeline"

    pipeline_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class PipelineLayoutRow(Base):
    """Canvas coordinates: presentation state, kept out of the recipe.

    It is a separate table rather than a column on `pipeline` so that moving a
    node writes a different row than the one the content hash is computed from.
    `design/data-model.md` puts layout outside the scientific record, and the
    executor's cache depends on that being true.
    """

    __tablename__ = "pipeline_layout"

    pipeline_id: Mapped[str] = mapped_column(
        String, ForeignKey("pipeline.pipeline_id"), primary_key=True
    )
    document: Mapped[str] = mapped_column(Text, nullable=False)


class ExperimentRow(Base):
    __tablename__ = "experiment"

    experiment_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    # Both are nullable because `Experiment` types them so: a pending run has
    # not started, and a running one has not finished. Ordering newest-first
    # puts a run with no start last, which is where it belongs.
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)
    document: Mapped[str] = mapped_column(Text, nullable=False)


class CacheEntryRow(Base):
    """A node's cache key against the array paths it produced - one per fold
    below a split. References, which is what this database is for; the arrays
    themselves stay files."""

    __tablename__ = "cache_entry"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    document: Mapped[str] = mapped_column(Text, nullable=False)


def database_path(directory: str | os.PathLike[str]) -> Path:
    """Where a project directory keeps its database."""
    return Path(directory) / DATABASE_FILE


#: One engine per resolved directory. Keyed on the resolved path so that a
#: relative path and an absolute one to the same project share a pool rather
#: than opening the file twice.
_ENGINES: dict[Path, Engine] = {}


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(connection: Any, _record: Any) -> None:
    """WAL, foreign keys and a bounded wait, on every connection.

    WAL is what makes two processes over one project directory defined rather
    than undefined: readers see the last committed state while a write is in
    flight, and writers serialise. `busy_timeout` is the difference between a
    second application instance waiting its turn and failing instantly with
    `database is locked`.

    Guarded on the connection type because this listens on the `Engine` class:
    a non-SQLite engine would reject the pragmas outright.
    """
    if not isinstance(connection, sqlite3.Connection):
        return
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def schema_version(engine: Engine) -> int:
    """The `user_version` stamped in the file. Zero means "never written"."""
    with engine.connect() as connection:
        return int(connection.execute(text("PRAGMA user_version")).scalar_one())


def engine_for(directory: str | os.PathLike[str], *, create: bool = False) -> Engine:
    """The engine for this project directory, created or opened.

    `create=True` writes the tables and stamps the version if the file is new;
    without it, a directory with no database is an error rather than a silently
    empty project. Opening a database stamped newer than `SCHEMA_VERSION` is
    refused either way, with both numbers in the message - a newer application
    may have written columns this one would read as absent.
    """
    path = Path(directory).resolve() / DATABASE_FILE
    cached = _ENGINES.get(path)
    if cached is not None:
        return cached

    if not path.exists() and not create:
        raise DatabaseError(f"{path} does not exist: this directory has no database")

    engine = create_engine(f"sqlite:///{path}", future=True)
    try:
        stamped = schema_version(engine)
        if stamped > SCHEMA_VERSION:
            raise DatabaseError(
                f"{path} was written by a newer version of the application: its schema "
                f"version is {stamped} and this one understands {SCHEMA_VERSION}"
            )
        if stamped == 0:
            # A fresh file, or one whose tables were never written. Both are
            # the same job: write them, then stamp, so a database that exists
            # is a database that is complete.
            Base.metadata.create_all(engine)
            with engine.begin() as connection:
                connection.execute(text(f"PRAGMA user_version={SCHEMA_VERSION}"))
    except Exception:
        engine.dispose()
        raise

    _ENGINES[path] = engine
    return engine


def open_session(directory: str | os.PathLike[str], *, create: bool = False) -> Session:
    """A session on this project's database, for one unit of work.

    Short-lived and never shared between threads: the executor runs in one and
    a SQLAlchemy `Session` is not thread-safe. The engine underneath *is*
    shared, which is the point of caching it.
    """
    return Session(engine_for(directory, create=create), future=True)


def dispose_all() -> None:
    """Close every cached engine. For tests, and for a server shutting down."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()
