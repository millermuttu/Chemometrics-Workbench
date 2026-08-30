"""A project directory written before the database existed still opens.

Phase 1.2 kept the index in five JSON files; `v0.3.0` shipped that shape and
seeded three end-to-end projects in it. Opening one reads it into a fresh
`project.db` once, and the files stay where they are.

The directory here is built by hand rather than by an old build, so what it
really tests is that the *reader* accepts the shape `v0.3.0` wrote. The shape
itself is pinned by the models, which have not changed: every document below is
a `model_dump_json` of the same class the old code wrote.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import pytest

from chemometrics_workbench import db
from chemometrics_workbench.models import (
    Dataset,
    DatasetVersion,
    Experiment,
    ExperimentStatus,
    Pipeline,
    Project,
    SourceNode,
    VariableAxis,
)
from chemometrics_workbench.project import (
    ProjectError,
    is_project,
    open_project,
    read_cache_index,
    read_datasets,
    read_experiment,
    read_layout,
    read_pipeline,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))


class Written(NamedTuple):
    """What the directory below holds, so a test can compare against it."""

    project: Project
    dataset: Dataset
    version: DatasetVersion
    pipeline: Pipeline
    layout: dict[str, dict[str, float]]
    experiment: Experiment


def write_json_project(path: Path) -> Written:
    """A project directory in the shape `v0.3.0` wrote, and what it holds."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "arrays").mkdir(exist_ok=True)

    project = Project(name="Corn", description="NIR", directory=str(path))
    dataset = Dataset(project_id=project.project_id, name="corn m5")
    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash="sha256:" + "1" * 64,
        n_samples=3,
        n_variables=2,
        axis=VariableAxis(kind="wavelength_nm", values=[1100.0, 1102.0], unit="nm"),
        array_path="arrays/" + "1" * 64 + ".npy",
    )
    pipeline = Pipeline(
        project_id=project.project_id,
        name="corn pipeline",
        nodes=[SourceNode(id="source", version_id=version.version_id)],
    )
    layout = {"source": {"x": 40.0, "y": 40.0}}
    experiment = Experiment(
        project_id=project.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=version.content_hash,
        status=ExperimentStatus.RUNNING,
        started_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )

    (path / "project.json").write_text(
        json.dumps(
            {
                "layout_version": 1,
                "project": project.model_dump(mode="json", exclude={"directory"}),
            }
        ),
        encoding="utf-8",
    )
    (path / "datasets.json").write_text(
        json.dumps(
            [
                {
                    "dataset": json.loads(dataset.model_dump_json()),
                    "versions": [json.loads(version.model_dump_json())],
                }
            ]
        ),
        encoding="utf-8",
    )
    (path / "pipeline.json").write_text(pipeline.model_dump_json(), encoding="utf-8")
    (path / "pipeline_layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (path / "experiment.json").write_text(experiment.model_dump_json(), encoding="utf-8")
    (path / "cache.json").write_text(
        json.dumps({"sha256:" + "9" * 64: ["arrays/" + "1" * 64 + ".npy"]}), encoding="utf-8"
    )

    return Written(project, dataset, version, pipeline, layout, experiment)


def test_a_directory_written_before_the_database_is_read_into_one(tmp_path: Path) -> None:
    held = write_json_project(tmp_path / "corn")

    opened = open_project(tmp_path / "corn")

    assert opened.project_id == held.project.project_id
    assert (tmp_path / "corn" / db.DATABASE_FILE).is_file()


def test_everything_the_files_held_reads_back(tmp_path: Path) -> None:
    held = write_json_project(tmp_path / "corn")
    root = tmp_path / "corn"

    open_project(root)

    entries = read_datasets(root)
    assert [entry.dataset.dataset_id for entry in entries] == [held.dataset.dataset_id]
    assert entries[0].versions == [held.version]
    assert read_pipeline(root) == held.pipeline
    assert read_layout(root) == held.layout
    assert read_experiment(root) == held.experiment


def test_the_executors_cache_comes_across_so_nothing_recomputes(tmp_path: Path) -> None:
    # Recomputing would be correct - the arrays are content-addressed and
    # would be rewritten identically - but it is a pipeline's worth of work to
    # arrive back where the project already was.
    root = tmp_path / "corn"
    write_json_project(root)

    open_project(root)

    assert read_cache_index(root) == {"sha256:" + "9" * 64: ["arrays/" + "1" * 64 + ".npy"]}


def test_the_files_are_left_where_they_are(tmp_path: Path) -> None:
    # They are not a second copy to keep in step; they are what the directory
    # used to be, and deleting a user's data to tidy up is not this function's
    # decision to make.
    root = tmp_path / "corn"
    write_json_project(root)
    before = {name: (root / name).read_bytes() for name in ("project.json", "pipeline.json")}

    open_project(root)

    assert {name: (root / name).read_bytes() for name in before} == before


def test_opening_twice_does_not_import_twice(tmp_path: Path) -> None:
    root = tmp_path / "corn"
    write_json_project(root)

    open_project(root)
    open_project(root)

    assert len(read_datasets(root)) == 1
    with db.open_session(root) as session:
        assert session.query(db.ProjectRow).count() == 1


def test_a_json_project_counts_as_a_project_before_it_is_read(tmp_path: Path) -> None:
    # api.py asks this to decide whether to create or open. Answering "no"
    # would send it to create_project, which refuses over an existing project.
    root = tmp_path / "corn"
    write_json_project(root)

    assert is_project(root) is True


def test_a_newer_layout_version_is_still_refused_by_name(tmp_path: Path) -> None:
    root = tmp_path / "corn"
    write_json_project(root)
    record = json.loads((root / "project.json").read_text(encoding="utf-8"))
    record["layout_version"] = 2
    (root / "project.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ProjectError, match="layout version"):
        open_project(root)


def test_a_project_with_only_its_project_file_opens(tmp_path: Path) -> None:
    """The empty project: imported nothing, built nothing, ran nothing."""
    root = tmp_path / "empty"
    write_json_project(root)
    for name in ("datasets.json", "pipeline.json", "pipeline_layout.json", "experiment.json"):
        (root / name).unlink()

    open_project(root)

    assert read_datasets(root) == []
    assert read_pipeline(root) is None
    assert read_experiment(root) is None
    assert read_layout(root) == {}


def test_an_unreadable_file_is_named_rather_than_skipped(tmp_path: Path) -> None:
    # A project whose pipeline cannot be parsed has lost work. Starting again
    # from nothing is the wrong answer to that.
    root = tmp_path / "corn"
    write_json_project(root)
    (root / "pipeline.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ProjectError, match="could not be read"):
        open_project(root)
