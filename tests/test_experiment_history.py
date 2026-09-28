"""Every run this project has recorded, not only the last one (#209).

The `experiment` table has kept them all since #121; `read_experiment` returned
the most recently started and its own docstring said why nothing read the rest.
`PROPOSAL.md` §8.3 wants lineage to be a query rather than a feature bolted on,
and this is the read side of that.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chemometrics_workbench import db
from chemometrics_workbench.models import (
    Dataset,
    DatasetVersion,
    Environment,
    Experiment,
    ExperimentStatus,
    Metrics,
    Pipeline,
    Project,
    SourceNode,
    VariableAxis,
)
from chemometrics_workbench.project import (
    create_project,
    read_experiment,
    read_experiments,
    write_experiment,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))


def _parts(project: Project) -> tuple[DatasetVersion, Pipeline]:
    dataset = Dataset(project_id=project.project_id, name="corn")
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
    return version, pipeline


def _run(project: Project, version: DatasetVersion, pipeline: Pipeline, minute: int) -> Experiment:
    return Experiment(
        project_id=project.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=version.content_hash,
        status=ExperimentStatus.SUCCEEDED,
        metrics=Metrics(rmsecv=round(0.4 + minute / 100, 2)),
        environment=Environment(app_version="0.7.0", python_version="3.12", platform="test"),
        started_at=datetime(2026, 9, 18, 10, minute, tzinfo=UTC),
        finished_at=datetime(2026, 9, 18, 10, minute, 30, tzinfo=UTC),
    )


def test_every_run_is_kept_and_read_back_newest_first(tmp_path: Path) -> None:
    """`read_experiment` stays the head of `read_experiments`: one order, so
    the two cannot disagree about which run is current."""
    directory = tmp_path / "history"
    project = create_project(directory, name="history")
    version, pipeline = _parts(project)

    made = [_run(project, version, pipeline, minute) for minute in range(3)]
    for experiment in made:
        write_experiment(directory, experiment)

    history = read_experiments(directory)
    assert [e.experiment_id for e in history] == [e.experiment_id for e in reversed(made)]
    assert [e.metrics.rmsecv for e in history if e.metrics] == [0.42, 0.41, 0.4]

    current = read_experiment(directory)
    assert current is not None
    assert current.experiment_id == history[0].experiment_id

    assert [e.experiment_id for e in read_experiments(directory, limit=2)] == [
        e.experiment_id for e in history[:2]
    ]


def test_a_project_with_nothing_run_has_an_empty_history(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    create_project(directory, name="empty")
    assert read_experiments(directory) == []
    assert read_experiment(directory) is None


def test_a_record_that_no_longer_validates_costs_its_own_row_and_not_the_history(
    tmp_path: Path,
) -> None:
    """A run written by a version whose schema has since moved should cost its
    own row. `read_experiment` keeps the stricter contract: a caller asking for
    *the* experiment gets an answer or an error, never silence."""
    directory = tmp_path / "mixed"
    project = create_project(directory, name="mixed")
    version, pipeline = _parts(project)
    good = _run(project, version, pipeline, 0)
    write_experiment(directory, good)

    with db.open_session(directory) as session:
        session.add(
            db.ExperimentRow(
                experiment_id="broken",
                dataset_version_id=str(version.version_id),
                status="succeeded",
                # Later than the good one, so it is first in the order and a
                # reader that gave up on the first bad row would return none.
                started_at="2026-09-18T11:00:00+00:00",
                finished_at=None,
                document='{"not": "an experiment"}',
            )
        )
        session.commit()

    assert [str(e.experiment_id) for e in read_experiments(directory)] == [str(good.experiment_id)]
