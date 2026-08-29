"""Tests for the job table and the executor's progress and cancellation hooks.

Concurrency tests that sleep are flaky tests, so the work here is driven by
`threading.Event` wherever timing decides the outcome: the test releases a node
when it is ready to, and nothing depends on how fast the machine is. The two
places a wait is unavoidable — waiting for a job to reach a state — go through
`_until`, which polls with a deadline and fails with what it actually saw.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import (
    ExecutorError,
    Progress,
    Run,
    RunCancelled,
    execute,
    node_label,
)
from chemometrics_workbench.jobs import Job, Jobs, JobStatus, Reporter, submit_run
from chemometrics_workbench.models import (
    SNV,
    Autoscale,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    PCASpec,
    Pipeline,
    PreprocessNode,
    RangeSelect,
    SourceNode,
    SplitNode,
)
from chemometrics_workbench.project import create_project, write_array

DEADLINE = 10.0


def _until(read: Callable[[], Job | None], done: Callable[[Job], bool], what: str) -> Job:
    """Poll until `done`, or fail saying what the job actually was."""
    deadline = time.monotonic() + DEADLINE
    job = read()
    while time.monotonic() < deadline:
        job = read()
        if job is not None and done(job):
            return job
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}; last saw {job}")


@pytest.fixture
def jobs() -> Iterator[Jobs]:
    table = Jobs()
    yield table
    table.shutdown()


@pytest.fixture(scope="module")
def tecator() -> Any:
    return load_tecator()


@pytest.fixture
def project(tmp_path: Path, tecator: Any) -> tuple[Path, DatasetVersion]:
    directory = tmp_path / "project"
    create_project(directory, "job tests")
    array_path, _ = write_array(directory, tecator.spectra)
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=tecator.source.file_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        array_path=array_path,
    )
    return directory, version


def _pipeline(version_id: Any, *nodes: Any) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=version_id), *nodes],
    )


def branch(version_id: Any) -> Pipeline:
    """A short real pipeline: four nodes and a PCA."""
    return _pipeline(
        version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(id="centre", inputs=("snv",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=5)),
    )


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_a_submitted_job_is_returned_before_it_has_started(jobs: Jobs) -> None:
    """§11: submitted, not waited on. The first answer is `queued`, honestly."""
    release = threading.Event()

    def work(reporter: Reporter) -> Any:
        release.wait(DEADLINE)
        return None

    job = jobs.submit("experiment", work)
    assert (job.status, job.progress, job.message) == (JobStatus.QUEUED, 0.0, "Queued")

    release.set()
    _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "the job to finish")


def test_a_job_that_finishes_reports_succeeded_and_keeps_its_result(jobs: Jobs) -> None:
    job = jobs.submit("experiment", lambda reporter: "the run")
    finished = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "success")

    assert finished.status is JobStatus.SUCCEEDED
    assert (finished.progress, finished.message, finished.node_id) == (1.0, "Done", None)
    assert finished.result == "the run"


def test_an_unknown_job_is_none_rather_than_a_guess(jobs: Jobs) -> None:
    assert jobs.get("nope") is None
    assert jobs.cancel("nope") is None


def test_the_payload_is_the_published_envelope_plus_the_node(jobs: Jobs) -> None:
    """The 1.1 shape is kept exactly; `node_id` is added beside it."""
    job = jobs.submit("experiment-1", lambda reporter: None)
    body = job.payload()

    assert set(body) == {"job_id", "experiment_id", "status", "progress", "message", "node_id"}
    assert body["experiment_id"] == "experiment-1"
    assert body["status"] == "queued"
    assert isinstance(body["status"], str), "a StrEnum on the wire is the string, not an object"


# --------------------------------------------------------------------------
# failure
# --------------------------------------------------------------------------


def test_a_failing_job_carries_the_cause_and_no_traceback(jobs: Jobs) -> None:
    def work(reporter: Reporter) -> Any:
        raise ExecutorError("node 'scale' (autoscale) failed: the column is dead")

    job = jobs.submit("experiment", work)
    failed = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "failure")

    assert failed.status is JobStatus.FAILED
    assert failed.message == "node 'scale' (autoscale) failed: the column is dead"
    assert "Traceback" not in failed.message
    assert 'File "' not in failed.message


def test_a_failure_with_nothing_to_say_still_says_something(jobs: Jobs) -> None:
    """An empty message would render an empty red box."""

    def work(reporter: Reporter) -> Any:
        raise RuntimeError()

    job = jobs.submit("experiment", work)
    failed = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "failure")
    assert failed.message == "RuntimeError"


def test_a_job_that_raises_does_not_take_the_table_with_it(jobs: Jobs) -> None:
    jobs.submit("experiment", lambda reporter: 1 / 0)
    later = jobs.submit("experiment", lambda reporter: "fine")

    finished = _until(lambda: jobs.get(later.job_id), lambda j: j.status.finished, "the next job")
    assert finished.status is JobStatus.SUCCEEDED


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------


def test_cancelling_a_queued_job_does_not_wait_for_a_worker(jobs: Jobs) -> None:
    """One worker, so the second submission is genuinely queued behind the first."""
    release = threading.Event()
    jobs.submit("experiment", lambda reporter: release.wait(DEADLINE))
    queued = jobs.submit("experiment", lambda reporter: "never runs")

    cancelled = jobs.cancel(queued.job_id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED

    release.set()


def test_a_running_job_stops_where_it_stood_and_stays_cancelled(jobs: Jobs) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def work(reporter: Reporter) -> Any:
        reporter.advance(Progress(1, 4, "snv", "Preprocessing: snv"))
        started.set()
        while not reporter.cancelled:
            time.sleep(0.005)
        stopped.set()
        raise RunCancelled("asked to stop")

    job = jobs.submit("experiment", work)
    started.wait(DEADLINE)
    running = jobs.get(job.job_id)
    assert running is not None and running.progress == 0.25

    jobs.cancel(job.job_id)
    assert stopped.wait(DEADLINE)

    final = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "cancellation")
    assert final.status is JobStatus.CANCELLED
    assert final.progress == 0.25, "it stopped where it stood, not at zero and not at one"


def test_a_cancelled_job_cannot_be_brought_back_by_a_late_progress_report(jobs: Jobs) -> None:
    """A node already running finishes, and its report must not undo the cancel."""
    cancelled_now = threading.Event()

    def work(reporter: Reporter) -> Any:
        cancelled_now.wait(DEADLINE)
        raise RunCancelled("stopped")

    job = jobs.submit("experiment", work)
    _until(lambda: jobs.get(job.job_id), lambda j: j.status is JobStatus.RUNNING, "it to start")
    jobs.cancel(job.job_id)
    cancelled_now.set()
    final = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "cancellation")

    reporter = Reporter(jobs, job.job_id, threading.Event())
    reporter.advance(Progress(9, 10, "late", "Preprocessing: late"))

    after = jobs.get(job.job_id)
    assert after is not None
    assert after.status is JobStatus.CANCELLED
    assert after.progress == final.progress


def test_cancelling_a_finished_job_leaves_its_result_alone(jobs: Jobs) -> None:
    job = jobs.submit("experiment", lambda reporter: "done")
    _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "success")

    cancelled = jobs.cancel(job.job_id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.SUCCEEDED


# --------------------------------------------------------------------------
# the executor's own hooks
# --------------------------------------------------------------------------


def test_progress_is_counted_per_node_and_never_interpolated(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    seen: list[Progress] = []
    execute(directory, branch(version.version_id), version, on_progress=seen.append)

    assert [step.node_id for step in seen] == ["source", "snv", "centre", "pca"]
    assert [step.completed for step in seen] == [1, 2, 3, 4]
    assert [step.fraction for step in seen] == [0.25, 0.5, 0.75, 1.0]
    assert [step.label for step in seen] == [
        "Reading the dataset",
        "Preprocessing: snv",
        "Preprocessing: mean_centre",
        "Fitting: pca",
    ]


def test_the_label_comes_from_the_recipe_rather_than_a_table_of_names() -> None:
    """A table of pretty names drifts the moment a step is added."""
    version_id = uuid4()
    labels = [
        node_label(node)
        for node in _pipeline(
            version_id,
            PreprocessNode(id="w", inputs=("source",), step=RangeSelect(start=1.0, end=2.0)),
            SplitNode(id="cv", inputs=("w",), spec=KFoldSplit(n_splits=5)),
            PreprocessNode(id="a", inputs=("cv",), step=Autoscale()),
            EstimatorNode(id="p", inputs=("a",), spec=PCASpec(n_components=2)),
        ).nodes
    ]
    assert labels == [
        "Reading the dataset",
        "Preprocessing: range_select",
        "Splitting: kfold",
        "Preprocessing: autoscale",
        "Fitting: pca",
    ]


def test_a_cancelled_execution_stops_and_keeps_what_finished(
    project: tuple[Path, DatasetVersion],
) -> None:
    """Cancellation leaves no half-written output, and does not throw away good ones."""
    directory, version = project
    pipeline = branch(version.version_id)
    done: list[str] = []

    def record(progress: Progress) -> None:
        done.append(progress.node_id)

    with pytest.raises(RunCancelled, match="cancelled before node 'centre'"):
        execute(
            directory,
            pipeline,
            version,
            on_progress=record,
            is_cancelled=lambda: len(done) >= 2,
        )

    assert done == ["source", "snv"]

    # The two nodes that finished are cached, so resuming does not repeat them,
    # and the two that did not are recomputed.
    resumed = execute(directory, pipeline, version)
    assert sorted(resumed.reused) == ["snv", "source"]
    assert resumed.computed == ["centre"]


def test_nothing_a_cancelled_run_stored_is_wrong(
    project: tuple[Path, DatasetVersion],
) -> None:
    """A node's key is its recipe and its data, never what ran after it."""
    directory, version = project
    pipeline = branch(version.version_id)
    seen: list[str] = []

    with pytest.raises(RunCancelled):
        execute(
            directory,
            pipeline,
            version,
            on_progress=lambda p: seen.append(p.node_id),
            is_cancelled=lambda: len(seen) >= 2,
        )

    cancelled_snv = execute(directory, pipeline, version).displays["snv"]
    fresh = execute(directory, pipeline, version, use_cache=False).displays["snv"]
    np.testing.assert_array_equal(cancelled_snv, fresh)


# --------------------------------------------------------------------------
# the two ends joined
# --------------------------------------------------------------------------


def test_a_real_run_advances_through_real_nodes_and_succeeds(
    jobs: Jobs, project: tuple[Path, DatasetVersion]
) -> None:
    directory, version = project
    job = submit_run(jobs, "experiment", directory, branch(version.version_id), version)

    finished = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "the run")
    assert finished.status is JobStatus.SUCCEEDED
    assert finished.progress == 1.0

    assert isinstance(finished.result, Run)
    assert sorted(finished.result.results) == ["pca"]
    assert finished.result.results["pca"].n_components == 5


def test_a_run_that_genuinely_fails_reaches_the_failed_state_without_an_affordance(
    jobs: Jobs, project: tuple[Path, DatasetVersion]
) -> None:
    """#49's failure screen, reached by a bad recipe rather than by `?fail=true`."""
    directory, version = project
    doomed = _pipeline(
        version.version_id,
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=850.0, end=850.1)),
        PreprocessNode(id="scatter", inputs=("window",), step=SNV()),
    )

    job = submit_run(jobs, "experiment", directory, doomed, version)
    failed = _until(lambda: jobs.get(job.job_id), lambda j: j.status.finished, "the failure")

    assert failed.status is JobStatus.FAILED
    assert "node 'scatter' (snv) failed" in failed.message
    assert "Traceback" not in failed.message


def test_a_ten_fold_cross_validation_finishes_within_the_budget_with_progress_visible(
    jobs: Jobs, project: tuple[Path, DatasetVersion]
) -> None:
    """§13: under 30 s, and progress that appears only at the end is not progress."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        SplitNode(id="cv", inputs=("snv",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="centre", inputs=("cv",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=5)),
    )

    started = time.monotonic()
    job = submit_run(jobs, "experiment", directory, pipeline, version)
    seen: set[float] = set()

    deadline = started + 30.0
    while time.monotonic() < deadline:
        current = jobs.get(job.job_id)
        assert current is not None
        seen.add(current.progress)
        if current.status.finished:
            break
        time.sleep(0.002)

    final = jobs.get(job.job_id)
    assert final is not None and final.status is JobStatus.SUCCEEDED
    assert time.monotonic() - started < 30.0
    assert len(seen) > 2, f"progress arrived in one jump: {sorted(seen)}"
