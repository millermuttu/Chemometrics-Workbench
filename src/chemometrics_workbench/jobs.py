"""Long-running work, submitted rather than waited on.

`PROPOSAL.md` §11: cross-validation over many preprocessing variants will
exceed a request timeout, so experiments are submitted as jobs with progress
reporting and cancellation, not as blocking HTTP calls — and designing that in
from Phase 1 is far cheaper than retrofitting it. In Phase 1.1 a job was
elapsed time against a fixture sequence, with no task, no lock and no
executor. This is the real thing.

## A thread, not a process

The work is NumPy: SVDs, matrix products, filter applications. Every one of
them releases the GIL for the duration, so a worker thread genuinely runs
alongside the server rather than starving it, and a thread shares the array
store and the cache index with no marshalling. A process pool would buy
pre-emptive cancellation and cost a copy of every matrix across a pipe.

**Cancellation is therefore cooperative**: the executor is asked between nodes
whether it should stop, and it stops there. A node already running finishes.
That is bounded by the slowest single node rather than by the whole run, which
is the property that matters — a user who cancels a ten-fold cross-validation
waits for one fold's fit, not for ten.

## Progress is counted, never interpolated

`Progress` comes from the executor as each node finishes, and the fraction is
nodes completed over nodes total. The 1.1 stub moved a number against the wall
clock, which looks the same to a user until a run is slower than the clock
expected and the bar sits at 100% while the work continues.

## Nothing survives a restart

The table is a dict in memory and the process owns it. Persistence is Phase
1.3's, and a job table that half-persists — surviving a restart with no worker
behind it, reporting `running` forever — is worse than one that admits it is
gone.

## The endpoint is not here

`POST /api/experiments/{id}/run` still belongs to the stub. A run needs an
experiment, which holds a pipeline snapshot, and there is nowhere to keep one
until #89's store — the same cut #99 describes, and the third feature to reach
it. What the endpoint would do is `submit_run`, and `Job.payload()` is what it
would return.

## What a failure carries

A cause and no traceback, which is `PROPOSAL.md` §6's rule for readers applied
to runs. `ExecutorError` already names the node it stopped at, so the message
is that sentence; anything else that escapes is caught, logged with its
traceback for the developer, and reported to the user as the exception's own
sentence. A failed experiment is a result (§8.2), not an absence.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from chemometrics_workbench.executor import Progress, Run, RunCancelled, execute
from chemometrics_workbench.models import DatasetVersion, Pipeline

__all__ = [
    "Job",
    "JobStatus",
    "Jobs",
    "Reporter",
    "submit_run",
]

_log = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """The five states `stub/fixtures/jobs.json` publishes, and #49 renders."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass(frozen=True)
class Job:
    """One submitted run, as the client sees it.

    The four fields the Phase 1.1 envelope publishes — `job_id`,
    `experiment_id`, `status`, `progress`, `message` — plus `node_id`, which is
    what a canvas needs to light up the node being worked on and which a
    message cannot carry structurally. Additive: a screen that ignores it
    renders exactly what it rendered before.

    Frozen, and the table replaces it rather than mutating it, so a snapshot
    handed to a request handler cannot change underneath it while it is being
    serialised.
    """

    job_id: str
    experiment_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = "Queued"
    node_id: str | None = None
    result: Any = field(default=None, repr=False)
    """Whatever the work returned. A `Run` for a pipeline execution, and the
    table deliberately does not care: it schedules work and reports on it."""

    def payload(self) -> dict[str, Any]:
        """The wire body, which is the 1.1 shape plus `node_id`."""
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "status": str(self.status),
            "progress": self.progress,
            "message": self.message,
            "node_id": self.node_id,
        }


class Reporter:
    """What a submitted piece of work is handed to talk to its job.

    It is the only thing the work sees of the table: it can say where it has
    got to, and it can be asked whether it should stop. Passing the table
    itself would let work reach into other jobs.
    """

    def __init__(self, jobs: Jobs, job_id: str, cancel: threading.Event) -> None:
        self._jobs = jobs
        self._job_id = job_id
        self._cancel = cancel

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def advance(self, progress: Progress) -> None:
        self._jobs._update(
            self._job_id,
            status=JobStatus.RUNNING,
            progress=progress.fraction,
            message=progress.label,
            node_id=progress.node_id,
        )


class Jobs:
    """The job table: submit, look at, cancel.

    One worker thread by default. Runs are CPU-bound and share one machine, so
    running two at once makes both slower and neither finish sooner; a second
    submission queues, which is what `queued` means.
    """

    def __init__(self, workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._cancels: dict[str, threading.Event] = {}

    # --- the table --------------------------------------------------------

    def submit(self, experiment_id: str, work: Callable[[Reporter], Any]) -> Job:
        """Queue `work` and return immediately, before it has started.

        The returned job is `queued` with zero progress, which is the honest
        report: nothing has happened yet. The 1.1 fixture's first frame says
        the same, so the screen has something true to render at once.
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, experiment_id=experiment_id)
        cancel = threading.Event()
        with self._lock:
            self._jobs[job_id] = job
            self._cancels[job_id] = cancel

        self._pool.submit(self._run, job_id, work, Reporter(self, job_id, cancel))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop, and report where it stood when asked.

        A job that has already finished is returned unchanged: cancelling a
        succeeded run cannot un-succeed it, and reporting otherwise would lose
        a result the user can see on their screen.

        A job that has not started yet goes straight to `cancelled` — there is
        no worker to ask, and leaving it `queued` would mean waiting for a run
        nobody wants before the table admitted it was cancelled.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status.finished:
                return job
            self._cancels[job_id].set()
            if job.status is JobStatus.QUEUED:
                job = replace(job, status=JobStatus.CANCELLED, message="Cancelled")
                self._jobs[job_id] = job
            return job

    def shutdown(self, wait: bool = True) -> None:
        """Stop the pool. Every unfinished job is asked to cancel first."""
        with self._lock:
            for cancel in self._cancels.values():
                cancel.set()
        self._pool.shutdown(wait=wait)

    # --- running ----------------------------------------------------------

    def _run(self, job_id: str, work: Callable[[Reporter], Any], reporter: Reporter) -> None:
        if reporter.cancelled:
            self._update(job_id, status=JobStatus.CANCELLED, message="Cancelled")
            return

        self._update(job_id, status=JobStatus.RUNNING, message="Running", progress=0.0)
        try:
            result = work(reporter)
        except RunCancelled:
            # Not a failure. The user asked, and the run stopped where it was.
            self._update(job_id, status=JobStatus.CANCELLED, message="Cancelled")
        except Exception as error:
            # Broad on purpose: a job that raises must not take the server with it.
            # The traceback is for whoever is reading the log, never for the
            # user: §6's rule is a specific diagnostic, not a stack trace.
            _log.exception("job %s failed", job_id)
            self._update(
                job_id,
                status=JobStatus.FAILED,
                message=str(error) or type(error).__name__,
            )
        else:
            self._update(
                job_id,
                status=JobStatus.SUCCEEDED,
                progress=1.0,
                message="Done",
                node_id=None,
                result=result,
            )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status.finished:
                # A cancelled job that reports one more node's progress must not
                # come back to life, and a finished one has nothing left to say.
                return
            self._jobs[job_id] = replace(job, **changes)


def submit_run(
    jobs: Jobs,
    experiment_id: str,
    directory: str | Path,
    pipeline: Pipeline,
    version: DatasetVersion,
) -> Job:
    """Submit one pipeline execution, wired to its job.

    Here rather than at the call site so that the endpoint and the tests wire
    the executor to the table the same way: progress in, cancellation out, and
    no other coupling between the two modules.
    """

    def work(reporter: Reporter) -> Run:
        return execute(
            directory,
            pipeline,
            version,
            on_progress=reporter.advance,
            is_cancelled=lambda: reporter.cancelled,
        )

    return jobs.submit(experiment_id, work)
