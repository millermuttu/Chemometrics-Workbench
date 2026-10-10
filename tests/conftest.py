"""Session-level wiring for the test suite.

Two things live here: one BLAS thread per process, and the parity run record.
Comparisons are collected by `tests/parity.py` as they happen and written out
once at the end, so a run produces one machine-readable document rather than a
file per test. Under
pytest-xdist each worker hands its comparisons to the controller, which writes
the one document.
"""

from __future__ import annotations

import os

# One BLAS thread per process: pytest-xdist already runs a worker per core, and
# eight workers each spawning eight BLAS threads ran no faster than one process.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from typing import Any  # noqa: E402

import pytest  # noqa: E402

from tests import parity  # noqa: E402


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write `parity-results.json` when a run made any comparison.

    Written even when comparisons failed — a failing parity claim is exactly
    what the report in #14 must be able to show.
    """
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput["parity"] = [r.as_dict() for r in parity.recorder.results]
        return
    if parity.recorder.results:
        parity.recorder.write()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    """Gather a finished xdist worker's comparisons into the controller's record."""
    for result in node.workeroutput.get("parity", []):
        parity.recorder.add(parity.ParityResult(**{**result, "tier": parity.Tier(result["tier"])}))
