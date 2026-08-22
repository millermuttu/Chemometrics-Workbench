"""Session-level wiring for the test suite.

The only thing here is the parity run record. Comparisons are collected by
`tests/parity.py` as they happen and written out once at the end, so a run
produces one machine-readable document rather than a file per test.
"""

from __future__ import annotations

import pytest

from tests import parity


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write `parity-results.json` when a run made any comparison.

    Written even when comparisons failed — a failing parity claim is exactly
    what the report in #14 must be able to show.
    """
    if parity.recorder.results:
        parity.recorder.write()
