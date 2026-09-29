"""The parity report's coverage table cannot fall behind the code (#236).

`PROPOSAL.md` §18: a published parity report covering every implemented
algorithm. The table in `tests/parity_report.py` is what says so, and these
tests are what keep it true: a kernel added without a row, a row naming an
entry the fixture does not have, or a fixture entry no row claims, each fails
here rather than leaving the report quietly incomplete.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from tests.parity_report import COVERAGE, KERNEL_MODULES, NOT_KERNELS

FIXTURE = Path(__file__).parent / "fixtures" / "reference_values.json"


def _fixture_entries() -> set[str]:
    entries = json.loads(FIXTURE.read_text())["entries"]
    return {".".join(entry["id"].split(".")[1:3]) for entry in entries}


def test_every_public_kernel_has_a_row_or_a_reason_it_is_not_one() -> None:
    public = {
        name
        for module in KERNEL_MODULES
        for name in importlib.import_module(f"chemometrics_workbench.{module}").__all__
    }
    covered = {kernel for row in COVERAGE for kernel in row.kernels}
    assert public - covered - set(NOT_KERNELS) == set()
    # And nothing stale: every name the table mentions still exists.
    assert (covered | set(NOT_KERNELS)) - public == set()


def test_every_row_is_compared_or_says_why_not() -> None:
    for row in COVERAGE:
        assert bool(row.entries) != bool(row.not_compared), row.name


def test_the_rows_and_the_fixture_name_the_same_entries() -> None:
    claimed = [entry for row in COVERAGE for entry in row.entries]
    assert len(claimed) == len(set(claimed)), "an entry is claimed by two rows"
    assert set(claimed) == _fixture_entries()
