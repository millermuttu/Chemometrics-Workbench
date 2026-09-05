"""The checker is shown catching each rule, not just passing on the real file.

A checker that cannot fail is worse than no checker: it makes the checklist
step look done. That is #131's second item — a test asserting a file that could
no longer exist either way — one level up, and the reason this file exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.check_feature_list import DEFAULT, ROOT, main, problems

LISTS = [DEFAULT, *sorted(ROOT.glob("docs/phase-*/feature_list.json"))]


def feature(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "a",
        "priority": 0,
        "area": "backend",
        "issue": 1,
        "title": "A thing happens",
        "depends_on": [],
        "user_visible_behavior": "The thing happens.",
        "status": "passing",
        "verification": ["uv run pytest"],
        "evidence": "2026-09-05. uv run pytest - 758 passed.",
        "notes": "",
    }
    return base | overrides


def document(*features: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "Phase N",
        "exit_criterion": "Something demonstrable.",
        "status_values": ["not_started", "in_progress", "blocked", "passing"],
        "features": list(features),
    }


def test_a_list_breaking_nothing_is_consistent() -> None:
    assert problems(document(feature())) == []


def test_passing_with_no_evidence_is_caught() -> None:
    (found,) = problems(document(feature(evidence="   ")))
    assert "passing with empty evidence" in found


def test_blocked_with_no_notes_is_caught() -> None:
    (found,) = problems(document(feature(status="blocked", evidence="", notes="")))
    assert "blocked with empty notes" in found


def test_an_unknown_status_is_caught() -> None:
    (found,) = problems(document(feature(status="nearly", evidence="")))
    assert "not one of" in found


def test_a_duplicate_id_is_caught() -> None:
    found = problems(document(feature(), feature()))
    assert any("appears 2 times" in problem for problem in found)


def test_a_dependency_on_nothing_is_caught() -> None:
    (found,) = problems(document(feature(depends_on=["ghost"])))
    assert "'ghost', which is not in this list" in found


def test_passing_on_an_unfinished_dependency_is_caught() -> None:
    found = problems(
        document(
            feature(id="a", status="not_started", evidence=""),
            feature(id="b", depends_on=["a"]),
        )
    )
    assert found == ["'b' is passing but depends on 'a', which is not_started."]


def test_an_unfinished_feature_may_depend_on_an_unfinished_one() -> None:
    assert (
        problems(
            document(
                feature(id="a", status="not_started", evidence=""),
                feature(id="b", status="not_started", evidence="", depends_on=["a"]),
            )
        )
        == []
    )


def test_two_features_in_progress_are_caught() -> None:
    found = problems(
        document(
            feature(id="a", status="in_progress", evidence=""),
            feature(id="b", status="in_progress", evidence=""),
        )
    )
    assert any("At most one may be." in problem for problem in found)


def test_one_feature_in_progress_is_fine() -> None:
    assert problems(document(feature(status="in_progress", evidence=""))) == []


@pytest.mark.parametrize("path", LISTS, ids=lambda p: p.parent.name)
def test_every_list_in_the_repository_is_consistent(path: Path) -> None:
    assert problems(json.loads(path.read_text(encoding="utf-8"))) == []


def test_main_prints_the_sentence_the_checklist_names(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(DEFAULT)]) == 0
    assert capsys.readouterr().out == "feature_list.json consistent\n"


def test_main_exits_non_zero_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "feature_list.json"
    broken.write_text(json.dumps(document(feature(evidence=""))), encoding="utf-8")
    assert main([str(broken)]) == 1
    assert "empty evidence" in capsys.readouterr().out
