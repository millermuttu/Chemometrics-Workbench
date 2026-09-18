"""A pipeline says which estimator nodes this build will not fit.

The silence #136 is about: a PLS node validated clean, the run reported
`succeeded` and `"Done"`, and the node was left `not_run` — the same state it
has before it has ever been run. Nothing told the two apart, though
`Run.pending_estimators` had named the node the whole time.

This is not #88. Nothing here fits PLS; it says that nothing will.

Every estimator spec has a kernel since #185, so the warning has nothing real
to fire on. The cases below take PLS-DA's kernel away through `_FITTED` - the
one tuple `has_kernel` reads - because the path has to keep working for the
next spec that lands before its kernel does.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from chemometrics_workbench import executor
from chemometrics_workbench.api import ESTIMATOR_NOT_FITTED, validation_payload
from chemometrics_workbench.checks import LEAK_BEFORE_SPLIT
from chemometrics_workbench.executor import has_kernel
from chemometrics_workbench.models import (
    SNV,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    PCASpec,
    Pipeline,
    PLSDASpec,
    PLSRegressionSpec,
    PreprocessNode,
    SourceNode,
    SplitNode,
)


def pipeline(*nodes: object) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=uuid4()), *nodes],
    )


def codes(payload: dict[str, Any]) -> list[str]:
    return [warning["code"] for warning in payload["warnings"]]


# --- The one place that knows -----------------------------------------------


@pytest.fixture(autouse=True)
def no_plsda_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take PLS-DA's kernel away, so the warning has something to say."""
    monkeypatch.setattr(executor, "_FITTED", (PCASpec, PLSRegressionSpec))


def test_every_estimator_has_a_kernel_and_the_tuple_is_where_that_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#142 added PLS and #185 PLS-DA by adding to one tuple, not by editing two
    files. With the tuple restored, all three fit; with PLS-DA removed from it,
    `has_kernel` says so and nothing else has to be told."""
    assert has_kernel(PCASpec(n_components=2))
    assert has_kernel(PLSRegressionSpec(n_components=2, target="fat"))
    assert not has_kernel(PLSDASpec(n_components=2, class_column="grade"))
    monkeypatch.setattr(executor, "_FITTED", (PCASpec, PLSRegressionSpec, PLSDASpec))
    assert has_kernel(PLSDASpec(n_components=2, class_column="grade"))


# --- What validate now says -------------------------------------------------


def test_a_pipeline_of_things_that_run_still_says_nothing() -> None:
    graph = pipeline(
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=2)),
    )
    payload = validation_payload(graph)
    assert payload["valid"] is True
    assert payload["warnings"] == []
    assert payload["problems"] == []


def test_a_pls_da_node_is_named_before_the_run_rather_than_after_it() -> None:
    graph = pipeline(
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls_dm",
            inputs=("centre",),
            spec=PLSDASpec(n_components=12, class_column="grade"),
        ),
    )
    payload = validation_payload(graph)

    assert payload["valid"] is False
    (warning,) = payload["warnings"]
    assert warning["code"] == ESTIMATOR_NOT_FITTED
    assert warning["node_id"] == "pls_dm"
    assert warning["severity"] == "info"
    assert "will not be fitted" in warning["message"]
    # The published envelope carries the sentence too, so the screen that only
    # renders `problems` says something rather than nothing.
    assert payload["problems"] == [warning["message"]]


def test_every_unfittable_node_is_named_not_just_the_first() -> None:
    graph = pipeline(
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=2)),
        EstimatorNode(
            id="plsda_a", inputs=("centre",), spec=PLSDASpec(n_components=3, class_column="a")
        ),
        EstimatorNode(
            id="plsda_b", inputs=("centre",), spec=PLSDASpec(n_components=3, class_column="b")
        ),
        # PLS is fitted since #142 and must not appear.
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    named = [w["node_id"] for w in validation_payload(graph)["warnings"]]
    assert named == ["plsda_a", "plsda_b"]


def test_it_does_not_displace_what_checks_py_had_to_say() -> None:
    """Two sources, one list. A recipe can be both wrong and partly unfittable."""
    graph = pipeline(
        # Centring above the split is #103's leak, and PLS below it has no kernel.
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        SplitNode(id="split", inputs=("centre",), spec=KFoldSplit(n_splits=10, seed=42)),
        EstimatorNode(
            id="plsda", inputs=("split",), spec=PLSDASpec(n_components=3, class_column="grade")
        ),
    )
    found = codes(validation_payload(graph))
    assert LEAK_BEFORE_SPLIT in found
    assert ESTIMATOR_NOT_FITTED in found


def test_the_two_kinds_are_told_apart_by_code_and_severity() -> None:
    """A screen filtering "what is wrong with my recipe" must not catch this one."""
    graph = pipeline(
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        EstimatorNode(
            id="plsda", inputs=("snv",), spec=PLSDASpec(n_components=3, class_column="grade")
        ),
    )
    by_code = {w["code"]: w for w in validation_payload(graph)["warnings"]}
    assert by_code[ESTIMATOR_NOT_FITTED]["severity"] == "info"
    # PLS with no centring above it is a real mistake and keeps its own severity.
    assert by_code["pls_without_centring"]["severity"] == "warning"


# The run itself is unchanged, and `tests/test_executor.py` already asserts
# that: `run.pending_estimators == ["pls"]` with the PCA node's result present.
# Repeating it here would be a second copy of one claim.
