"""A pipeline says which estimator nodes this build will not fit.

The silence #136 is about: a PLS node validated clean, the run reported
`succeeded` and `"Done"`, and the node was left `not_run` — the same state it
has before it has ever been run. Nothing told the two apart, though
`Run.pending_estimators` had named the node the whole time.

This is not #88. Nothing here fits PLS; it says that nothing will.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

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


def test_pca_has_a_kernel_and_the_two_pls_specs_do_not() -> None:
    """#88 flips the last two by adding to one tuple, not by editing two files."""
    assert has_kernel(PCASpec(n_components=2))
    assert not has_kernel(PLSRegressionSpec(n_components=2, target="fat"))
    assert not has_kernel(PLSDASpec(n_components=2, class_column="grade"))


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


def test_a_pls_node_is_named_before_the_run_rather_than_after_it() -> None:
    graph = pipeline(
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls_dm",
            inputs=("centre",),
            spec=PLSRegressionSpec(n_components=12, target="target_DM"),
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
            id="pls_a", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
        EstimatorNode(
            id="plsda_b", inputs=("centre",), spec=PLSDASpec(n_components=3, class_column="grade")
        ),
    )
    named = [w["node_id"] for w in validation_payload(graph)["warnings"]]
    assert named == ["pls_a", "plsda_b"]


def test_it_does_not_displace_what_checks_py_had_to_say() -> None:
    """Two sources, one list. A recipe can be both wrong and partly unfittable."""
    graph = pipeline(
        # Centring above the split is #103's leak, and PLS below it has no kernel.
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        SplitNode(id="split", inputs=("centre",), spec=KFoldSplit(n_splits=10, seed=42)),
        EstimatorNode(
            id="pls", inputs=("split",), spec=PLSRegressionSpec(n_components=3, target="fat")
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
            id="pls", inputs=("snv",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    by_code = {w["code"]: w for w in validation_payload(graph)["warnings"]}
    assert by_code[ESTIMATOR_NOT_FITTED]["severity"] == "info"
    # PLS with no centring above it is a real mistake and keeps its own severity.
    assert by_code["pls_without_centring"]["severity"] == "warning"


# The run itself is unchanged, and `tests/test_executor.py` already asserts
# that: `run.pending_estimators == ["pls"]` with the PCA node's result present.
# Repeating it here would be a second copy of one claim.
