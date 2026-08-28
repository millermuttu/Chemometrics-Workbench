"""Tests for the two warnings nothing used to emit.

Both catch a mistake whose symptom is a plausible number, so what is asserted
is not only that they fire but that they fire on the right node and say what
the consequence is. A warning reading "centring before split" would pass a
weaker test and teach the user nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from chemometrics_workbench.api import validation_payload
from chemometrics_workbench.checks import (
    LEAK_BEFORE_SPLIT,
    PLS_WITHOUT_CENTRING,
    check_pipeline,
)
from chemometrics_workbench.models import (
    SNV,
    Autoscale,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    Normalise,
    PCASpec,
    Pipeline,
    PLSDASpec,
    PLSRegressionSpec,
    PreprocessNode,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "contract"


def pipeline(*nodes: Any) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=uuid4()), *nodes],
    )


# --------------------------------------------------------------------------
# centring above a split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", [MeanCentre(), Autoscale(), Autoscale(ddof=0)])
def test_centring_above_a_split_warns_and_names_both_nodes(step: Any) -> None:
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre", inputs=("source",), step=step),
            SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=10)),
            EstimatorNode(id="pca", inputs=("cv",), spec=PCASpec(n_components=3)),
        )
    )

    assert [warning.code for warning in found] == [LEAK_BEFORE_SPLIT]
    assert found[0].node_id == "centre"
    assert found[0].related == ("cv",)


def test_the_warning_states_the_consequence_not_only_the_rule() -> None:
    """ "RMSECV will be optimistic" is the part a user can act on."""
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=5)),
        )
    )
    message = found[0].message

    assert "optimistic" in message
    assert "'centre'" in message and "'cv'" in message
    assert "below the split" in message, "and what to do about it"


def test_the_leak_is_found_through_intervening_nodes() -> None:
    """Fitted before the split is fitted on everything, however far before."""
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            PreprocessNode(
                id="savgol",
                inputs=("centre",),
                step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
            ),
            PreprocessNode(id="snv", inputs=("savgol",), step=SNV()),
            SplitNode(id="cv", inputs=("snv",), spec=KFoldSplit(n_splits=5)),
        )
    )
    assert [(w.code, w.node_id) for w in found] == [(LEAK_BEFORE_SPLIT, "centre")]


def test_centring_below_the_split_is_the_correct_arrangement_and_is_silent() -> None:
    found = check_pipeline(
        pipeline(
            SplitNode(id="cv", inputs=("source",), spec=KFoldSplit(n_splits=10)),
            PreprocessNode(id="centre", inputs=("cv",), step=MeanCentre()),
            EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=3)),
        )
    )
    assert found == []


def test_centring_on_a_branch_that_never_reaches_a_split_is_silent() -> None:
    """A split somewhere in the graph is not a split above this node.

    This is the fixture pipeline's shape: `centre_a` and `split_d` are in the
    same pipeline and have nothing to do with each other.
    """
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre_a", inputs=("source",), step=MeanCentre()),
            EstimatorNode(id="pca_a", inputs=("centre_a",), spec=PCASpec(n_components=3)),
            SplitNode(id="cv", inputs=("source",), spec=KFoldSplit(n_splits=10)),
            PreprocessNode(id="centre_d", inputs=("cv",), step=MeanCentre()),
        )
    )
    assert found == []


def test_a_step_that_estimates_nothing_may_sit_above_a_split() -> None:
    """§9 names them: Savitzky-Golay, derivatives, SNV, range selection.

    None depends on which other samples were present, so fitting once on
    everything gives each row what it would have had on its own.
    """
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="snv", inputs=("source",), step=SNV()),
            PreprocessNode(
                id="savgol",
                inputs=("snv",),
                step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
            ),
            PreprocessNode(id="norm", inputs=("savgol",), step=Normalise()),
            SplitNode(id="cv", inputs=("norm",), spec=KFoldSplit(n_splits=10)),
        )
    )
    assert found == []


def test_two_splits_below_one_centring_are_both_named() -> None:
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            SplitNode(id="cv_a", inputs=("centre",), spec=KFoldSplit(n_splits=5)),
            SplitNode(id="cv_b", inputs=("centre",), spec=KFoldSplit(n_splits=10)),
        )
    )
    assert found[0].related == ("cv_a", "cv_b")
    assert "'cv_a', 'cv_b'" in found[0].message


# --------------------------------------------------------------------------
# PLS without centring
# --------------------------------------------------------------------------


def test_a_pls_node_with_no_centring_above_it_warns() -> None:
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="snv", inputs=("source",), step=SNV()),
            EstimatorNode(
                id="pls",
                inputs=("snv",),
                spec=PLSRegressionSpec(n_components=6, target="fat"),
            ),
        )
    )

    assert [warning.code for warning in found] == [PLS_WITHOUT_CENTRING]
    assert found[0].node_id == "pls"
    assert "first component" in found[0].message
    assert "intercept" in found[0].message


def test_pls_da_is_the_same_model_and_gets_the_same_warning() -> None:
    found = check_pipeline(
        pipeline(
            EstimatorNode(
                id="plsda", inputs=("source",), spec=PLSDASpec(n_components=4, class_column="grade")
            ),
        )
    )
    assert [(w.code, w.node_id) for w in found] == [(PLS_WITHOUT_CENTRING, "plsda")]


@pytest.mark.parametrize("step", [MeanCentre(), Autoscale()])
def test_either_kind_of_centring_above_a_pls_node_silences_it(step: Any) -> None:
    """`Autoscale` subtracts the column means before it divides, so it centres."""
    found = check_pipeline(
        pipeline(
            SplitNode(id="cv", inputs=("source",), spec=KFoldSplit(n_splits=10)),
            PreprocessNode(id="centre", inputs=("cv",), step=step),
            EstimatorNode(
                id="pls",
                inputs=("centre",),
                spec=PLSRegressionSpec(n_components=6, target="fat"),
            ),
        )
    )
    assert found == []


def test_the_centring_may_be_any_distance_above_the_pls_node() -> None:
    found = check_pipeline(
        pipeline(
            SplitNode(id="cv", inputs=("source",), spec=KFoldSplit(n_splits=10)),
            PreprocessNode(id="centre", inputs=("cv",), step=MeanCentre()),
            PreprocessNode(id="snv", inputs=("centre",), step=SNV()),
            EstimatorNode(
                id="pls", inputs=("snv",), spec=PLSRegressionSpec(n_components=6, target="fat")
            ),
        )
    )
    assert found == []


def test_a_pca_node_with_no_centring_is_not_warned_about() -> None:
    """Uncentred PCA is a choice with a literature behind it; uncentred PLS is not.

    Only `pls-regression.md` §3 says "almost always wrong", so only PLS is
    warned about. Extending this to PCA would be a preference with no document
    behind it.
    """
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="snv", inputs=("source",), step=SNV()),
            EstimatorNode(id="pca", inputs=("snv",), spec=PCASpec(n_components=5)),
        )
    )
    assert found == []


# --------------------------------------------------------------------------
# both at once, and the pipeline that deserves neither
# --------------------------------------------------------------------------


def test_a_pipeline_can_earn_both_warnings_and_gets_both() -> None:
    found = check_pipeline(
        pipeline(
            # One branch centres above its split; the other never centres at all.
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=10)),
            EstimatorNode(id="pca", inputs=("cv",), spec=PCASpec(n_components=3)),
            PreprocessNode(id="snv", inputs=("source",), step=SNV()),
            EstimatorNode(
                id="pls", inputs=("snv",), spec=PLSRegressionSpec(n_components=6, target="fat")
            ),
        )
    )
    assert {(w.code, w.node_id) for w in found} == {
        (LEAK_BEFORE_SPLIT, "centre"),
        (PLS_WITHOUT_CENTRING, "pls"),
    }


def test_a_pls_fed_by_a_leaky_centring_is_warned_about_once_not_twice() -> None:
    """The centring is there, so the PLS has what it needs; where it sits is
    the other warning's subject, and it already names the node."""
    found = check_pipeline(
        pipeline(
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=10)),
            EstimatorNode(
                id="pls", inputs=("cv",), spec=PLSRegressionSpec(n_components=6, target="fat")
            ),
        )
    )
    assert [(w.code, w.node_id) for w in found] == [(LEAK_BEFORE_SPLIT, "centre")]


def test_the_fixture_pipeline_earns_neither_warning() -> None:
    """The recipe the artboards draw is arranged correctly, and stays a check on these rules."""
    published = Pipeline.model_validate(
        json.loads((FIXTURES / "pipeline.json").read_text(encoding="utf-8"))
    )
    assert check_pipeline(published) == []


# --------------------------------------------------------------------------
# the envelope
# --------------------------------------------------------------------------


def test_the_response_keeps_the_published_shape_and_adds_to_it() -> None:
    """The GUESS envelope is kept exactly, so no screen changes to read it."""
    leaky = pipeline(
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=10)),
    )
    payload = validation_payload(leaky)

    assert set(payload) == {"pipeline_id", "valid", "problems", "warnings"}
    assert payload["pipeline_id"] == str(leaky.pipeline_id)
    assert payload["valid"] is False
    assert payload["problems"] == [warning.message for warning in check_pipeline(leaky)]

    # The structured form carries what a sentence cannot: the node to point at.
    assert payload["warnings"][0]["node_id"] == "centre"
    assert payload["warnings"][0]["related"] == ["cv"]
    assert payload["warnings"][0]["severity"] == "warning"
    assert payload["warnings"][0]["code"] == LEAK_BEFORE_SPLIT


def test_a_pipeline_with_nothing_to_say_about_it_is_valid_and_empty() -> None:
    clean = pipeline(
        SplitNode(id="cv", inputs=("source",), spec=KFoldSplit(n_splits=10)),
        PreprocessNode(id="centre", inputs=("cv",), step=MeanCentre()),
    )
    assert validation_payload(clean) == {
        "pipeline_id": str(clean.pipeline_id),
        "valid": True,
        "problems": [],
        "warnings": [],
    }


def test_a_warning_does_not_stop_the_pipeline_running(tmp_path: Path) -> None:
    """The user is told, not stopped. The recipe is the record of what was done."""
    from chemometrics_workbench.datasets import load_tecator
    from chemometrics_workbench.executor import execute
    from chemometrics_workbench.models import DatasetVersion
    from chemometrics_workbench.project import create_project, write_array

    tecator = load_tecator()
    directory = tmp_path / "project"
    create_project(directory, "warnings do not block")
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

    leaky = Pipeline(
        project_id=uuid4(),
        name="leaky",
        nodes=[
            SourceNode(id="source", version_id=version.version_id),
            PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
            SplitNode(id="cv", inputs=("centre",), spec=KFoldSplit(n_splits=10, seed=42)),
            EstimatorNode(id="pca", inputs=("cv",), spec=PCASpec(n_components=3)),
        ],
    )

    assert [warning.code for warning in check_pipeline(leaky)] == [LEAK_BEFORE_SPLIT]
    run = execute(directory, leaky, version)
    assert run.results["pca"].n_components == 3
