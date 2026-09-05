"""A node under a range selection is on its own axis, and both payloads use it.

`RangeSelect` is the one step that changes the variable count, so the axis a
node's output is on stops being the dataset's the moment one is in its
ancestry. Two payloads pair values with an axis and both were wrong about it —
`spectra/{node}` refused with a 500, and `results/{node}` silently published a
306-value axis beside 167 loadings, which puts every peak on the loadings plot
at the wrong wavelength. #134.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from chemometrics_workbench.api import node_axis, results_payload, spectra_payload
from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import (
    AxisKind,
    DatasetVersion,
    EstimatorNode,
    MeanCentre,
    PCASpec,
    Pipeline,
    PreprocessNode,
    RangeSelect,
    SourceNode,
    VariableAxis,
)

#: 285-1200 nm at 3 nm, the mango dry-matter set's axis and the one the bug was
#: found on. A selection of 500-1000 nm keeps 167 of its 306 channels.
AXIS = np.arange(285.0, 1200.0 + 1e-9, 3.0)
N_SELECTED = 167


def version(axis: np.ndarray = AXIS) -> DatasetVersion:
    return DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash="sha256:" + "0" * 64,
        n_samples=4,
        n_variables=axis.size,
        axis=VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=axis.tolist(), unit="nm"),
        sample_ids=[f"S{i}" for i in range(4)],
        array_path="arrays/synthetic.npy",
    )


def pipeline(*nodes: object) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=uuid4()), *nodes],
    )


def test_the_source_is_on_the_datasets_own_axis() -> None:
    assert node_axis(pipeline(), "source", version()).tolist() == AXIS.tolist()


def test_a_selection_shortens_the_axis_to_what_it_kept() -> None:
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0))
    )
    axis = node_axis(graph, "window", version())
    assert axis.size == N_SELECTED
    assert (axis >= 500.0).all() and (axis <= 1000.0).all()
    assert axis[0] == 501.0 and axis[-1] == 999.0


def test_the_shortened_axis_is_inherited_by_everything_downstream() -> None:
    """The symptom: the selecting node was one of many broken, not the only one."""
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0)),
        PreprocessNode(id="centre", inputs=("window",), step=MeanCentre()),
        PreprocessNode(id="again", inputs=("centre",), step=MeanCentre()),
    )
    for node_id in ("window", "centre", "again"):
        assert node_axis(graph, node_id, version()).size == N_SELECTED


def test_two_selections_compose_in_order() -> None:
    graph = pipeline(
        PreprocessNode(id="wide", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0)),
        PreprocessNode(id="narrow", inputs=("wide",), step=RangeSelect(start=700.0, end=800.0)),
    )
    axis = node_axis(graph, "narrow", version())
    assert axis[0] == 702.0 and axis[-1] == 798.0
    assert axis.size == 33


def test_a_sibling_branch_without_a_selection_keeps_the_whole_axis() -> None:
    """A selection belongs to its branch, not to the pipeline."""
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0)),
        PreprocessNode(id="plain", inputs=("source",), step=MeanCentre()),
    )
    assert node_axis(graph, "window", version()).size == N_SELECTED
    assert node_axis(graph, "plain", version()).size == AXIS.size


def test_a_descending_axis_is_selected_the_same_way() -> None:
    """Wavenumber axes run downwards and an interval is still an interval."""
    descending = np.arange(4000.0, 399.0, -4.0)
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=1000.0, end=1800.0))
    )
    axis = node_axis(graph, "window", version(descending))
    assert axis[0] > axis[-1]
    assert (axis >= 1000.0).all() and (axis <= 1800.0).all()


# --- The two payloads -----------------------------------------------------


def test_the_spectra_payload_draws_against_the_axis_it_is_given() -> None:
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0))
    )
    axis = node_axis(graph, "window", version())
    values = np.linspace(0.0, 1.0, 4 * N_SELECTED).reshape(4, N_SELECTED)

    payload = spectra_payload("window", values, version(), axis=axis)
    assert len(payload["axis"]["values"]) == N_SELECTED
    assert payload["axis"]["values"][0] == 501.0
    assert len(payload["traces"][0]["y"]) == N_SELECTED


def test_a_width_no_step_accounts_for_is_still_refused() -> None:
    """The 500 stays for a real mismatch. It fired on a case it could answer."""
    values = np.zeros((4, 20))
    with pytest.raises(Exception, match="has to be in `node_axis`"):
        spectra_payload("window", values, version())


def result(n_variables: int) -> EstimatorResult:
    return EstimatorResult(
        node_id="pca",
        key="k",
        task="pca",
        n_components=2,
        n_samples=4,
        n_variables=n_variables,
        rank=2,
        fold=None,
        rows=[0, 1, 2, 3],
        scores=[[0.0, 0.0]] * 4,
        loadings=[[0.0] * n_variables] * 2,
        eigenvalues=[1.0, 0.5],
        explained_variance_ratio=[0.6, 0.3],
        cumulative_explained_variance=[0.6, 0.9],
        hotelling_t2=[0.0] * 4,
        hotelling_t2_limit=1.0,
        spe=[0.0] * 4,
        spe_limit=1.0,
        alpha=0.05,
    )


def test_the_loadings_axis_is_the_nodes_own() -> None:
    """The silent half. This published 306 values beside 167 loadings."""
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0)),
        PreprocessNode(id="centre", inputs=("window",), step=MeanCentre()),
    )
    axis = node_axis(graph, "centre", version())

    payload = results_payload(result(N_SELECTED), version(), axis=axis)
    published = payload["loadings"]["axis"]["values"]
    assert len(published) == N_SELECTED == len(payload["loadings"]["components"][0])
    assert published[0] == 501.0
    assert payload["loadings"]["axis"]["unit"] == "nm"


def test_a_loadings_axis_that_does_not_match_is_refused_rather_than_served() -> None:
    """It used to be served. A wrong plot is worse than a missing one."""
    with pytest.raises(Exception, match="has to be in `node_axis`"):
        results_payload(result(N_SELECTED), version())


def test_an_estimator_on_the_whole_axis_is_unchanged() -> None:
    """A pipeline with no selection publishes exactly what it published before."""
    graph = pipeline(PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()))
    axis = node_axis(graph, "centre", version())
    payload = results_payload(result(AXIS.size), version(), axis=axis)
    assert payload["loadings"]["axis"]["values"] == AXIS.tolist()


def test_an_estimator_node_inherits_the_selection_too() -> None:
    graph = pipeline(
        PreprocessNode(id="window", inputs=("source",), step=RangeSelect(start=500.0, end=1000.0)),
        PreprocessNode(id="centre", inputs=("window",), step=MeanCentre()),
    )
    graph = Pipeline(
        project_id=graph.project_id,
        name=graph.name,
        nodes=[
            *graph.nodes,
            EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=2)),
        ],
    )
    assert node_axis(graph, "pca", version()).size == N_SELECTED
