"""The select-variables step (#280): what "Apply selection" writes.

It holds explicit column positions, fits nothing, and folds into exported
coefficients, so a selection made by VIP, iPLS or CARS is part of the recipe
and travels with the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from chemometrics_workbench.api import node_axis
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import execute
from chemometrics_workbench.export import json_model, python_snippet
from chemometrics_workbench.models import (
    DatasetVersion,
    EstimatorNode,
    MeanCentre,
    Pipeline,
    PLSRegressionSpec,
    PreprocessNode,
    PreprocessStep,
    RangeSelect,
    SelectVariables,
    SourceNode,
)
from chemometrics_workbench.preprocessing import (
    Selection,
    SelectVariablesTransformer,
    from_spec,
)
from chemometrics_workbench.project import create_project, write_array
from chemometrics_workbench.regression import FOLDABLE

KEPT = [3, 10, 11, 12, 40, 41, 77, 98]


def test_positions_are_stored_sorted_once_and_round_trip() -> None:
    step = SelectVariables(indices=[12, 3, 12, 10])
    assert step.indices == [3, 10, 12]
    assert step == SelectVariables(indices=[3, 10, 12])
    parsed: object = TypeAdapter(PreprocessStep).validate_json(step.model_dump_json())
    assert parsed == step and isinstance(parsed, SelectVariables)


@pytest.mark.parametrize("indices", [[], [-1, 2]])
def test_an_empty_or_negative_selection_is_refused(indices: list[int]) -> None:
    with pytest.raises(ValidationError):
        SelectVariables(indices=indices)


def test_it_keeps_the_named_columns_and_fits_nothing() -> None:
    rng = np.random.default_rng(0)
    fitted_on, applied_to = rng.standard_normal((5, 100)), rng.standard_normal((7, 100))
    transformer = from_spec(SelectVariables(indices=KEPT))
    assert isinstance(transformer, SelectVariablesTransformer)
    assert isinstance(transformer, FOLDABLE) and isinstance(transformer, Selection)
    out = transformer.fit(fitted_on).transform(applied_to)
    np.testing.assert_array_equal(out, applied_to[:, KEPT])


def test_a_position_past_the_input_is_refused_naming_the_width() -> None:
    with pytest.raises(ValueError, match="has 50 variables"):
        SelectVariablesTransformer([3, 60]).fit(np.zeros((2, 50)))


def test_the_axis_follows_a_selection_after_a_range() -> None:
    tecator = load_tecator()
    axis = np.asarray(tecator.axis.values)
    version = _version(tecator, "arrays/none.npy")
    graph = Pipeline(
        project_id=uuid4(),
        name="axis",
        nodes=[
            SourceNode(id="source", version_id=version.version_id),
            PreprocessNode(
                id="window",
                inputs=("source",),
                step=RangeSelect(start=float(axis[20]), end=float(axis[80])),
            ),
            PreprocessNode(id="pick", inputs=("window",), step=SelectVariables(indices=[0, 5, 60])),
        ],
    )
    assert node_axis(graph, "pick", version).tolist() == axis[[20, 25, 80]].tolist()


def test_a_pls_under_a_selection_runs_folds_and_exports(tmp_path: Path) -> None:
    """The step's whole job downstream: run, fold to zeros where it dropped a
    variable, and export as a bare coefficient vector the snippet predicts with."""
    tecator = load_tecator()
    directory = tmp_path / "project"
    create_project(directory, "select")
    array_path, _ = write_array(directory, tecator.spectra)
    version = _version(tecator, array_path)
    pipeline = Pipeline(
        project_id=uuid4(),
        name="select",
        nodes=[
            SourceNode(id="source", version_id=version.version_id),
            PreprocessNode(id="pick", inputs=("source",), step=SelectVariables(indices=KEPT)),
            PreprocessNode(id="centre", inputs=("pick",), step=MeanCentre()),
            EstimatorNode(
                id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
            ),
        ],
    )
    result = execute(directory, pipeline, version).results["pls"]
    assert result.n_variables == len(KEPT)

    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)
    assert model["preprocessing"] == []
    coefficients = np.asarray(model["coefficients"])
    dropped = np.setdiff1d(np.arange(version.n_variables), KEPT)
    assert coefficients.size == version.n_variables and np.all(coefficients[dropped] == 0.0)

    namespace: dict[str, Any] = {}
    exec(compile(python_snippet(model), "exported.py", "exec"), namespace)
    predicted = namespace["predict"](tecator.spectra)[np.asarray(result.rows)]
    np.testing.assert_allclose(predicted, result.predicted, rtol=1e-4, atol=1e-6)


def _version(tecator: Any, array_path: str) -> DatasetVersion:
    return DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=tecator.source.file_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        targets={"fat": [float(v) for v in tecator.targets["fat"]]},
        array_path=array_path,
    )


@pytest.mark.parametrize(
    "above",
    [
        RangeSelect(start=0.0, end=1.0),  # replaced below by a real window
        SelectVariables(indices=list(range(10, 80))),
    ],
)
def test_a_range_selection_below_another_selection_runs(tmp_path: Path, above: Any) -> None:
    """#312: the executor handed every step the dataset's axis, so a range
    selection below any selection refused an axis longer than its input."""
    tecator = load_tecator()
    axis = np.asarray(tecator.axis.values)
    if isinstance(above, RangeSelect):
        above = RangeSelect(start=float(axis[10]), end=float(axis[80]))
    directory = tmp_path / "project"
    create_project(directory, "nested")
    array_path, _ = write_array(directory, tecator.spectra)
    version = _version(tecator, array_path)
    pipeline = Pipeline(
        project_id=uuid4(),
        name="nested",
        nodes=[
            SourceNode(id="source", version_id=version.version_id),
            PreprocessNode(id="outer", inputs=("source",), step=above),
            PreprocessNode(
                id="inner",
                inputs=("outer",),
                step=RangeSelect(start=float(axis[20]), end=float(axis[50])),
            ),
            PreprocessNode(id="pick", inputs=("inner",), step=SelectVariables(indices=[0, 30])),
            EstimatorNode(
                id="pls", inputs=("pick",), spec=PLSRegressionSpec(n_components=1, target="fat")
            ),
        ],
    )
    result = execute(directory, pipeline, version).results["pls"]
    assert result.n_variables == 2
    # And the axis it is on is the real one, end to end.
    assert node_axis(pipeline, "pls", version).tolist() == axis[[20, 50]].tolist()
