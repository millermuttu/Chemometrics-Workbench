"""Coefficients against the raw axis, and the chains that cannot have them.

`PROPOSAL.md` §9 promises a portable model: a coefficient vector plus a snippet
that depends only on NumPy. `EstimatorResult.coefficients` is not that — it is
`b` on whatever the last preprocessing node produced. #144 folds the chain back
out, and the claim it makes is testable in one line: `intercept + X_raw @ b`
must reproduce what the model predicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from chemometrics_workbench.api import folded_coefficients
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import EstimatorResult, execute
from chemometrics_workbench.models import (
    SNV,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    Pipeline,
    PLSRegressionSpec,
    PreprocessNode,
    RangeSelect,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
)
from chemometrics_workbench.project import create_project, read_array, write_array

SAVGOL = SavitzkyGolay(window_length=11, polyorder=2, deriv=1)


@pytest.fixture(scope="module")
def tecator() -> Any:
    return load_tecator()


@pytest.fixture
def project(tmp_path: Path, tecator: Any) -> tuple[Path, DatasetVersion]:
    directory = tmp_path / "project"
    create_project(directory, "folded coefficients")
    array_path, content_hash = write_array(directory, tecator.spectra)
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=content_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        targets={name: [float(v) for v in values] for name, values in tecator.targets.items()},
        array_path=array_path,
    )
    return directory, version


def fit(directory: Path, version: DatasetVersion, *nodes: Any) -> tuple[Pipeline, EstimatorResult]:
    pipeline = Pipeline(
        project_id=uuid4(),
        name="test",
        nodes=[SourceNode(id="source", version_id=version.version_id), *nodes],
    )
    run = execute(directory, pipeline, version, use_cache=False)
    return pipeline, run.results["pls"]


def test_a_foldable_chain_reproduces_the_models_own_predictions(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The whole claim, in one assertion.

    Compared against the **stored** array rather than the original float64
    spectra, because that is the matrix the model was fitted through: every
    array reaches a kernel via the float32 store (#83). Against the float64
    file the agreement is 3.6e-05 rather than 1.4e-06, and the difference is
    the storage boundary, not the folding.
    """
    directory, version = project
    pipeline, result = fit(
        directory,
        version,
        PreprocessNode(id="win", inputs=("source",), step=RangeSelect(start=860.0, end=1040.0)),
        PreprocessNode(id="sg", inputs=("win",), step=SAVGOL),
        PreprocessNode(id="centre", inputs=("sg",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=6, target="fat")
        ),
    )
    folded = folded_coefficients(directory, pipeline, "pls", version, result)

    assert folded["available"] is True
    assert folded["target"] == "fat"
    # One coefficient per *raw* variable, not per surviving one: the range
    # selection is folded in, so the dropped channels carry a zero.
    assert len(folded["coefficients"]) == version.n_variables
    assert len(folded["axis"]["values"]) == version.n_variables

    raw = read_array(directory, version.array_path)
    b = np.asarray(folded["coefficients"])
    predicted = folded["intercept"] + raw @ b
    assert predicted == pytest.approx(result.predicted, abs=1e-4)


def test_the_channels_a_range_selection_dropped_carry_no_weight(
    project: tuple[Path, DatasetVersion],
) -> None:
    directory, version = project
    pipeline, result = fit(
        directory,
        version,
        PreprocessNode(id="win", inputs=("source",), step=RangeSelect(start=900.0, end=1000.0)),
        PreprocessNode(id="centre", inputs=("win",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    folded = folded_coefficients(directory, pipeline, "pls", version, result)

    axis = np.asarray(version.axis.values)
    b = np.asarray(folded["coefficients"])
    outside = (axis < 900.0) | (axis > 1000.0)
    assert outside.any(), "the selection has to drop something for this to mean anything"
    assert np.allclose(b[outside], 0.0)
    assert np.any(b[~outside] != 0.0)


def test_below_a_split_it_folds_fold_zeros_parameters(
    project: tuple[Path, DatasetVersion],
) -> None:
    """The model is fold zero's, so the chain refitted here must be too."""
    directory, version = project
    pipeline, result = fit(
        directory,
        version,
        SplitNode(id="split", inputs=("source",), spec=KFoldSplit(n_splits=5, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    folded = folded_coefficients(directory, pipeline, "pls", version, result)

    raw = read_array(directory, version.array_path)
    b = np.asarray(folded["coefficients"])
    predicted = folded["intercept"] + raw[np.asarray(result.rows)] @ b
    assert predicted == pytest.approx(result.predicted, abs=1e-4)


def test_an_snv_makes_it_unavailable_and_says_which_step(
    project: tuple[Path, DatasetVersion],
) -> None:
    """§7's "says so when it is not available".

    Not an error: SNV depends on the sample being predicted, so the chain has
    no fixed linear map and the honest answer is the sentence naming the step
    rather than an approximation nobody asked for.
    """
    directory, version = project
    pipeline, result = fit(
        directory,
        version,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(id="centre", inputs=("snv",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    folded = folded_coefficients(directory, pipeline, "pls", version, result)

    assert folded["available"] is False
    assert "SNVTransformer" in folded["reason"]
    assert "cannot be folded" in folded["reason"]
    assert "coefficients" not in folded


def test_it_survives_a_cache_hit(project: tuple[Path, DatasetVersion]) -> None:
    """The reason this is derived rather than stored beside the arrays.

    A node whose key hits the cache never builds a transformer, so anything
    that kept the fitted chain would have nothing to offer on the second run.
    Refitting from the recipe gives the same answer either way.
    """
    directory, version = project
    nodes = (
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    pipeline = Pipeline(
        project_id=uuid4(),
        name="cached",
        nodes=[SourceNode(id="source", version_id=version.version_id), *nodes],
    )
    cold = execute(directory, pipeline, version).results["pls"]
    hot_run = execute(directory, pipeline, version)
    assert hot_run.reused, "the second run has to have hit the cache for this to test anything"

    first = folded_coefficients(directory, pipeline, "pls", version, cold)
    second = folded_coefficients(directory, pipeline, "pls", version, hot_run.results["pls"])
    assert first == second
