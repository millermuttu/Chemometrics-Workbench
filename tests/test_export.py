"""The portable export forms (#213), against `docs/model-export.md`.

The claim that matters is `PROPOSAL.md` §9's constraint - *exported predictions
must match in-application predictions to within a stated numerical tolerance* -
and §5 of the export document states that tolerance with its reason. The test
that holds it runs the exported snippet in a subprocess where this package is
unreachable, so what is measured is the file rather than the code that wrote it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import EstimatorResult, execute
from chemometrics_workbench.export import (
    SCHEMA_VERSION,
    THRESHOLD,
    ExportError,
    json_model,
    python_snippet,
)
from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    BaselineCorrect,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    PCASpec,
    Pipeline,
    PLSDASpec,
    PLSRegressionSpec,
    PreprocessNode,
    RangeSelect,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
)
from chemometrics_workbench.project import create_project, write_array

#: `docs/model-export.md` §5. Wider than the parity harness's prediction class
#: because the application predicts from float32-stored arrays and an export
#: computes in float64 - measured at 7.09e-6 relative in
#: `docs/phase-2/exit-run.md`, and this is an order of magnitude above it.
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture(scope="module")
def tecator() -> Any:
    return load_tecator()


@pytest.fixture
def project(tmp_path: Path, tecator: Any) -> tuple[Path, DatasetVersion]:
    directory = tmp_path / "project"
    create_project(directory, "export tests")
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
        metadata_columns={
            "fat_class": [
                "high" if value > float(np.median(tecator.targets["fat"])) else "low"
                for value in tecator.targets["fat"]
            ]
        },
        array_path=array_path,
    )
    return directory, version


def _pipeline(version_id: Any, *nodes: Any) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="export",
        nodes=[SourceNode(id="source", version_id=version_id), *nodes],
    )


def _run(
    directory: Path, version: DatasetVersion, pipeline: Pipeline, node: str
) -> EstimatorResult:
    return execute(directory, pipeline, version).results[node]


def _predict(snippet: str, spectra: np.ndarray) -> np.ndarray:
    namespace: dict[str, Any] = {}
    exec(compile(snippet, "exported.py", "exec"), namespace)
    return np.asarray(namespace["predict"](spectra))


# --------------------------------------------------------------------------
# what is exported
# --------------------------------------------------------------------------


def test_a_foldable_chain_exports_a_bare_coefficient_vector(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """§1's ideal case: nothing unfoldable, so the residual chain is empty and
    the whole recipe lives in `b` and the intercept."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(
            id="savgol",
            inputs=("source",),
            step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        ),
        PreprocessNode(id="centre", inputs=("savgol",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=5, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    assert model["preprocessing"] == []
    assert len(model["coefficients"]) == version.n_variables
    assert model["model"]["threshold"] is None
    assert model["schema_version"] == SCHEMA_VERSION
    assert model["provenance"]["dataset_content_hash"] == version.content_hash
    assert model["provenance"]["pipeline_hash"] == pipeline.content_hash()

    predicted = _predict(python_snippet(model), tecator.spectra)
    np.testing.assert_allclose(
        predicted[np.asarray(result.rows)], result.predicted, rtol=RTOL, atol=ATOL
    )


def test_an_unfoldable_chain_carries_its_residual_steps(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """§1: the chain is cut at the *last* unfoldable step. SNV then Savitzky-
    Golay then centring carries the SNV and folds the other two."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(
            id="savgol",
            inputs=("snv",),
            step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        ),
        PreprocessNode(id="centre", inputs=("savgol",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=5, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    assert [step["kind"] for step in model["preprocessing"]] == ["snv"]
    predicted = _predict(python_snippet(model), tecator.spectra)
    np.testing.assert_allclose(
        predicted[np.asarray(result.rows)], result.predicted, rtol=RTOL, atol=ATOL
    )


def test_msc_carries_the_reference_it_was_fitted_with(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """§4: MSC's reference is a fitted parameter, so it travels with the step.
    Without it a snippet would regress against something else entirely."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="msc", inputs=("source",), step=MSC(reference="median")),
        PreprocessNode(id="centre", inputs=("msc",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=5, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    [step] = model["preprocessing"]
    assert step["kind"] == "msc" and step["reference"] == "median"
    assert len(step["reference_spectrum"]) == version.n_variables

    predicted = _predict(python_snippet(model), tecator.spectra)
    np.testing.assert_allclose(
        predicted[np.asarray(result.rows)], result.predicted, rtol=RTOL, atol=ATOL
    )


def test_a_range_selection_folds_and_the_export_keeps_the_raw_axis(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """A range selection drops coefficients rather than variables from the
    input: the exported model still takes a full raw spectrum, and the
    channels the selection dropped carry no weight."""
    directory, version = project
    axis = np.asarray(version.axis.values)
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(
            id="window",
            inputs=("source",),
            step=RangeSelect(start=float(axis[10]), end=float(axis[60])),
        ),
        PreprocessNode(id="centre", inputs=("window",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    assert len(model["coefficients"]) == version.n_variables
    assert len(model["axis"]["values"]) == version.n_variables
    coefficients = np.asarray(model["coefficients"])
    assert np.all(coefficients[:10] == 0.0), "outside the selection, no weight"

    predicted = _predict(python_snippet(model), tecator.spectra)
    np.testing.assert_allclose(
        predicted[np.asarray(result.rows)], result.predicted, rtol=RTOL, atol=ATOL
    )


def test_a_classification_exports_its_classes_and_its_threshold(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """§2 and `pls-da.md` §5: the cut is stated in the file rather than left
    for a reader to assume, and the snippet returns labels."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(id="centre", inputs=("snv",), step=MeanCentre()),
        EstimatorNode(
            id="plsda",
            inputs=("centre",),
            spec=PLSDASpec(n_components=5, class_column="fat_class"),
        ),
    )
    result = _run(directory, version, pipeline, "plsda")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    assert model["model"]["task"] == "classification"
    assert model["model"]["classes"] == ["high", "low"]
    assert model["model"]["threshold"] == THRESHOLD

    labels = _predict(python_snippet(model), tecator.spectra)
    served = [model["model"]["classes"][index] for index in result.predicted_class]
    assert list(labels[np.asarray(result.rows)]) == served


def test_a_model_below_a_split_exports_the_fold_it_was_fitted_on(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    """The model the application serves is fold zero's, so the export is too -
    fitted on its training rows, and reproducing its predictions on them."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        SplitNode(id="split", inputs=("snv",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=Autoscale()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    predicted = _predict(python_snippet(model), tecator.spectra)
    np.testing.assert_allclose(
        predicted[np.asarray(result.rows)], result.predicted, rtol=RTOL, atol=ATOL
    )
    # And on the rows it never saw, against what the application predicted for
    # them through the same fitted model.
    np.testing.assert_allclose(
        predicted[np.asarray(result.held_out)], result.held_out_predicted, rtol=RTOL, atol=ATOL
    )


# --------------------------------------------------------------------------
# what is refused, by name
# --------------------------------------------------------------------------


def test_a_baseline_is_refused_by_name(project: tuple[Path, DatasetVersion], tecator: Any) -> None:
    """§1: a solve per spectrum is not a few lines of arithmetic, and the
    honest answer is the sentence naming the step."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="baseline", inputs=("source",), step=BaselineCorrect(method="asls")),
        PreprocessNode(id="centre", inputs=("baseline",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=3, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    with pytest.raises(ExportError, match="'asls' baseline at 'baseline'"):
        json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)


def test_a_decomposition_has_no_prediction_to_export(
    project: tuple[Path, DatasetVersion], tecator: Any
) -> None:
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=3)),
    )
    result = _run(directory, version, pipeline, "pca")
    with pytest.raises(ExportError, match="is a decomposition, which has no prediction"):
        json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)


# --------------------------------------------------------------------------
# §9's constraint, in a subprocess that has nothing of ours
# --------------------------------------------------------------------------


def test_the_snippet_predicts_with_nothing_but_numpy_and_agrees_within_the_stated_tolerance(
    project: tuple[Path, DatasetVersion], tecator: Any, tmp_path: Path
) -> None:
    """`PROPOSAL.md` §9's constraint, held at `docs/model-export.md` §5's number.

    The snippet runs in a subprocess that makes this package unreachable, so
    what is measured is the exported file rather than the code that wrote it.
    The difference that remains is the float32 store's, which §5 explains and
    `docs/phase-2/exit-run.md` measured.
    """
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(
            id="savgol",
            inputs=("snv",),
            step=SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
        ),
        PreprocessNode(id="centre", inputs=("savgol",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=5, target="fat")
        ),
    )
    result = _run(directory, version, pipeline, "pls")
    model = json_model(result, pipeline=pipeline, version=version, raw=tecator.spectra)

    snippet = tmp_path / "exported.py"
    snippet.write_text(python_snippet(model), encoding="utf-8")
    spectra = tmp_path / "spectra.npy"
    np.save(spectra, np.asarray(tecator.spectra)[np.asarray(result.rows)])

    driver = tmp_path / "run_it.py"
    driver.write_text(
        """
import json, sys
import numpy as np


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "chemometrics_workbench":
            raise ImportError("the application is deliberately unreachable here")
        return None


sys.meta_path.insert(0, Blocked())
sys.path.insert(0, str(sys.argv[1]))
import exported  # noqa: E402

print(json.dumps({
    "predictions": [float(v) for v in exported.predict(np.load(sys.argv[2]))],
    "ours": sorted(n for n in sys.modules if n.startswith("chemometrics")),
}))
""",
        encoding="utf-8",
    )

    finished = subprocess.run(
        [sys.executable, "-I", str(driver), str(tmp_path), str(spectra)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    read = json.loads(finished.stdout)

    assert read["ours"] == [], "nothing of ours was imported to predict"
    np.testing.assert_allclose(read["predictions"], result.predicted, rtol=RTOL, atol=ATOL)

    # And the difference that remains is the store's, not an error: it is the
    # size `docs/phase-2/exit-run.md` measured, an order of magnitude inside
    # the tolerance §5 states.
    served = np.asarray(result.predicted)
    relative = np.max(np.abs(np.asarray(read["predictions"]) - served) / np.abs(served))
    assert relative < 1e-4, f"outside the stated tolerance at {relative:g}"
