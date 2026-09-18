"""The model artifact (#211), against `docs/model-artifact.md`.

The claim that matters is the last clause of `PROPOSAL.md` §8.4 - *readable
without this application* - and the test that proves it runs in a subprocess
where importing `chemometrics_workbench` fails. Everything else here is the
round trip: every number the application fitted comes back out of the file
unchanged.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from chemometrics_workbench.artifact import (
    MANIFEST,
    SCHEMA_VERSION,
    ArtifactError,
    read_artifact,
    write_artifact,
)
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import capture_environment, execute
from chemometrics_workbench.models import (
    SNV,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    PCASpec,
    Pipeline,
    PLSRegressionSpec,
    PreprocessNode,
    SourceNode,
    SplitNode,
)
from chemometrics_workbench.project import create_project, write_array


def _pipeline(version_id: object, *nodes: object) -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        name="artifact",
        nodes=[SourceNode(id="source", version_id=version_id), *nodes],
    )


@pytest.fixture(scope="module")
def tecator() -> object:
    return load_tecator()


@pytest.fixture
def project(tmp_path: Path, tecator: object) -> tuple[Path, DatasetVersion]:
    """A project holding Tecator and the version describing it, as
    `test_executor.py`'s fixture builds one - written out here rather than
    imported, because a fixture imported across modules is a name two files
    have to keep in step."""
    directory = tmp_path / "project"
    create_project(directory, "artifact tests")
    array_path, _ = write_array(directory, tecator.spectra)  # type: ignore[attr-defined]
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=tecator.source.file_hash,  # type: ignore[attr-defined]
        n_samples=tecator.n_samples,  # type: ignore[attr-defined]
        n_variables=tecator.n_variables,  # type: ignore[attr-defined]
        axis=tecator.axis,  # type: ignore[attr-defined]
        sample_ids=list(tecator.sample_ids),  # type: ignore[attr-defined]
        targets={
            name: [float(v) for v in values]
            for name, values in tecator.targets.items()  # type: ignore[attr-defined]
        },
        array_path=array_path,
    )
    return directory, version


@pytest.fixture
def fitted(project: tuple[Path, DatasetVersion]) -> tuple[Path, DatasetVersion, Pipeline, object]:
    """A PLS below a ten-fold split: the model with the most to carry."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        SplitNode(id="split", inputs=("snv",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
        EstimatorNode(
            id="pls", inputs=("centre",), spec=PLSRegressionSpec(n_components=4, target="fat")
        ),
    )
    run = execute(directory, pipeline, version)
    return directory, version, pipeline, run


def test_every_number_the_model_was_fitted_with_comes_back_out(
    fitted: tuple[Path, DatasetVersion, Pipeline, object],
) -> None:
    """§7: the arrays are float64 and exact. The application stores its
    intermediates as float32; an artifact is a record, not a cache."""
    directory, version, pipeline, run = fitted
    result = run.results["pls"]  # type: ignore[attr-defined]
    axis = np.asarray(version.axis.values, dtype=np.float64)

    path = directory / "pls.cwmodel"
    content_hash = write_artifact(
        path,
        result,
        pipeline=pipeline,
        version=version,
        node_axis=axis,
        split=run.resolved_splits[0],  # type: ignore[attr-defined]
        environment=capture_environment(),
    )
    assert content_hash.startswith("sha256:") and len(content_hash) == 71

    read = read_artifact(path)
    assert read.schema_version == SCHEMA_VERSION
    assert read.task == "regression"

    np.testing.assert_array_equal(read.arrays["coefficients"], np.asarray(result.coefficients))
    np.testing.assert_array_equal(read.arrays["x_mean"], np.asarray(result.x_mean))
    np.testing.assert_array_equal(read.arrays["vip"], np.asarray(result.vip))
    np.testing.assert_array_equal(read.arrays["loadings"], np.asarray(result.loadings))
    np.testing.assert_array_equal(read.arrays["rotations"], np.asarray(result.rotations))
    np.testing.assert_array_equal(read.arrays["train_indices"], np.asarray(result.rows))
    np.testing.assert_array_equal(read.arrays["test_indices"], np.asarray(result.held_out))
    assert all(read.arrays[name].dtype == np.float64 for name in ("coefficients", "x_mean", "vip"))
    assert read.arrays["train_indices"].dtype == np.int64

    model = read.manifest["model"]
    assert model["y_mean"] == result.y_mean
    assert (model["target"], model["n_components"]) == ("fat", result.n_components)
    assert read.manifest["metrics"]["rmsecv"] == result.metrics["rmsecv"]
    assert read.manifest["split"] == {"node_id": "split", "fold": 0, "n_folds": 10}
    assert read.manifest["dataset"]["content_hash"] == version.content_hash
    assert read.manifest["environment"]["app_version"]


def test_the_pipeline_travels_by_value_and_parses_back(
    fitted: tuple[Path, DatasetVersion, Pipeline, object],
) -> None:
    """§3: a pipeline gets edited, and an artifact whose recipe pointed at one
    would lose its meaning the moment it was."""
    directory, version, pipeline, run = fitted
    path = directory / "pls.cwmodel"
    write_artifact(
        path,
        run.results["pls"],  # type: ignore[attr-defined]
        pipeline=pipeline,
        version=version,
        node_axis=np.asarray(version.axis.values),
        split=run.resolved_splits[0],  # type: ignore[attr-defined]
    )

    carried = read_artifact(path).pipeline
    assert [node.id for node in carried.nodes] == [node.id for node in pipeline.nodes]
    assert carried.content_hash() == pipeline.content_hash()


def test_a_decomposition_carries_what_it_has_and_not_what_it_does_not(
    project: tuple[Path, DatasetVersion],
) -> None:
    """§7: a reader must not assume any array beyond `dataset_axis`, so absent
    is absent rather than an empty array that reads as a model with no
    coefficients."""
    directory, version = project
    pipeline = _pipeline(
        version.version_id,
        PreprocessNode(id="centre", inputs=("source",), step=MeanCentre()),
        EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=3)),
    )
    run = execute(directory, pipeline, version)

    path = directory / "pca.cwmodel"
    write_artifact(
        path,
        run.results["pca"],
        pipeline=pipeline,
        version=version,
        node_axis=np.asarray(version.axis.values),
    )
    read = read_artifact(path)

    assert read.task == "decomposition"
    assert "coefficients" not in read.arrays
    assert "x_mean" not in read.arrays
    assert "train_indices" not in read.arrays, "no split above it"
    assert read.manifest["split"] is None
    assert read.manifest["model"]["y_mean"] is None
    assert read.arrays["loadings"].shape == (3, version.n_variables)


def test_a_file_written_by_a_newer_application_is_refused_by_name(tmp_path: Path) -> None:
    """§2, and the rule `db.py` applies to a database: a newer writer may have
    added a field whose absence this reader would take as a default."""
    path = tmp_path / "future.cwmodel"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MANIFEST, json.dumps({"schema_version": SCHEMA_VERSION + 3, "arrays": {}}))

    with pytest.raises(ArtifactError, match=f"schema version is {SCHEMA_VERSION + 3} and this one"):
        read_artifact(path)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda path: path.write_bytes(b"not a zip"), "is not a zip archive"),
        (
            lambda path: zipfile.ZipFile(path, "w").writestr("readme.txt", "hello"),
            "holds no manifest.json",
        ),
        (
            lambda path: zipfile.ZipFile(path, "w").writestr(MANIFEST, "{not json"),
            "is not valid JSON",
        ),
        (
            lambda path: zipfile.ZipFile(path, "w").writestr(MANIFEST, json.dumps({"a": 1})),
            "states no schema_version",
        ),
        (
            lambda path: zipfile.ZipFile(path, "w").writestr(
                MANIFEST,
                json.dumps(
                    {"schema_version": 1, "arrays": {"loadings": {"file": "arrays/gone.npy"}}}
                ),
            ),
            "which the archive does not hold",
        ),
    ],
)
def test_an_unreadable_artifact_is_a_sentence_not_a_stack_trace(
    tmp_path: Path, build: object, message: str
) -> None:
    path = tmp_path / "broken.cwmodel"
    build(path)  # type: ignore[operator]
    with pytest.raises(ArtifactError, match=message):
        read_artifact(path)


READER = '''
"""Read a model artifact with nothing but the standard library and NumPy.

The point is not that this package happens to be absent - in a developer's
environment it is installed - but that reading never reaches for it. So it is
made unreachable: any import of it raises, and if the format needed it, this
script would fail rather than quietly succeed.
"""
import json, sys, zipfile
import numpy as np


class Blocked:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] == "chemometrics_workbench" else None

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "chemometrics_workbench":
            raise ImportError("the application is deliberately unreachable here")
        return None


sys.meta_path.insert(0, Blocked())
try:
    import chemometrics_workbench  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("the block did not work, so this proves nothing")

with zipfile.ZipFile(sys.argv[1]) as archive:
    manifest = json.loads(archive.read("manifest.json"))
    with archive.open(manifest["arrays"]["coefficients"]["file"]) as handle:
        coefficients = np.load(handle)
    with archive.open(manifest["arrays"]["x_mean"]["file"]) as handle:
        x_mean = np.load(handle)

print(json.dumps({
    "schema_version": manifest["schema_version"],
    "task": manifest["model"]["task"],
    "target": manifest["model"]["target"],
    "y_mean": manifest["model"]["y_mean"],
    "nodes": [node["id"] for node in manifest["pipeline"]["nodes"]],
    "n_coefficients": int(coefficients.size),
    "coefficient_0": float(coefficients[0]),
    "x_mean_0": float(x_mean[0]),
    "modules": sorted(
        name for name in sys.modules if name.startswith("chemometrics")
    ),
}))
'''


def test_an_artifact_opens_with_nothing_but_the_standard_library_and_numpy(
    fitted: tuple[Path, DatasetVersion, Pipeline, object], tmp_path: Path
) -> None:
    """`PROPOSAL.md` §8.4's last clause, and the reason the format is a zip of
    JSON and `.npy` rather than a pickle.

    The subprocess makes this package *unreachable* - a meta-path finder that
    raises on any import of it - rather than relying on it being absent, which
    in a developer's environment it is not. If the format needed us, the
    script would fail rather than quietly succeed.
    """
    directory, version, pipeline, run = fitted
    result = run.results["pls"]  # type: ignore[attr-defined]
    path = directory / "pls.cwmodel"
    write_artifact(
        path,
        result,
        pipeline=pipeline,
        version=version,
        node_axis=np.asarray(version.axis.values),
        split=run.resolved_splits[0],  # type: ignore[attr-defined]
    )

    script = tmp_path / "read_it.py"
    script.write_text(READER, encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-I", str(script), str(path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    read = json.loads(finished.stdout)

    assert read["schema_version"] == SCHEMA_VERSION
    assert (read["task"], read["target"]) == ("regression", "fat")
    assert read["y_mean"] == result.y_mean
    assert read["nodes"] == [node.id for node in pipeline.nodes]
    assert read["n_coefficients"] == result.n_variables
    assert read["coefficient_0"] == result.coefficients[0]
    assert read["x_mean_0"] == result.x_mean[0]
    assert read["modules"] == [], "nothing of ours was imported to read it"


def test_the_archive_is_a_plain_zip_anyone_can_list(
    fitted: tuple[Path, DatasetVersion, Pipeline, object],
) -> None:
    """Copyable between machines means openable by whatever is to hand."""
    directory, version, pipeline, run = fitted
    path = directory / "pls.cwmodel"
    write_artifact(
        path,
        run.results["pls"],  # type: ignore[attr-defined]
        pipeline=pipeline,
        version=version,
        node_axis=np.asarray(version.axis.values),
        split=run.resolved_splits[0],  # type: ignore[attr-defined]
    )

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert MANIFEST in names
        assert all(name.startswith("arrays/") for name in names - {MANIFEST})
        assert archive.testzip() is None, "no corrupt member"
        # The manifest states each array's dtype and shape, so a reader can see
        # what it is about to load without loading it (§7).
        manifest = json.loads(archive.read(MANIFEST))
        for name, entry in manifest["arrays"].items():
            with archive.open(entry["file"]) as handle:
                values = np.load(io.BytesIO(handle.read()))
            assert list(values.shape) == entry["shape"], name
            assert str(values.dtype) == entry["dtype"], name
