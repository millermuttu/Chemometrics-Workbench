"""Peak resident memory of a ten-fold branch: the number #176 is judged by.

Not a test - a probe, run by hand and quoted in the feature's evidence:

    uv run python -m tests.memory_probe 6000 1200

It builds a project holding an `n x p` random matrix, runs source, SNV, a
ten-fold split, mean centring and a PCA, and prints the process's peak
resident set beside its size before the run. One float64 `n x p` array is
`8np` bytes - 57.6 MB at 6000 x 1200 - so the peak above baseline reads as a
count of arrays held at once. What the executor guarantees since #176 is that
the count is the current node's k fold arrays plus a transient, not every
node's; the fold arrays of the node being computed are the floor, and the
next step past it would be keeping them on disk and reading per fold in the
estimators.
"""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from chemometrics_workbench.executor import execute
from chemometrics_workbench.models import (
    SNV,
    AxisKind,
    DatasetVersion,
    EstimatorNode,
    KFoldSplit,
    MeanCentre,
    PCASpec,
    Pipeline,
    PreprocessNode,
    SourceNode,
    SplitNode,
    VariableAxis,
)
from chemometrics_workbench.project import create_project, write_array


def main(n: int, p: int, k: int = 10) -> None:
    with tempfile.TemporaryDirectory(prefix="chemometrics-memory-probe-") as temporary:
        directory = Path(temporary) / "project"
        project = create_project(directory, "memory probe")
        matrix = np.random.default_rng(0).normal(size=(n, p)) + 5.0
        array_path, content_hash = write_array(directory, matrix)
        del matrix
        version = DatasetVersion(
            dataset_id=uuid4(),
            version=1,
            content_hash=content_hash,
            n_samples=n,
            n_variables=p,
            axis=VariableAxis(kind=AxisKind.INDEX, values=[float(i) for i in range(p)]),
            array_path=array_path,
        )
        pipeline = Pipeline(
            project_id=project.project_id,
            name="memory probe",
            nodes=[
                SourceNode(id="source", version_id=version.version_id),
                PreprocessNode(id="snv", inputs=("source",), step=SNV()),
                SplitNode(id="split", inputs=("snv",), spec=KFoldSplit(n_splits=k, seed=42)),
                PreprocessNode(id="centre", inputs=("split",), step=MeanCentre()),
                EstimatorNode(id="pca", inputs=("centre",), spec=PCASpec(n_components=5)),
            ],
        )
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        run = execute(directory, pipeline, version)
        elapsed = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        per_array = 8 * n * p / 2**20
        held = (peak - baseline) / 1024 / per_array
        print(
            f"{n} x {p}, {k}-fold: peak RSS {peak / 1024:.0f} MB, baseline "
            f"{baseline / 1024:.0f} MB, {held:.1f} arrays of {per_array:.1f} MB above it, "
            f"{elapsed:.1f} s; displays are {type(run.displays).__name__}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 10)
