"""Generate the Phase 1.1 contract fixtures.

Phase 1.1 has no executor behind it. These files are what the stub server
(#53) returns, and they are generated rather than written by hand for one
reason: a payload someone typed encodes an API shape nobody agreed to, and
Phase 1.2 then rewrites both sides. Every number below comes out of a kernel
in `chemometrics_workbench`, so the fixture *is* the contract and 1.2 has to
either meet it or change it deliberately.

Run it:

    uv run python stub/generate_fixtures.py

## This module computes nothing

It loads, it calls kernels, it reshapes their output into a response body. The
one piece of arithmetic it owns is decimation, which is a presentation
concern with no kernel and is marked where it happens. If you find yourself
adding a formula here, it belongs in `chemometrics_workbench` — the parity
report is generated under the same rule and for the same reason.

## What is real and what is a guess

**Real, and 1.2 must reproduce it:** every array, every metric, every
confidence limit, the dataset's content hash and source record, the pipeline's
content hash, and the fold indices.

**A guess, and 1.2 is free to change it:** the envelope shapes that
`models.py` does not cover — how a list is paginated (it is not, yet), what an
error body looks like, how a job reports progress, and how node run-state is
attached to a pipeline. Each is marked GUESS at the point it is built.

## Determinism

Every id is a UUID5 of a fixed namespace and every timestamp is pinned. The
schema's `default_factory` would otherwise put a fresh UUID and the wall clock
into the output, and the file could never be checked for having changed.

## The dataset is Tecator, and why

It is the only reference dataset committed to the repository, so this script
runs on a fresh checkout with no network. Corn and gasoline are downloaded on
first use, which would make the fixtures unbuildable offline and their content
dependent on a cache.

**Publishing a result from this dataset obliges you to name the instrument and
company (Tecator).** The stub server's dataset payload carries that note.

One consequence worth stating: at 100 variables, nothing is dropped along the
wavelength axis, so the `variables_kept` field is populated but the x-axis
decimation path is not exercised. The trace cap and the density band are
exercised — 240 spectra against a cap of 60. Real x-decimation belongs to 1.2,
where `PROPOSAL.md` §13 puts it, against a dataset big enough to need it.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import numpy as np
import scipy
from numpy.typing import NDArray
from pydantic import TypeAdapter

import chemometrics_workbench as cw
from chemometrics_workbench import preprocessing, validation
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.decomposition import PCA
from chemometrics_workbench.models import (
    MSC,
    SNV,
    Autoscale,
    Dataset,
    DatasetVersion,
    Environment,
    EstimatorNode,
    Experiment,
    ExperimentStatus,
    KFoldSplit,
    MeanCentre,
    Metrics,
    PCASpec,
    Pipeline,
    PreprocessNode,
    PreprocessStep,
    Project,
    ResolvedSplit,
    SavitzkyGolay,
    SourceNode,
    SplitNode,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Fixed so that ids are stable across runs; the value is arbitrary and means
# nothing beyond "these fixtures".
NAMESPACE = UUID("6f1f7a1e-1c2b-4f8a-9d3e-0a5b7c9d1e2f")

# Pinned so the output can be diffed. Not "now", deliberately.
PINNED = datetime(2026, 8, 24, 9, 0, 0, tzinfo=UTC)

# How many spectra are drawn individually before the rest becomes a band, and
# how many points a trace may carry. Both are presentation limits, not
# scientific ones; 1.2 owns their real values (PROPOSAL.md §13).
MAX_TRACES = 60
MAX_POINTS = 1000


def _id(name: str) -> UUID:
    return uuid5(NAMESPACE, name)


def _round(values: object, places: int = 6) -> list[float]:
    """Serialise an array for the wire.

    Six decimal places is far finer than a plot can draw and keeps the files
    readable. The *limits and metrics* are not rounded — a confidence limit is
    a number the UI displays, and rounding it here would mean the interface
    shows something the kernel did not produce.
    """
    return [round(float(v), places) for v in np.asarray(values).ravel()]


# --------------------------------------------------------------------------
# the project, its dataset and the recipe
# --------------------------------------------------------------------------


def build_domain() -> dict[str, Any]:
    """The entities, straight through Pydantic. No shape is invented here."""
    tecator = load_tecator()

    project = Project(
        project_id=_id("project"),
        name="Tecator meat study",
        description="Phase 1.1 fixture project. Not a real study.",
        directory="~/lab/tecator-meat",
        created_at=PINNED,
    )
    dataset = Dataset(
        dataset_id=_id("dataset"),
        project_id=project.project_id,
        name="tecator_raw",
        description=(
            "240 meat samples, 100 NIT absorbance channels over 850-1050 nm. "
            "Measured on a Tecator Infratec Food and Feed Analyzer; publishing a "
            "result from this data obliges you to name the instrument and company."
        ),
        created_at=PINNED,
    )
    version = DatasetVersion(
        version_id=_id("version"),
        dataset_id=dataset.dataset_id,
        version=1,
        # The dataset's identity is the hash of the file it came from, which
        # the loader already computed and verified against a committed sum.
        content_hash=tecator.source.file_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        targets={name: _round(values, 4) for name, values in tecator.targets.items()},
        source=tecator.source.model_copy(update={"imported_at": PINNED}),
        array_path="arrays/tecator_v1.npy",
        created_at=PINNED,
    )
    return {"tecator": tecator, "project": project, "dataset": dataset, "version": version}


def build_pipeline(project_id: UUID, version_id: UUID) -> Pipeline:
    """Four preprocessing branches off one source, as the artboard draws it.

    Branch D carries the split node, placed *downstream* of preprocessing and
    upstream of centring — which is the ordering `metrics-and-validation.md` §9
    calls correct, and the opposite of the mistake the 1.2 validator will warn
    about.
    """
    savgol = SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
    nodes = [
        SourceNode(id="source", version_id=version_id),
        # A: SNV
        PreprocessNode(id="snv", inputs=("source",), step=SNV()),
        PreprocessNode(id="centre_a", inputs=("snv",), step=MeanCentre()),
        EstimatorNode(id="pca_a", inputs=("centre_a",), spec=PCASpec(n_components=5)),
        # B: MSC
        PreprocessNode(id="msc", inputs=("source",), step=MSC()),
        PreprocessNode(id="centre_b", inputs=("msc",), step=MeanCentre()),
        EstimatorNode(id="pca_b", inputs=("centre_b",), spec=PCASpec(n_components=5)),
        # C: Savitzky-Golay first derivative
        PreprocessNode(id="savgol", inputs=("source",), step=savgol),
        PreprocessNode(id="autoscale_c", inputs=("savgol",), step=Autoscale()),
        EstimatorNode(id="pca_c", inputs=("autoscale_c",), spec=PCASpec(n_components=5)),
        # D: SNV then Savitzky-Golay - the path the exit criterion walks
        PreprocessNode(id="snv_savgol", inputs=("snv",), step=savgol),
        SplitNode(id="split_d", inputs=("snv_savgol",), spec=KFoldSplit(n_splits=10, seed=42)),
        PreprocessNode(id="centre_d", inputs=("split_d",), step=MeanCentre()),
        EstimatorNode(id="pca_d", inputs=("centre_d",), spec=PCASpec(n_components=5)),
    ]
    return Pipeline(
        pipeline_id=_id("pipeline"),
        project_id=project_id,
        name="Scatter correction comparison",
        nodes=nodes,
        created_at=PINNED,
    )


# --------------------------------------------------------------------------
# running the recipe
# --------------------------------------------------------------------------


def run_preprocessing(
    pipeline: Pipeline, spectra: NDArray[np.float64], axis: object
) -> dict[str, NDArray[np.float64]]:
    """Walk the graph and keep every intermediate array.

    This is a miniature of what 1.2's executor does properly — topological
    order, one output per node, `from_spec` as the seam. It is here because
    the spectra view needs the array at every node, not only at the end, and
    it is *not* the executor: no caching, no jobs, no fold handling beyond
    what branch D needs.
    """
    by_id = {node.id: node for node in pipeline.nodes}
    outputs: dict[str, NDArray[np.float64]] = {}

    def output_of(node_id: str) -> NDArray[np.float64]:
        if node_id in outputs:
            return outputs[node_id]
        node = by_id[node_id]
        if node.type == "source":
            result = spectra
        elif node.type == "preprocess":
            transformer = preprocessing.from_spec(node.step, axis=axis)
            result = transformer.fit_transform(output_of(node.inputs[0]))
        elif node.type == "split":
            # A split does not change the array. It changes which rows the
            # nodes below it are fitted on, which is branch D's business
            # below - here it passes through so the spectra view has
            # something to draw at every node.
            result = output_of(node.inputs[0])
        else:
            raise AssertionError(f"{node.type} is not a preprocessing node")
        outputs[node_id] = result
        return result

    for node in pipeline.nodes:
        if node.type != "estimator":
            output_of(node.id)
    return outputs


def fit_pca(matrix: NDArray[np.float64], n_components: int) -> PCA:
    return PCA(n_components=n_components).fit(matrix)


# --------------------------------------------------------------------------
# response bodies
# --------------------------------------------------------------------------


def spectra_payload(
    node_id: str,
    label: str,
    matrix: NDArray[np.float64],
    axis_values: list[float],
    axis_unit: str,
    sample_ids: list[str],
    ordinate: str,
) -> dict[str, Any]:
    """One spectra plot's worth of data, decimated for the wire.

    GUESS: the envelope is this module's invention - `models.py` describes
    datasets, not plot payloads. What 1.2 must keep is the split between
    individually drawn traces and a band, because that is what `PROPOSAL.md`
    §13 requires of any implementation; the field names are negotiable.
    """
    n_spectra, n_variables = matrix.shape

    # ponytail: strided x-decimation. It is exact at 100 variables because the
    # stride is 1, and it would clip a narrow peak on a wide axis - 1.2 should
    # use a min/max envelope per bucket once a dataset needs one.
    stride = max(1, int(np.ceil(n_variables / MAX_POINTS)))
    kept = np.arange(0, n_variables, stride)

    banded = n_spectra > MAX_TRACES
    if banded:
        # An evenly spaced subset, so the drawn traces span the set rather
        # than showing its first sixty samples.
        drawn = np.linspace(0, n_spectra - 1, MAX_TRACES).round().astype(int)
    else:
        drawn = np.arange(n_spectra)

    payload: dict[str, Any] = {
        "node_id": node_id,
        "label": label,
        "axis": {
            "kind": "wavelength_nm",
            "unit": axis_unit,
            "values": [round(float(axis_values[i]), 4) for i in kept],
        },
        "ordinate": {"label": ordinate},
        "n_spectra": int(n_spectra),
        "decimation": {
            "variables_total": int(n_variables),
            "variables_kept": int(kept.size),
            "traces_total": int(n_spectra),
            "traces_drawn": int(drawn.size),
            "banded": banded,
        },
        "traces": [
            {
                "index": int(i),
                "sample_id": sample_ids[i],
                "y": _round(matrix[i, kept]),
            }
            for i in drawn
        ],
    }
    if banded:
        # The band is taken over every spectrum, not over the undrawn
        # remainder: it describes the distribution, and leaving out the drawn
        # ones would make it describe a subset nobody asked about.
        payload["band"] = {
            "n_spectra": int(n_spectra),
            "y_lower": _round(np.percentile(matrix[:, kept], 5, axis=0)),
            "y_median": _round(np.percentile(matrix[:, kept], 50, axis=0)),
            "y_upper": _round(np.percentile(matrix[:, kept], 95, axis=0)),
        }
    return payload


def pca_payload(
    node_id: str,
    model: PCA,
    matrix: NDArray[np.float64],
    axis_values: list[float],
    axis_unit: str,
    sample_ids: list[str],
    sample_rows: NDArray[np.intp],
) -> dict[str, Any]:
    """Everything the analysis screen draws. Every number is the kernel's.

    `spe(X)` needs the matrix because it measures the part of X the model does
    not span; `hotelling_t2()` does not, because the model kept the scores.
    """
    t2 = model.hotelling_t2()
    spe = model.spe(matrix)
    return {
        "node_id": node_id,
        "task": "decomposition",
        "n_components": model.n_components,
        "n_samples": int(model.n_samples_ or 0),
        "n_variables": int(model.n_variables_ or 0),
        "rank": int(model.rank_ or 0),
        "samples": [{"index": int(row), "sample_id": sample_ids[int(row)]} for row in sample_rows],
        "scores": [_round(row) for row in np.asarray(model.scores_)],
        "loadings": {
            "axis": {
                "kind": "wavelength_nm",
                "unit": axis_unit,
                "values": [round(float(v), 4) for v in axis_values],
            },
            "components": [_round(column) for column in np.asarray(model.loadings_).T],
        },
        "eigenvalues": _round(np.asarray(model.eigenvalues_)[: model.n_components], 8),
        "explained_variance_ratio": _round(model.explained_variance_ratio(), 8),
        "cumulative_explained_variance": _round(model.cumulative_explained_variance(), 8),
        "diagnostics": {
            # Not rounded: these are displayed, and a limit the interface
            # shows should be the limit the kernel computed.
            "hotelling_t2": _round(t2, 8),
            "hotelling_t2_limit": float(model.hotelling_t2_limit()),
            "spe": _round(spe, 8),
            "spe_limit": float(model.spe_limit()),
            "alpha": 0.05,
        },
    }


def job_sequences(experiment_id: UUID) -> dict[str, list[dict[str, Any]]]:
    """The states a run passes through, one sequence per outcome.

    Both endings are here because #49 needs a failure that is reachable
    without editing code, and because `PROPOSAL.md` §8.2 is explicit that a
    failed experiment is a result rather than an absence.

    GUESS, all of it: `design/data-model.md` lists job and progress state
    under "not yet modelled". #53 replays a sequence over a few seconds rather
    than returning it; the UI never sees these arrays whole.
    """

    def stages(job: str, entries: list[tuple[str, float, str]]) -> list[dict[str, Any]]:
        base = {"job_id": str(_id(job)), "experiment_id": str(experiment_id)}
        return [
            {"status": status, "progress": progress, "message": message, **base}
            for status, progress, message in entries
        ]

    running = [
        ("queued", 0.0, "Queued"),
        ("running", 0.15, "Preprocessing: SNV"),
        ("running", 0.40, "Preprocessing: Savitzky-Golay (window 11, poly 2, deriv 1)"),
        ("running", 0.65, "Resolving split: 10-fold, seed 42"),
    ]
    return {
        "succeeded": stages(
            "job-ok",
            [
                *running,
                ("running", 0.85, "Fitting PCA, 5 components"),
                ("succeeded", 1.0, "Complete"),
            ],
        ),
        "failed": stages(
            "job-failed",
            [
                *running,
                ("running", 0.85, "Fitting PCA, 5 components"),
                (
                    "failed",
                    0.85,
                    "5 components were asked of a matrix of rank 4. Reduce n_components, "
                    "or add samples or variables.",
                ),
            ],
        ),
        "cancelled": stages(
            "job-cancelled",
            [*running, ("cancelled", 0.65, "Cancelled after 4.2 s")],
        ),
    }


def error_payload() -> dict[str, Any]:
    """The body a failure returns.

    GUESS: nothing in `models.py` describes an error envelope. The one thing
    that is not negotiable is `PROPOSAL.md` §6's rule - a reader that cannot
    parse a file produces a specific diagnostic naming what it choked on,
    never a stack trace - so the shape carries a place for that and no field
    for a traceback.
    """
    return {
        "error": {
            "code": "reader_failed",
            "message": (
                "tecator.txt: expected a multiple of 125 values per row, found 124 "
                "on row 87. The file may be truncated."
            ),
            "detail": {"file": "tecator.txt", "row": 87, "reader": "tecator_txt"},
        }
    }


def step_schema() -> dict[str, Any]:
    """The preprocessing steps' own JSON Schema, straight out of `models.py`.

    NOT a guess, and deliberately not written by hand: the inspector builds its
    parameter forms from this, so a field's type, its bounds, its enum and its
    default come from the same place the backend enforces them. Restating them
    in TypeScript is how a form ends up refusing what the schema allows, or
    allowing what it refuses.

    The cross-field rules - an odd Savitzky-Golay window, `polyorder` below it,
    `start` below `end` - live in `model_validator` and have no JSON Schema
    equivalent. The stub server validates against the model itself instead of
    restating them; see `/steps/validate`.
    """
    return TypeAdapter(PreprocessStep).json_schema()


def node_states(pipeline: Pipeline) -> dict[str, Any]:
    """Run state per node, for the canvas.

    GUESS: `PipelineNode` carries no state and should not - state belongs to a
    run, not to a recipe. This attaches it alongside, keyed by node id, so the
    canvas can render all five states in #46 and #49. Layout coordinates live
    here too, deliberately outside `Pipeline.content_hash()`: moving a node
    must not change the science.
    """
    complete = {"state": "complete"}
    states: dict[str, dict[str, Any]] = {node.id: dict(complete) for node in pipeline.nodes}
    states["pca_a"] = {"state": "complete", "headline": {"label": "PC1-5 var", "value": None}}
    states["msc"] = {"state": "running", "progress": 0.45}
    states["pca_b"] = {"state": "queued"}
    states["savgol"] = {"state": "stale", "reason": "edited - downstream stale"}
    states["autoscale_c"] = {"state": "stale", "reason": "upstream changed"}
    states["pca_c"] = {"state": "not_run"}
    states["centre_b"] = {"state": "not_run"}
    # The fifth state. Its message is the one jobs.json's failing run ends on,
    # because the node that failed and the run that failed are the same event
    # seen from two places.
    states["pca_d"] = {
        "state": "failed",
        "message": (
            "5 components were asked of a matrix of rank 4. Reduce n_components, "
            "or add samples or variables."
        ),
    }
    return {
        "pipeline_id": str(pipeline.pipeline_id),
        "nodes": states,
        "layout": {
            node.id: {"x": 40 + 170 * column, "y": 40 + 130 * row}
            for node, column, row in _layout(pipeline)
        },
    }


def _layout(pipeline: Pipeline) -> list[tuple[Any, int, int]]:
    """Column by depth, row by branch. Presentation only, and outside the hash."""
    by_id = {node.id: node for node in pipeline.nodes}
    depth: dict[str, int] = {}

    def depth_of(node_id: str) -> int:
        if node_id not in depth:
            node = by_id[node_id]
            depth[node_id] = 0 if not node.inputs else 1 + max(map(depth_of, node.inputs))
        return depth[node_id]

    rows: dict[int, int] = {}
    placed = []
    for node in pipeline.nodes:
        column = depth_of(node.id)
        row = rows.get(column, 0)
        rows[column] = row + 1
        placed.append((node, column, row))
    return placed


def import_preview(version: DatasetVersion, tecator: Any) -> dict[str, Any]:
    """What the import screen shows before anything is committed.

    Every field is read off the real file and the real loader, not invented:
    the hash and reader come from `SourceFile`, the axis and shape from the
    loaded dataset. 1.2's readers produce this from a file the user chose;
    what they must keep is that nothing is committed until it is confirmed.

    GUESS: the field names, and `alternatives` - the list a user picks from
    when a detection is wrong, which #44 needs and no document specifies.
    """
    axis = version.axis.values
    return {
        "source": {
            "filename": version.source.filename if version.source else "tecator.txt",
            "file_hash": version.source.file_hash if version.source else "",
            "reader": version.source.reader if version.source else "",
            "reader_version": version.source.reader_version if version.source else "",
            "size_bytes": len(
                (
                    Path(__file__).resolve().parents[1]
                    / "src/chemometrics_workbench/data/tecator/tecator.txt"
                ).read_bytes()
            ),
        },
        "detected": {
            "delimiter": {"value": "whitespace", "alternatives": [",", ";", "\t", "|"]},
            "decimal": {"value": ".", "alternatives": [","]},
            "orientation": {
                "value": "samples_in_rows",
                "alternatives": ["samples_in_columns"],
            },
            "n_samples": version.n_samples,
            "n_variables": version.n_variables,
            "axis": {
                "kind": version.axis.kind.value,
                "unit": version.axis.unit,
                "start": round(float(axis[0]), 4),
                "end": round(float(axis[-1]), 4),
                "reconstructed": True,
                "note": (
                    "The file carries no axis. It is reconstructed as 100 evenly "
                    "spaced points over 850-1050 nm."
                ),
            },
            "metadata_columns": [],
            "targets": sorted(version.targets),
            "discarded": [
                {
                    "what": "22 principal components",
                    "why": "preprocessing supplied by the file, not raw data",
                }
            ],
        },
        "head": {
            "sample_ids": version.sample_ids[:5],
            "rows": [_round(row[:6], 4) for row in tecator.spectra[:5]],
        },
    }


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def write(name: str, body: Any) -> Path:
    path = FIXTURES / f"{name}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n")
    return path


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    domain = build_domain()
    tecator = domain["tecator"]
    project: Project = domain["project"]
    dataset: Dataset = domain["dataset"]
    version: DatasetVersion = domain["version"]

    pipeline = build_pipeline(project.project_id, version.version_id)
    axis_values = version.axis.values
    axis_unit = version.axis.unit or ""
    sample_ids = version.sample_ids

    outputs = run_preprocessing(pipeline, tecator.spectra, tecator.axis.values)

    # Branch D is fitted on the training rows of fold 0, because its recipe
    # says there is a split above it. The folds are the real ones, from the
    # real splitter, and they are recorded so the run can be repeated.
    folds = validation.k_fold(version.n_samples, n_splits=10, seed=42)
    train_rows = folds[0].train

    fits: dict[str, tuple[PCA, NDArray[np.float64], NDArray[np.intp]]] = {}
    for estimator in (n for n in pipeline.nodes if n.type == "estimator"):
        matrix = outputs[estimator.inputs[0]]
        rows = train_rows if estimator.id == "pca_d" else np.arange(version.n_samples)
        fitted_on = matrix[rows]
        fits[estimator.id] = (fit_pca(fitted_on, estimator.spec.n_components), fitted_on, rows)

    resolved = ResolvedSplit(
        node_id="split_d",
        train_indices=[fold.train.tolist() for fold in folds],
        test_indices=[fold.test.tolist() for fold in folds],
    )

    pca_d = fits["pca_d"][0]
    experiment = Experiment(
        experiment_id=_id("experiment"),
        project_id=project.project_id,
        pipeline_snapshot=pipeline,
        dataset_version_id=version.version_id,
        dataset_content_hash=version.content_hash,
        status=ExperimentStatus.SUCCEEDED,
        resolved_splits=[resolved],
        metrics=Metrics(
            explained_variance=_round(pca_d.explained_variance_ratio(), 8),
            extra={
                "hotelling_t2_limit": float(pca_d.hotelling_t2_limit()),
                "spe_limit": float(pca_d.spe_limit()),
            },
        ),
        environment=Environment(
            app_version=cw.__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            packages={"numpy": np.__version__, "scipy": scipy.__version__},
            recorded_at=PINNED,
        ),
        started_at=PINNED,
        finished_at=PINNED,
    )

    labels = {
        "source": "tecator_raw",
        "snv": "SNV",
        "msc": "MSC",
        "savgol": "SG d1 w11",
        "snv_savgol": "SNV + SG d1",
        "centre_a": "Mean centre",
        "centre_b": "Mean centre",
        "centre_d": "Mean centre",
        "autoscale_c": "Autoscale",
        "split_d": "K-fold 10",
    }

    written = [
        write("project", json.loads(project.model_dump_json())),
        write(
            "datasets",
            [
                {
                    "dataset": json.loads(dataset.model_dump_json()),
                    "versions": [json.loads(version.model_dump_json())],
                }
            ],
        ),
        write("pipeline", json.loads(pipeline.model_dump_json())),
        write("pipeline_state", node_states(pipeline)),
        write("experiment", json.loads(experiment.model_dump_json())),
        write("import_preview", import_preview(version, tecator)),
        write("jobs", job_sequences(experiment.experiment_id)),
        write("error", error_payload()),
        write("step_schema", step_schema()),
        write(
            "spectra",
            {
                node_id: spectra_payload(
                    node_id,
                    labels.get(node_id, node_id),
                    matrix,
                    axis_values,
                    axis_unit,
                    sample_ids,
                    "Absorbance" if node_id in {"source", "snv", "msc"} else "d/dx Absorbance",
                )
                for node_id, matrix in outputs.items()
            },
        ),
        write(
            "pca",
            {
                node_id: pca_payload(
                    node_id, model, matrix, axis_values, axis_unit, sample_ids, rows
                )
                for node_id, (model, matrix, rows) in fits.items()
            },
        ),
    ]

    # The pipeline hash is the thing lineage is keyed on, so it is worth
    # printing: if it moves, the recipe moved.
    print(f"pipeline {pipeline.content_hash()}")
    print(f"dataset  {version.content_hash}")
    for path in written:
        print(f"{path.stat().st_size:>9,} B  {path.relative_to(FIXTURES.parent)}")


if __name__ == "__main__":
    main()
