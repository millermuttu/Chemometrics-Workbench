"""The two portable export forms: a JSON model and a Python snippet.

`docs/model-export.md` is the specification and this implements it. Where the
two disagree, one of them is a bug — decide which before changing either.

`PROPOSAL.md` §9: a fitted PLS model is, at prediction time, a preprocessing
recipe plus a coefficient vector, and making that free and easy to take with
you is a genuine differentiator. It also places one constraint on the codebase,
and it is a test rather than a promise: **exported predictions must match
in-application predictions to within a stated numerical tolerance**.
`docs/model-export.md` §5 states that tolerance and says where the number came
from — the float32 array store, measured in `docs/phase-2/exit-run.md`.

## The split

`pls-regression.md` §7 folds a step into the coefficients when it is a fixed
linear map on X; SNV, MSC and the baselines are not, because each depends on
the sample being predicted. So the chain is cut at the **last** unfoldable
step: everything after it folds into `b`, everything up to and including it is
carried and re-executed. A chain with nothing unfoldable exports as a bare
coefficient vector; a chain ending in an SNV folds nothing and says so by
carrying all of it.

A baseline is refused by name rather than emitted. AsLS is a penalised sparse
solve per spectrum; putting one inside a file whose point is that it can be
pasted into an instrument PC would make it neither short nor checkable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench import __version__, preprocessing
from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import (
    MSC,
    SNV,
    BaselineCorrect,
    DatasetVersion,
    Normalise,
    Pipeline,
    PreprocessNode,
)
from chemometrics_workbench.regression import FOLDABLE, coefficients_original_units

__all__ = [
    "EXPORTABLE_RESIDUAL",
    "SCHEMA_VERSION",
    "THRESHOLD",
    "ExportError",
    "json_model",
    "python_snippet",
]

#: `docs/model-export.md` §2.
SCHEMA_VERSION = 1

#: `pls-da.md` §5's fixed cut, stated in the export rather than left for a
#: reader to assume.
THRESHOLD = 0.5

#: The unfoldable steps a snippet can re-execute: each is a few lines of NumPy
#: on one row. A baseline is deliberately absent - §1.
EXPORTABLE_RESIDUAL = (SNV, MSC, Normalise)


class ExportError(Exception):
    """A model could not be exported, naming what about it cannot be.

    One exception with good messages, for the reason `ProjectError` is one: the
    only caller that distinguishes the cases turns all of them into one error
    body, and what it needs is the sentence.
    """


def json_model(
    result: EstimatorResult,
    *,
    pipeline: Pipeline,
    version: DatasetVersion,
    raw: NDArray[np.float64],
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """The plain JSON model (§2), or an `ExportError` naming why there is none.

    `raw` is the dataset's own matrix, needed to refit the chain: the executor
    keeps no fitted transformers - `_transform` builds one, uses it and drops
    it - so the chain is measured again here from the recipe and the stored
    source array, which is `folded_coefficients`' shape and for the same
    reason. A pure function of the pipeline and the source cannot disagree with
    the cache or move a content hash.
    """
    if result.task == "decomposition":
        raise ExportError(
            f"node {result.node_id!r} is a decomposition, which has no prediction to export. "
            "A PCA produces scores; export the artifact instead (docs/model-artifact.md)."
        )

    chain = _chain(pipeline, result.node_id)
    residual, foldable = _split(chain)

    axis = np.asarray(version.axis.values, dtype=np.float64)
    rows = np.asarray(result.rows, dtype=np.intp)
    values = np.asarray(raw, dtype=np.float64)

    # Fit and apply the residual chain, so the foldable tail is measured
    # against the matrix it actually saw.
    steps: list[dict[str, Any]] = []
    for node in residual:
        transformer = preprocessing.from_spec(node.step, axis=axis)
        transformer.fit(values[rows])
        steps.append(_step_payload(node, transformer))
        values = transformer.transform(values)

    fitted: list[preprocessing.Transformer] = []
    for node in foldable:
        transformer = preprocessing.from_spec(node.step, axis=axis)
        transformer.fit(values[rows])
        values = transformer.transform(values)
        if isinstance(transformer, preprocessing.RangeSelectTransformer):
            axis = transformer.selected_axis()
        fitted.append(transformer)

    # The estimator's own centring folds in too: the model computes
    # `(chain(X) - x̄)·b + ȳ`, and passing `ȳ - x̄·b` as the response mean puts
    # that where the helper's intercept belongs.
    coefficients = np.asarray(result.coefficients, dtype=np.float64)
    x_mean = np.asarray(result.x_mean, dtype=np.float64)
    y_mean = float(result.y_mean or 0.0)
    try:
        folded, intercept = coefficients_original_units(
            coefficients,
            fitted,
            n_variables=values.shape[1] if residual else version.n_variables,
            y_mean=y_mean - float(x_mean @ coefficients),
        )
    except ValueError as error:  # pragma: no cover - _split already refused these
        raise ExportError(str(error)) from error

    # The axis the residual chain hands on, which is the dataset's unless a
    # range selection sits in the residual part - and one cannot, because a
    # range selection is foldable.
    residual_axis = np.asarray(version.axis.values, dtype=np.float64)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "application": {"name": "chemometrics-workbench", "version": __version__},
        "model": {
            "node_id": result.node_id,
            "task": result.task,
            "target": result.target,
            "classes": result.classes or None,
            "n_components": result.n_components,
            "threshold": THRESHOLD if result.task == "classification" else None,
        },
        "axis": {
            "kind": version.axis.kind.value,
            "unit": version.axis.unit,
            "values": [float(value) for value in residual_axis],
        },
        "preprocessing": steps,
        "coefficients": [float(value) for value in folded],
        "intercept": float(intercept),
        "provenance": {
            "dataset_content_hash": version.content_hash,
            "pipeline_hash": pipeline.content_hash(),
            "experiment_id": experiment_id,
            "metrics": dict(result.metrics),
        },
    }


def python_snippet(model: dict[str, Any]) -> str:
    """The JSON model as one self-contained file (§3).

    Generated from the JSON model rather than from the result, so the two
    cannot drift: anything §4 says of one is true of the other because the
    second is written from the first.
    """
    task = model["model"]["task"]
    classes = model["model"]["classes"]
    provenance = model["provenance"]
    metrics = ", ".join(
        f"{name} {value:.6g}" for name, value in sorted(provenance["metrics"].items())
    )

    body = [
        '"""Predict with a model exported from Chemometrics Workbench.',
        "",
        f"    Model:    {model['model']['node_id']}, {task}"
        + (f" of {model['model']['target']}" if model["model"]["target"] else ""),
        f"    Exported: {model['created_at']} by {model['application']['name']} "
        f"{model['application']['version']}",
        f"    Dataset:  {provenance['dataset_content_hash']}",
        f"    Pipeline: {provenance['pipeline_hash']}",
        f"    Scored:   {metrics or 'no metrics recorded'}",
        "",
        "Needs nothing but NumPy. `predict` takes an n x p array of raw spectra on",
        "AXIS and returns n predictions; one spectrum is `predict(x[None, :])[0]`.",
        '"""',
        "",
        "import numpy as np",
        "",
        f"# {model['axis']['kind']}"
        + (f", {model['axis']['unit']}" if model["axis"]["unit"] else ""),
        f"AXIS = np.array({_literal(model['axis']['values'])})",
        f"COEFFICIENTS = np.array({_literal(model['coefficients'])})",
        f"INTERCEPT = {model['intercept']!r}",
    ]
    if task == "classification":
        body += [
            f"CLASSES = {classes!r}",
            f"THRESHOLD = {model['model']['threshold']!r}",
        ]

    for index, step in enumerate(model["preprocessing"]):
        body += ["", *_step_source(index, step)]

    body += [
        "",
        "",
        "def predict(X):",
        '    """Raw spectra in, predictions out. Shapes are n x p, never transposed."""',
        "    X = np.asarray(X, dtype=np.float64)",
        "    if X.ndim != 2:",
        "        raise ValueError(f'expected an n x p array, got shape {X.shape}')",
        "    if X.shape[1] != AXIS.size:",
        "        raise ValueError(",
        "            f'the model expects {AXIS.size} variables and was given {X.shape[1]}'",
        "        )",
    ]
    for index, _ in enumerate(model["preprocessing"]):
        body.append(f"    X = _step_{index}(X)")
    body += [
        "    y = X @ COEFFICIENTS + INTERCEPT",
    ]
    if task == "classification":
        body += [
            "    above = y >= THRESHOLD",
            "    return np.array([CLASSES[1] if hit else CLASSES[0] for hit in above])",
        ]
    else:
        body += ["    return y"]
    return "\n".join(body) + "\n"


# --- the chain -------------------------------------------------------------


def _chain(pipeline: Pipeline, node_id: str) -> list[PreprocessNode]:
    """The preprocess nodes from the source down to this one, in order."""
    by_id = {node.id: node for node in pipeline.nodes}
    chain: list[PreprocessNode] = []
    current = node_id
    while True:
        node = by_id[current]
        if isinstance(node, PreprocessNode):
            chain.append(node)
        if not node.inputs:
            return list(reversed(chain))
        current = node.inputs[0]


def _split(chain: list[PreprocessNode]) -> tuple[list[PreprocessNode], list[PreprocessNode]]:
    """Cut at the last unfoldable step (§1), refusing a baseline by name."""
    cut = 0
    for index, node in enumerate(chain, start=1):
        if isinstance(node.step, BaselineCorrect):
            raise ExportError(
                f"the chain has a {node.step.method!r} baseline at {node.id!r}, which an export "
                "cannot carry: it is a solve per spectrum rather than a few lines of arithmetic, "
                "and it is not a fixed linear map on X so it cannot be folded into the "
                "coefficients either (docs/model-export.md section 1). The model artifact carries "
                "it, and the application still predicts with it."
            )
        if isinstance(node.step, EXPORTABLE_RESIDUAL):
            cut = index
        elif not isinstance(
            preprocessing.from_spec(node.step, axis=np.zeros(1)), FOLDABLE
        ):  # pragma: no cover - every remaining kind is foldable
            raise ExportError(
                f"{node.step.kind!r} at {node.id!r} can be neither folded into the coefficients "
                "nor re-executed by an exported snippet."
            )
    return chain[:cut], chain[cut:]


def _step_payload(node: PreprocessNode, transformer: object) -> dict[str, Any]:
    """One residual step, with whatever fitted parameter it needs (§4)."""
    payload: dict[str, Any] = json.loads(node.step.model_dump_json())
    if isinstance(transformer, preprocessing.MSCTransformer):
        reference = transformer.reference_
        assert reference is not None, "a fitted MSC has its reference"
        payload["reference_spectrum"] = [float(value) for value in reference]
    return payload


# --- the snippet's bodies --------------------------------------------------


def _literal(values: list[float]) -> str:
    """A NumPy-readable list at full precision: `repr` of a float round-trips."""
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def _step_source(index: int, step: dict[str, Any]) -> list[str]:
    kind = step["kind"]
    if kind == "snv":
        ddof = int(step.get("ddof", 1))
        return [
            f"def _step_{index}(X):",
            '    """SNV: each row centred and scaled by its own moments."""',
            "    spread = X.std(axis=1, ddof=" + str(ddof) + ", keepdims=True)",
            "    if not np.all(spread > 0):",
            "        raise ValueError('a spectrum has no scatter to correct')",
            "    return (X - X.mean(axis=1, keepdims=True)) / spread",
        ]
    if kind == "msc":
        return [
            f"_REFERENCE_{index} = np.array({_literal(step['reference_spectrum'])})",
            "",
            f"def _step_{index}(X):",
            '    """MSC: each row regressed against the reference fitted at calibration."""',
            f"    reference = _REFERENCE_{index} - _REFERENCE_{index}.mean()",
            "    centred = X - X.mean(axis=1, keepdims=True)",
            "    slope = (centred @ reference) / float(reference @ reference)",
            "    if not np.all(slope != 0):",
            "        raise ValueError('a spectrum regresses flat against the reference')",
            f"    intercept = X.mean(axis=1) - slope * _REFERENCE_{index}.mean()",
            "    return (X - intercept[:, None]) / slope[:, None]",
        ]
    if kind == "normalise":
        norm = step.get("norm", "l2")
        divisor = {
            "l1": "np.abs(X).sum(axis=1)",
            "l2": "np.sqrt((X**2).sum(axis=1))",
            "max": "np.abs(X).max(axis=1)",
            "area": "X.sum(axis=1)",
        }[norm]
        return [
            f"def _step_{index}(X):",
            f'    """Normalise: each row divided by its {norm} norm."""',
            f"    divisor = {divisor}",
            "    if not np.all(divisor != 0):",
            "        raise ValueError('a spectrum has a zero norm')",
            "    return X / divisor[:, None]",
        ]
    raise ExportError(  # pragma: no cover - _split refuses these first
        f"{kind!r} has no snippet form."
    )
