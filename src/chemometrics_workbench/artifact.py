"""The model artifact: one file holding a fitted model, readable without us.

`docs/model-artifact.md` is the format and this implements it. Where the two
disagree, one of them is a bug — decide which before changing either.

`PROPOSAL.md` §8.4 asks for one self-describing file carrying the pipeline
definition, the fitted parameters, the provenance record and the metrics, in an
openly documented format, copyable between machines, **readable without this
application**. That last clause is the whole design: a zip of `manifest.json`
and one `.npy` per array, which `zipfile`, `json` and any NumPy open. A pickle
would fail it twice — it needs this package importable, and unpickling a file
someone sent you runs their code.

## What is not here

**Prediction.** The artifact carries what predicting needs and `docs/model-artifact.md`
§8 writes the arithmetic out, but turning that into a portable function is
`json-and-snippet-export`'s job, which states which form it produced.

**Where the file lives.** `write_artifact` takes a path and returns the hash of
what it wrote. Recording that in the project's `model` table is
`model-registry`.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench import __version__
from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import DatasetVersion, Environment, Pipeline, ResolvedSplit

__all__ = [
    "MANIFEST",
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactError",
    "read_artifact",
    "write_artifact",
]

#: `docs/model-artifact.md` §2. Bumped when a field is removed, renamed or
#: changes meaning; adding an optional one does not bump it, because a reader
#: that ignores an unknown key loses nothing.
SCHEMA_VERSION = 1

MANIFEST = "manifest.json"
ARRAYS = "arrays"


class ArtifactError(Exception):
    """An artifact could not be written or read, saying which file and why.

    One exception with good messages rather than a hierarchy, for the reason
    `ProjectError` is one: the only caller that would distinguish the cases
    turns all of them into the same error body.
    """


@dataclass(frozen=True)
class Artifact:
    """One artifact, read back.

    `manifest` is the document as it was written — a reader wanting a field
    this dataclass does not name reads it there rather than waiting for a
    field to be added. `arrays` are float64, or int64 for the index sets.
    """

    manifest: dict[str, Any]
    arrays: dict[str, NDArray[Any]]

    @property
    def schema_version(self) -> int:
        return int(self.manifest["schema_version"])

    @property
    def task(self) -> str:
        return str(self.manifest["model"]["task"])

    @property
    def pipeline(self) -> Pipeline:
        """The recipe that produced the estimator's input, by value (§3)."""
        return Pipeline.model_validate(self.manifest["pipeline"])


def write_artifact(
    path: str | Path,
    result: EstimatorResult,
    *,
    pipeline: Pipeline,
    version: DatasetVersion,
    node_axis: object,
    split: ResolvedSplit | None = None,
    environment: Environment | None = None,
) -> str:
    """Write one fitted estimator as an artifact, and return its content hash.

    `node_axis` is the axis the model's own matrix is on, which is not the
    dataset's under a range selection (#134) — the caller computes it because
    deriving it needs the pipeline and this module deliberately computes
    nothing.

    The bytes are assembled in memory and written through a temporary file, so
    an interrupted write leaves no half-artifact; the hash is of the bytes that
    landed, which is what a later read can be checked against.
    """
    target = Path(path)
    arrays = _arrays(result, version, node_axis, split)
    manifest = _manifest(result, pipeline, version, split, environment, arrays)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
        for name, values in arrays.items():
            member = io.BytesIO()
            np.save(member, values, allow_pickle=False)
            archive.writestr(f"{ARRAYS}/{name}.npy", member.getvalue())
    blob = buffer.getvalue()

    temporary = target.with_name(f"{target.name}.{hashlib.sha256(blob).hexdigest()[:8]}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(blob)
        temporary.replace(target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot write {target}: {error.strerror}") from error
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def read_artifact(path: str | Path) -> Artifact:
    """Read an artifact back, refusing one this build does not understand.

    A file stamped a newer schema version is refused by name, as `db.py`
    refuses a newer database and for the same reason: a newer writer may have
    added a field whose absence this reader would silently take as a default.
    """
    file = Path(path)
    try:
        with zipfile.ZipFile(file) as archive:
            try:
                document = json.loads(archive.read(MANIFEST))
            except KeyError as error:
                raise ArtifactError(
                    f"{file.name} holds no {MANIFEST}, so it is not a model artifact."
                ) from error
            except ValueError as error:
                raise ArtifactError(f"{file.name}'s {MANIFEST} is not valid JSON: {error}") from (
                    error
                )

            if not isinstance(document, dict) or "schema_version" not in document:
                raise ArtifactError(f"{file.name}'s {MANIFEST} states no schema_version.")
            stamped = document["schema_version"]
            if not isinstance(stamped, int):
                raise ArtifactError(
                    f"{file.name} states a schema_version of {stamped!r}, which is not a number."
                )
            if stamped > SCHEMA_VERSION:
                raise ArtifactError(
                    f"{file.name} was written by a newer version of the application: its schema "
                    f"version is {stamped} and this one understands {SCHEMA_VERSION}."
                )

            arrays: dict[str, NDArray[Any]] = {}
            for name, entry in document.get("arrays", {}).items():
                member = entry.get("file") if isinstance(entry, dict) else None
                if not isinstance(member, str):
                    raise ArtifactError(f"{file.name}'s array {name!r} names no file.")
                try:
                    with archive.open(member) as handle:
                        arrays[name] = np.load(handle, allow_pickle=False)
                except KeyError as error:
                    raise ArtifactError(
                        f"{file.name}'s manifest names {member}, which the archive does not hold."
                    ) from error
                except ValueError as error:
                    raise ArtifactError(f"{file.name}'s {member} could not be read: {error}") from (
                        error
                    )
    except zipfile.BadZipFile as error:
        raise ArtifactError(f"{file.name} is not a zip archive: {error}") from error
    except OSError as error:
        raise ArtifactError(f"cannot read {file.name}: {error.strerror}") from error

    return Artifact(manifest=document, arrays=arrays)


# --- what goes in ----------------------------------------------------------


def _float64(values: object) -> NDArray[np.float64]:
    """float64, always. §7: an artifact is a record rather than a cache, and
    narrowing a coefficient vector to save 16 kB trades the thing it exists
    for."""
    return np.asarray(values, dtype=np.float64)


def _arrays(
    result: EstimatorResult,
    version: DatasetVersion,
    node_axis: object,
    split: ResolvedSplit | None,
) -> dict[str, NDArray[Any]]:
    """Every array §7 names that this model has. A reader must not assume any
    of them beyond `dataset_axis`, so absent is absent rather than empty."""
    arrays: dict[str, NDArray[Any]] = {
        "dataset_axis": _float64(version.axis.values),
        "node_axis": _float64(node_axis),
        "loadings": _float64(result.loadings),
        "eigenvalues": _float64(result.eigenvalues),
        "explained_variance_ratio": _float64(result.explained_variance_ratio),
    }
    if result.rotations:
        arrays["rotations"] = _float64(result.rotations)

    if result.task in ("regression", "classification"):
        arrays["x_mean"] = _float64(result.x_mean)
        arrays["coefficients"] = _float64(result.coefficients)
        arrays["y_loadings"] = _float64(result.y_loadings)
        arrays["vip"] = _float64(result.vip)
        arrays["y_explained_variance_ratio"] = _float64(result.y_explained_variance_ratio)

    if split is not None and result.fold is not None:
        arrays["train_indices"] = np.asarray(result.rows, dtype=np.int64)
        arrays["test_indices"] = np.asarray(result.held_out, dtype=np.int64)
    return arrays


def _manifest(
    result: EstimatorResult,
    pipeline: Pipeline,
    version: DatasetVersion,
    split: ResolvedSplit | None,
    environment: Environment | None,
    arrays: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "application": {"name": "chemometrics-workbench", "version": __version__},
        "model": {
            "node_id": result.node_id,
            "task": result.task,
            "n_components": result.n_components,
            "n_variables": result.n_variables,
            "n_samples": result.n_samples,
            "rank": result.rank,
            "target": result.target,
            "classes": result.classes or None,
            "y_mean": result.y_mean,
            "alpha": result.alpha,
            "hotelling_t2_limit": result.hotelling_t2_limit,
            "spe_limit": result.spe_limit,
            "spe_limit_caveat": result.spe_limit_caveat,
        },
        # By value, never a reference: a pipeline gets edited, and an artifact
        # whose recipe pointed at one would lose its meaning the moment it was
        # (`PROPOSAL.md` §8.2).
        "pipeline": json.loads(pipeline.model_dump_json()),
        "dataset": {
            "version_id": str(version.version_id),
            "content_hash": version.content_hash,
            "n_samples": version.n_samples,
            "n_variables": version.n_variables,
            "axis": {"kind": version.axis.kind.value, "unit": version.axis.unit},
        },
        "split": None
        if split is None or result.fold is None
        else {
            "node_id": split.node_id,
            "fold": result.fold,
            "n_folds": len(split.test_indices),
        },
        "metrics": dict(result.metrics),
        "environment": None if environment is None else json.loads(environment.model_dump_json()),
        "arrays": {
            name: {
                "file": f"{ARRAYS}/{name}.npy",
                "dtype": str(values.dtype),
                "shape": list(values.shape),
            }
            for name, values in arrays.items()
        },
    }
