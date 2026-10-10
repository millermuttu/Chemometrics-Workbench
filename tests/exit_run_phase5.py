"""Phase 5's exit criterion, demonstrated rather than reasoned (#289).

`PROPOSAL.md` §16, Phase 5: "an open multi-class spectral dataset is cleaned,
its variables selected, and PLS-DA, SIMCA, LDA and kNN compared under the same
stratified cross-validation, each matching its reference within stated
tolerance; a file of each new format imports and matches its source".
Decision 0006 splits it into two claims, because the open multi-class sets are
CSV files, not one of the new formats. This script demonstrates both against
the served application, the way `tests/exit_run.py` did Phase 2's.

**Claim 1, the workflow, on the Quadram meat set** (`docs/examples/meat.csv`):

1. Import it through `/import/preview` and `/import`.
2. **Clean.** Fit a PCA on the centred spectra and read its outlier flags.
   This script stands in for the user, and its rule is stated: exclude every
   sample that breaks `RULES_TO_EXCLUDE` or more rules. The exclusion goes
   through the same endpoint the outlier table calls, so it is a derived
   dataset version and the pipeline moves onto it.
3. **Select.** Put a ten-fold cross-validation stratified by `meat`, mean
   centring and a PLS-DA on the cleaned version, run it, and keep the
   variables whose VIP is at least 1: a `select_variables` step on the
   centring, recorded as chosen by VIP.
4. **Compare.** Put PLS-DA, LDA, kNN and SIMCA below that step, run them as one
   job, and rebuild every one of them independently with scikit-learn and
   SciPy on the experiment's own resolved folds: the same rows, the same
   refitted centring, the same columns. The cross-validated confusion and
   acceptance tables must be equal; the PLS-DA RMSECV curve and SIMCA's
   reduced distances must agree within the parity tolerances.

**Claim 2, the formats.** One file of each new format, from the sources
decision 0006 names and committed as reader fixtures, is imported through the
application on a project of its own and read back through `/spectra/source`.
Each is compared with its source's own export where the source has one, and
with the nearest independent reading where it does not; the record says which.

**Claim 3, PCR and a selected PLS on Tecator**, which issue #289 asks for
beside the two: SNV, Savitzky-Golay, a ten-fold split, centring, PCR and PLS on
`fat`; then the variables with PLS VIP at least 1, as a `select_variables` step
above a second PLS. Both RMSECV curves are rebuilt with scikit-learn on the
served folds. Claim 1 also reads lineage back: the cleaning run and the
comparison run name different source versions, and the derived one names its
parent and the rows it left out.

**The store is float32** (`PROPOSAL.md` §13), so the served arrays are
float32-rounded at every node. The reference rounds at the same points - the
source, the centred matrix - and fits in float64. That is why it can demand
equal decisions rather than nearly equal ones.

**SIMCA has no external implementation here** (`simca.md` §9: no R). Its
reference is a per-class `sklearn.decomposition.PCA` with the `pca.md` §7 and
§8 limits computed in SciPy, which is the independent reading the parity suite
also rests on.

Run it:

    uv run python -m tests.exit_run_phase5
    uv run python -m tests.exit_run_phase5 --tighten 1e-9

It exits 0 when every claim is met and 1 when any is not, and rewrites the
record either way, because a failed run is a result too. `--tighten` scales
every numeric tolerance by the factor given, to show the comparison can fail.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from tests import parity
from tests.exit_run import Served, _csv_matrix, _ok, _savgol, _snv
from tests.seed_e2e import tecator_csv

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "phase-5" / "exit-run.md"
MEAT = ROOT / "docs" / "examples" / "meat.csv"
READERS = ROOT / "tests" / "fixtures" / "readers"

CLASS_COLUMN = "meat"
N_SPLITS, SEED = 10, 42
A_CLEAN, RULES_TO_EXCLUDE = 5, 3
A_PLSDA, A_LDA, A_KNN, K, A_SIMCA = 5, 5, 5, 5, 3
VIP_CUT = 1.0
TARGET, A_REGRESSION = "fat", 5
ALPHA = 0.05
#: The outlier table's names for its rules.
RULES = {
    "t2": "T²",
    "q": "Q",
    "leverage": "leverage",
    "residual": "residual",
    "robust": "robust distance",
}

#: The store's float32, for reading back what a reader wrote (§13): half an
#: ulp is 6e-8 relative, and the margin covers the reader's own float64
#: arithmetic before the narrowing. Set here, not adjusted to pass.
STORED = parity.Tolerance(
    rtol=1e-6,
    atol=1e-12,
    reason="A value read back from the float32 store, against its source in float64.",
)


@dataclass
class Check:
    claim: int
    quantity: str
    #: A parity tolerance class, `"stored"`, or `None` for an exact comparison.
    tolerance: str | None
    ours: NDArray[np.float64]
    theirs: NDArray[np.float64]
    against: str
    tighten: float = 1.0

    def _tolerance(self) -> parity.Tolerance:
        return STORED if self.tolerance == "stored" else parity.TOLERANCES[str(self.tolerance)]

    @property
    def bounds(self) -> str:
        if self.tolerance is None:
            return "equal"
        tol = self._tolerance()
        return f"{self.tolerance} ({tol.rtol * self.tighten:.0e}, {tol.atol * self.tighten:.0e})"

    @property
    def max_abs(self) -> float:
        if self.ours.shape != self.theirs.shape:
            return float("inf")
        return float(np.max(np.abs(self.ours - self.theirs))) if self.ours.size else 0.0

    @property
    def passed(self) -> bool:
        if self.ours.shape != self.theirs.shape:
            return False
        if self.tolerance is None:
            return bool(np.array_equal(self.ours, self.theirs))
        tol = self._tolerance()
        return bool(
            np.allclose(
                self.ours, self.theirs, rtol=tol.rtol * self.tighten, atol=tol.atol * self.tighten
            )
        )


def _f32(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def _run(client: httpx.Client, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    _ok(client.put("/pipelines/current", json={"nodes": nodes}))
    job = _ok(client.post("/experiments/current/run"))
    deadline = time.monotonic() + 600.0
    while job["status"] in ("queued", "running"):
        if time.monotonic() > deadline:
            raise RuntimeError(f"the run did not finish within 600 s: {job}")
        time.sleep(0.2)
        job = _ok(client.get(f"/jobs/{job['job_id']}"))
    if job["status"] != "succeeded":
        raise RuntimeError(f"the run ended {job['status']}: {job['message']}")
    return dict(job)


def _source(client: httpx.Client) -> dict[str, Any]:
    pipeline = _ok(client.get("/pipelines/current"))
    return dict(next(node for node in pipeline["nodes"] if node["type"] == "source"))


def _import(client: httpx.Client, name: str, data: bytes) -> dict[str, Any]:
    files = {"file": (name, data)}
    preview = _ok(client.post("/import/preview", files=files))
    entry = _ok(client.post("/import", files=files, data={"corrections": "{}"}))
    return {"preview": preview, "entry": entry}


# --------------------------------------------------------------------------
# claim 1: the workflow, served
# --------------------------------------------------------------------------


def _estimators(input_id: str) -> list[dict[str, Any]]:
    specs = {
        "plsda": {
            "kind": "plsda",
            "n_components": A_PLSDA,
            "algorithm": "nipals",
            "class_column": CLASS_COLUMN,
        },
        "lda": {"kind": "lda", "n_components": A_LDA, "class_column": CLASS_COLUMN},
        "knn": {"kind": "knn", "k": K, "n_components": A_KNN, "class_column": CLASS_COLUMN},
        "simca": {"kind": "simca", "n_components": A_SIMCA, "class_column": CLASS_COLUMN},
    }
    return [
        {"id": node, "type": "estimator", "inputs": [input_id], "spec": spec}
        for node, spec in specs.items()
    ]


def drive_workflow(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, MEAT.name, MEAT.read_bytes())
    first = imported["entry"]["versions"][0]

    # Clean: a PCA on every sample, its flags, and the stated rule.
    source = _source(client)
    centre_all = {
        "id": "centre_all",
        "type": "preprocess",
        "inputs": [source["id"]],
        "step": {"kind": "mean_centre"},
    }
    pca = {
        "id": "pca",
        "type": "estimator",
        "inputs": ["centre_all"],
        "spec": {"kind": "pca", "n_components": A_CLEAN},
    }
    _run(client, [source, centre_all, pca])
    cleaning = _ok(client.get("/experiments/current"))
    pca_result = _ok(client.get("/results/pca"))
    outliers = _ok(client.get("/results/pca/outliers"))
    flagged = [
        {
            "row": pca_result["samples"][flag["index"]]["index"],
            "sample_id": pca_result["samples"][flag["index"]]["sample_id"],
            "rules": flag["rules"],
        }
        for flag in outliers["flags"]
    ]
    excluded = [flag for flag in flagged if len(flag["rules"]) >= RULES_TO_EXCLUDE]
    derived = _ok(
        client.post(
            f"/datasets/{outliers['dataset_id']}/versions",
            json={
                "from_version_id": outliers["version_id"],
                "exclude": [flag["row"] for flag in excluded],
            },
        )
    )
    cleaned = derived["versions"][-1]
    source = _source(client)
    if source["version_id"] != cleaned["version_id"]:
        raise RuntimeError("the pipeline did not move onto the cleaned version")

    # Select: VIP of a PLS-DA under the stratified split.
    split = {
        "id": "kfold",
        "type": "split",
        "inputs": [source["id"]],
        "spec": {
            "kind": "kfold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "seed": SEED,
            "stratify_by": CLASS_COLUMN,
        },
    }
    centre = {
        "id": "centre",
        "type": "preprocess",
        "inputs": ["kfold"],
        "step": {"kind": "mean_centre"},
    }
    _run(client, [source, split, centre, *_estimators("centre")[:1]])
    vip = _ok(client.get("/results/plsda"))["regression"]["vip"]
    selected = [index for index, value in enumerate(vip) if value >= VIP_CUT]
    select = {
        "id": "select",
        "type": "preprocess",
        "inputs": ["centre"],
        "step": {"kind": "select_variables", "indices": selected, "chosen_by": "vip"},
    }

    # Compare: the four classifiers below the selection, one job.
    nodes = [source, split, centre, select, *_estimators("select")]
    _ok(client.put("/pipelines/current", json={"nodes": nodes}))
    validation = _ok(client.post("/pipelines/current/validate"))
    job = _run(client, nodes)
    return {
        "preview": imported["preview"],
        "first": first,
        "cleaned": cleaned,
        "flagged": flagged,
        "excluded": excluded,
        "outlier_limits": outliers["limits"],
        "selected": selected,
        "n_variables": len(vip),
        "validation": validation,
        "job": job,
        "results": {node: _ok(client.get(f"/results/{node}")) for node in _names()},
        "experiment": _ok(client.get("/experiments/current")),
        "cleaning": _ok(client.get(f"/experiments/{cleaning['experiment_id']}")),
    }


def _names() -> list[str]:
    return ["plsda", "lda", "knn", "simca"]


# --------------------------------------------------------------------------
# claim 1: the reference, independently
# --------------------------------------------------------------------------


def _meat() -> tuple[list[str], NDArray[np.float64], list[str]]:
    rows = list(csv.reader(io.StringIO(MEAT.read_text(encoding="utf-8"))))
    body = rows[1:]
    values = np.asarray([[float(v) for v in row[3:]] for row in body], dtype=np.float64)
    return [row[0] for row in body], values, [row[1] for row in body]


def _confusion(observed: NDArray[np.intp], assigned: NDArray[np.intp], n: int) -> NDArray[Any]:
    table = np.zeros((n, n), dtype=np.float64)
    for one, other in zip(observed, assigned, strict=True):
        table[one, other] += 1
    return table


def _jackson_mudholkar(discarded: NDArray[np.float64]) -> float:
    """`pca.md` §8, from the eigenvalues a class model leaves out."""
    theta1, theta2, theta3 = (float(np.sum(discarded**m)) for m in (1, 2, 3))
    h0 = 1.0 - 2.0 * theta1 * theta3 / (3.0 * theta2**2)
    c = stats.norm.ppf(1.0 - ALPHA)
    bracket = (
        c * np.sqrt(2.0 * theta2 * h0**2) / theta1 + 1.0 + theta2 * h0 * (h0 - 1.0) / theta1**2
    )
    return float(theta1 * bracket ** (1.0 / h0))


def _simca_distances(
    train: NDArray[np.float64], codes: NDArray[np.intp], test: NDArray[np.float64], n: int
) -> NDArray[np.float64]:
    """`simca.md` §2 to §4: per-class PCA, the new-sample T² limit, Q's limit."""
    h = np.zeros((test.shape[0], n))
    for k in range(n):
        rows = train[codes == k]
        n_k = rows.shape[0]
        model = PCA(svd_solver="full").fit(rows)
        loadings = model.components_[:A_SIMCA].T
        eigenvalues = model.explained_variance_
        rank = int(np.sum(eigenvalues > eigenvalues[0] * 1e-12))
        centred = test - model.mean_
        scores = centred @ loadings
        t2 = np.sum(scores**2 / eigenvalues[:A_SIMCA], axis=1)
        q = np.sum((centred - scores @ loadings.T) ** 2, axis=1)
        a = A_SIMCA
        t2_limit = a * (n_k**2 - 1) / (n_k * (n_k - a)) * stats.f.ppf(1.0 - ALPHA, a, n_k - a)
        q_limit = _jackson_mudholkar(eigenvalues[a:rank])
        h[:, k] = np.maximum(t2 / t2_limit, q / q_limit)
    return h


def reference(served: dict[str, Any]) -> dict[str, Any]:
    _, raw, labels = _meat()
    excluded = {flag["row"] for flag in served["excluded"]}
    kept = [row for row in range(raw.shape[0]) if row not in excluded]
    values = _f32(raw[kept])
    names = sorted(set(labels))
    codes = np.asarray([names.index(labels[row]) for row in kept], dtype=np.intp)
    n_classes = len(names)
    dummy = np.eye(n_classes)[codes]
    columns = np.asarray(served["selected"], dtype=np.intp)
    split = next(
        one for one in served["experiment"]["resolved_splits"] if one["node_id"] == "kfold"
    )

    curve = np.zeros((A_PLSDA, *dummy.shape))
    lda = np.zeros(len(kept), dtype=np.intp)
    knn = np.zeros(len(kept), dtype=np.intp)
    h = np.zeros((len(kept), n_classes))
    for train_list, test_list in zip(split["train_indices"], split["test_indices"], strict=True):
        train, test = np.asarray(train_list), np.asarray(test_list)
        centred = _f32(values - values[train].mean(axis=0))
        x = centred[:, columns]
        for a in range(1, A_PLSDA + 1):
            pls = PLSRegression(n_components=a, scale=False, tol=1e-14, max_iter=10_000).fit(
                x[train], dummy[train]
            )
            curve[a - 1][test] = pls.predict(x[test])
        pca = PCA(n_components=A_LDA, svd_solver="full").fit(x[train])
        lda[test] = (
            LinearDiscriminantAnalysis()
            .fit(pca.transform(x[train]), codes[train])
            .predict(pca.transform(x[test]))
        )
        pca = PCA(n_components=A_KNN, svd_solver="full").fit(x[train])
        knn[test] = (
            KNeighborsClassifier(n_neighbors=K)
            .fit(pca.transform(x[train]), codes[train])
            .predict(pca.transform(x[test]))
        )
        h[test] = _simca_distances(x[train], codes[train], x[test], n_classes)

    accepted = h <= 1.0
    return {
        "classes": names,
        "n": len(kept),
        "plsda_curve": np.asarray(
            [np.sqrt(np.mean((dummy - curve[a]) ** 2)) for a in range(A_PLSDA)]
        ),
        "plsda": _confusion(codes, curve[-1].argmax(axis=1), n_classes),
        "lda": _confusion(codes, lda, n_classes),
        "knn": _confusion(codes, knn, n_classes),
        "simca_table": np.asarray(
            [
                [float(np.sum(accepted[codes == j, k])) for k in range(n_classes)]
                for j in range(n_classes)
            ]
        ),
        "simca_none": np.asarray(
            [float(np.sum(~accepted[codes == j].any(axis=1))) for j in range(n_classes)]
        ),
        "simca_h": h,
    }


def compare_workflow(served: dict[str, Any], ref: dict[str, Any], tighten: float) -> list[Check]:
    results = served["results"]
    sklearn_ = f"scikit-learn {sklearn.__version__}"

    def table(node: str) -> NDArray[np.float64]:
        confusion = results[node]["classification"]["confusion"]["cross_validation"]
        return np.asarray(confusion, dtype=np.float64)

    simca = results["simca"]["simca"]
    sets = simca["sets"]["cross_validation"]
    checks = [
        Check(
            1,
            "PLS-DA: classes, in order",
            None,
            np.asarray(
                [
                    float(c == r)
                    for c, r in zip(
                        results["plsda"]["classification"]["classes"], ref["classes"], strict=True
                    )
                ]
            ),
            np.ones(len(ref["classes"])),
            "the class column, sorted",
        ),
        Check(
            1,
            f"PLS-DA: RMSECV curve, A = 1…{A_PLSDA} (dummy response, pooled)",
            "metrics",
            np.asarray(
                [results["plsda"]["metrics"][f"rmsecv_a{a}"] for a in range(1, A_PLSDA + 1)]
            ),
            ref["plsda_curve"],
            f"{sklearn_} `PLSRegression(scale=False)`",
            tighten,
        ),
        Check(
            1,
            "PLS-DA: cross-validated confusion",
            None,
            table("plsda"),
            ref["plsda"],
            f"{sklearn_} `PLSRegression`, largest dummy prediction",
        ),
        Check(
            1,
            "LDA: cross-validated confusion",
            None,
            table("lda"),
            ref["lda"],
            f"{sklearn_} `PCA` then `LinearDiscriminantAnalysis()`",
        ),
        Check(
            1,
            "kNN: cross-validated confusion",
            None,
            table("knn"),
            ref["knn"],
            f"{sklearn_} `PCA` then `KNeighborsClassifier({K})`",
        ),
        Check(
            1,
            "SIMCA: cross-validated acceptance",
            None,
            np.asarray(sets["table"], dtype=np.float64),
            ref["simca_table"],
            f"per-class {sklearn_} `PCA`, limits in SciPy {scipy.__version__}",
        ),
        Check(
            1,
            "SIMCA: accepted by no model",
            None,
            np.asarray(sets["none"], dtype=np.float64),
            ref["simca_none"],
            "the same",
        ),
    ]
    checks.extend(_lineage(served))
    distances = sets.get("distances")
    if distances is not None:
        checks.append(
            Check(
                1,
                "SIMCA: cross-validated reduced distances h",
                "predictions",
                np.asarray(distances, dtype=np.float64),
                ref["simca_h"],
                "the same",
                tighten,
            )
        )
    return checks


def _source_version(experiment: dict[str, Any]) -> str:
    nodes = experiment["pipeline_snapshot"]["nodes"]
    return str(next(node for node in nodes if node["type"] == "source")["version_id"])


def _lineage(served: dict[str, Any]) -> list[Check]:
    """The exclusion is in lineage: the cleaning run read the imported
    version, the comparison runs read the derived one, and the derived one
    names its parent and the rows it left out."""
    first, cleaned = served["first"], served["cleaned"]
    rows = sorted(flag["row"] for flag in served["excluded"])
    same = [
        (_source_version(served["cleaning"]), first["version_id"]),
        (_source_version(served["experiment"]), cleaned["version_id"]),
        (str(cleaned.get("derived_from")), first["version_id"]),
    ]
    return [
        Check(
            1,
            "Lineage: source version of the cleaning run, then of the comparison run, "
            "then the derived version's parent",
            None,
            np.asarray([float(ours == theirs) for ours, theirs in same]),
            np.ones(len(same)),
            "the imported and derived version ids",
        ),
        Check(
            1,
            "Lineage: rows the derived version left out",
            None,
            np.asarray(sorted(cleaned.get("excluded_samples", [])), dtype=np.float64),
            np.asarray(rows, dtype=np.float64),
            "the rows the stated rule excluded",
        ),
    ]


# --------------------------------------------------------------------------
# claim 3: PCR and a selected PLS on Tecator
# --------------------------------------------------------------------------


def drive_tecator(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, "tecator.csv", tecator_csv())
    source = _source(client)
    nodes = [
        source,
        {"id": "snv", "type": "preprocess", "inputs": [source["id"]], "step": {"kind": "snv"}},
        {
            "id": "savgol",
            "type": "preprocess",
            "inputs": ["snv"],
            "step": {"kind": "savgol", "window_length": 11, "polyorder": 2, "deriv": 1},
        },
        {
            "id": "kfold",
            "type": "split",
            "inputs": ["savgol"],
            "spec": {"kind": "kfold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
        },
        {
            "id": "centre",
            "type": "preprocess",
            "inputs": ["kfold"],
            "step": {"kind": "mean_centre"},
        },
        {
            "id": "pcr",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {"kind": "pcr", "n_components": A_REGRESSION, "target": TARGET},
        },
        {
            "id": "pls",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "pls",
                "n_components": A_REGRESSION,
                "algorithm": "nipals",
                "target": TARGET,
            },
        },
    ]
    _run(client, nodes)
    vip = _ok(client.get("/results/pls"))["regression"]["vip"]
    selected = [index for index, value in enumerate(vip) if value >= VIP_CUT]
    nodes += [
        {
            "id": "select",
            "type": "preprocess",
            "inputs": ["centre"],
            "step": {"kind": "select_variables", "indices": selected, "chosen_by": "vip"},
        },
        {
            "id": "pls_selected",
            "type": "estimator",
            "inputs": ["select"],
            "spec": {
                "kind": "pls",
                "n_components": min(A_REGRESSION, len(selected)),
                "algorithm": "nipals",
                "target": TARGET,
            },
        },
    ]
    _run(client, nodes)
    return {
        "version": imported["entry"]["versions"][0],
        "selected": selected,
        "n_variables": len(vip),
        "pcr": _ok(client.get("/results/pcr")),
        "pls_selected": _ok(client.get("/results/pls_selected")),
        "experiment": _ok(client.get("/experiments/current")),
    }


def reference_tecator(served: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    """SNV, Savitzky-Golay and the per-fold centring, each narrowed to float32
    where the store narrows; then scikit-learn on the served folds."""
    spectra, response = _csv_matrix()
    values = _f32(_savgol(_f32(_snv(_f32(spectra)))))
    columns = np.asarray(served["selected"], dtype=np.intp)
    a_selected = served["pls_selected"]["n_components"]
    split = next(
        one for one in served["experiment"]["resolved_splits"] if one["node_id"] == "kfold"
    )
    pcr = np.zeros((A_REGRESSION, response.size))
    pls = np.zeros((a_selected, response.size))
    for train_list, test_list in zip(split["train_indices"], split["test_indices"], strict=True):
        train, test = np.asarray(train_list), np.asarray(test_list)
        centred = _f32(values - values[train].mean(axis=0))
        for a in range(1, A_REGRESSION + 1):
            pca = PCA(n_components=a, svd_solver="full").fit(centred[train])
            fit = LinearRegression().fit(pca.transform(centred[train]), response[train])
            pcr[a - 1][test] = fit.predict(pca.transform(centred[test]))
        x = centred[:, columns]
        for a in range(1, a_selected + 1):
            model = PLSRegression(n_components=a, scale=False).fit(x[train], response[train])
            pls[a - 1][test] = np.asarray(model.predict(x[test])).ravel()

    def curve(predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(np.mean((predicted - response) ** 2, axis=1))

    return {"pcr": curve(pcr), "pls_selected": curve(pls)}


def compare_tecator(
    served: dict[str, Any], ref: dict[str, NDArray[np.float64]], tighten: float
) -> list[Check]:
    sklearn_ = f"scikit-learn {sklearn.__version__}"
    against = {
        "pcr": f"{sklearn_} `PCA(svd_solver='full')` then `LinearRegression()`",
        "pls_selected": f"{sklearn_} `PLSRegression(scale=False)` on the selected columns",
    }
    labels = {"pcr": "PCR", "pls_selected": "PLS on the VIP-selected variables"}
    checks = []
    for node in ("pcr", "pls_selected"):
        metrics = served[node]["metrics"]
        n = len(ref[node])
        checks.append(
            Check(
                3,
                f"{labels[node]}: RMSECV curve, A = 1…{n}",
                "metrics",
                np.asarray([metrics[f"rmsecv_a{a}"] for a in range(1, n + 1)]),
                ref[node],
                against[node],
                tighten,
            )
        )
    return checks


# --------------------------------------------------------------------------
# claim 2: the formats
# --------------------------------------------------------------------------


@dataclass
class Format:
    name: str
    file: str
    source: str
    export: str
    upload: Callable[[], tuple[str, bytes]]
    expected: Callable[[], tuple[NDArray[np.float64], NDArray[np.float64] | None]]
    #: Turns the served spectra into what `expected` gives, when the export is
    #: not the spectra themselves.
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None


def _mat() -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    folder = READERS / "mat"
    data = np.loadtxt(folder / "mlnir_slice_data.csv", delimiter=",").T
    return data, np.loadtxt(folder / "mlnir_slice_axis.csv", delimiter=",")


def _spc() -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    table = np.loadtxt(READERS / "spc" / "rohanisaac" / "nir.spc.txt")
    return table[:, 1:].T, table[:, 0]


def _spa() -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    # OMNIC's export ascends and pads one zero point below the range.
    twin = np.loadtxt(READERS / "spa" / "ecoflex-1.CSV", delimiter=",")[1:][::-1]
    return twin[:, 1][None, :], twin[:, 0]


ASD_FILES = sorted((READERS / "asd").glob("*.asd"))


def _asd_zip() -> tuple[str, bytes]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for file in ASD_FILES:
            bundle.write(file, file.name)
    return "eaton.zip", buffer.getvalue()


def _asd_published() -> NDArray[np.float64]:
    with (READERS / "asd" / "measurements.csv").open(encoding="utf-8") as file:
        rows = {row[1]: row[2:] for row in list(csv.reader(file))[1:]}
    return np.asarray([[float(v) for v in rows[f.name]] for f in ASD_FILES])


def _asd_factors(served: NDArray[np.float64]) -> NDArray[np.float64]:
    """Published over served, with the panel curve and each sample's VNIR
    splice divided out: all ones if the two differ by exactly those."""
    ratio: NDArray[np.float64] = _asd_published() / served
    panel = ratio[0].copy()
    ratio = ratio / panel
    ratio[:, :651] /= ratio[:, :1]
    return ratio


FORMATS = [
    Format(
        "MATLAB .mat",
        "mat/mlnir_slice.mat",
        "MLNIRdata, 10.5281/zenodo.16781223 (CC-BY-4.0), a 12-sample slice",
        "the publisher's CSV of the same matrix and its axis",
        lambda: ("mlnir_slice.mat", (READERS / "mat" / "mlnir_slice.mat").read_bytes()),
        _mat,
    ),
    Format(
        "Galactic SPC",
        "spc/rohanisaac/nir.spc",
        "rohanisaac/spc test data (GPL-3.0), new format, 20 subfiles",
        "rohanisaac/spc's own text conversion; the bone SPC files decision 0006 names "
        "carry no export of their own",
        lambda: ("nir.spc", (READERS / "spc" / "rohanisaac" / "nir.spc").read_bytes()),
        _spc,
    ),
    Format(
        "Thermo OMNIC SPA",
        "spa/ecoflex-1.SPA",
        "MXene-Based Ecoflex Dry Electrodes, 10.5281/zenodo.17691644 (CC-BY-4.0)",
        "OMNIC's own CSV export of the same spectrum",
        lambda: ("ecoflex-1.SPA", (READERS / "spa" / "ecoflex-1.SPA").read_bytes()),
        _spa,
    ),
    Format(
        "ASD FieldSpec",
        "asd/*.asd (a zip of four)",
        "Eaton Fire field spectroscopy, 10.5281/zenodo.16538735 (CC-BY-4.0)",
        "the record's published reflectance, less the panel calibration curve and one VNIR "
        "splice factor per sample (`tests/test_readers_asd.py`)",
        _asd_zip,
        lambda: (np.ones((len(ASD_FILES), 2151)), None),
        _asd_factors,
    ),
]


def drive_format(client: httpx.Client, one: Format) -> dict[str, Any]:
    name, data = one.upload()
    imported = _import(client, name, data)
    version = imported["entry"]["versions"][0]
    _run(client, [_source(client)])
    rows = ",".join(str(i) for i in range(version["n_samples"]))
    spectra = _ok(client.get("/spectra/source", params={"highlight": rows}))
    return {
        "version": version,
        "values": np.asarray([t["y"] for t in spectra["highlighted"]["traces"]]),
        "axis": np.asarray(spectra["highlighted"]["axis"]["values"]),
    }


def compare_format(one: Format, served: dict[str, Any], tighten: float) -> list[Check]:
    expected, axis = one.expected()
    values = one.transform(served["values"]) if one.transform else served["values"]
    checks = [Check(2, f"{one.name}: values", "stored", values, expected, one.export, tighten)]
    if axis is not None:
        checks.append(
            Check(2, f"{one.name}: axis", "stored", served["axis"], axis, one.export, tighten)
        )
    return checks


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def _row(check: Check) -> str:
    result = "pass" if check.passed else "**FAIL**"
    shape = "×".join(str(s) for s in check.ours.shape) or "1"
    return (
        f"| {check.quantity} | {shape} | {check.max_abs:.3g} | {check.bounds} "
        f"| {check.against} | {result} |"
    )


def _table(checks: list[Check]) -> list[str]:
    return [
        "| Quantity | Shape | max abs diff | Tolerance (rtol, atol) | Against | Result |",
        "| --- | --- | --- | --- | --- | --- |",
        *(_row(check) for check in checks),
    ]


def _matrix(names: list[str], table: Any, columns: list[str]) -> list[str]:
    head = "| | " + " | ".join(columns) + " |"
    rule = "| --- |" + " --- |" * len(columns)
    body = [
        f"| {name} | " + " | ".join(str(int(v)) for v in row) + " |"
        for name, row in zip(names, table, strict=True)
    ]
    return [head, rule, *body]


def record(
    workflow: dict[str, Any],
    tecator: dict[str, Any],
    formats: list[tuple[Format, dict[str, Any]]],
    checks: list[Check],
    started: datetime,
    tighten: float,
) -> tuple[str, bool]:
    first_met = all(check.passed for check in checks if check.claim == 1)
    second_met = all(check.passed for check in checks if check.claim == 2)
    third_met = all(check.passed for check in checks if check.claim == 3)
    met = first_met and second_met and third_met
    results = workflow["results"]
    env = workflow["experiment"]["environment"]
    classes = results["plsda"]["classification"]["classes"]
    n = workflow["cleaned"]["n_samples"]
    warnings = [w for w in workflow["validation"].get("warnings", [])]
    verdict = (
        "**Met.** All three claims hold at the stated tolerances."
        if met
        else "**Not met.** "
        + ("Claim 1 fails. " if not first_met else "")
        + ("Claim 2 fails. " if not second_met else "")
        + ("Claim 3 fails. " if not third_met else "")
        + "The failing rows are marked below."
    )
    excluded = (
        ", ".join(
            f"{flag['sample_id']} ({' · '.join(RULES[rule] for rule in flag['rules'])})"
            for flag in workflow["excluded"]
        )
        or "none"
    )
    simca = results["simca"]["simca"]["sets"]["cross_validation"]
    lines = [
        "# Phase 5 exit run",
        "",
        "Generated by `uv run python -m tests.exit_run_phase5"
        + (f" --tighten {tighten:g}" if tighten != 1.0 else "")
        + f"` on {started:%Y-%m-%d} at {started:%H:%M} UTC. **Do not edit by hand**: rerun "
        "the script.",
        "",
        "`PROPOSAL.md` §16, Phase 5: *an open multi-class spectral dataset is cleaned, its "
        "variables selected, and PLS-DA, SIMCA, LDA and kNN compared under the same stratified "
        "cross-validation, each matching its reference within stated tolerance; a file of each "
        "new format imports and matches its source.* Decision 0006 splits it into two claims. "
        "Issue #289 adds a third: PCR and variable-selected PLS on Tecator match their "
        "references. This is the run of all three, over HTTP against the served application.",
        "",
        f"## Verdict: {verdict}",
        "",
        "## What was run",
        "",
        f"- **Application:** chemometrics-workbench {env['app_version']}, Python "
        f"{env['python_version']}, numpy {env['packages']['numpy']}, scipy "
        f"{env['packages']['scipy']}, on {env['platform']}.",
        f"- **Reference:** scikit-learn {sklearn.__version__}, SciPy {scipy.__version__}, "
        f"NumPy {np.__version__}, on the experiment's own resolved folds. "
        "`PLSRegression` runs with `tol=1e-14`: at its default `tol=1e-06` its NIPALS "
        "stops early on a three-column response, and the reference itself is about 1e-5 "
        "off the converged fit.",
        f"- **Data:** `docs/examples/meat.csv`, derived from the Quadram Institute's "
        "fresh-meat FTIR set (CC0-1.0; Al-Jowder, Kemsley and Wilson, *Food Chemistry* 59, "
        "1997); `docs/examples/meat-source.md` says how. "
        f"{workflow['first']['n_samples']} samples × {workflow['first']['n_variables']} "
        f"wavenumbers, classes {', '.join(classes)}.",
        "",
        "## Claim 1: clean, select, compare",
        "",
        f"**Cleaned.** A PCA with {A_CLEAN} components on the centred spectra flagged "
        f"{len(workflow['flagged'])} of {workflow['first']['n_samples']} samples. This run "
        f"stands in for the user, and its rule is to exclude a sample that breaks "
        f"{RULES_TO_EXCLUDE} or more rules. Excluded: {excluded}. The exclusion is a derived "
        f"dataset version of {n} samples, and the pipeline moved onto it.",
        "",
        f"**Selected.** A PLS-DA with {A_PLSDA} latent variables under the split below gave VIP "
        f"≥ {VIP_CUT:g} for {len(workflow['selected'])} of {workflow['n_variables']} "
        "variables, applied as a `select_variables` step recorded as chosen by VIP. "
        + (
            "Validation warned, as it should: "
            + "; ".join(f"`{w['code']}` on `{w['node_id']}`" for w in warnings)
            + ". The selection saw every sample, so the cross-validated errors below are "
            "optimistic as estimates of a new sample's error; the comparison with the reference "
            "is unaffected, because both sides use the same columns."
            if warnings
            else "Validation raised no warning."
        ),
        "",
        f"**Compared** under one {N_SPLITS}-fold cross-validation stratified by "
        f"`{CLASS_COLUMN}`, seed {SEED}: PLS-DA {A_PLSDA} LV, LDA on {A_LDA} PCs, kNN "
        f"(k = {K}) on {A_KNN} PCs, SIMCA with {A_SIMCA} PCs per class, all below the same "
        "centring and selection.",
        "",
        "| Classifier | Accuracy (CV) | Sensitivity (CV) | Specificity (CV) |",
        "| --- | --- | --- | --- |",
        *(
            f"| {label} | {results[node]['metrics'].get('accuracy_cv', float('nan')):.3f} | – | – |"
            for node, label in (("plsda", "PLS-DA"), ("lda", "LDA"), ("knn", "kNN"))
        ),
        f"| SIMCA | – | {results['simca']['metrics']['sensitivity_cv']:.3f} "
        f"| {results['simca']['metrics']['specificity_cv']:.3f} |",
        "",
        "PLS-DA's cross-validated confusion (rows observed, columns assigned):",
        "",
        *_matrix(
            classes,
            results["plsda"]["classification"]["confusion"]["cross_validation"],
            classes,
        ),
        "",
        "SIMCA's cross-validated acceptance (rows observed, columns the class model):",
        "",
        *_matrix(
            classes,
            [[*row, none] for row, none in zip(simca["table"], simca["none"], strict=True)],
            [*classes, "none"],
        ),
        "",
        "Every served quantity against its independent reference. A table of counts must be "
        "equal; a continuous quantity must agree within the parity tolerance named.",
        "",
        *_table([check for check in checks if check.claim == 1]),
        "",
        "## Claim 2: a file of each new format",
        "",
        "Each file is imported on a project of its own, through `/import/preview` and "
        "`/import`, and read back at full resolution through `/spectra/source`. The store "
        f"is float32, so the tolerance is `stored` ({STORED.rtol:.0e}, {STORED.atol:.0e}).",
        "",
        "| Format | File | Source | Imported |",
        "| --- | --- | --- | --- |",
        *(
            f"| {one.name} | `{one.file}` | {one.source} | "
            f"{served['version']['n_samples']} × {served['version']['n_variables']} |"
            for one, served in formats
        ),
        "",
        *_table([check for check in checks if check.claim == 2]),
        "",
        "## Claim 3: PCR and a selected PLS on Tecator",
        "",
        f"Tecator ({tecator['version']['n_samples']} × {tecator['version']['n_variables']}, "
        f"`{TARGET}`), on a project of its own: SNV, Savitzky-Golay (11, 2, first "
        f"derivative), {N_SPLITS}-fold cross-validation with seed {SEED}, mean centring, then "
        f"PCR and PLS with {A_REGRESSION} components. PLS's VIP ≥ {VIP_CUT:g} kept "
        f"{len(tecator['selected'])} of {tecator['n_variables']} variables, applied as a "
        "`select_variables` step above a second PLS with "
        f"{tecator['pls_selected']['n_components']} components. The reference is "
        "rebuilt on the experiment's resolved folds, narrowed to float32 after SNV, "
        "Savitzky-Golay and the per-fold centring, where the store narrows. As in claim 1, "
        "the selection saw every sample, so the selected PLS's RMSECV is optimistic as an "
        "estimate of a new sample's error; the comparison is unaffected.",
        "",
        "| Model | RMSECV (served) | Q² (served) |",
        "| --- | --- | --- |",
        *(
            f"| {label} | {tecator[node]['metrics']['rmsecv']:.4f} "
            f"| {tecator[node]['metrics']['q2']:.4f} |"
            for node, label in (("pcr", "PCR"), ("pls_selected", "PLS, VIP-selected"))
        ),
        "",
        *_table([check for check in checks if check.claim == 3]),
        "",
    ]
    return "\n".join(lines), met


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tighten", type=float, default=1.0)
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory() as directory:
        with Served(Path(directory) / "workflow") as client:
            workflow = drive_workflow(client)
        with Served(Path(directory) / "tecator") as client:
            tecator = drive_tecator(client)
        formats = []
        for index, one in enumerate(FORMATS):
            with Served(Path(directory) / f"format_{index}") as client:
                formats.append((one, drive_format(client, one)))
    checks = compare_workflow(workflow, reference(workflow), args.tighten)
    checks.extend(compare_tecator(tecator, reference_tecator(tecator), args.tighten))
    for one, served in formats:
        checks.extend(compare_format(one, served, args.tighten))
    text, met = record(workflow, tecator, formats, checks, started, args.tighten)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(text, encoding="utf-8")
    for check in checks:
        print(
            f"{'pass' if check.passed else 'FAIL'}  {check.quantity}  max abs {check.max_abs:.3g}"
        )
    print(f"{'met' if met else 'NOT MET'}: {RECORD.relative_to(ROOT)}")
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
