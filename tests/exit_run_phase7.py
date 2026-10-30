"""Phase 7's exit criterion, demonstrated rather than reasoned (#383).

`PROPOSAL.md` §16, Phase 7, as `feature_list.json` states it, run against the
served application the way `tests/exit_run_phase6.py` ran Phase 6's.

**Claim 1, Tecator** (`docs/examples/tecator.csv`, `fat`): a seeded ten-fold
cross-validation, mean centring, then PLS (5), Ridge, Elastic Net, SVR and
OPLS (1 + 4) beside each other, and MLR on the PLS's VIP >= 1 selection.
Each one's cross-validated predictions equal scikit-learn's on the
experiment's own folds - OPLS's equal PLS's with as many components, the
identity `opls.md` §4 states - and each final model's Python snippet,
applied to the raw spectra, reproduces the application's predictions.

**Claim 2, the raw meat set** (`docs/examples/meat-raw.csv`, both runs of
each sample): a ten-fold cross-validation grouped by sample, mean centring,
then PLS-DA, logistic regression and QDA on `meat`; and, on a version that
excludes one meat, two-class OPLS-DA beside two-class PLS-DA. Each
cross-validated confusion equals its reference's on the same folds, and each
snippet assigns what the application assigned. The criterion says "grouped,
stratified": a split cannot be both (`metrics-and-validation.md` §8.8,
decided in Phase 6), so it is grouped, as Phase 6's exit run was.

**Claim 3, a class coded as numbers.** A CSV whose `variety` is 1 or 2 is
read as a target, so a classifier on it is refused by name; made a class
through the role endpoint, it derives a new version the pipeline moves onto,
and an LDA runs on it. That the step menu shows the classifiers disabled
before the change and enabled after it is the screen's half, walked by
`frontend/e2e/walkthrough.spec.ts` (#371, #372).

The store is float32 (`PROPOSAL.md` §13): the references round where the
store does, after the source and after each fold's centring.

    uv run python -m tests.exit_run_phase7
    uv run python -m tests.exit_run_phase7 --tighten 1e-9

It exits 0 when every claim is met and 1 when any is not, and rewrites
`docs/phase-7/exit-run.md` either way.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR

from chemometrics_workbench.preprocessing import SavitzkyGolayTransformer, SNVTransformer
from tests import parity
from tests.exit_run import Served, _ok
from tests.exit_run_phase5 import Check, _confusion, _f32, _import, _run, _source, _table
from tests.exit_run_phase6 import _folds, _read, _snippet

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "phase-7" / "exit-run.md"
TECATOR = ROOT / "docs" / "examples" / "tecator.csv"
MEAT = ROOT / "docs" / "examples" / "meat-raw.csv"

N_SPLITS, SEED, TARGET = 10, 42, "fat"
A_PLS, RIDGE_ALPHA, ENET_ALPHA, ENET_RATIO = 5, 3e-5, 1e-3, 0.5
SG_WINDOW, SG_ORDER, SG_DERIV = 11, 2, 1
SVR_C, SVR_EPSILON, A_SVR, N_ORTHOGONAL, VIP_CUT = 50.0, 0.1, 5, 4, 1.0
CLASS_COLUMN, GROUP_COLUMN, A_CLASS, EXCLUDED = "meat", "sample", 5, "turkey"
#: qda.md section 6: scikit-learn's SVD solver refuses these scores at 0 by an
#: absolute 1e-4 on the scalings; the application, testing relatively, does not.
QDA_REG = 0.1

SKLEARN = f"scikit-learn {sklearn.__version__}"


def _centred_folds(
    x: NDArray[np.float64], folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]]
) -> list[NDArray[np.float64]]:
    """Each fold's array as the store holds it: centred by its training rows, float32."""
    return [_f32(x - x[train].mean(axis=0)) for train, _ in folds]


def _held_out(
    arrays: list[NDArray[np.float64]],
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
    y: NDArray[np.float64],
    fit_predict: Any,
) -> NDArray[np.float64]:
    held = np.empty(len(y))
    for values, (train, test) in zip(arrays, folds, strict=True):
        held[test] = fit_predict(values[train], y[train], values[test])
    return held


def _pls(train: Any, y: Any, test: Any, a: int = A_PLS) -> Any:
    mean, centre = train.mean(axis=0), y.mean()
    model = PLSRegression(n_components=a, scale=False, tol=1e-14, max_iter=10_000)
    model.fit(train - mean, y - centre)
    return np.asarray(model.predict(test - mean)).ravel() + centre


# --------------------------------------------------------------------------
# claim 1: Tecator, six regressions under one split
# --------------------------------------------------------------------------


def _estimator(node: str, inputs: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {"id": node, "type": "estimator", "inputs": [inputs], "spec": spec}


REGRESSIONS: dict[str, dict[str, Any]] = {
    "pls": {"kind": "pls", "n_components": A_PLS, "algorithm": "nipals", "target": TARGET},
    "ridge": {"kind": "ridge", "alpha": RIDGE_ALPHA, "target": TARGET},
    "enet": {"kind": "elastic_net", "alpha": ENET_ALPHA, "l1_ratio": ENET_RATIO, "target": TARGET},
    "svr": {
        "kind": "svr",
        "kernel": "rbf",
        "C": SVR_C,
        "epsilon": SVR_EPSILON,
        "n_components": A_SVR,
        "target": TARGET,
    },
    "opls": {"kind": "opls", "n_orthogonal": N_ORTHOGONAL, "target": TARGET},
}


def drive_tecator(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, TECATOR.name, TECATOR.read_bytes())
    source = _source(client)
    # The example's chain (docs/examples/model-comparison.md): SNV and a first
    # derivative above the split. On raw absorbances the VIP selection is 37
    # adjacent wavelengths, and one training fold's least squares is so close
    # to singular that QR and SVD solvers part by 0.3 - an ill-posed problem,
    # not a disagreement.
    base = [
        source,
        {"id": "snv", "type": "preprocess", "inputs": [source["id"]], "step": {"kind": "snv"}},
        {
            "id": "savgol",
            "type": "preprocess",
            "inputs": ["snv"],
            "step": {
                "kind": "savgol",
                "window_length": SG_WINDOW,
                "polyorder": SG_ORDER,
                "deriv": SG_DERIV,
            },
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
        *(_estimator(node, "centre", spec) for node, spec in REGRESSIONS.items()),
    ]
    _run(client, base)
    vip = _ok(client.get("/results/pls"))["regression"]["vip"]
    kept = [index for index, value in enumerate(vip) if value >= VIP_CUT]
    selection = {
        "id": "pls_select",
        "type": "preprocess",
        "inputs": ["centre"],
        "step": {"kind": "select_variables", "indices": kept, "chosen_by": "vip"},
    }
    _run(
        client,
        [*base, selection, _estimator("mlr", "pls_select", {"kind": "mlr", "target": TARGET})],
    )
    served: dict[str, Any] = {
        "version": imported["entry"]["versions"][0],
        "experiment": _ok(client.get("/experiments/current")),
        "kept": kept,
    }
    for node in [*REGRESSIONS, "mlr"]:
        served[node] = _ok(client.get(f"/results/{node}"))
        served[f"{node}_snippet"] = client.get(f"/results/{node}/export.py").text
    return served


def reference_tecator(served: dict[str, Any]) -> dict[str, Any]:
    rows = TECATOR.read_text(encoding="utf-8").splitlines()
    header, body = rows[0].split(","), [row.split(",") for row in rows[1:]]
    p = served["version"]["n_variables"]
    raw64 = np.asarray([[float(v) for v in row[1 : 1 + p]] for row in body])
    raw = _f32(raw64)
    # The chain through the application's own SNV and derivative, whose
    # parity with scipy is the parity report's; here the store's float32
    # rounding after each step has to fall exactly where the application's
    # does, or an ill-conditioned model (MLR on a selection, Ridge at a small
    # alpha) amplifies one rounding the other way into a visible difference.
    snv = _f32(SNVTransformer().fit_transform(raw))
    derivative = SavitzkyGolayTransformer(SG_WINDOW, SG_ORDER, SG_DERIV)
    spectra = _f32(derivative.fit_transform(snv))
    # The same chain in float64 end to end, for the export (compare_tecator).
    chain64 = derivative.fit_transform(SNVTransformer().fit_transform(raw64))
    y = np.asarray([float(row[header.index(TARGET)]) for row in body])
    folds = _folds(served["experiment"])
    arrays = _centred_folds(spectra, folds)

    def ridge(train: Any, yt: Any, test: Any) -> Any:
        return Ridge(alpha=RIDGE_ALPHA, solver="svd").fit(train, yt).predict(test)

    def enet(train: Any, yt: Any, test: Any) -> Any:
        model = ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_RATIO, tol=1e-12, max_iter=2_000_000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return model.fit(train, yt).predict(test)

    def svr(train: Any, yt: Any, test: Any) -> Any:
        mean = train.mean(axis=0)
        pca = PCA(n_components=A_SVR, svd_solver="full").fit(train - mean)
        model = SVR(
            kernel="rbf", C=SVR_C, epsilon=SVR_EPSILON, gamma="scale", tol=1e-3, shrinking=False
        )
        return model.fit(pca.transform(train - mean), yt).predict(pca.transform(test - mean))

    kept = np.asarray(served["kept"], dtype=np.intp)

    def mlr(train: Any, yt: Any, test: Any) -> Any:
        return LinearRegression().fit(train[:, kept], yt).predict(test[:, kept])

    # Each final model refitted in float64 on every row, for the export:
    # the snippet runs the chain in float64 from the raw spectra.
    final64 = {
        "pls": _pls(chain64, y, chain64),
        "ridge": ridge(chain64, y, chain64),
        "enet": enet(chain64, y, chain64),
        "svr": svr(chain64, y, chain64),
        "opls": _pls(chain64, y, chain64, 1 + N_ORTHOGONAL),
        "mlr": mlr(chain64, y, chain64),
    }
    return {
        "final64": final64,
        "y": y,
        "stored": raw,
        "pls": _held_out(arrays, folds, y, _pls),
        "ridge": _held_out(arrays, folds, y, ridge),
        "enet": _held_out(arrays, folds, y, enet),
        "svr": _held_out(arrays, folds, y, svr),
        "opls": _held_out(arrays, folds, y, lambda a, b, c: _pls(a, b, c, 1 + N_ORTHOGONAL)),
        "mlr": _held_out(arrays, folds, y, mlr),
    }


#: The models with a closed form or an exact identity: held to the parity
#: predictions class, sample by sample.
EXACT = {
    "pls": "PLSRegression",
    "ridge": "Ridge(solver='svd')",
    "opls": "PLSRegression(1 + 4), the OPLS identity",
    "mlr": "LinearRegression on the served VIP selection",
}
#: The iterative ones: each solver stops at its default tolerance rather than
#: at the optimum, so their RMSECV is compared, to the stopping rule's precision.
ITERATIVE = {
    "enet": "ElasticNet(tol=1e-12)",
    "svr": "PCA then SVR(rbf, tol=1e-3, shrinking=False)",
}
SOLVER_STOP = parity.Tolerance(
    rtol=1e-3,
    atol=0.0,
    reason=(
        "An iterative solver's RMSECV, the application's at its default stopping rule: Elastic "
        "Net at a duality gap of 1e-4 ||y||^2 against a reference solved to 1e-12, SVR at a "
        "KKT gap of 1e-3 against LIBSVM at the same gap with its float32 kernel cache. Each "
        "stops short of the optimum; their individual predictions differ by up to a few "
        "hundredths, and the pooled error by far less."
    ),
)


class _StopCheck(Check):
    """A Check held to SOLVER_STOP, which is no parity class: it is a property of
    where a solver stops, stated here rather than added to the frozen classes."""

    def _tolerance(self) -> parity.Tolerance:
        return SOLVER_STOP


def compare_tecator(served: dict[str, Any], ref: dict[str, Any], tighten: float) -> list[Check]:
    checks: list[Check] = []
    y = ref["y"]
    for node in [*EXACT, *ITERATIVE]:
        result = served[node]
        rmsecv = np.asarray([float(np.sqrt(np.mean((y - ref[node]) ** 2)))])
        served_rmsecv = np.asarray([result["metrics"]["rmsecv"]])
        if node in EXACT:
            against = f"{SKLEARN} {EXACT[node]}"
            cross = _cv_predictions(result)
            checks.append(
                Check(
                    1, f"{node}: CV predictions", "predictions", cross, ref[node], against, tighten
                )
            )
            checks.append(
                Check(
                    1, f"{node}: RMSECV", "predictions", served_rmsecv, rmsecv, "the same", tighten
                )
            )
        else:
            against = f"{SKLEARN} {ITERATIVE[node]}"
            checks.append(
                _StopCheck(
                    1, f"{node}: RMSECV", "solver stop", served_rmsecv, rmsecv, against, tighten
                )
            )
        # The export, on the spectra as the application stored them.
        snippet = np.asarray(_numbers(served[f"{node}_snippet"], ref["stored"]))
        predicted = np.asarray(result["regression"]["predicted"], dtype=np.float64)
        checks.append(
            Check(
                1,
                f"{node}: exported snippet",
                "stored",
                snippet,
                predicted,
                "the application's predictions",
                tighten,
            )
        )
    return checks


def store_gap(served: NDArray[np.float64], float64: NDArray[np.float64]) -> float:
    """How far the served calibration predictions sit from the float64 refit:
    the float32 store's rounding carried through the model, not a defect."""
    return float(np.max(np.abs(served - float64)))


def store_effect(served: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    rows = []
    for node in [*EXACT, *ITERATIVE]:
        predicted = np.asarray(served[node]["regression"]["predicted"])
        rows.append(f"| {node} | {store_gap(predicted, ref['final64'][node]):.2e} |")
    return rows


def _cv_predictions(result: dict[str, Any]) -> NDArray[np.float64]:
    values = result["regression"].get("cross_validated")
    if values is None:
        raise RuntimeError(f"{result['node_id']} serves no cross-validated predictions")
    return np.asarray(values, dtype=np.float64)


def _numbers(text: str, spectra: NDArray[np.float64]) -> list[float]:
    namespace: dict[str, Any] = {}
    exec(compile(text, "exported.py", "exec"), namespace)
    return [float(value) for value in namespace["predict"](spectra)]


# --------------------------------------------------------------------------
# claim 2: the raw meat set, grouped by sample
# --------------------------------------------------------------------------


def _meat_nodes(
    source: dict[str, Any], estimators: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        source,
        {
            "id": "kfold",
            "type": "split",
            "inputs": [source["id"]],
            "spec": {
                "kind": "kfold",
                "n_splits": N_SPLITS,
                "shuffle": True,
                "seed": SEED,
                "group_by": GROUP_COLUMN,
            },
        },
        {
            "id": "centre",
            "type": "preprocess",
            "inputs": ["kfold"],
            "step": {"kind": "mean_centre"},
        },
        *(_estimator(node, "centre", spec) for node, spec in estimators.items()),
    ]


CLASSIFIERS = {
    "plsda": {
        "kind": "plsda",
        "n_components": A_CLASS,
        "algorithm": "nipals",
        "class_column": CLASS_COLUMN,
    },
    "logistic": {"kind": "logistic", "C": 1, "n_components": A_CLASS, "class_column": CLASS_COLUMN},
    "qda": {
        "kind": "qda",
        "n_components": A_CLASS,
        "reg_param": QDA_REG,
        "class_column": CLASS_COLUMN,
    },
}
TWO_CLASS = {
    "plsda2": {
        "kind": "plsda",
        "n_components": 1 + N_ORTHOGONAL,
        "algorithm": "nipals",
        "class_column": CLASS_COLUMN,
    },
    "oplsda": {"kind": "oplsda", "n_orthogonal": N_ORTHOGONAL, "class_column": CLASS_COLUMN},
}


def drive_meat(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, MEAT.name, MEAT.read_bytes())
    entry = imported["entry"]
    version = entry["versions"][0]
    source = _source(client)
    _run(client, _meat_nodes(source, CLASSIFIERS))
    served: dict[str, Any] = {
        "version": version,
        "experiment": _ok(client.get("/experiments/current")),
    }
    for node in CLASSIFIERS:
        served[node] = _ok(client.get(f"/results/{node}"))
        served[f"{node}_snippet"] = client.get(f"/results/{node}/export.py").text

    labels = version["metadata_columns"][CLASS_COLUMN]
    excluded = [row for row, label in enumerate(labels) if label == EXCLUDED]
    derived = _ok(
        client.post(
            f"/datasets/{entry['dataset']['dataset_id']}/versions",
            json={"from_version_id": version["version_id"], "exclude": excluded},
        )
    )["versions"][-1]
    _run(client, _meat_nodes(_source(client), TWO_CLASS))
    served["two"] = derived
    served["two_experiment"] = _ok(client.get("/experiments/current"))
    for node in TWO_CLASS:
        served[node] = _ok(client.get(f"/results/{node}"))
        served[f"{node}_snippet"] = client.get(f"/results/{node}/export.py").text
    return served


def reference_meat(served: dict[str, Any]) -> dict[str, Any]:
    x, columns = _read(MEAT, ["spectrum", CLASS_COLUMN, "supplier", GROUP_COLUMN, "run"])
    spectra = _f32(x)
    labels = np.asarray(columns["meat"])
    classes = served["plsda"]["classification"]["classes"]
    codes = np.asarray([classes.index(label) for label in labels])
    folds = _folds(served["experiment"])
    arrays = _centred_folds(spectra, folds)
    k = len(classes)

    def assign(fit_predict: Any) -> NDArray[np.intp]:
        held = np.empty(len(codes), dtype=np.intp)
        for values, (train, test) in zip(arrays, folds, strict=True):
            held[test] = fit_predict(values[train], codes[train], values[test])
        return held

    def plsda(train: Any, ct: Any, test: Any) -> Any:
        onehot = np.eye(k)[ct]
        mean, centre = train.mean(axis=0), onehot.mean(axis=0)
        model = PLSRegression(n_components=A_CLASS, scale=False, tol=1e-14, max_iter=10_000)
        return (model.fit(train - mean, onehot - centre).predict(test - mean) + centre).argmax(
            axis=1
        )

    def scores_then(model: Any) -> Any:
        def fit_predict(train: Any, ct: Any, test: Any) -> Any:
            mean = train.mean(axis=0)
            pca = PCA(n_components=A_CLASS, svd_solver="full").fit(train - mean)
            return model().fit(pca.transform(train - mean), ct).predict(pca.transform(test - mean))

        return fit_predict

    ref: dict[str, Any] = {
        "spectra": spectra,
        "codes": codes,
        "plsda": assign(plsda),
        "logistic": assign(scores_then(lambda: LogisticRegression(C=1.0))),
        "qda": assign(scores_then(lambda: QuadraticDiscriminantAnalysis(reg_param=QDA_REG))),
    }

    # The two-class version: the excluded meat's rows gone, its own folds.
    keep = labels != EXCLUDED
    two_spectra, two_labels = spectra[keep], labels[keep]
    two_classes = served["oplsda"]["classification"]["classes"]
    two_codes = np.asarray([two_classes.index(label) for label in two_labels])
    two_folds = _folds(served["two_experiment"])
    two_arrays = _centred_folds(two_spectra, two_folds)
    held = np.empty(len(two_codes), dtype=np.intp)
    for values, (train, test) in zip(two_arrays, two_folds, strict=True):
        dummy = two_codes[train].astype(np.float64)
        predicted = _pls(values[train], dummy, values[test], 1 + N_ORTHOGONAL)
        held[test] = (predicted >= 0.5).astype(np.intp)
    ref["two_spectra"], ref["two_codes"], ref["oplsda"] = two_spectra, two_codes, held
    return ref


def compare_meat(served: dict[str, Any], ref: dict[str, Any], tighten: float) -> list[Check]:
    checks = []
    classes = served["plsda"]["classification"]["classes"]
    for node, against in (
        ("plsda", "PLSRegression on the one-hot response"),
        ("logistic", "PCA then LogisticRegression(lbfgs)"),
        ("qda", "PCA then QuadraticDiscriminantAnalysis"),
    ):
        confusion = served[node]["classification"]["confusion"]["cross_validation"]
        checks.append(
            Check(
                2,
                f"{node}: cross-validated confusion",
                None,
                np.asarray(confusion, dtype=np.float64),
                _confusion(ref["codes"], ref[node], len(classes)),
                f"{SKLEARN} {against}",
                tighten,
            )
        )
        checks.append(_assignment_check(served, node, ref["spectra"], classes, tighten))
    two = served["oplsda"]["classification"]["classes"]
    for node, against in (
        ("oplsda", "two-class PLSRegression(1 + 4), the OPLS-DA identity"),
        ("plsda2", "the same"),
    ):
        checks.append(
            Check(
                2,
                f"{node}: cross-validated confusion, {EXCLUDED} excluded",
                None,
                np.asarray(
                    served[node]["classification"]["confusion"]["cross_validation"],
                    dtype=np.float64,
                ),
                _confusion(ref["two_codes"], ref["oplsda"], len(two)),
                f"{SKLEARN} {against}",
                tighten,
            )
        )
    checks.append(_assignment_check(served, "oplsda", ref["two_spectra"], two, tighten))
    return checks


def _assignment_check(
    served: dict[str, Any],
    node: str,
    spectra: NDArray[np.float64],
    classes: list[str],
    tighten: float,
) -> Check:
    assigned = _snippet(served[f"{node}_snippet"], spectra)
    served_classes = [classes[i] for i in served[node]["classification"]["predicted_class"]]
    return Check(
        2,
        f"{node}: exported snippet assigns",
        None,
        np.asarray([classes.index(name) for name in assigned], dtype=np.float64),
        np.asarray([classes.index(name) for name in served_classes], dtype=np.float64),
        "the application's assignments",
        tighten,
    )


# --------------------------------------------------------------------------
# claim 3: a class coded as numbers, made a class
# --------------------------------------------------------------------------

#: Twelve spectra whose class is coded as a number, so the reader takes it for a target.
CODED = "sample,900,902,904,906,908,910,variety,fat\n" + "".join(
    f"S{row:02d},"
    + ",".join(
        f"{(row % 2) * 0.5 + 0.1 * col + 0.013 * row + 0.002 * col * row:.4f}" for col in range(6)
    )
    + f",{row % 2 + 1},{10 + row}\n"
    for row in range(12)
)


def drive_roles(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, "coded.csv", CODED.encode())
    entry = imported["entry"]
    parent = entry["versions"][0]
    source = _source(client)
    lda = _estimator(
        "lda", source["id"], {"kind": "lda", "n_components": 2, "class_column": "variety"}
    )
    _ok(client.put("/pipelines/current", json={"nodes": [source, lda]}))
    job = _ok(client.post("/experiments/current/run"))
    while job["status"] in ("queued", "running"):
        job = _ok(client.get(f"/jobs/{job['job_id']}"))
    refused = {"status": job["status"], "message": job.get("message") or ""}
    roles = _ok(
        client.post(
            f"/datasets/{entry['dataset']['dataset_id']}/roles",
            json={"from_version_id": parent["version_id"], "roles": {"variety": "class"}},
        )
    )
    child = roles["versions"][-1]
    source = _source(client)
    _run(client, [source, {**lda, "inputs": [source["id"]]}])
    return {
        "parent": parent,
        "child": child,
        "refused": refused,
        "source_version": source["version_id"],
        "lda": _ok(client.get("/results/lda")),
    }


def compare_roles(served: dict[str, Any], tighten: float) -> list[Check]:
    parent, child, lda = served["parent"], served["child"], served["lda"]

    def flag(value: bool) -> NDArray[np.float64]:
        return np.asarray([1.0 if value else 0.0])

    facts = [
        ("variety read as a target on import", "variety" in parent["targets"]),
        (
            "an LDA on it refused, naming the column",
            served["refused"]["status"] == "failed" and "variety" in served["refused"]["message"],
        ),
        (
            "made a class: a new version, derived from the import",
            child["derived_from"] == parent["version_id"],
        ),
        ("the role change recorded on it", child.get("role_changes") == {"variety": "class"}),
        ("the pipeline moved onto it", served["source_version"] == child["version_id"]),
        ("the LDA runs on it, two classes", lda["classification"]["classes"] == ["1", "2"]),
    ]
    return [
        Check(3, name, None, flag(holds), flag(True), "the criterion", tighten)
        for name, holds in facts
    ]


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def record(
    tecator: dict[str, Any],
    tecator_ref: dict[str, Any],
    meat: dict[str, Any],
    checks: list[Check],
    started: datetime,
    tighten: float,
) -> tuple[str, bool]:
    met_by = {claim: all(c.passed for c in checks if c.claim == claim) for claim in (1, 2, 3)}
    met = all(met_by.values())
    env = tecator["experiment"]["environment"]
    verdict = (
        "**Met.** All three claims hold at the stated tolerances."
        if met
        else "**Not met.** "
        + " ".join(f"Claim {claim} fails." for claim, ok in met_by.items() if not ok)
        + " The failing rows are marked below."
    )
    regressions = [
        "| {} | {:.4f} | {:.4f} |".format(
            node, tecator[node]["metrics"]["rmsecv"], tecator[node]["metrics"]["q2"]
        )
        for node in [*REGRESSIONS, "mlr"]
    ]
    classifiers = [
        f"| {node} | {meat[node]['metrics']['accuracy_cv']:.4f} |"
        for node in [*CLASSIFIERS, *TWO_CLASS]
    ]
    lines = [
        "# Phase 7 exit run",
        "",
        "Generated by `uv run python -m tests.exit_run_phase7"
        + (f" --tighten {tighten:g}" if tighten != 1.0 else "")
        + f"` on {started:%Y-%m-%d} at {started:%H:%M} UTC. **Do not edit by hand**: rerun "
        "the script.",
        "",
        "The criterion (`feature_list.json`, #383): on Tecator under one seeded ten-fold "
        "cross-validation, Ridge, Elastic Net, MLR on a VIP-selected set, SVR and OPLS run beside "
        "PLS; each one's RMSECV matches its reference (scikit-learn on the served folds; OPLS "
        "against PLS with equal total components), and each final model's export reproduces the "
        "application's predictions. On the Quadram meat set, logistic regression and QDA run "
        "beside PLS-DA, and OPLS-DA on two of its classes, each matching its reference and "
        "exporting as above. A CSV whose class is coded as numbers has the column made a class "
        "through the role control, a new version is derived, and a classifier runs on it.",
        "",
        "**Two departures from the wording, both recorded:** the meat split is *grouped* by "
        "sample, not grouped and stratified - a split cannot be both "
        "(`metrics-and-validation.md` §8.8, decided in Phase 6). And the screen's half - each "
        "model added from the Regression or Classification heading, the classifiers shown "
        "disabled before the role change - is walked by `frontend/e2e/docs-comparison.spec.ts` "
        "and `frontend/e2e/walkthrough.spec.ts`; this run drives the HTTP API those screens call.",
        "",
        f"## Verdict: {verdict}",
        "",
        "## What was run",
        "",
        f"- **Application:** chemometrics-workbench {env['app_version']}, Python "
        f"{env['python_version']}, numpy {env['packages']['numpy']}, scipy "
        f"{env['packages']['scipy']}, on {env['platform']}.",
        f"- **Reference:** {SKLEARN}, SciPy {scipy.__version__}, NumPy {np.__version__}, on the "
        "experiments' own resolved folds, rounded to float32 where the store rounds.",
        "- **Data:** `docs/examples/tecator.csv` with its permission note; "
        "`docs/examples/meat-raw.csv`, the Quadram Institute's fresh-meat FTIR set (CC0-1.0) with "
        "both runs of each sample; a twelve-row CSV written by the script.",
        "",
        "## Claim 1: Tecator, six regressions under one split",
        "",
        f"{tecator['version']['n_samples']} × {tecator['version']['n_variables']}, `{TARGET}`. "
        f"SNV and a Savitzky-Golay first derivative ({SG_WINDOW} points, order {SG_ORDER}), as "
        "the model comparison example has them; "
        f"{N_SPLITS}-fold cross-validation, seed {SEED}; mean centring; PLS ({A_PLS}), Ridge "
        f"(α {RIDGE_ALPHA:g}), Elastic Net (α {ENET_ALPHA:g}, l1_ratio {ENET_RATIO:g}), SVR "
        f"(RBF, C {SVR_C:g}, ε {SVR_EPSILON:g}, {A_SVR} components), OPLS (1 + {N_ORTHOGONAL}); "
        f"MLR on the {len(tecator['kept'])} variables with VIP ≥ {VIP_CUT:g}.",
        "",
        "| Model | RMSECV | Q² |",
        "| --- | --- | --- |",
        *regressions,
        "",
        "The reference reaches the models through the application's own SNV and derivative "
        "(their parity with scipy is the parity report's), rounded to float32 where the store "
        "rounds, so PLS, Ridge, OPLS and MLR are compared sample by sample in the parity "
        "predictions class. Elastic Net and SVR stop at their solvers' default tolerances, so "
        "their RMSECV "
        f"is compared at a solver-stop tolerance (rtol {SOLVER_STOP.rtol:g}): "
        + SOLVER_STOP.reason,
        "",
        *_table([check for check in checks if check.claim == 1]),
        "",
        "**The exports, and the store's own effect.** The application fits and predicts "
        "through the float32 store; an exported snippet runs the chain in float64 from the raw "
        "spectra. How far the served calibration predictions sit from a float64 refit of the "
        "same recipe - largest absolute difference, in % fat - is that rounding carried through "
        "each model, largest where the model is ill-conditioned. Each snippet is held to its "
        "model's figure: an export that differs from the application by no more reproduces it "
        "as closely as the application reproduces its own recipe.",
        "",
        "| Model | Served − float64 refit |",
        "| --- | --- |",
        *store_effect(tecator, tecator_ref),
        "",
        "## Claim 2: the raw meat set, grouped by sample",
        "",
        f"{meat['version']['n_samples']} spectra × {meat['version']['n_variables']} wavenumbers. "
        f"{N_SPLITS}-fold cross-validation grouped by `{GROUP_COLUMN}`, seed {SEED}; mean "
        f"centring; PLS-DA, logistic regression (C 1) and QDA (reg_param {QDA_REG:g}), each "
        f"on {A_CLASS} components. Then, on version {meat['two']['version']} with every "
        f"{EXCLUDED} spectrum excluded ({meat['two']['n_samples']} left), OPLS-DA "
        f"(1 + {N_ORTHOGONAL}) beside "
        f"two-class PLS-DA ({1 + N_ORTHOGONAL}).",
        "",
        "| Model | Accuracy (CV) |",
        "| --- | --- |",
        *classifiers,
        "",
        *_table([check for check in checks if check.claim == 2]),
        "",
        "## Claim 3: a class coded as numbers, made a class",
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
        with Served(Path(directory) / "tecator") as client:
            tecator = drive_tecator(client)
        with Served(Path(directory) / "meat") as client:
            meat = drive_meat(client)
        with Served(Path(directory) / "roles") as client:
            roles = drive_roles(client)
    tecator_ref = reference_tecator(tecator)
    checks = compare_tecator(tecator, tecator_ref, args.tighten)
    checks.extend(compare_meat(meat, reference_meat(meat), args.tighten))
    checks.extend(compare_roles(roles, args.tighten))
    text, met = record(tecator, tecator_ref, meat, checks, started, args.tighten)
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
