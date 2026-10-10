"""Phase 6's exit criterion, demonstrated rather than reasoned (#341).

`PROPOSAL.md` §16, Phase 6, as `feature_list.json` states it: on the raw
Quadram meat set, grouped by sample, the validation a user reports is the one
an independent implementation computes; and on Tecator, the final model and
its bootstrap intervals are. Run against the served application, the way
`tests/exit_run_phase5.py` ran Phase 5's.

**Claim 1, the raw meat set** (`docs/examples/meat-raw.csv`, 120 spectra, both
runs of each of 60 samples): a ten-fold cross-validation grouped by `sample`,
mean centring, then a PLS-DA and an SVM on `meat`.

1. **Grouped.** In every resolved fold, no sample has a run on both sides.
2. **PLS-DA.** The cross-validated confusion equals scikit-learn's
   `PLSRegression` on a one-hot response, assigned by the largest column, on
   the experiment's own folds.
3. **Permutation.** The served test of 100 permutations, seed 0, gives the
   null and p-value of scikit-learn's `permutation_test_score` handed the same
   permutations and folds.
4. **Final model.** The model reported below the split is fitted on all 120
   spectra; its predictions equal `PLSRegression` fitted on all of them.
5. **Nested.** A VIP ≥ 1 selection validated in the outer loop gives the
   outer RMSECV, and the per-fold selections, of a scikit-learn rebuild.
6. **SVM.** The cross-validated confusion equals `PCA` then `SVC` on the same
   folds.
7. **Export.** The JSON model and the Python snippet of the final PLS-DA,
   each applied to the raw spectra, assign what the application assigned.

**Claim 2, Tecator.** SNV-free and short: ten-fold cross-validation, mean
centring and a five-component PLS on `fat`. The exported coefficients equal
`PLSRegression` fitted on every sample, and 200 bootstrap resamples, seed 0,
give the percentile intervals of the same refit on the same resampled rows.

**The store is float32** (`PROPOSAL.md` §13): the reference rounds where the
store does, after the source and after each fold's centring.

    uv run python -m tests.exit_run_phase6
    uv run python -m tests.exit_run_phase6 --tighten 1e-9

It exits 0 when every claim is met and 1 when any is not, and rewrites
`docs/phase-6/exit-run.md` either way. `--tighten` scales every numeric
tolerance, to show the comparison can fail.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import permutation_test_score
from sklearn.svm import SVC

from tests.exit_run import Served, _ok
from tests.exit_run_phase5 import Check, _confusion, _f32, _import, _matrix, _run, _source, _table

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "phase-6" / "exit-run.md"
MEAT = ROOT / "docs" / "examples" / "meat-raw.csv"
TECATOR = ROOT / "docs" / "examples" / "tecator.csv"

CLASS_COLUMN, GROUP_COLUMN = "meat", "sample"
N_SPLITS, SEED = 10, 42
A_PLSDA, A_SVM, VIP_CUT = 5, 5, 1.0
N_PERMUTATIONS, PERMUTATION_SEED = 100, 0
TARGET, A_PLS = "fat", 5
N_RESAMPLES, LEVEL, BOOTSTRAP_SEED = 200, 0.95, 0

SKLEARN = f"scikit-learn {sklearn.__version__}"


def _pls(x: NDArray[np.float64], y: NDArray[np.float64], a: int) -> Any:
    """`PLSRegression` iterated to its fixed point: at its default tolerance its
    NIPALS stops early on a several-column response."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return PLSRegression(n_components=a, scale=False, tol=1e-14, max_iter=10_000).fit(x, y)


def _vip(model: Any) -> NDArray[np.float64]:
    """`pls-regression.md` §8 from scikit-learn's own fit, the response sum of
    squares summed over every column for PLS2 (`variable-selection.md` §9)."""
    w, t = np.asarray(model.x_weights_), np.asarray(model.x_scores_)
    q = np.asarray(model.y_loadings_)
    explained = np.sum(q**2, axis=0) * np.sum(t**2, axis=0)
    w = w / np.linalg.norm(w, axis=0)
    return np.asarray(np.sqrt(w.shape[0] * (w**2 @ explained) / explained.sum()))


class _CentredPlsDa(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """The served chain in scikit-learn's interface: centre on the training
    rows, round to the store's float32, PLS2 on the one-hot, largest column."""

    def __init__(self, n_components: int = A_PLSDA, n_classes: int = 3) -> None:
        self.n_components = n_components
        self.n_classes = n_classes

    def fit(self, x: NDArray[np.float64], y: NDArray[np.intp]) -> _CentredPlsDa:
        self.mean_ = x.mean(axis=0)
        self.model_ = _pls(_f32(x - self.mean_), np.eye(self.n_classes)[y], self.n_components)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.intp]:
        return np.asarray(self.model_.predict(_f32(x - self.mean_)).argmax(axis=1))


def _read(path: Path, labels: list[str]) -> tuple[NDArray[np.float64], dict[str, list[str]]]:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    header, body = rows[0], rows[1:]
    columns = {name: [row[header.index(name)] for row in body] for name in labels}
    first = max(header.index(name) for name in labels) + 1 if labels else 1
    return np.asarray([[float(v) for v in row[first:]] for row in body]), columns


def _folds(experiment: dict[str, Any]) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    split = next(one for one in experiment["resolved_splits"] if one["node_id"] == "kfold")
    return [
        (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for train, test in zip(split["train_indices"], split["test_indices"], strict=True)
    ]


# --------------------------------------------------------------------------
# claim 1: the raw meat set, grouped by sample
# --------------------------------------------------------------------------


def drive_meat(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, MEAT.name, MEAT.read_bytes())
    source = _source(client)
    nodes = [
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
        {
            "id": "plsda",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "plsda",
                "n_components": A_PLSDA,
                "algorithm": "nipals",
                "class_column": CLASS_COLUMN,
            },
        },
        {
            "id": "svm",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "svm",
                "kernel": "rbf",
                "C": 1,
                "n_components": A_SVM,
                "class_column": CLASS_COLUMN,
            },
        },
    ]
    _run(client, nodes)
    job = _ok(
        client.post(
            f"/results/plsda/permutation?n_permutations={N_PERMUTATIONS}&seed={PERMUTATION_SEED}"
        )
    )
    while job["status"] in ("queued", "running"):
        time.sleep(0.2)
        job = _ok(client.get(f"/jobs/{job['job_id']}"))
    if job["status"] != "succeeded":
        raise RuntimeError(f"the permutation test ended {job['status']}: {job['message']}")
    return {
        "version": imported["entry"]["versions"][0],
        "experiment": _ok(client.get("/experiments/current")),
        "plsda": _ok(client.get("/results/plsda")),
        "svm": _ok(client.get("/results/svm")),
        "permutation": _ok(client.get(f"/permutations/{job['job_id']}")),
        "nested": _ok(client.get(f"/results/plsda/nested?method=vip&cut={VIP_CUT}")),
        "json": _ok(client.get("/results/plsda/export.json")),
        "snippet": client.get("/results/plsda/export.py").text,
    }


def reference_meat(served: dict[str, Any]) -> dict[str, Any]:
    raw, columns = _read(MEAT, ["spectrum", CLASS_COLUMN, "supplier", GROUP_COLUMN, "run"])
    x = _f32(raw)
    names = sorted(set(columns[CLASS_COLUMN]))
    codes = np.asarray([names.index(label) for label in columns[CLASS_COLUMN]], dtype=np.intp)
    groups = columns[GROUP_COLUMN]
    dummy = np.eye(len(names))[codes]
    folds = _folds(served["experiment"])

    split_groups = sum(
        len({groups[i] for i in train} & {groups[i] for i in test}) for train, test in folds
    )
    plsda = np.empty(len(codes), dtype=np.intp)
    svm = np.empty(len(codes), dtype=np.intp)
    outer = np.empty_like(dummy)
    selections: list[list[int]] = []
    for train, test in folds:
        centred = _f32(x - x[train].mean(axis=0))
        plsda[test] = _pls(centred[train], dummy[train], A_PLSDA).predict(centred[test]).argmax(1)
        pca = PCA(n_components=A_SVM, svd_solver="full").fit(centred[train])
        svc = SVC(kernel="rbf", C=1.0, gamma="scale", shrinking=False)
        svm[test] = svc.fit(pca.transform(centred[train]), codes[train]).predict(
            pca.transform(centred[test])
        )
        # variable-selection.md §8: VIP from a PLS on the outer training rows
        # alone, then the selected model scored on the held-out rows.
        xt, yt = centred[train], dummy[train]
        chosen = np.flatnonzero(_vip(_pls(xt - xt.mean(axis=0), yt, A_PLSDA)) >= VIP_CUT)
        block = xt[:, chosen]
        model = _pls(block - block.mean(axis=0), yt - yt.mean(axis=0), A_PLSDA)
        outer[test] = model.predict(centred[np.ix_(test, chosen)] - block.mean(axis=0)) + yt.mean(
            axis=0
        )
        selections.append([int(j) for j in chosen])

    observed, null, p_value = permutation_test_score(
        _CentredPlsDa(A_PLSDA, len(names)),
        x,
        codes,
        cv=folds,
        scoring="accuracy",
        n_permutations=N_PERMUTATIONS,
        random_state=_OurStream(PERMUTATION_SEED),
    )
    centred_all = _f32(x - x.mean(axis=0))
    final = _pls(centred_all, dummy, A_PLSDA)
    return {
        "classes": names,
        "codes": codes,
        "raw": raw,
        "split_groups": split_groups,
        "plsda": _confusion(codes, plsda, len(names)),
        "svm": _confusion(codes, svm, len(names)),
        "observed": observed,
        "null": np.asarray(null),
        "p_value": p_value,
        "final": np.asarray(final.predict(centred_all)),
        "final_assigned": np.asarray(final.predict(centred_all).argmax(axis=1)),
        "outer_rmsecv": float(np.sqrt(np.mean((dummy - outer) ** 2))),
        "selections": selections,
    }


class _OurStream(np.random.RandomState):
    """A `RandomState` whose `permutation` draws from `default_rng(seed)`, so
    scikit-learn shuffles exactly as §14 does (`tests/test_permutation.py`)."""

    def __init__(self, seed: int) -> None:
        super().__init__(0)
        self._rng = np.random.default_rng(seed)

    def permutation(self, x: Any) -> Any:
        return self._rng.permutation(x)


def _snippet(text: str, spectra: NDArray[np.float64]) -> list[str]:
    namespace: dict[str, Any] = {}
    exec(compile(text, "exported.py", "exec"), namespace)
    return [str(label) for label in namespace["predict"](spectra)]


def _json_assign(model: dict[str, Any], spectra: NDArray[np.float64]) -> list[str]:
    """`model-export.md` §4: a mean centre folds away, so no residual chain."""
    if model["preprocessing"]:
        raise RuntimeError(f"expected an empty residual chain, got {model['preprocessing']}")
    y = spectra @ np.asarray(model["coefficients"]) + np.asarray(model["intercept"])
    return [model["model"]["classes"][k] for k in y.argmax(axis=1)]


def compare_meat(served: dict[str, Any], ref: dict[str, Any], tighten: float) -> list[Check]:
    plsda, svm, permutation = served["plsda"], served["svm"], served["permutation"]
    classes = ref["classes"]
    assigned = np.asarray(plsda["classification"]["predicted_class"], dtype=np.float64)
    rows = np.asarray([sample["index"] for sample in plsda["samples"]], dtype=np.intp)
    application = [classes[int(k)] for k in assigned]
    json_labels = _json_assign(served["json"], ref["raw"][rows])
    snippet_labels = _snippet(served["snippet"], ref["raw"][rows])

    def labels(values: list[str]) -> NDArray[np.float64]:
        return np.asarray([classes.index(v) for v in values], dtype=np.float64)

    def check(quantity: str, tolerance: str | None, ours: Any, theirs: Any, against: str) -> Check:
        return Check(
            1,
            quantity,
            tolerance,
            np.asarray(ours, dtype=np.float64),
            np.asarray(theirs, dtype=np.float64),
            against,
            tighten,
        )

    return [
        check("samples split across a fold's sides", None, 0, ref["split_groups"], "the folds"),
        check(
            "PLS-DA confusion (CV)",
            None,
            plsda["classification"]["confusion"]["cross_validation"],
            ref["plsda"],
            f"{SKLEARN} PLSRegression, one-hot",
        ),
        check(
            "PLS-DA accuracy (CV)",
            "metrics",
            plsda["metrics"]["accuracy_cv"],
            np.trace(ref["plsda"]) / ref["plsda"].sum(),
            f"{SKLEARN} PLSRegression, one-hot",
        ),
        check(
            "permutation: observed accuracy",
            "metrics",
            permutation["observed"],
            ref["observed"],
            f"{SKLEARN} permutation_test_score",
        ),
        check(
            f"permutation: null ({N_PERMUTATIONS})",
            "metrics",
            permutation["null"],
            ref["null"],
            f"{SKLEARN} permutation_test_score",
        ),
        check(
            "permutation: p-value",
            "metrics",
            permutation["p_value"],
            ref["p_value"],
            f"{SKLEARN} permutation_test_score",
        ),
        check(
            "final model: rows fitted",
            None,
            plsda["n_samples"],
            len(ref["codes"]),
            "the dataset",
        ),
        check(
            "final model: assignments (calibration)",
            None,
            assigned,
            ref["final_assigned"][rows],
            f"{SKLEARN} PLSRegression on all 120",
        ),
        check(
            "nested VIP: outer RMSECV",
            "metrics",
            served["nested"]["outer_rmsecv"],
            ref["outer_rmsecv"],
            f"{SKLEARN} PLSRegression, VIP rebuilt",
        ),
        check(
            "nested VIP: variables kept per outer fold",
            None,
            served["nested"]["selected_per_fold"],
            [len(s) for s in ref["selections"]],
            f"{SKLEARN} PLSRegression, VIP rebuilt",
        ),
        check(
            "SVM confusion (CV)",
            None,
            svm["classification"]["confusion"]["cross_validation"],
            ref["svm"],
            f"{SKLEARN} PCA then SVC",
        ),
        check(
            "export: JSON model's assignments",
            None,
            labels(json_labels),
            labels(application),
            "the application",
        ),
        check(
            "export: Python snippet's assignments",
            None,
            labels(snippet_labels),
            labels(application),
            "the application",
        ),
    ]


# --------------------------------------------------------------------------
# claim 2: Tecator's final model and its bootstrap intervals
# --------------------------------------------------------------------------


def drive_tecator(client: httpx.Client) -> dict[str, Any]:
    imported = _import(client, TECATOR.name, TECATOR.read_bytes())
    source = _source(client)
    nodes = [
        source,
        {
            "id": "kfold",
            "type": "split",
            "inputs": [source["id"]],
            "spec": {"kind": "kfold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
        },
        {
            "id": "centre",
            "type": "preprocess",
            "inputs": ["kfold"],
            "step": {"kind": "mean_centre"},
        },
        {
            "id": "pls",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "pls",
                "n_components": A_PLS,
                "algorithm": "nipals",
                "target": TARGET,
            },
        },
    ]
    _run(client, nodes)
    query = f"n_resamples={N_RESAMPLES}&level={LEVEL}&seed={BOOTSTRAP_SEED}"
    return {
        "version": imported["entry"]["versions"][0],
        "pls": _ok(client.get("/results/pls")),
        "json": _ok(client.get("/results/pls/export.json")),
        "bootstrap": _ok(client.get(f"/results/pls/bootstrap?{query}")),
    }


def reference_tecator() -> dict[str, NDArray[np.float64]]:
    rows = list(csv.reader(io.StringIO(TECATOR.read_text(encoding="utf-8"))))
    header, body = rows[0], rows[1:]
    p = sum(1 for name in header[1:] if _is_number(name))
    x = _f32(np.asarray([[float(v) for v in row[1 : 1 + p]] for row in body]))
    y = np.asarray([float(row[header.index(TARGET)]) for row in body])

    def fit(xb: NDArray[np.float64], yb: NDArray[np.float64]) -> Any:
        centred = xb - xb.mean(axis=0)
        return _pls(centred - centred.mean(axis=0), yb - yb.mean(), A_PLS)

    # The final model is fitted on the store's float32 centred matrix; the
    # bootstrap refits its chain in float64 (pls-regression.md §16).
    centred = _f32(x - x.mean(axis=0))
    final = _pls(centred - centred.mean(axis=0), y - y.mean(), A_PLS)
    b = np.asarray(final.coef_).ravel()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vips, coefficients = [], []
    for _ in range(N_RESAMPLES):
        resample = rng.integers(0, len(y), len(y))
        model = fit(x[resample], y[resample])
        vips.append(_vip(model))
        coefficients.append(np.asarray(model.coef_).ravel())
    tail = (1 - LEVEL) / 2
    vip_bounds = np.quantile(np.asarray(vips), [tail, 1 - tail], axis=0)
    b_bounds = np.quantile(np.asarray(coefficients), [tail, 1 - tail], axis=0)
    return {
        "coefficients": b,
        "intercept": np.asarray(y.mean() - x.mean(axis=0) @ b),
        "vip_lower": vip_bounds[0],
        "vip_upper": vip_bounds[1],
        "b_lower": b_bounds[0],
        "b_upper": b_bounds[1],
    }


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def compare_tecator(
    served: dict[str, Any], ref: dict[str, NDArray[np.float64]], tighten: float
) -> list[Check]:
    bands = served["bootstrap"]
    against = f"{SKLEARN} PLSRegression"
    pairs: list[tuple[str, str, Any, NDArray[np.float64]]] = [
        (
            "final model: coefficients",
            "coefficients",
            served["json"]["coefficients"],
            ref["coefficients"],
        ),
        ("final model: intercept", "coefficients", served["json"]["intercept"], ref["intercept"]),
        ("bootstrap: VIP lower", "coefficients", bands["vip"]["lower"], ref["vip_lower"]),
        ("bootstrap: VIP upper", "coefficients", bands["vip"]["upper"], ref["vip_upper"]),
        (
            "bootstrap: coefficient lower",
            "coefficients",
            bands["coefficients"]["lower"],
            ref["b_lower"],
        ),
        (
            "bootstrap: coefficient upper",
            "coefficients",
            bands["coefficients"]["upper"],
            ref["b_upper"],
        ),
    ]
    return [
        Check(
            2,
            quantity,
            tolerance,
            np.asarray(ours, dtype=np.float64),
            np.asarray(theirs, dtype=np.float64),
            against + (" on every sample" if quantity.startswith("final") else ", same resamples"),
            tighten,
        )
        for quantity, tolerance, ours, theirs in pairs
    ]


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------


def record(
    meat: dict[str, Any],
    tecator: dict[str, Any],
    checks: list[Check],
    started: datetime,
    tighten: float,
) -> tuple[str, bool]:
    first_met = all(check.passed for check in checks if check.claim == 1)
    second_met = all(check.passed for check in checks if check.claim == 2)
    met = first_met and second_met
    env = meat["experiment"]["environment"]
    plsda, svm, permutation, nested = (
        meat["plsda"],
        meat["svm"],
        meat["permutation"],
        meat["nested"],
    )
    classes = plsda["classification"]["classes"]
    verdict = (
        "**Met.** Both claims hold at the stated tolerances."
        if met
        else "**Not met.** "
        + ("Claim 1 fails. " if not first_met else "")
        + ("Claim 2 fails. " if not second_met else "")
        + "The failing rows are marked below."
    )
    lines = [
        "# Phase 6 exit run",
        "",
        "Generated by `uv run python -m tests.exit_run_phase6"
        + (f" --tighten {tighten:g}" if tighten != 1.0 else "")
        + f"` on {started:%Y-%m-%d} at {started:%H:%M} UTC. **Do not edit by hand**: rerun "
        "the script.",
        "",
        "The criterion (`feature_list.json`, #341): on the raw Quadram meat set, 120 spectra "
        "with both runs of each sample kept and grouped by sample, grouped cross-validation keeps "
        "every pair together; PLS-DA's grouped cross-validated accuracy, its permutation p-value "
        "and its all-sample final model each match their references; a VIP-selected model's "
        "nested outer error matches a reference rebuild; SVM's cross-validated accuracy matches "
        "scikit-learn `SVC`; the final model, exported as JSON and as the Python snippet, "
        "reproduces the application's predictions. On Tecator, PLS's final model and its "
        "bootstrap intervals match a reference. Signed packages are not part of it.",
        "",
        f"## Verdict: {verdict}",
        "",
        "## What was run",
        "",
        f"- **Application:** chemometrics-workbench {env['app_version']}, Python "
        f"{env['python_version']}, numpy {env['packages']['numpy']}, scipy "
        f"{env['packages']['scipy']}, on {env['platform']}.",
        f"- **Reference:** {SKLEARN}, SciPy {scipy.__version__}, NumPy {np.__version__}, on "
        "the experiment's own resolved folds, rounded to float32 where the store rounds. "
        "`PLSRegression` runs with `tol=1e-14`.",
        "- **Data:** `docs/examples/meat-raw.csv`, the Quadram Institute's fresh-meat FTIR set "
        "(CC0-1.0; Al-Jowder, Kemsley and Wilson, *Food Chemistry* 59, 1997) with both runs kept "
        "(`docs/examples/meat-source.md`); `docs/examples/tecator.csv` with its permission note.",
        "",
        "## Claim 1: the raw meat set, grouped by sample",
        "",
        f"{meat['version']['n_samples']} spectra × {meat['version']['n_variables']} "
        f"wavenumbers, classes {', '.join(classes)}. A {N_SPLITS}-fold cross-validation "
        f"grouped by `{GROUP_COLUMN}`, seed {SEED}; mean centring; PLS-DA with {A_PLSDA} latent "
        f"variables and an RBF SVM (C = 1, gamma 'scale') on {A_SVM} principal components, both "
        f"on `{CLASS_COLUMN}`.",
        "",
        "| Quantity | Served |",
        "| --- | --- |",
        f"| PLS-DA accuracy (CV) | {plsda['metrics']['accuracy_cv']:.4f} |",
        f"| SVM accuracy (CV) | {svm['metrics']['accuracy_cv']:.4f} |",
        f"| Permutation p-value ({permutation['n_permutations']} permutations, seed "
        f"{permutation['seed']}) | {permutation['p_value']:.4f} |",
        f"| Nested RMSECV, VIP ≥ {VIP_CUT:g} | {nested['outer_rmsecv']:.4f} |",
        f"| RMSECV of that selection made on every sample | {nested['inner_rmsecv']:.4f} |",
        f"| Final model fitted on | {plsda['n_samples']} spectra |",
        "",
        "PLS-DA's cross-validated confusion (rows observed, columns assigned):",
        "",
        *_matrix(classes, plsda["classification"]["confusion"]["cross_validation"], classes),
        "",
        "Every served quantity against its independent reference. A table of counts must be "
        "equal; a continuous quantity must agree within the parity tolerance named.",
        "",
        *_table([check for check in checks if check.claim == 1]),
        "",
        "## Claim 2: Tecator's final model and its bootstrap intervals",
        "",
        f"Tecator ({tecator['version']['n_samples']} × {tecator['version']['n_variables']}, "
        f"`{TARGET}`): {N_SPLITS}-fold cross-validation, seed {SEED}; mean centring; PLS with "
        f"{A_PLS} components. The JSON model's coefficients are the final model's, on the raw "
        f"axis. Bootstrap: {N_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, {LEVEL:.0%} "
        "percentile intervals, each resample refitting the centring and the PLS.",
        "",
        *_table([check for check in checks if check.claim == 2]),
        "",
    ]
    return "\n".join(lines), met


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tighten", type=float, default=1.0)
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    with tempfile.TemporaryDirectory() as directory:
        with Served(Path(directory) / "meat") as client:
            meat = drive_meat(client)
        with Served(Path(directory) / "tecator") as client:
            tecator = drive_tecator(client)
    checks = compare_meat(meat, reference_meat(meat), args.tighten)
    checks.extend(compare_tecator(tecator, reference_tecator(), args.tighten))
    text, met = record(meat, tecator, checks, started, args.tighten)
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
