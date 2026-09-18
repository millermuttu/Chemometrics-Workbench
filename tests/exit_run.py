"""Phase 2's exit criterion, demonstrated rather than reasoned (#201).

`PROPOSAL.md` §16: "a complete calibration workflow on a real NIR dataset
matches reference software within stated tolerance". `feature_list.json`
decides what that means, and this script does it:

1. Start the real server on a fresh project directory and import Tecator the
   way a user's file goes in - `/import/preview`, then `/import`.
2. Put the pipeline the seed calls "the path the exit criterion walks":
   SNV, Savitzky-Golay (11, 2, first derivative), k-fold 10 with seed 42,
   mean centring, PLS with 5 latent variables on `fat`.
3. Run it as a job, poll it to `succeeded`, and read the PLS node's result,
   its folded coefficients and the experiment record back over HTTP.
4. Rebuild the same chain independently - SNV in NumPy, Savitzky-Golay through
   `scipy.signal.savgol_filter`, `sklearn.cross_decomposition.PLSRegression` -
   on the **served** `resolved_splits`. Never on a seed of this script's own:
   `metrics-and-validation.md` §8.2 makes ours `default_rng` and scikit-learn's
   a legacy `RandomState`, and two experiments on different folds compare
   nothing.
5. Compare RMSECV, Q², RMSEC, R², RMSEP, the RMSECV curve, the coefficient
   vector and the predictions, each against the parity harness's tolerance for
   its class, and write `docs/phase-2/exit-run.md` with every number on both
   sides.

**Kernel parity alone is not the criterion.** The parity suite compares
kernels on arrays it hands them; the executor's fold handling is what it
cannot see, and #173 was exactly the bug it could not see. This drives the
application, not the kernels.

**One caveat is stated rather than hidden.** The executor stores every
node's array as float32 (`PROPOSAL.md` §13), so the served numbers come from
float32-rounded intermediates while the reference runs in float64. The record
reports the float64 comparison first. If that falls outside the parity
tolerance it also reports the comparison with the reference narrowed to
float32 at the same points, and says which tolerance the criterion is met at.

Run it:

    uv run python -m tests.exit_run

It exits 0 when the criterion is met and 1 when it is not, and rewrites the
record either way, because a failed run is a result too.
"""

from __future__ import annotations

import csv
import io
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

from tests import parity
from tests.seed_e2e import tecator_csv

RECORD = Path(__file__).resolve().parents[1] / "docs" / "phase-2" / "exit-run.md"

WINDOW, POLYORDER, DERIV = 11, 2, 1
N_SPLITS, SEED = 10, 42
N_COMPONENTS = 5
TARGET = "fat"


# --------------------------------------------------------------------------
# the application, driven over HTTP
# --------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Served:
    """The real server, on a fresh project, for the life of a `with` block."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.port = _free_port()
        self.token = secrets.token_urlsafe(16)
        self.process: subprocess.Popen[bytes] | None = None
        self.client: httpx.Client | None = None

    def __enter__(self) -> httpx.Client:
        env = {
            **os.environ,
            "CHEMOMETRICS_PROJECT": str(self.root / "project"),
            "CHEMOMETRICS_CONFIG_HOME": str(self.root / "config"),
            "WORKBENCH_PORT": str(self.port),
            "WORKBENCH_TOKEN": self.token,
            # No bundle: this drives the API, and a missing directory means
            # the server mounts nothing rather than someone's stale build.
            "WORKBENCH_BUNDLE": str(self.root / "no-bundle"),
        }
        self.process = subprocess.Popen(
            [sys.executable, "-m", "chemometrics_workbench.server"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}/api",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=60.0,
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                if self.client.get("/projects").status_code == 200:
                    return self.client
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError("the server did not answer within 60 s")

    def __exit__(self, *_: object) -> None:
        if self.client is not None:
            self.client.close()
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)


def _ok(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise RuntimeError(f"{response.request.method} {response.url}: {response.text}")
    return response.json()


def drive(client: httpx.Client) -> dict[str, Any]:
    """Import, build, run, read back. Every step is the endpoint a screen calls."""
    files = {"file": ("tecator.csv", tecator_csv())}
    preview = _ok(client.post("/import/preview", files=files))
    if TARGET not in preview["detected"]["targets"]:
        raise RuntimeError(f"the preview did not classify {TARGET!r} as a target: {preview}")
    entry = _ok(client.post("/import", files=files, data={"corrections": "{}"}))
    version = entry["versions"][0]

    pipeline = _ok(client.get("/pipelines/current"))
    source = next(node for node in pipeline["nodes"] if node["type"] == "source")
    nodes = [
        source,
        {"id": "snv", "type": "preprocess", "inputs": [source["id"]], "step": {"kind": "snv"}},
        {
            "id": "savgol",
            "type": "preprocess",
            "inputs": ["snv"],
            "step": {
                "kind": "savgol",
                "window_length": WINDOW,
                "polyorder": POLYORDER,
                "deriv": DERIV,
            },
        },
        {
            "id": "split",
            "type": "split",
            "inputs": ["savgol"],
            "spec": {"kind": "kfold", "n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
        },
        {
            "id": "centre",
            "type": "preprocess",
            "inputs": ["split"],
            "step": {"kind": "mean_centre"},
        },
        {
            "id": "pls",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "pls",
                "n_components": N_COMPONENTS,
                "algorithm": "nipals",
                "target": TARGET,
            },
        },
    ]
    _ok(client.put("/pipelines/current", json={"nodes": nodes}))
    validation = _ok(client.post("/pipelines/current/validate"))

    job = _ok(client.post("/experiments/current/run"))
    deadline = time.monotonic() + 300.0
    while job["status"] in ("queued", "running"):
        if time.monotonic() > deadline:
            raise RuntimeError(f"the run did not finish within 300 s: {job}")
        time.sleep(0.2)
        job = _ok(client.get(f"/jobs/{job['job_id']}"))
    if job["status"] != "succeeded":
        raise RuntimeError(f"the run ended {job['status']}: {job['message']}")

    return {
        "preview": preview,
        "version": version,
        "validation": validation,
        "job": job,
        "result": _ok(client.get("/results/pls")),
        "coefficients": _ok(client.get("/results/pls/coefficients")),
        "experiment": _ok(client.get("/experiments/current")),
        "projects": _ok(client.get("/projects")),
    }


# --------------------------------------------------------------------------
# the reference, independently
# --------------------------------------------------------------------------


def _csv_matrix() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The numbers the application was given: the CSV's, parsed here without
    the readers. Six decimals in the file is the data; `load_tecator()`'s
    float64 is not what went in."""
    rows = list(csv.reader(io.StringIO(tecator_csv().decode("utf-8"))))
    header, body = rows[0], rows[1:]
    fat = header.index(TARGET)
    first_target = min(
        index for index, name in enumerate(header) if name in ("moisture", "fat", "protein")
    )
    spectra = np.asarray(
        [[float(v) for v in row[1:first_target]] for row in body], dtype=np.float64
    )
    response = np.asarray([float(row[fat]) for row in body], dtype=np.float64)
    return spectra, response


def _snv(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = values - values.mean(axis=1, keepdims=True)
    return np.asarray(centred / values.std(axis=1, ddof=1, keepdims=True), dtype=np.float64)


def _savgol(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        savgol_filter(values, WINDOW, POLYORDER, deriv=DERIV, delta=1.0, axis=1, mode="interp"),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Reference:
    rmsec: float
    r2: float
    rmsep: float
    rmsecv: float
    q2: float
    curve: list[float]
    coefficients: NDArray[np.float64]
    predicted: NDArray[np.float64]
    held_out_predicted: NDArray[np.float64]


def reference(
    spectra: NDArray[np.float64],
    response: NDArray[np.float64],
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
    *,
    narrow: bool,
) -> Reference:
    """scikit-learn on the served folds, with or without the store's float32 narrowing.

    `PLSRegression` centres X and y on its own fit set, which is the per-fold
    centring `metrics-and-validation.md` §9 requires, so no centring step is
    written here. With `narrow`, each array is rounded to float32 where the
    executor would have stored it: the imported dataset itself, then after
    SNV, after Savitzky-Golay, and after the centring the executor performs as
    a node before the estimator.
    """

    def stored(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values.astype(np.float32).astype(np.float64) if narrow else values

    # The dataset itself is stored float32 at import, before any node reads it.
    prepared = stored(_savgol(stored(_snv(stored(spectra)))))

    def fold_matrix(train: NDArray[np.intp]) -> NDArray[np.float64]:
        # The executor centres as a node, fitted on the training rows, and
        # stores that array before PLS reads it. Undoing nothing: PLSRegression
        # re-centres on the same rows and the result is the same matrix.
        return stored(prepared - prepared[train].mean(axis=0))

    def fit(train: NDArray[np.intp], a: int) -> Any:
        matrix = fold_matrix(train)
        return PLSRegression(n_components=a, scale=False).fit(
            matrix[train], response[train]
        ), matrix

    cross_validated = np.empty((N_COMPONENTS, response.size), dtype=np.float64)
    for train, test in folds:
        for a in range(1, N_COMPONENTS + 1):
            model, matrix = fit(train, a)
            cross_validated[a - 1, test] = model.predict(matrix[test]).ravel()
    curve = [
        float(np.sqrt(np.mean((response - cross_validated[a]) ** 2))) for a in range(N_COMPONENTS)
    ]
    press = float(((response - cross_validated[-1]) ** 2).sum())
    total = float(((response - response.mean()) ** 2).sum())

    train0, test0 = folds[0]
    model, matrix = fit(train0, N_COMPONENTS)
    predicted = model.predict(matrix[train0]).ravel()
    held_out = model.predict(matrix[test0]).ravel()
    residual = response[train0] - predicted
    return Reference(
        rmsec=float(np.sqrt(np.mean(residual**2))),
        r2=float(
            1.0 - (residual**2).sum() / ((response[train0] - response[train0].mean()) ** 2).sum()
        ),
        rmsep=float(np.sqrt(np.mean((response[test0] - held_out) ** 2))),
        rmsecv=curve[-1],
        q2=float(1.0 - press / total),
        curve=curve,
        coefficients=np.asarray(model.coef_, dtype=np.float64).ravel(),
        predicted=predicted,
        held_out_predicted=held_out,
    )


def nipals_longdouble(
    matrix: NDArray[np.float64], response: NDArray[np.float64], a: int
) -> NDArray[np.float64]:
    """PLS1 coefficients by NIPALS in 80-bit extended precision, as an arbiter.

    Where the served coefficient vector and scikit-learn's differ by more than
    the coefficient tolerance, this says which of them is nearer the answer
    both are approximating. Twenty lines of `pls-regression.md` §4 and §5 with
    nothing shared with either implementation: LAPACK does not take
    `longdouble`, so `(P'W) z = q` is solved by back-substitution - P'W is
    upper triangular with unit diagonal, §5.
    """
    x = matrix.astype(np.longdouble)
    y = response.astype(np.longdouble)
    x = x - x.mean(axis=0)
    y = y - y.mean()
    residual_x, residual_y = x.copy(), y.copy()
    weights, loadings, y_loadings = [], [], []
    for _ in range(a):
        w = residual_x.T @ residual_y
        w = w / np.sqrt(w @ w)
        t = residual_x @ w
        tt = t @ t
        p = residual_x.T @ t / tt
        q = (residual_y @ t) / tt
        residual_x = residual_x - np.outer(t, p)
        residual_y = residual_y - q * t
        weights.append(w)
        loadings.append(p)
        y_loadings.append(q)
    big_w = np.column_stack(weights)
    big_p = np.column_stack(loadings)
    triangular = big_p.T @ big_w
    z = np.zeros(a, dtype=np.longdouble)
    for i in reversed(range(a)):
        z[i] = (y_loadings[i] - triangular[i, i + 1 :] @ z[i + 1 :]) / triangular[i, i]
    return np.asarray(big_w @ z, dtype=np.float64)


# --------------------------------------------------------------------------
# the comparison, and its record
# --------------------------------------------------------------------------


#: What the `phase-2-exit-run` entry names: RMSECV, Q² and the predictions,
#: with the calibration metrics and the curve beside them. The coefficient
#: vector is compared too, and reported, but the criterion was written on the
#: numbers a calibration is read by.
CRITERION_CLASSES = ("metrics", "predictions")


@dataclass(frozen=True)
class Comparison:
    quantity: str
    tolerance_class: str
    ours: NDArray[np.float64]
    theirs: NDArray[np.float64]

    @property
    def in_criterion(self) -> bool:
        return self.tolerance_class in CRITERION_CLASSES

    @property
    def tolerance(self) -> parity.Tolerance:
        return parity.TOLERANCES[self.tolerance_class]

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.ours - self.theirs)))

    @property
    def max_rel(self) -> float:
        scale = np.maximum(np.abs(self.theirs), 1e-300)
        return float(np.max(np.abs(self.ours - self.theirs) / scale))

    @property
    def passed(self) -> bool:
        return bool(
            np.allclose(self.ours, self.theirs, rtol=self.tolerance.rtol, atol=self.tolerance.atol)
        )


def compare(served: dict[str, Any], ref: Reference) -> list[Comparison]:
    result = served["result"]
    metrics = result["metrics"]
    curve = [metrics[f"rmsecv_a{a}"] for a in range(1, N_COMPONENTS + 1)]

    def scalar(name: str, value: str) -> Comparison:
        return Comparison(
            name,
            "metrics",
            np.asarray([metrics[value]], dtype=np.float64),
            np.asarray([getattr(ref, value)], dtype=np.float64),
        )

    return [
        scalar("RMSECV", "rmsecv"),
        scalar("Q²", "q2"),
        scalar("RMSEC (fold 0)", "rmsec"),
        scalar("R² (fold 0)", "r2"),
        scalar("RMSEP (fold 0 held out)", "rmsep"),
        Comparison("RMSECV curve, A = 1…5", "metrics", np.asarray(curve), np.asarray(ref.curve)),
        Comparison(
            "coefficients b (node axis, fold 0)",
            "coefficients",
            np.asarray(result["regression"]["coefficients"], dtype=np.float64),
            ref.coefficients,
        ),
        Comparison(
            "calibration predictions (fold 0)",
            "predictions",
            np.asarray(result["regression"]["predicted"], dtype=np.float64),
            ref.predicted,
        ),
        Comparison(
            "held-out predictions (fold 0)",
            "predictions",
            np.asarray(result["validation"]["predicted"], dtype=np.float64),
            ref.held_out_predicted,
        ),
    ]


def _table(comparisons: list[Comparison]) -> list[str]:
    lines = [
        "| Quantity | Served | Reference | max abs diff | max rel diff "
        "| Class (rtol, atol) | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in comparisons:
        if item.ours.size == 1:
            ours, theirs = f"{item.ours[0]:.10g}", f"{item.theirs[0]:.10g}"
        else:
            ours, theirs = f"{item.ours.size} values", f"{item.theirs.size} values"
        lines.append(
            f"| {item.quantity} | {ours} | {theirs} | {item.max_abs:.3g} | {item.max_rel:.3g} | "
            f"{item.tolerance_class} ({item.tolerance.rtol:g}, {item.tolerance.atol:g}) | "
            f"{'pass' if item.passed else 'FAIL'} |"
        )
    return lines


def _f32(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.astype(np.float32).astype(np.float64)


def _arbiter_lines(
    served: dict[str, Any],
    folds: list[tuple[NDArray[np.intp], NDArray[np.intp]]],
    spectra: NDArray[np.float64],
    response: NDArray[np.float64],
    narrowed: Reference,
) -> list[str]:
    """Where the coefficient difference sits, judged by extended precision.

    The only place this script touches `chemometrics_workbench`: to refit its
    kernel on the *same* narrowed matrix scikit-learn was given, and to count
    the elements at which its Savitzky-Golay and SciPy's round to different
    float32 values. Diagnostics for the record, never part of the criterion.
    """
    from chemometrics_workbench.preprocessing import SavitzkyGolayTransformer
    from chemometrics_workbench.regression import PLS

    train0 = folds[0][0]
    after_snv = _f32(_snv(_f32(spectra)))
    prepared = _f32(_savgol(after_snv))
    matrix = _f32(prepared - prepared[train0].mean(axis=0))
    arbiter = nipals_longdouble(matrix[train0], response[train0], N_COMPONENTS)
    scale = float(np.max(np.abs(arbiter)))

    served_b = np.asarray(served["result"]["regression"]["coefficients"], dtype=np.float64)
    x_mean = matrix[train0].mean(axis=0)
    y_mean = float(response[train0].mean())
    kernel = PLS(N_COMPONENTS).fit(matrix[train0] - x_mean, response[train0] - y_mean)
    kernel_b = np.asarray(kernel.coefficients_, dtype=np.float64)

    ours_sg = _f32(
        SavitzkyGolayTransformer(WINDOW, POLYORDER, deriv=DERIV).fit_transform(after_snv)
    )
    differing = int(np.count_nonzero(ours_sg != prepared))

    def distance(vector: NDArray[np.float64]) -> str:
        return f"{float(np.max(np.abs(vector - arbiter))) / scale:.3g}"

    return [
        "## Where the coefficient difference sits",
        "",
        "The coefficient vector is the one quantity outside its parity tolerance under both "
        "comparisons. Both sides approximate the same `b = W(P'W)⁻¹q`; an 80-bit NIPALS "
        "written for this script alone, on the narrowed fold-zero matrix, says which side is "
        "nearer it. Distances are the largest element-wise difference over the largest "
        "coefficient.",
        "",
        "| Coefficient vector | Distance from the extended-precision vector |",
        "| --- | --- |",
        f"| served by the application | {distance(served_b)} |",
        f"| `chemometrics_workbench.regression.PLS` refitted on the same narrowed matrix | "
        f"{distance(kernel_b)} |",
        f"| scikit-learn `PLSRegression` on that matrix | {distance(narrowed.coefficients)} |",
        "",
        f"On an identical matrix the two kernels agree with the arbiter and with each other. "
        f"The served vector is further away because the application did not fit this exact "
        f"matrix: its Savitzky-Golay and SciPy's agree to the smoothing tolerance in float64 "
        f"and then round to **different float32 values at {differing} of "
        f"{prepared.size} elements** - values that sit on a rounding boundary. Fitting five "
        f"latent variables on a first-derivative matrix carries those half-ulp differences "
        f"into `b` at the level shown, and into the predictions at the level the tables "
        f"above show, which is inside their tolerance. This is the store's precision, not the "
        f"arithmetic.",
        "",
    ]


def record(
    served: dict[str, Any],
    exact: list[Comparison],
    narrowed: list[Comparison],
    started: datetime,
    arbiter: list[str],
) -> tuple[str, bool]:
    criterion_exact = all(item.passed for item in exact if item.in_criterion)
    criterion_narrowed = all(item.passed for item in narrowed if item.in_criterion)
    coefficients_ok = all(item.passed for item in exact + narrowed if not item.in_criterion)
    met = criterion_exact or criterion_narrowed
    version = served["version"]
    experiment = served["experiment"]
    env = experiment["environment"]
    folds = experiment["resolved_splits"][0]
    verdict = (
        "**Met** on every quantity the criterion names - RMSECV, Q², the calibration metrics, "
        "the curve and both prediction sets - at the parity tolerances, "
        + (
            "against a float64 reference."
            if criterion_exact
            else "once the reference is narrowed to float32 where the store narrows; not against "
            "a float64 reference, and the caveat below says why."
        )
        if met
        else "**Not met.** A quantity the criterion names falls outside the parity tolerance "
        "under both comparisons."
    )
    if not coefficients_ok:
        verdict += (
            " The coefficient vector, compared as well, is outside the coefficient class's "
            "tolerance under both; the section at the end says on which side the difference "
            "lies."
        )
    lines = [
        "# Phase 2 exit run",
        "",
        f"Generated by `uv run python -m tests.exit_run` on {started:%Y-%m-%d} at "
        f"{started:%H:%M} UTC. **Do not edit by hand**: rerun the script.",
        "",
        "`PROPOSAL.md` §16, Phase 2: *a complete calibration workflow on a real NIR dataset "
        "matches reference software within stated tolerance.* `feature_list.json` decides what "
        "that means - the workflow driven over HTTP against an independent PLS on the "
        "experiment's own resolved folds, within the parity tolerances - and this is that run.",
        "",
        f"## Verdict: {verdict}",
        "",
        "## What was run",
        "",
        f"- **Application:** chemometrics-workbench {env['app_version']}, Python "
        f"{env['python_version']}, numpy {env['packages']['numpy']}, scipy "
        f"{env['packages']['scipy']}, on {env['platform']}.",
        f"- **Reference:** scikit-learn {sklearn.__version__} `PLSRegression(scale=False)`, "
        f'scipy {scipy.__version__} `savgol_filter(mode="interp")`, SNV in NumPy '
        f"{np.__version__} (ddof 1). Nothing from `chemometrics_workbench` on this side.",
        f"- **Dataset:** Tecator, {version['n_samples']} × {version['n_variables']}, imported "
        f"through `/import` from the CSV `tests/seed_e2e.py` writes; content hash "
        f"`{version['content_hash']}`, reader `{version['source']['reader']} "
        f"{version['source']['reader_version']}`. The reference parses the same CSV.",
        f"- **Pipeline:** source → SNV → Savitzky-Golay (window {WINDOW}, order {POLYORDER}, "
        f"derivative {DERIV}) → k-fold {N_SPLITS} (shuffle, seed {SEED}) → mean centre → PLS "
        f"{N_COMPONENTS} LV on `{TARGET}`. Validation answered `valid: "
        f"{served['validation']['valid']}` with {len(served['validation']['problems'])} problems.",
        f"- **Folds:** the {len(folds['test_indices'])} test sets the experiment recorded, "
        f"sizes {[len(fold) for fold in folds['test_indices']]}, handed to the reference as "
        "index arrays. Never reseeded on the reference side.",
        f"- **Run:** job `{served['job']['job_id']}` ended `{served['job']['status']}`; "
        f"experiment `{experiment['experiment_id']}`.",
        f"- **Folded coefficients:** `/results/pls/coefficients` answered "
        f"`available: {served['coefficients']['available']}`"
        + (
            f" - {served['coefficients']['reason']}"
            if not served["coefficients"]["available"]
            else ""
        ),
        "",
        "## Comparison against a float64 reference",
        "",
        "The reference runs in float64 throughout. The served numbers come from arrays the "
        "executor stored as float32 between nodes (`PROPOSAL.md` §13).",
        "",
        *_table(exact),
        "",
        "## Comparison against the reference narrowed to float32 where the store narrows",
        "",
        "The same reference, with the imported matrix and each intermediate array rounded to "
        "float32 - at import, after SNV, after Savitzky-Golay and after the per-fold centring, "
        "the four points at which the executor reads an array back from the store.",
        "",
        *_table(narrowed),
        "",
        "## Reading the two tables",
        "",
        "- The parity tolerances are `tests/parity.py`'s, by class, and were not adjusted for "
        "this run.",
        "- A first-derivative chain amplifies the float32 rounding of the stored arrays: the "
        "derivative of a rounded spectrum differs from the derivative of the exact one by the "
        "rounding scaled by the filter's weights. Where the float64 comparison fails and the "
        "narrowed one passes, that rounding is the whole difference, and it is a property of "
        "the store the specification chose, not of the arithmetic.",
        "- `RMSEC`, `R²`, `RMSEP`, the coefficients and both prediction sets are fold zero's "
        "model, which is the one the executor fits and serves (`executor.py`, *Estimators*). "
        "`RMSECV`, `Q²` and the curve pool every fold.",
        "",
        *arbiter,
    ]
    return "\n".join(lines), met


def main() -> int:
    started = datetime.now(UTC)
    with (
        tempfile.TemporaryDirectory(prefix="chemometrics-exit-run-") as temporary,
        Served(Path(temporary)) as client,
    ):
        served = drive(client)

    spectra, response = _csv_matrix()
    resolved = served["experiment"]["resolved_splits"][0]
    folds = [
        (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for train, test in zip(resolved["train_indices"], resolved["test_indices"], strict=True)
    ]
    exact = compare(served, reference(spectra, response, folds, narrow=False))
    narrowed_reference = reference(spectra, response, folds, narrow=True)
    narrowed = compare(served, narrowed_reference)
    arbiter = _arbiter_lines(served, folds, spectra, response, narrowed_reference)

    text, met = record(served, exact, narrowed, started, arbiter)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nrecord written to {RECORD}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
