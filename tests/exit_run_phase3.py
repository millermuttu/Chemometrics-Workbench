"""Phase 3's exit criterion, demonstrated rather than reasoned (#227).

`PROPOSAL.md` §16, Phase 3: "two models differing only in preprocessing can be
compared step by step; an exported model reproduces application predictions
within tolerance in a clean environment". This script does both halves against
the served application, the way `tests/exit_run.py` did Phase 2's:

1. Start the real server on a fresh project directory and import Tecator
   through `/import/preview` and `/import`.
2. **Export.** Put SNV, Savitzky-Golay (11, 2, first derivative), a 25 % train
   and test split, mean centring and PLS with 5 latent variables on `fat`, and
   run it. Fetch `/results/pls/export.py` and run it in a virtual environment
   made for this run with **nothing but NumPy installed**, on the raw spectra
   of the CSV that was imported. Compare its predictions with the ones the
   application served, for the calibration rows and for the held-out rows, at
   `docs/model-export.md` §5's tolerance.
3. **Comparison.** Change the Savitzky-Golay window to 15, run again, and
   read both experiments back from the history. Their pipeline snapshots are
   diffed by the rule `frontend/src/lineage/diff.ts` applies - nodes matched
   by id, a changed node named with the parameters that differ - and the
   comparison has to name that one node and that one parameter. The view that
   draws it is exercised by `runs.spec.ts`'s "two runs compare step by step,
   and the differing node is named"; this is the same claim, on the record.
4. Write `docs/phase-3/exit-run.md` with every number and its tolerance.

**Why the chain is SNV then a first derivative.** SNV is re-executed by the
snippet (`model-export.md` §1), so the residual chain is not empty, and a first
derivative is the worst case for the float32 store §5's tolerance is set
against. An easy chain would demonstrate less.

**The clean environment is `uv venv` plus `uv pip install numpy`**, because uv
is how this project makes environments already, and it installs from its cache
when it has one. The record lists every distribution the environment held.

Run it:

    uv run python -m tests.exit_run_phase3

It exits 0 when both halves are met and 1 when either is not, and rewrites the
record either way, because a failed run is a result too.
"""

from __future__ import annotations

import json
import shutil
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
from numpy.typing import NDArray

from tests.exit_run import Served, _csv_matrix, _ok
from tests.seed_e2e import tecator_csv

RECORD = Path(__file__).resolve().parents[1] / "docs" / "phase-3" / "exit-run.md"

WINDOW, CHANGED_WINDOW, POLYORDER, DERIV = 11, 15, 2, 1
TEST_SIZE, SEED = 0.25, 42
N_COMPONENTS = 5
TARGET = "fat"

#: `docs/model-export.md` §5. Not adjusted for this run.
RTOL, ATOL = 1e-4, 1e-6


# --------------------------------------------------------------------------
# the application, driven over HTTP
# --------------------------------------------------------------------------


def _nodes(source: dict[str, Any], window: int) -> list[dict[str, Any]]:
    return [
        source,
        {"id": "snv", "type": "preprocess", "inputs": [source["id"]], "step": {"kind": "snv"}},
        {
            "id": "savgol",
            "type": "preprocess",
            "inputs": ["snv"],
            "step": {
                "kind": "savgol",
                "window_length": window,
                "polyorder": POLYORDER,
                "deriv": DERIV,
            },
        },
        {
            "id": "split",
            "type": "split",
            "inputs": ["savgol"],
            "spec": {"kind": "train_test", "test_size": TEST_SIZE, "seed": SEED},
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


def _run(client: httpx.Client, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Put the pipeline, run it, and read back what a screen would."""
    _ok(client.put("/pipelines/current", json={"nodes": nodes}))
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
        "job": job,
        "result": _ok(client.get("/results/pls")),
        "experiment": _ok(client.get("/experiments/current")),
    }


def drive(client: httpx.Client) -> dict[str, Any]:
    files = {"file": ("tecator.csv", tecator_csv())}
    _ok(client.post("/import/preview", files=files))
    version = _ok(client.post("/import", files=files, data={"corrections": "{}"}))["versions"][0]
    source = next(
        n for n in _ok(client.get("/pipelines/current"))["nodes"] if n["type"] == "source"
    )

    first = _run(client, _nodes(source, WINDOW))
    # Read before the second run replaces what `/results/pls` answers with.
    first["model"] = _ok(client.get("/results/pls/export.json"))
    snippet = client.get("/results/pls/export.py")
    snippet.raise_for_status()
    first["snippet"] = snippet.text

    second = _run(client, _nodes(source, CHANGED_WINDOW))
    history = _ok(client.get("/experiments"))
    return {
        "version": version,
        "first": first,
        "second": second,
        "left": _ok(client.get(f"/experiments/{first['experiment']['experiment_id']}")),
        "right": _ok(client.get(f"/experiments/{second['experiment']['experiment_id']}")),
        "history": history,
    }


# --------------------------------------------------------------------------
# the export half: the snippet, somewhere with nothing but NumPy
# --------------------------------------------------------------------------

DRIVER = """
import importlib.metadata, importlib.util, json, sys
import numpy as np

sys.path.insert(0, sys.argv[1])
import exported

print(json.dumps({
    "predictions": [float(v) for v in exported.predict(np.load(sys.argv[2]))],
    "python": sys.version.split()[0],
    "distributions": sorted(
        f"{d.metadata['Name']} {d.version}" for d in importlib.metadata.distributions()
    ),
    "application_importable": importlib.util.find_spec("chemometrics_workbench") is not None,
    "modules": sorted(
        n for n in sys.modules
        if n.split(".")[0] in ("chemometrics_workbench", "scipy", "sklearn")
    ),
}))
"""


def predict_in_clean_environment(snippet: str, spectra: NDArray[np.float64]) -> dict[str, Any]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH; it is what makes the clean environment")
    with tempfile.TemporaryDirectory(prefix="chemometrics-clean-") as temporary:
        root = Path(temporary)
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run([uv, "venv", "--quiet", "--python", python, str(root / "venv")], check=True)
        interpreter = root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
        interpreter = interpreter / ("python.exe" if sys.platform == "win32" else "python")
        subprocess.run(
            [uv, "pip", "install", "--quiet", "--python", str(interpreter), "numpy"], check=True
        )
        (root / "exported.py").write_text(snippet, encoding="utf-8")
        (root / "run_it.py").write_text(DRIVER, encoding="utf-8")
        np.save(root / "spectra.npy", spectra)
        # -I: no user site, no PYTHONPATH, no current directory. What it can
        # import is what the environment holds.
        finished = subprocess.run(
            [str(interpreter), "-I", str(root / "run_it.py"), str(root), str(root / "spectra.npy")],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )
        if finished.returncode != 0:
            raise RuntimeError(f"the snippet failed in the clean environment:\n{finished.stderr}")
        read: dict[str, Any] = json.loads(finished.stdout)
        return read


@dataclass(frozen=True)
class Agreement:
    rows: str
    indices: NDArray[np.intp]
    served: NDArray[np.float64]
    exported: NDArray[np.float64]

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.exported - self.served)))

    @property
    def max_rel(self) -> float:
        return float(np.max(np.abs(self.exported - self.served) / np.abs(self.served)))

    @property
    def passed(self) -> bool:
        return bool(np.allclose(self.exported, self.served, rtol=RTOL, atol=ATOL))


def agreements(result: dict[str, Any], predictions: NDArray[np.float64]) -> list[Agreement]:
    def one(rows: str, block: dict[str, Any], samples: list[dict[str, Any]]) -> Agreement:
        indices = np.asarray([s["index"] for s in samples], dtype=np.intp)
        return Agreement(
            rows, indices, np.asarray(block["predicted"], dtype=np.float64), predictions[indices]
        )

    return [
        one("calibration rows", result["regression"], result["samples"]),
        one("held-out rows", result["validation"], result["validation"]["samples"]),
    ]


# --------------------------------------------------------------------------
# the comparison half: `diff.ts`'s rule, on the two records
# --------------------------------------------------------------------------


def _parameters(node: dict[str, Any]) -> dict[str, Any]:
    return node.get("step") or node.get("spec") or {}


def diff(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`diffPipelines` in `frontend/src/lineage/diff.ts`, written again here so
    the record does not depend on a browser. Matched by id; a changed node
    carries the names of what differs."""
    before = {node["id"]: node for node in left}
    after = {node["id"]: node for node in right}
    order = [n["id"] for n in left] + [n["id"] for n in right if n["id"] not in before]
    out = []
    for node_id in order:
        if node_id not in after:
            out.append({"id": node_id, "change": "removed", "fields": []})
        elif node_id not in before:
            out.append({"id": node_id, "change": "added", "fields": []})
        else:
            a, b = before[node_id], after[node_id]
            fields = [f for f in ("type", "inputs") if a[f] != b[f]]
            pa, pb = _parameters(a), _parameters(b)
            fields += [k for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)]
            out.append(
                {"id": node_id, "change": "changed" if fields else "unchanged", "fields": fields}
            )
    return out


# --------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------

METRICS = ("rmsec", "r2", "rmsep", "bias")


def record(
    served: dict[str, Any],
    clean: dict[str, Any],
    agreed: list[Agreement],
    nodes: list[dict[str, Any]],
    started: datetime,
) -> tuple[str, bool]:
    version, first = served["version"], served["first"]
    left, right = served["left"], served["right"]
    model = first["model"]
    env = left["environment"]

    differing = [n for n in nodes if n["change"] != "unchanged"]
    comparison_met = (
        len(differing) == 1
        and differing[0]["id"] == "savgol"
        and differing[0]["change"] == "changed"
        and differing[0]["fields"] == ["window_length"]
    )
    isolated = not clean["application_importable"] and clean["modules"] == []
    export_met = isolated and all(item.passed for item in agreed)
    met = export_met and comparison_met

    def verdict(ok: bool) -> str:
        return "**met**" if ok else "**not met**"

    residual = ", ".join(step["kind"] for step in model["preprocessing"]) or "none"
    lines = [
        "# Phase 3 exit run",
        "",
        f"Generated by `uv run python -m tests.exit_run_phase3` on {started:%Y-%m-%d} at "
        f"{started:%H:%M} UTC. **Do not edit by hand**: rerun the script.",
        "",
        "`PROPOSAL.md` §16, Phase 3: *two models differing only in preprocessing can be compared "
        "step by step; an exported model reproduces application predictions within tolerance in "
        "a clean environment.* Both halves are driven over HTTP against the served application.",
        "",
        f"## Verdict: {'**Met.**' if met else '**Not met.**'} Export {verdict(export_met)}; "
        f"comparison {verdict(comparison_met)}.",
        "",
        "## What was run",
        "",
        f"- **Application:** chemometrics-workbench {env['app_version']}, Python "
        f"{env['python_version']}, numpy {env['packages']['numpy']}, on {env['platform']}.",
        f"- **Dataset:** Tecator, {version['n_samples']} × {version['n_variables']}, imported "
        f"through `/import` from the CSV `tests/seed_e2e.py` writes; content hash "
        f"`{version['content_hash']}`.",
        f"- **Pipeline:** source → SNV → Savitzky-Golay (window {WINDOW}, order {POLYORDER}, "
        f"derivative {DERIV}) → train/test split (test size {TEST_SIZE}, seed {SEED}) → mean "
        f"centre → PLS {N_COMPONENTS} LV on `{TARGET}`. Experiment "
        f"`{left['experiment_id']}`, job `{first['job']['job_id']}`.",
        "",
        "## Export: the snippet in a clean environment",
        "",
        f"- **Exported:** `/results/pls/export.py`, {len(first['snippet'].splitlines())} lines. "
        f"Residual chain re-executed by the snippet: {residual}. Everything after it folds "
        f"into {len(model['coefficients'])} coefficients and an intercept of "
        f"{model['intercept']:.10g}.",
        f"- **Environment:** a virtual environment made for this run by `uv venv`, then "
        f"`uv pip install numpy`. Python {clean['python']}. It held exactly: "
        + ", ".join(f"`{d}`" for d in clean["distributions"])
        + ".",
        f"- **Isolation:** run with `python -I`. `chemometrics_workbench` importable: "
        f"{clean['application_importable']}. Modules of ours, SciPy or scikit-learn loaded while "
        f"predicting: {clean['modules'] or 'none'}.",
        f"- **Input:** the {version['n_samples']} raw spectra parsed from the imported CSV, "
        f"float64. The application's own predictions come from arrays its store narrowed to "
        f"float32 (`PROPOSAL.md` §13), which is why §5's tolerance is what it is.",
        "",
        f"Tolerance: `rtol = {RTOL:g}`, `atol = {ATOL:g}`, from `docs/model-export.md` §5.",
        "",
        "| Rows | n | max abs diff | max rel diff | Result |",
        "| --- | --- | --- | --- | --- |",
        *(
            f"| {a.rows} | {a.indices.size} | {a.max_abs:.3g} | {a.max_rel:.3g} | "
            f"{'pass' if a.passed else 'FAIL'} |"
            for a in agreed
        ),
        "",
        "The first five held-out rows, as a sample of the numbers compared:",
        "",
        "| Row | Served | Exported | rel diff |",
        "| --- | --- | --- | --- |",
        *(
            f"| {i} | {s:.8g} | {e:.8g} | {abs(e - s) / abs(s):.3g} |"
            for i, s, e in list(
                zip(agreed[1].indices, agreed[1].served, agreed[1].exported, strict=True)
            )[:5]
        ),
        "",
        "## Comparison: two runs, one node apart",
        "",
        f"The second run changed the Savitzky-Golay window from {WINDOW} to {CHANGED_WINDOW} and "
        f"nothing else. Both records were read back from `/experiments/{{id}}`; the history "
        f"held {len(served['history'])} experiments. The snapshots are diffed by "
        f"`frontend/src/lineage/diff.ts`'s rule, written again in the script.",
        "",
        "| Node | Change | Fields |",
        "| --- | --- | --- |",
        *(f"| `{n['id']}` | {n['change']} | {', '.join(n['fields']) or '-'} |" for n in nodes),
        "",
        f"| Metric | Left (window {WINDOW}) | Right (window {CHANGED_WINDOW}) |",
        "| --- | --- | --- |",
        *(
            f"| {name} | {left['metrics'][name]:.6g} | {right['metrics'][name]:.6g} |"
            for name in METRICS
            if left["metrics"].get(name) is not None and right["metrics"].get(name) is not None
        ),
        "",
        "The same comparison drawn by the application is exercised by `frontend/e2e/runs.spec.ts`,"
        ' "two runs compare step by step, and the differing node is named".',
        "",
    ]
    return "\n".join(lines), met


def main() -> int:
    started = datetime.now(UTC)
    with (
        tempfile.TemporaryDirectory(prefix="chemometrics-exit-run-") as temporary,
        Served(Path(temporary)) as client,
    ):
        served = drive(client)

    spectra, _ = _csv_matrix()
    clean = predict_in_clean_environment(served["first"]["snippet"], spectra)
    predictions = np.asarray(clean["predictions"], dtype=np.float64)
    agreed = agreements(served["first"]["result"], predictions)
    nodes = diff(
        served["left"]["pipeline_snapshot"]["nodes"], served["right"]["pipeline_snapshot"]["nodes"]
    )

    text, met = record(served, clean, agreed, nodes, started)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nrecord written to {RECORD}")
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
