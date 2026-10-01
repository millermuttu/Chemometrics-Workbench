"""The packaged application, launched and driven the way a user's first minute
drives it (#233).

    uv run python -m tests.smoke_package dist/ChemometricsWorkbench

1. Refuse a bundle that carries scikit-learn or chemotools: both are
   development dependencies (`PROPOSAL.md` §7), and a package holding either
   is a wrapper around what the parity report compares against.
2. Launch the executable on a fresh config directory, with no project set, so
   it opens `<config dir>/projects/default` as a double-click would, and read
   the launch URL it prints.
3. Load the page, import Tecator, run mean centring and a two-component PCA,
   and check 240 x 2 finite scores come back.
4. Print the unpacked size, which CI records beside the packed one.

It exits 0 when every step holds and 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from tests.seed_e2e import tecator_csv

FORBIDDEN = ("sklearn", "chemotools")


def executable(bundle: Path) -> Path:
    return bundle / ("ChemometricsWorkbench.exe" if os.name == "nt" else "ChemometricsWorkbench")


def forbidden(bundle: Path) -> list[Path]:
    return [path for path in bundle.rglob("*") if path.name in FORBIDDEN]


def size_mb(bundle: Path) -> float:
    return (
        sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file() and not p.is_symlink()) / 1e6
    )


def launch_url(process: subprocess.Popen[str], timeout: float = 120.0) -> str:
    """The URL the launcher prints once its socket is listening."""
    found: list[str] = []

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"  | {line.rstrip()}")
            if "Launch URL:" in line and not found:
                found.append(line.split("Launch URL:", 1)[1].strip())

    threading.Thread(target=read, daemon=True).start()
    deadline = time.monotonic() + timeout
    while not found:
        if process.poll() is not None:
            raise RuntimeError(f"the executable exited with {process.returncode}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"no launch URL within {timeout:.0f} s")
        time.sleep(0.1)
    return found[0]


def ok(response: httpx.Response) -> dict:  # type: ignore[type-arg]
    if response.status_code >= 400:
        raise RuntimeError(f"{response.request.method} {response.url}: {response.text}")
    return response.json()  # type: ignore[no-any-return]


def drive(url: str) -> list[list[float]]:
    base, token = url.split("/?token=")
    page = httpx.get(url, timeout=30)
    if page.status_code != 200 or "Chemometrics Workbench" not in page.text:
        raise RuntimeError(f"the page did not load: {page.status_code}")
    with httpx.Client(
        base_url=f"{base}/api", headers={"Authorization": f"Bearer {token}"}, timeout=120
    ) as client:
        ok(
            client.post(
                "/import",
                files={"file": ("tecator.csv", tecator_csv())},
                data={"corrections": "{}"},
            )
        )
        source = next(
            n for n in ok(client.get("/pipelines/current"))["nodes"] if n["type"] == "source"
        )
        nodes = [
            source,
            {
                "id": "centre",
                "type": "preprocess",
                "inputs": [source["id"]],
                "step": {"kind": "mean_centre"},
            },
            {
                "id": "pca",
                "type": "estimator",
                "inputs": ["centre"],
                "spec": {"kind": "pca", "n_components": 2},
            },
        ]
        ok(client.put("/pipelines/current", json={"nodes": nodes}))
        job = ok(client.post("/experiments/current/run"))
        deadline = time.monotonic() + 300
        while job["status"] in ("queued", "running"):
            if time.monotonic() > deadline:
                raise RuntimeError(f"the run did not finish: {job}")
            time.sleep(0.2)
            job = ok(client.get(f"/jobs/{job['job_id']}"))
        if job["status"] != "succeeded":
            raise RuntimeError(f"the run ended {job['status']}: {job.get('message')}")
        return ok(client.get("/results/pca"))["scores"]  # type: ignore[no-any-return]


def main(argv: list[str]) -> int:
    bundle = Path(argv[0]).resolve()
    bad = forbidden(bundle)
    if bad:
        print(f"FAIL: the bundle carries development dependencies: {bad}")
        return 1
    with tempfile.TemporaryDirectory() as root:
        env = {
            k: v for k, v in os.environ.items() if not k.startswith(("WORKBENCH_", "CHEMOMETRICS_"))
        }
        # A browser that is not there: the launcher still hands it the URL.
        env |= {"CHEMOMETRICS_CONFIG_HOME": root, "BROWSER": "none"}
        process = subprocess.Popen(
            [str(executable(bundle))],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            url = launch_url(process)
            scores = drive(url)
        finally:
            process.terminate()
            process.wait(timeout=30)
        shape = (len(scores), len(scores[0]) if scores else 0)
        finite = all(abs(v) < float("inf") for row in scores for v in row)
        if shape != (240, 2) or not finite:
            print(f"FAIL: scores came back {shape}, finite={finite}")
            return 1
        if not (Path(root) / "projects" / "default").is_dir():
            print("FAIL: the default project was not created")
            return 1
    print(f"OK: {url.split('/?')[0]} served a 240 x 2 PCA; unpacked {size_mb(bundle):.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
