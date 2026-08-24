"""The Phase 1.1 stub server: real endpoints, generated fixtures, real jobs.

Run it:

    uv run python stub/server.py

It prints the launch URL — `http://127.0.0.1:<port>/?token=<token>` — which is
what the desktop shell will hand the frontend in Phase 1.2. The token is
required on every `/api` request as `Authorization: Bearer <token>`.

## What this module is, and what it is not

It routes and it delays. It does not compute: every response body is a fixture
from `stub/fixtures`, generated from the real kernels by
`stub/generate_fixtures.py`, and returned unmodified. There is no database, no
persistence and no executor — job state is a dict that dies with the process.
Phase 1.2 replaces these handlers one at a time behind unchanged URLs; each
one says below what replaces it.

Phase 1.2's issues are not cut yet, so the handlers name the work rather than a
number. When those issues exist, the marker to search for is `Phase 1.2:`.

## Jobs advance because static fixtures cannot fail or take time

`POST /api/experiments/{id}/run` starts an in-memory job that walks the status
sequence in `jobs.json` against the wall clock: the current step is
`elapsed / STEP_SECONDS`, so no background task, no lock and no executor is
involved. `?fail=true` walks the failing sequence instead, which is how #49's
failed state is reached without editing code. `X-Stub-Fail: 1` on any request
returns the documented error body, which is how a failed *request* is reached.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import TypeAdapter, ValidationError

from chemometrics_workbench.models import PreprocessStep

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Production mode serves the built frontend from here. STUB_BUNDLE overrides it,
# which is how the mount is exercised before #42 has produced a bundle.
_BUNDLE_DEFAULT = Path(__file__).resolve().parents[1] / "frontend" / "dist"
BUNDLE = Path(os.environ.get("STUB_BUNDLE") or _BUNDLE_DEFAULT)

# The token is a real check, not a placeholder: an unauthenticated localhost
# server never gets authentication retrofitted (PROPOSAL.md section 4.3). Set
# STUB_TOKEN to keep it stable across restarts while developing.
TOKEN = os.environ.get("STUB_TOKEN") or secrets.token_urlsafe(32)

# How long each job step lasts. Six steps at 1.2s is the "handful of seconds"
# the UI needs to show progress moving; the tests turn it down to milliseconds.
STEP_SECONDS = float(os.environ.get("STUB_JOB_STEP_SECONDS", "1.2"))

# The Vite dev server, so the frontend can run on 5173 against this on its own
# port. In production mode the bundle is served from here and no origin is.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

# Zero means an ephemeral port, which is what PROPOSAL.md section 4.3 asks for
# and what the packaged application will do. STUB_PORT pins it instead, because
# Vite's dev proxy needs a target it can be told about in advance (#42).
PORT = int(os.environ.get("STUB_PORT", "0"))


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def maybe_fail(x_stub_fail: Annotated[str | None, Header()] = None) -> None:
    """Make any request fail on demand, so the UI's failure state is reachable."""
    if x_stub_fail:
        raise HTTPException(status_code=422, detail=fixture("error")["error"])


app = FastAPI(title="Chemometrics Workbench stub server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One router rather than a mounted sub-application: a mount carries its own
# exception middleware, and the error body below would then not apply to it.
api = APIRouter(prefix="/api", dependencies=[Depends(require_token), Depends(maybe_fail)])


@app.exception_handler(HTTPException)
async def error_body(request: Request, exc: HTTPException) -> JSONResponse:
    """Every failure has a body, not only a status code - the shape error.json documents."""
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    code = {401: "unauthorized", 404: "not_found"}.get(exc.status_code, "request_failed")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(detail), "detail": {}}},
    )


# --- Projects, datasets, import ------------------------------------------
# Phase 1.2: the project directory reader and the file readers replace these.


@api.get("/projects")
def list_projects() -> Any:
    return [fixture("project")]


@api.get("/projects/{project_id}")
def get_project(project_id: str) -> Any:
    return fixture("project")


@api.get("/projects/{project_id}/datasets")
def list_datasets(
    project_id: str,
    empty: Annotated[bool, Query()] = False,
    oversize: Annotated[bool, Query()] = False,
) -> Any:
    # `?empty=true` is the same kind of affordance as `?fail=true` on a run:
    # the empty-project state (#44) has to be reachable without editing code,
    # and a project with no datasets is otherwise unreachable from a fixture
    # that always has one.
    if empty:
        return []
    entries = fixture("datasets")
    if oversize:
        # `?oversize=true` reports the committed dataset as one past the
        # envelope in PROPOSAL.md section 13, so the overloaded state (#49) can
        # be entered without committing a 320 MB fixture to git. Only the
        # declared shape changes; no array is fabricated.
        for entry in entries:
            for version in entry["versions"]:
                version["n_samples"] = 42_000
                version["n_variables"] = 6_200
    return entries


@api.post("/import/preview")
def import_preview() -> Any:
    return fixture("import_preview")


@api.post("/import")
def import_dataset() -> Any:
    return fixture("datasets")[0]


@api.get("/schema/steps")
def step_schema() -> Any:
    """The preprocessing steps' JSON Schema, generated from models.py (#41).

    Phase 1.2: served from the live models rather than from a file, which is a
    change of source and not of shape.
    """
    return fixture("step_schema")


@api.post("/steps/validate")
def validate_step(step: dict[str, Any]) -> Any:
    """Validate one step against the model that will enforce it.

    This is the one place the stub server computes something, and it does so
    on purpose. The cross-field rules - an odd Savitzky-Golay window,
    `polyorder` below it, `start` below `end` - live in `model_validator` and
    have no JSON Schema equivalent, so a form that checked them itself would be
    restating `models.py` in TypeScript and drifting from it. Validating
    against the model means the message the user reads is the model's own.

    Phase 1.2: the same call, with the step going on to be stored.
    """
    try:
        TypeAdapter(PreprocessStep).validate_python(step)
    except ValidationError as error:
        return {
            "valid": False,
            "errors": [
                {
                    # Pydantic prefixes its own message; the part after the
                    # comma is what a person needs to read.
                    "field": ".".join(str(part) for part in problem["loc"]) or "step",
                    "message": problem["msg"].removeprefix("Value error, "),
                }
                for problem in error.errors()
            ],
        }
    return {"valid": True, "errors": []}


# --- Pipelines -----------------------------------------------------------
# Phase 1.2: the pipeline store and the validator replace these.


@api.get("/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: str) -> Any:
    return fixture("pipeline")


@api.get("/pipelines/{pipeline_id}/state")
def get_pipeline_state(pipeline_id: str) -> Any:
    return fixture("pipeline_state")


@api.post("/pipelines/{pipeline_id}/validate")
def validate_pipeline(pipeline_id: str) -> Any:
    # GUESS, like the envelope shapes in generate_fixtures.py: models.py does
    # not cover a validation response, so 1.2 is free to change this.
    return {"pipeline_id": fixture("pipeline")["pipeline_id"], "valid": True, "problems": []}


# --- Results -------------------------------------------------------------
# Phase 1.2: the executor's stored outputs replace these.


@api.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> Any:
    return fixture("experiment")


@api.get("/spectra/{node_id}")
def get_spectra(node_id: str) -> Any:
    spectra = fixture("spectra")
    if node_id not in spectra:
        raise HTTPException(status_code=404, detail=f"No spectra for node {node_id!r}")
    return spectra[node_id]


@api.get("/results/{node_id}")
def get_results(node_id: str) -> Any:
    pca = fixture("pca")
    if node_id not in pca:
        raise HTTPException(status_code=404, detail=f"No results for node {node_id!r}")
    return pca[node_id]


# --- Jobs ----------------------------------------------------------------
# Phase 1.2: the real executor and its job table replace these. In-memory only:
# a job is its status sequence, when it started, and when it was cancelled.

_jobs: dict[str, dict[str, Any]] = {}


def _snapshot(job: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = job["steps"]
    index = min(int((time.monotonic() - job["started"]) / STEP_SECONDS), len(steps) - 1)
    if job["cancelled_at"] is not None:
        index = min(index, job["cancelled_at"])
        last = dict(steps[index])
        last.update(status="cancelled", message="Cancelled")
        return last | {"job_id": job["job_id"]}
    return dict(steps[index]) | {"job_id": job["job_id"]}


@api.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str, fail: Annotated[bool, Query()] = False) -> Any:
    job_id = str(uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "steps": fixture("jobs")["failed" if fail else "succeeded"],
        "started": time.monotonic(),
        "cancelled_at": None,
    }
    return _snapshot(_jobs[job_id])


@api.get("/jobs/{job_id}")
def get_job(job_id: str) -> Any:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
    return _snapshot(_jobs[job_id])


@api.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Any:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
    job = _jobs[job_id]
    if job["cancelled_at"] is None:
        elapsed = int((time.monotonic() - job["started"]) / STEP_SECONDS)
        job["cancelled_at"] = min(elapsed, len(job["steps"]) - 1)
    return _snapshot(job)


# --- The bundle ----------------------------------------------------------
# Production mode serves the built frontend; development mode leaves it to the
# Vite dev server, which reaches the API across the CORS origins above.

app.include_router(api)

if BUNDLE.is_dir():

    @app.get("/{path:path}")
    def bundle(path: str) -> FileResponse:
        """Serve the built file, or index.html so a deep link reaches the app.

        The frontend routes on the path itself (#42), so /tokens has to arrive
        as the application rather than as a 404. Anything under /api never
        reaches here - those routes are registered above this one.
        """
        candidate = (BUNDLE / path).resolve()
        # A path from the client never escapes the bundle: PROPOSAL.md section
        # 4.3 confines filesystem access, and ../ in a URL is the cheapest way
        # to test whether anyone meant it.
        inside = candidate.is_relative_to(BUNDLE.resolve())
        if path and inside and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(BUNDLE / "index.html")


def main() -> None:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info"))
    # An ephemeral port is only knowable after the socket is bound, so the URL
    # is printed from the socket rather than from the config.
    original = server.startup

    async def startup(sockets: Any = None) -> None:
        await original(sockets=sockets)
        port = server.servers[0].sockets[0].getsockname()[1]
        print(f"\n  Launch URL: http://127.0.0.1:{port}/?token={TOKEN}\n", flush=True)

    server.startup = startup  # type: ignore[method-assign]
    server.run()


if __name__ == "__main__":
    main()
