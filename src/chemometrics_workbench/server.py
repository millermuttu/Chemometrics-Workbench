"""The application: one server, on the loopback interface, behind a token.

Run it:

    uv run python -m chemometrics_workbench.server

It prints the launch URL — `http://127.0.0.1:<port>/?token=<token>` — which is
what the desktop shell hands the browser.

This module assembles; `api.py` computes. What is here is the things a server
has and a router does not: the port, the token, the error envelope, the CORS
origins the Vite dev server needs, and the mount that serves the built frontend
in production.

## The token is a real check

`PROPOSAL.md` §4.3 calls localhost a trust boundary rather than a private room.
Every `/api` request carries `Authorization: Bearer <token>`; an
unauthenticated localhost server never gets authentication retrofitted, because
by then something depends on its absence.

## The port is ephemeral

§4.3 again: the packaged application binds port zero and prints what it got, so
two copies never fight over a number and nothing has to be configured.
`WORKBENCH_PORT` pins it instead, which is what the Vite dev proxy needs — it
has to be told a target in advance.

## Every failure has a body

`{"error": {"code", "message", "detail"}}`, which is the shape every screen
renders and which `tests/fixtures/error.json` documents. Handlers raise
`HTTPException` with that dict as the detail; anything that arrives without one
— a 404 from the router itself, say — is wrapped in the same shape here, so a
client never has to tell two error formats apart.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from chemometrics_workbench.api import JOBS, router
from chemometrics_workbench.db import dispose_all

__all__ = ["BUNDLE", "TOKEN", "app", "main"]

#: Production mode serves the built frontend from here. `WORKBENCH_BUNDLE`
#: overrides it, which is how the mount is exercised without a build.
_BUNDLE_DEFAULT = Path(__file__).resolve().parents[2] / "frontend" / "dist"
BUNDLE = Path(os.environ.get("WORKBENCH_BUNDLE") or _BUNDLE_DEFAULT)

#: Set `WORKBENCH_TOKEN` to keep the token stable across restarts while
#: developing; otherwise it is fresh every time, as a launched application's is.
TOKEN = os.environ.get("WORKBENCH_TOKEN") or secrets.token_urlsafe(32)

#: The Vite dev server, so the frontend can run on 5173 against this on its own
#: port. In production mode the bundle is served from here and no origin is.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

PORT = int(os.environ.get("WORKBENCH_PORT", "0"))


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Ask every unfinished run to stop before the process goes, and let go of
    the database.

    Disposing matters least on the way out of a process and most on Windows,
    where an undisposed handle is what keeps a file locked after the program
    that held it has gone.
    """
    yield
    JOBS.shutdown(wait=False)
    dispose_all()


app = FastAPI(title="Chemometrics Workbench", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def error_body(request: Request, exc: HTTPException) -> JSONResponse:
    """Every failure has a body, not only a status code."""
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content={"error": detail})
    code = {401: "unauthorized", 404: "not_found"}.get(exc.status_code, "request_failed")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(detail), "detail": {}}},
    )


@app.exception_handler(RequestValidationError)
async def malformed_body(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A body FastAPI could not parse gets the same envelope as everything else.

    FastAPI answers its own validation failures with `{"detail": [...]}`, which
    is a second error shape - exactly what the envelope exists to prevent, and
    invisible until a client sends a malformed body rather than a wrong one.
    The first error is reported because a list of twenty is not a sentence
    anyone reads; the rest are in `detail.errors` for whoever wants them.
    """
    errors = exc.errors()
    first = errors[0] if errors else {}
    # `loc` starts with "body"; the field is the rest of the path.
    field = ".".join(str(part) for part in first.get("loc", ())[1:])
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "bad_request",
                "message": str(first.get("msg", "the request body is not valid")),
                "detail": {"field": field, "errors": len(errors)},
            }
        },
    )


# One router rather than a mounted sub-application: a mount carries its own
# exception middleware, and the error body above would not apply to it.
app.include_router(router, prefix="/api", dependencies=[Depends(require_token)])


if BUNDLE.is_dir():

    @app.get("/{path:path}")
    def bundle(path: str) -> FileResponse:
        """Serve the built file, or `index.html` so a deep link reaches the app.

        The frontend routes on the path itself, so `/tokens` has to arrive as
        the application rather than as a 404. Anything under `/api` never
        reaches here — those routes are registered above this one.
        """
        candidate = (BUNDLE / path).resolve()
        # A path from the client never escapes the bundle: §4.3 confines
        # filesystem access, and `../` in a URL is the cheapest way to find out
        # whether anyone meant it.
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
