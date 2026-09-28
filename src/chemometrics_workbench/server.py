"""The application: one server, on the loopback interface, behind a token.

Run it:

    uv run python -m chemometrics_workbench.server

It prints the launch URL — `http://127.0.0.1:<port>/?token=<token>` — which is
what the desktop shell hands the browser.

This module assembles; `api.py` computes. What is here is the things a server
has and a router does not: the port, the token, the Host and Origin checks, the
error envelope, and the mount that serves the built frontend in production.

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

## Host and Origin are checked on every request

§4.3's defence against DNS rebinding. A page on `attacker.example` whose name
has been re-pointed at 127.0.0.1 reaches this server with its own name in
`Host`, so a request is refused unless `Host` names the loopback interface. A
browser request that carries an `Origin` is refused unless the origin is this
server's own, so nothing another page runs can talk to it. Both apply to every
route, the bundle included, and both are checked before the token.

There is no CORS: the frontend calls `/api` on its own origin in production,
and through the Vite proxy in development, which rewrites `Host` but forwards
the browser's `Origin: http://localhost:5173`. `WORKBENCH_DEV=1` accepts those
dev-server origins; a packaged application never sets it.

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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
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

#: The Vite dev server's origins, accepted only when `DEV` is set: the proxy
#: forwards the browser's `Origin` unchanged.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEV = os.environ.get("WORKBENCH_DEV") == "1"

#: The names the loopback interface answers to. The server binds 127.0.0.1
#: only, so nothing else is a name a legitimate request can carry.
LOOPBACK = ("127.0.0.1", "localhost")

PORT = int(os.environ.get("WORKBENCH_PORT", "0"))


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    # Constant-time, so the comparison does not leak how much of a guess matched.
    if not secrets.compare_digest(authorization or "", f"Bearer {TOKEN}"):
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


def _refusal(request: Request) -> str | None:
    """Why this request is refused before anything reads it, or `None`."""
    host = request.headers.get("host", "")
    # `Host` is `name` or `name:port`; the server binds IPv4 only, so a
    # bracketed IPv6 literal is not a name it answers to either.
    if host.rsplit(":", 1)[0] not in LOOPBACK:
        return f"Host {host!r} is not this machine's loopback interface"
    origin = request.headers.get("origin")
    if origin is not None and origin != f"http://{host}" and not (DEV and origin in DEV_ORIGINS):
        return f"Origin {origin!r} is not this application"
    return None


@app.middleware("http")
async def same_machine_same_origin(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    reason = _refusal(request)
    if reason is not None:
        return JSONResponse(
            status_code=403,
            content={"error": {"code": "forbidden", "message": reason, "detail": {}}},
        )
    return await call_next(request)


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
