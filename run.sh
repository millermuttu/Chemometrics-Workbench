#!/usr/bin/env bash
# Launch the workbench: one process on the loopback interface, serving the
# built frontend. It prints the URL to open — the token is in it, and every
# /api request needs that token, so a bare localhost address will not do.
#
#   ./run.sh            # build the bundle if it is missing, then serve
#   ./run.sh --build    # rebuild first, for when frontend/src has moved on
#
# The dev loop is not this script: it is two processes, and Vite wants its own
# terminal. See "Running it" in README.md.
set -euo pipefail
cd "$(dirname "$0")"

uv sync

if [ "${1:-}" = "--build" ] || [ ! -f frontend/dist/index.html ]; then
    # pnpm, and --frozen-lockfile, because that is what CI installs with:
    # frontend/pnpm-lock.yaml is the only lockfile here, and `npm ci` refuses a
    # project it cannot find a package-lock.json in.
    (cd frontend && pnpm install --frozen-lockfile && pnpm build)
fi

exec uv run python -m chemometrics_workbench.server
