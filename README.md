# Chemometrics Workbench

An open-source, local-first chemometrics workbench: a Python backend and a React interface
shipped as one double-clickable desktop application, aimed at research and academic users of
closed tools such as Unscrambler, SIMCA and OPUS. Your data never leaves your machine.

**Status: Phase 1 is complete and released, `v0.5.0`.** The application runs: import a
dataset, build a preprocessing pipeline on a canvas, fit PCA or PLS, cross-validate, and read
the result. Everything stands on the numerical foundation Phase 0 laid — the algorithm
kernels, their specifications, and the evidence that their numbers are right.

Phase 2 is under way. `PROPOSAL.md` §16 still wants PLS-DA, VIP scores, contribution plots, a
train/test splitting interface and the Bruker OPUS reader.

## Running it

```bash
./run.sh
```

It syncs the environment, builds the frontend bundle if there is not one, and starts the
server. Open the URL it prints — `http://127.0.0.1:<port>/?token=<token>`. The port is
ephemeral so two copies never fight over a number, and the token is a real check: every `/api`
request carries it, so a bare `127.0.0.1` address without the token gets a 401.

`./run.sh --build` rebuilds the bundle first, which is what you want after changing anything
under `frontend/src`.

Working on the interface is two processes instead, so Vite can serve its own:

```bash
WORKBENCH_PORT=8000 WORKBENCH_TOKEN=dev uv run python -m chemometrics_workbench.server
cd frontend && npm run dev          # http://localhost:5173
```

The port is pinned because the Vite proxy has to be told a target in advance, and the token
because a fresh one on every restart is tedious to paste. `localhost:5173` is already an
allowed origin.

## The parity report

**[`docs/parity-report.md`](docs/parity-report.md)** — every quantity this project computes,
next to an independent implementation's value for the same thing, with a claim tier, a
tolerance that has a stated reason, and a citation.

It is regenerated in CI on every push, and a change that moves a scientific number fails the
build — the parity suite is the gate, and the tolerances it compares against are frozen in a
test of their own, so weakening one cannot pass quietly. The report also lists what could
*not* be compared and why, because a report that shows only its coverage overstates it.

No competing open-source chemometrics project publishes one, and `PROPOSAL.md` §10.4 is the
argument for why this one does: it is the artifact that turns "an open-source tool" into "a
tool I can put in a paper".

## Documents

| | |
| --- | --- |
| [`PROPOSAL.md`](PROPOSAL.md) | The specification for the whole project: scope, architecture, phases, open questions |
| [`docs/algorithms/`](docs/algorithms/) | Normative algorithm specifications — PCA, PLS, metrics and validation, smoothing and baselines |
| [`docs/decisions/`](docs/decisions/) | Decisions taken with evidence, numbered and dated, with the script that reproduces the numbers |
| [`docs/parity-report.md`](docs/parity-report.md) | The parity report |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to set up, verify and land a change |

## Working on it

```bash
uv sync
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest   # once, to populate the dataset cache

uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

Only the Tecator dataset is committed. Corn and gasoline are downloaded on first use into
`~/.cache/chemometrics-workbench/datasets` and verified against a pinned SHA-256; the licence
finding for each is recorded beside its loader. **Publishing a result obtained with the
Tecator dataset obliges you to name the instrument and the company, Tecator** — a condition of
use, not a courtesy.

`scikit-learn` and `chemotools` are development dependencies and must stay that way: they are
the reference implementations the parity fixtures are generated against, and a kernel built on
one of them would be a wrapper around the thing this project claims parity with.

## Licence

MIT. See [`LICENSE`](LICENSE).
