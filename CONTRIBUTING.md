# Contributing

Single-maintainer project for now. These are the commands the rest of the repository's process documents refer to by role — `clean-state-checklist.md` in particular points here rather than hard-coding them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync
```

That is the whole setup path. If a step is ever needed beyond this line, it belongs in this section rather than in someone's memory.

## Verification

The full suite. All four must exit 0 before anything is merged.

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

Run a single test file or case with the usual pytest selectors:

```bash
uv run pytest tests/test_models.py
uv run pytest tests/test_models.py::test_content_hash_tracks_every_parameter
uv run pytest -k content_hash
```

### Reference datasets

The corn and gasoline benchmarks are downloaded rather than committed, so
their tests skip on a machine that has never fetched them. To run them:

```bash
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest tests/test_datasets.py
```

The download is cached under `~/.cache/chemometrics-workbench/datasets` and
verified against a pinned SHA-256 on every read. CI sets the same variable,
so all three datasets are checksum-asserted there.

## Layout

| Path | Holds |
| --- | --- |
| `src/chemometrics_workbench/` | The package. Algorithm kernels stay pure functions over arrays with no knowledge of the application. |
| `src/chemometrics_workbench/data/` | Reference datasets, one directory each, carrying the source URL, the terms of use and a checksum. Only Tecator's raw file is committed; see each README for why. |
| `tests/` | Test suite, mirroring the package layout. |
| `docs/algorithms/` | One specification per algorithm — the variant implemented, its conventions, and the definition of every quantity it reports. |
| `design/` | Design brief, data-model diagrams, and the artboard sources for the UI. Not shipped code; excluded from linting. |

## How to add an algorithm

The order matters, and it is the order Phase 0 itself follows.

1. **Write the specification first**, in `docs/algorithms/`. Name the variant, the centring and scaling conventions, the sign convention, and the exact definition of every quantity that will be reported. "PLS" is not a specification.
2. **Find reference values.** Published literature, or an established open implementation with its version recorded. A kernel with nothing to be checked against cannot be trusted.
3. **Write the kernel** as a pure function over arrays — no application knowledge, no global random state, seeds threaded explicitly, and never mutating the caller's array.
4. **Add parity tests** through the shared harness, with an explicit tolerance and a claim tier. Comparisons of scores and loadings must be sign-invariant.
5. **Wire it into the schema** in `models.py` as a new member of the relevant discriminated union, so an invalid configuration fails at parse time.

## Conventions worth knowing before you write code

- **Array shape is `n_samples × n_variables`**, always. Never silently transposed.
- **Never mutate a caller's array.**
- **Seeds are threaded explicitly.** No global `numpy.random` state.
- **A pipeline is data.** Executing a serialisable DAG is the only path from a dataset to a result — do not add a second, direct one.
- **Scientific numbers do not move silently.** A change that alters a reported value must update the parity fixtures deliberately, in the same commit, with the reason in the message.

## Branching and commits

See `CLAUDE.md` for the branching model and the commit-message convention. In short: branch from `dev`, merge back into `dev`, and `main` receives a merge only at the end of a phase.
