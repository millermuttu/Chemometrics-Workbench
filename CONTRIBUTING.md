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

### The parity suite

```bash
uv run pytest -m parity
```

Every claim the parity report renders is made by a test in `tests/test_parity.py`. A run writes `parity-results.json` at the repository root — one record per comparison, plus the fixture entries nothing was checked against, which is what stops the report overstating its own coverage. The file is gitignored; it is rebuilt by running the suite.

`tests/test_parity_harness.py` tests the harness rather than any scientific claim, and deliberately provokes failures. It restores the recorder around every case so those never reach the run record.

### The parity report

```bash
CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest      # writes parity-results.json
uv run python -m tests.parity_report                # rewrites docs/parity-report.md
```

`docs/parity-report.md` is generated and must never be edited by hand. CI regenerates it after the suite and runs `git diff --exit-code` on it, so a scientific number that moved fails the build **even if every test still passes** — a widened tolerance, for instance, keeps the suite green and changes the published claim. The tests are the gate on correctness; the diff is the gate on what the project says in public.

The renderer refuses to run against a partial suite: if `not_compared` in `parity-results.json` is non-empty, a report built from it would understate coverage while looking complete, so it exits with the command to run instead.

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
| `tests/fixtures/` | Parity fixtures and the script that regenerates them. `reference_values.json` is the numbers every kernel is checked against. |
| `tests/parity.py` | The parity harness: tolerance policy, sign alignment, claim tiers, run record. Every kernel's parity test goes through it. |
| `tests/parity_report.py` | Renders `docs/parity-report.md` from the run record. It renders and does not compute: two sources of truth for one number is one too many. |
| `src/chemometrics_workbench/preprocessing.py` | Scaling and scatter-correction kernels. `fit`/`transform`, duck-compatible with a scikit-learn transformer and importing nothing from it. |
| `docs/algorithms/` | One specification per algorithm — the variant implemented, its conventions, and the definition of every quantity it reports. |
| `design/` | Design brief, data-model diagrams, and the artboard sources for the UI. Not shipped code; excluded from linting. |

## How to add an algorithm

The order matters, and it is the order Phase 0 itself follows.

1. **Write the specification first**, in `docs/algorithms/`. Name the variant, the centring and scaling conventions, the sign convention, and the exact definition of every quantity that will be reported. "PLS" is not a specification.
2. **Find reference values.** Published literature, or an established open implementation with its version recorded. A kernel with nothing to be checked against cannot be trusted. They live in `tests/fixtures/reference_values.json`, one entry per value, each recording its preprocessing chain, algorithm variant, split, software and version, and citation. Regenerate the generated entries with `uv run python tests/fixtures/generate_reference_values.py`, and say in the commit message what moved and why. A value that cannot be sourced is written into the fixture as `status: "unsourced"` with the reason — never omitted, and never filled in with a plausible number.
3. **Write the kernel** as a pure function over arrays — no application knowledge, no global random state, seeds threaded explicitly, and never mutating the caller's array.
4. **Add parity tests** through the shared harness in `tests/parity.py`, never with a bare `assert_allclose`. Call `parity.check(entry_id, ours)`; it picks the tolerance for the quantity's class, aligns signs where the quantity is sign-invariant, tags the claim tier and records the result for the report. Where a quantity differs from a reference by documented convention, call `parity.record_divergence(entry_id, reason)` instead of loosening a tolerance. **Tolerances are not knobs** — a comparison that fails is a finding, and widening the tolerance to make it pass converts a finding into a lie in the one artifact this project cannot afford to have lying in it.
5. **Wire it into the schema** in `models.py` as a new member of the relevant discriminated union, so an invalid configuration fails at parse time.

## Conventions worth knowing before you write code

- **Array shape is `n_samples × n_variables`**, always. Never silently transposed.
- **Never mutate a caller's array.**
- **Seeds are threaded explicitly.** No global `numpy.random` state.
- **A pipeline is data.** Executing a serialisable DAG is the only path from a dataset to a result — do not add a second, direct one.
- **Scientific numbers do not move silently.** A change that alters a reported value must update the parity fixtures deliberately, in the same commit, with the reason in the message.

## Branching and commits

See `CLAUDE.md` for the branching model and the commit-message convention. In short: branch from `dev`, merge back into `dev`, and `main` receives a merge only at the end of a phase.
