# Contributing

Single-maintainer project for now. These are the commands the rest of the repository's process documents refer to by role — `clean-state-checklist.md` in particular points here rather than hard-coding them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) with Python 3.12 or newer, and for the interface
Node 22 with [pnpm](https://pnpm.io/) (the version is pinned by `packageManager` in
`frontend/package.json`; `corepack enable` provides it).

```bash
uv sync
cd frontend && pnpm install --frozen-lockfile && cd ..
./run.sh            # builds the bundle if missing, serves, opens the browser
```

That is the whole setup path. If a step is ever needed beyond these lines, it belongs in this section rather than in someone's memory. `./run.sh --build` rebuilds the bundle after a change under `frontend/src`; the two-process development loop with Vite is in the README.

## Verification

What CI runs on every pull request. All of it must exit 0 before anything is merged.

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

The interface, from `frontend/`:

```bash
pnpm typecheck
pnpm lint
pnpm test
pnpm build
pnpm exec playwright install chromium   # once
pnpm test:e2e
```

**Build before the end-to-end suite.** Playwright drives the real server serving `frontend/dist`, so a suite run against a stale bundle tests old code and can pass for the wrong reason. It starts seven servers on ports 8765 to 8771, each on a project of its own; free them first if a previous run left one behind.

The documentation site takes its screenshots from the running application, then builds strictly (a broken link or a missing image fails it):

```bash
cd frontend && pnpm exec playwright test e2e/docs-screens.spec.ts --project seeded && cd ..
uv run mkdocs build --strict
```

CI also builds the packaged application on all three platforms and smoke-tests it; to do the same locally, after `pnpm build`:

```bash
uv run pyinstaller packaging/workbench.spec --noconfirm
uv run python -m tests.smoke_package dist/ChemometricsWorkbench
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

`docs/parity-report.md` is generated and must never be edited by hand. Regenerate it whenever a claim, a gap or a tolerance changes, in the same commit, and say what moved.

**It is deliberately not byte-compared in CI.** That was tried and it is flaky by construction (#38): the last bits of any difference depend on the BLAS the local NumPy was built against, and several comparisons sit within a factor of two of the 32-ulp line between *identical within floating point* and *agrees within tolerance*, so two correct runners disagree on their tier. What guards the published claim instead lives in the suite, where it is machine-independent:

- `test_the_committed_report_covers_every_claim_and_gap_in_the_fixture` fails when a claim or a gap is added and the report is not regenerated.
- `test_the_published_tolerances_are_the_ones_the_project_agreed_to` freezes every tolerance. Widening one keeps the rest of the suite green and changes what the project says in public, which is exactly the failure the byte-diff was meant to catch. Change the number in `parity.py` and in that test together, with the reason in the commit message.

CI still regenerates the report, so a renderer that crashes or a run that missed an entry fails the build. The renderer also refuses to run against a partial suite: if `not_compared` in `parity-results.json` is non-empty, a report built from it would understate coverage while looking complete, so it exits with the command to run instead.

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
| `src/chemometrics_workbench/models.py` | The reproducibility schema: every step, split and estimator as a Pydantic model. |
| `src/chemometrics_workbench/executor.py` | Runs a pipeline DAG over the array store, fold by fold. |
| `src/chemometrics_workbench/api.py`, `server.py` | The HTTP surface, and the server around it: loopback only, token, Host and Origin checks. |
| `src/chemometrics_workbench/db.py`, `project.py` | A project directory: `project.db` (SQLite, references only), `arrays/`, `results/`. |
| `frontend/` | The React interface; `e2e/` is the Playwright suite. |
| `packaging/` | The PyInstaller spec for the packaged application. |
| `docs/algorithms/` | One specification per algorithm — the variant implemented, its conventions, and the definition of every quantity it reports. |
| `docs/` | Also the documentation site's user pages; `mkdocs.yml` is its table of contents. |
| `design/` | Design brief, data-model diagrams, and the artboard sources for the UI. Not shipped code; excluded from linting. |

## How to add an algorithm

**[docs/adding-a-step.md](docs/adding-a-step.md)** walks the whole path on a real step, `Normalise`: the specification, reference values, the kernel, its parity claim, the pipeline model, the executor and the screen. The order is the point: each layer is checked before the next is written, and a number that cannot be checked against something independent is not shipped.

### Regenerating the R reference values

Most of `tests/fixtures/reference_values.json` regenerates with one command. The nine `r_mdatools` entries do not, because R is not a dependency of this project and is not installed in CI: their values are read from `tests/fixtures/r_mdatools_values.json`, which is committed. Re-derive that file only when a matrix, a component count or a confidence level changes.

```bash
conda create -n r-parity -c conda-forge r-base r-mdatools r-jsonlite   # once; no root needed
uv run python tests/fixtures/export_for_r.py build/r-reference
~/anaconda3/envs/r-parity/bin/Rscript tests/fixtures/r_mdatools_reference.R \
    build/r-reference tests/fixtures/r_mdatools_values.json
uv run python tests/fixtures/generate_reference_values.py
```

The matrices are exported from Python rather than loaded in R on purpose: a reference value must differ from ours because the algorithm differs, never because two readers disagreed about a file. They are exported already centred, and every fit passes `center = FALSE` — `mdatools` centres by default and would otherwise centre twice.

## Conventions worth knowing before you write code

- **Array shape is `n_samples × n_variables`**, always. Never silently transposed.
- **Never mutate a caller's array.**
- **Seeds are threaded explicitly.** No global `numpy.random` state.
- **A pipeline is data.** Executing a serialisable DAG is the only path from a dataset to a result — do not add a second, direct one.
- **Scientific numbers do not move silently.** A change that alters a reported value must update the parity fixtures deliberately, in the same commit, with the reason in the message.

## Branching and pull requests

- **One issue per change.** Something found mid-change that is outside its scope becomes a new issue, not a wider branch.
- **Branch from a freshly pulled `dev`**, named `feature/<issue>_<short-name>` or `fix/<issue>_<short-name>`. Check before opening the pull request that the branch really starts from `dev`: `git merge-base --is-ancestor origin/dev HEAD`.
- **Open the pull request into `dev`**, never `main`, with `Closes #<issue>` in the body. It is merged only when CI is green; a red check is read and fixed, not re-run until it passes.
- **`main` is the release line.** It receives `dev` only at the end of a phase, and a `v*` tag on it builds and publishes the packages (`.github/workflows/release.yml`).
- Commit messages are one short line saying what changed, then a body saying why when it is not obvious. The maintainer's commits use the form `[Chemometrics_toolbox](Name):short description`.

## Licence

The project is MIT-licensed (`LICENSE`). Contributions are accepted on the same terms — **inbound = outbound**: by opening a pull request you agree that your contribution is licensed under the MIT licence of this repository. There is no separate contributor agreement.
