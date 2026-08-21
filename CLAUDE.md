# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Pre-implementation.** There is no source code, no commits, and no build tooling yet. `PROPOSAL.md` is the entire repository and is the specification — read it before proposing or writing anything. Everything below is a summary of decisions recorded there that are easy to violate accidentally.

The project is an open-source, local-first chemometrics workbench: a Python/FastAPI backend and React UI shipped as one double-clickable desktop application, aimed at replacing closed tools such as Unscrambler, SIMCA and OPUS for research and academic users.

## Locked decisions — do not re-litigate without being asked

| Decision | Value |
| --- | --- |
| Distribution | Single downloadable application (PyInstaller onedir + system default browser). Not Docker, not pip, not WASM/Pyodide |
| Licence | MIT |
| Data scope | 2-D spectra only (samples × variables). Hyperspectral cubes and 3-way data are out |
| Audience | Research and academia. GxP / 21 CFR Part 11 is out of scope |

A native window shell (Tauri, pywebview) is deliberately deferred; the default browser is the shipped UX until browser-tab UX proves to be a real complaint.

## Constraints that are expensive to retrofit

**Localhost is a trust boundary.** Any backend work must preserve: bind `127.0.0.1` only (never `0.0.0.0`), ephemeral port, per-session bearer token required on every request (header, never a cookie), strict `Origin` and `Host` validation against DNS rebinding, and filesystem access confined to the user-chosen project directory — the API must never accept an arbitrary server-side path from the client.

**Parity before UI.** Phase 0 is numerical correctness with no user interface. Algorithm kernels are pure functions over arrays with no knowledge of the application, so they stay testable in isolation and reusable as a library. Any change that moves a scientific number must fail CI unless the parity fixtures are updated deliberately. Algorithm variants (NIPALS vs SIMPLS), centring/scaling conventions, sign conventions and metric definitions are documented per algorithm — "PLS" alone is never a sufficient specification.

**The pipeline is data.** An analysis is a serialisable JSON DAG of typed steps, and executing one is the *only* path from a dataset to a result. Do not add a second, direct path — lineage, reproducibility and model export all depend on this being the single route. Datasets are identified by content hash, not filename. Splits store strategy, seed *and* the resulting index sets.

**Dependency rule.** Take a dependency for the tedious and well-solved (instrument file formats, numerics primitives); own the scientifically load-bearing and small (SNV, MSC, baseline correction, Hotelling T², SPE/Q, VIP). SpectroChemPy and process-improve were evaluated and deliberately rejected — do not reintroduce them. `chemotools` is provisional, pending Phase 0 parity evaluation.

**Deferral is deliberate.** PCR, SIMCA, permutation testing, bootstrap, variable selection, additional classifiers, PostgreSQL, object storage, multi-user/auth and self-hosted mode are all post-1.0 by decision, not by oversight. The test each must pass: it stays *additive* against the 1.0 data model. Do not build for them in advance.

**Plot performance is a design constraint, not an optimisation.** Server-side decimation, a cap on individually drawn traces with the remainder as a density band, and WebGL (`scattergl`) are required from the first plot. Target envelope: ~20,000 spectra × ~4,000 variables as float32.

## Intended toolchain (not yet scaffolded)

When scaffolding, use these — they are the recorded stack, and deviating from them is a decision worth surfacing:

- **Backend:** Python, `uv` for environment and dependency management, FastAPI, Pydantic, NumPy, SciPy, pandas, scikit-learn. Lint with `ruff`, type-check with `mypy`, test with `pytest`.
- **Frontend:** Node.js with `pnpm`, Vite, React, TypeScript, Tailwind CSS, shadcn/ui, TanStack Query, Plotly.js. Test with `vitest`, end-to-end with `playwright`.
- **Data:** SQLite via SQLAlchemy for metadata, pipelines, experiments, metrics and lineage. Files (datasets, processed arrays, model artifacts, reports) live in the project directory on disk; the database stores references, never contents.
- **Packaging:** PyInstaller, three-platform GitHub Actions matrix.

Docker is a developer-environment tool only. End users never see a container.

## Branching and release

Three levels, and work never skips one.

```
feature/<n>_<short-name>  ──merge──►  dev  ──merge at phase end──►  main  ──►  release tag
fix/<n>_<short-name>      ──merge──►
```

- **`main`** is the release line. It only ever receives a merge from `dev`, and only when a whole phase is complete. Never commit to it directly.
- **`dev`** is the integration line. Every feature and fix branch is cut from `dev` and merged back into `dev`. It is the default base for all new work.
- **`feature/<n>_<short-name>` / `fix/<n>_<short-name>`** — one per GitHub issue, `<n>` being the issue number. See the `new-branch` skill for the naming rules.

At the end of a phase: merge `dev` into `main`, tag a release, and only then start the next phase's branches.

The repository's release branch is named `main` — there is no `master`.

## Documents

`PROPOSAL.md` is canonical and git-diffable. A designed HTML rendering of the same content is published as an artifact at https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488 — when `PROPOSAL.md` changes materially, the artifact should be republished to that same URL so the two do not drift.
