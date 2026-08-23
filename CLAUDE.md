# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Phase 0 is released and tagged `v0.1.0`.** The algorithm kernels, the parity programme and the reproducibility schema exist and are green in CI. There is no application yet: no HTTP server, no database, no user interface. Phase 1 builds those.

- `PROPOSAL.md` — the specification. Read it before proposing or writing anything.
- `feature_list.json` — the **live** task list. It currently covers Phase 1.1.
- `docs/phase-0/feature_list.json` — Phase 0's completed list, kept as its record. Do not add to it.
- `design/DESIGN_BRIEF.md` — screens, states and plot rules for the UI.
- `src/chemometrics_workbench/` — the kernels: preprocessing, PCA, PLS, validation, reference datasets.
- `src/chemometrics_workbench/models.py` — the Pydantic schema for the reproducibility model; its invariants are exercised by `tests/test_models.py`.
- `design/data-model.md` — the same schema as mermaid diagrams, plus what is deliberately not modelled yet.
- `design/canvas/` — artboard sources for the five core screens.
- `docs/parity-report.md` — generated, never hand-edited.

**Phase 1 runs in three sub-phases**, because the interface and the backend are independently riskiest and a frontend built against invented JSON would encode an API contract nobody agreed to:

- **1.1 — frontend only.** The React shell and the core screens, built against fixtures generated from the real kernels. No server, no database.
- **1.2 — backend.** Readers, the pipeline executor, jobs and the HTTP surface, meeting the contract 1.1's fixtures established.
- **1.3 — database and integration.** SQLite per project directory, and the two halves joined.

Everything below summarises decisions recorded in those documents that are easy to violate accidentally.

The project is an open-source, local-first chemometrics workbench: a Python/FastAPI backend and React UI shipped as one double-clickable desktop application, aimed at replacing closed tools such as Unscrambler, SIMCA and OPUS for research and academic users.

## Working protocol

Follow this on every session. It exists because the failure mode in a long solo project is not bad code — it is half-finished work with no record of what was actually verified.

**Read the state before starting.** Never begin from this file's summary alone. Read, in order: `session-handoff.md` (where the last session stopped and what to pick up), `feature_list.json` (what is done, in progress and blocked), `git log` on `dev` (what actually landed), and the open GitHub issues (what the task really asks for). If those three disagree, the repository is the truth and the disagreement is itself worth fixing first.

**One feature at a time.** Pick the highest-priority feature whose `status` is `not_started` and whose every `depends_on` entry is `passing`. Set it to `in_progress`. **At most one feature may be `in_progress` at any moment.** Anything discovered mid-feature that falls outside its scope becomes a new GitHub issue and a new `feature_list.json` entry — never a quietly widened branch.

**Evidence before done.** A feature becomes `passing` only after its `verification` steps have actually been run, with the result recorded in `evidence`: the command, its real output or the path to the artifact, and the date. Never mark `passing` from reasoning, from a code review, or because the implementation looks correct. If a verification step cannot be run, the status is `blocked` with the reason in `notes` — not `passing` with a caveat.

**Blocked is a real status.** Use it. Record in `notes` what is blocking and what would unblock it. A blocked feature that is honestly labelled is worth more than an optimistic `in_progress` that hides a dead end.

**A session ends clean when all of these hold** — `clean-state-checklist.md` is the runnable version of this list, and takes precedence when the two differ:
- No feature is left `in_progress` without a note recording exactly where it stands and what the next step is.
- `feature_list.json` is committed if any status, evidence or note changed.
- The working tree is clean, or every remaining change is explained in the handover.
- The branch is pushed, and a completed feature has its pull request open or merged.
- The next feature to pick up is named.
- `session-handoff.md` is rewritten to match the state just described, and committed.

## Intended toolchain (not yet scaffolded)

When scaffolding, use these — they are the recorded stack, and deviating from them is a decision worth surfacing:

- **Backend:** Python, `uv` for environment and dependency management, FastAPI, Pydantic, NumPy, SciPy, pandas. Lint with `ruff`, type-check with `mypy`, test with `pytest`.
- **`scikit-learn` and `chemotools` are development dependencies, not runtime ones**, and must not be added to `[project.dependencies]`. They are the open reference implementations the parity fixtures are generated against; a kernel that imported either would be a wrapper around the thing we claim parity with. `chemotools` was evaluated in Phase 0 (#13) and rejected for the runtime for that reason plus its weight — it requires scikit-learn and installs 20 MB, 17 MB of it bundled example data. It is adopted as a reference for SNV, MSC and the baselines, which have no other. The evidence is in `docs/decisions/0001-chemotools.md`.
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

**Merges happen through pull requests, not local `git merge`.** A feature branch reaches `dev` by:

1. Push the branch.
2. Open a pull request **with `dev` as the base** — never `main`. GitHub's default base is the repository's default branch, so if that is still `main` the base must be set explicitly or the pull request targets the release line.
3. Wait for CI to pass. The pull request is the gate; a red check is not merged around.
4. Merge, then delete the branch locally and on origin.
5. Reference the issue in the pull request body (`Closes #n`) so it closes on merge.

`gh` is not installed on this machine — use the GitHub MCP tools to open and merge pull requests.

At the end of a phase: open a pull request from `dev` into `main`, merge it, tag a release, and only then start the next phase's branches.

The repository's release branch is named `main` — there is no `master`.

## Documents

Decisions that were taken with evidence and should not be re-argued from preference live in `docs/decisions/`, numbered and dated. Read the relevant one before revisiting a dependency or a convention it covers.

`PROPOSAL.md` is canonical and git-diffable. A designed HTML rendering of the same content is published as an artifact at https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488 — when `PROPOSAL.md` changes materially, the artifact should be republished to that same URL so the two do not drift.
