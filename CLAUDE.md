# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

**Pre-implementation.** There is no source code and no build tooling yet — Phase 0 creates both. What exists is specification and design:

- `PROPOSAL.md` — the specification. Read it before proposing or writing anything.
- `feature_list.json` — the Phase 0 task list, mirroring GitHub issues 1–14.
- `design/DESIGN_BRIEF.md` — screens, states and plot rules for the UI.
- `design/data_models.py` — the Pydantic schema for the reproducibility model; run it directly for its self-check.
- `design/data-model.md` — the same schema as mermaid diagrams.
- `design/canvas/` — artboard sources for the five core screens.

Everything below summarises decisions recorded in those documents that are easy to violate accidentally.

The project is an open-source, local-first chemometrics workbench: a Python/FastAPI backend and React UI shipped as one double-clickable desktop application, aimed at replacing closed tools such as Unscrambler, SIMCA and OPUS for research and academic users.

## Working protocol

Follow this on every session. It exists because the failure mode in a long solo project is not bad code — it is half-finished work with no record of what was actually verified.

**Read the state before starting.** Never begin from this file's summary alone. Read, in order: `feature_list.json` (what is done, in progress and blocked), `git log` on `dev` (what actually landed), and the open GitHub issues (what the task really asks for). If those three disagree, the repository is the truth and the disagreement is itself worth fixing first.

**One feature at a time.** Pick the highest-priority feature whose `status` is `not_started` and whose every `depends_on` entry is `passing`. Set it to `in_progress`. **At most one feature may be `in_progress` at any moment.** Anything discovered mid-feature that falls outside its scope becomes a new GitHub issue and a new `feature_list.json` entry — never a quietly widened branch.

**Evidence before done.** A feature becomes `passing` only after its `verification` steps have actually been run, with the result recorded in `evidence`: the command, its real output or the path to the artifact, and the date. Never mark `passing` from reasoning, from a code review, or because the implementation looks correct. If a verification step cannot be run, the status is `blocked` with the reason in `notes` — not `passing` with a caveat.

**Blocked is a real status.** Use it. Record in `notes` what is blocking and what would unblock it. A blocked feature that is honestly labelled is worth more than an optimistic `in_progress` that hides a dead end.

**A session ends clean when all of these hold:**
- No feature is left `in_progress` without a note recording exactly where it stands and what the next step is.
- `feature_list.json` is committed if any status, evidence or note changed.
- The working tree is clean, or every remaining change is explained in the handover.
- The branch is pushed.
- The next feature to pick up is named.

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
