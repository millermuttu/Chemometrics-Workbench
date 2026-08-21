# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

Phase 0 has started. The package and toolchain exist and the full verification suite runs green.

Issue #1 (`scaffold`) is complete and merged into `dev`. The schema now lives at `src/chemometrics_workbench/models.py`, its old self-check is `tests/test_models.py` (20 tests), and `CONTRIBUTING.md` holds the canonical setup and verification commands.

Nothing else has been started. No algorithm specifications exist yet, so no kernel work can begin.

## Repository

| | |
| --- | --- |
| Current branch | `dev` |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 13 of 14 remaining |
| Feature statuses | 1 × `passing`, 13 × `not_started` |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green on `dev` |

## Active feature

None. Nothing is `in_progress`.

## Next action

Start **`spec-metrics-cv`** — issue #5, priority 1, no dependencies.

```bash
git fetch origin
git checkout -b feature/5_metrics-cv-spec origin/dev
```

Deliverable is `docs/algorithms/metrics-and-validation.md`. Documentation only, no code. It blocks #7, which blocks the entire parity chain, and it fixes the definitions that will explain most future "why does this not match Unscrambler" reports.

After it: #4 (`spec-pls`), then #3 (`spec-pca`). Those three unblock #7 and everything downstream. #2 (`ci`) is also ready now that #1 has landed, and is short.

## Waiting on the user

- **GitHub default branch is still `main`.** It should be `dev`, or pull requests will target the release line by default. Repository setting; cannot be changed from here with the current tools.
- **Dataset redistribution terms** (#6) need checking per dataset before raw files are committed.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name, so they only need updating in one place.
- **`design/canvas` is excluded from ruff** — it generates design artboards and is not shipped code. Roughly 200 of the initial 214 lint errors came from it.
- **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography (en dashes, ×) that those rules flag as homoglyphs.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py` plus the seeding step; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues; `/new-branch` will ask for the issue number rather than looking up its title.
- **`uv.lock` is a universal lockfile.** Compiled packages list many wheels because it resolves for every supported platform. That is deliberate — the Phase 4 three-platform build matrix depends on it. Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
