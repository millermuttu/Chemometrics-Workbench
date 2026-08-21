# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

Phase 0 has started. The package and toolchain exist and the full verification suite runs green.

Issues #1 (`scaffold`), #2 (`ci`) and #3 (`spec-pca`) are complete and merged into `dev`. The schema lives at `src/chemometrics_workbench/models.py`, its old self-check is `tests/test_models.py` (20 tests), `CONTRIBUTING.md` holds the canonical commands, CI gates every pull request on Python 3.12 and 3.13, and `docs/algorithms/pca.md` fixes the PCA conventions.

Two of the three specification tasks remain. Until #4 and #5 land, #7 is blocked and therefore so is every kernel.

## Repository

| | |
| --- | --- |
| Current branch | `dev` |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 11 of 14 remaining |
| Feature statuses | 3 × `passing`, 11 × `not_started` |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green on `dev`, and enforced by CI |

## Active feature

None. Nothing is `in_progress`.

## Next action

Start **`spec-metrics-cv`** — issue #5, priority 1, no dependencies.

```bash
git fetch origin
git checkout -b feature/5_metrics-cv-spec origin/dev
```

Deliverable is `docs/algorithms/metrics-and-validation.md`. Documentation only, no code. It blocks #7, which blocks the entire parity chain, and it fixes the definitions that will explain most future "why does this not match Unscrambler" reports.

After it: #4 (`spec-pls`). Those two are all that stand between here and #7, which gates every kernel. #6 (`reference-datasets`) also has no dependencies and can run alongside.

`docs/algorithms/pca.md` is the template to follow — formulas rather than names, a table of every reported quantity against its defining section, and a list of known divergences from other packages.

## Waiting on the user

- **GitHub default branch is still `main`.** Now that merges go through pull requests this matters: any pull request opened without an explicit base will target the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Dataset redistribution terms** (#6) need checking per dataset before raw files are committed.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name, so they only need updating in one place.
- **`design/canvas` is excluded from ruff** — it generates design artboards and is not shipped code. Roughly 200 of the initial 214 lint errors came from it.
- **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography (en dashes, ×) that those rules flag as homoglyphs.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py` plus the seeding step; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues and pull requests; `/new-branch` will ask for the issue number rather than looking up its title.
- **Merging pull requests works** as of 2026-08-22, after the token was granted Pull requests: read and write. Merge queue permission is a different thing and does not grant this.
- **CI takes about 20 seconds** per matrix job. Check runs appear a few seconds after a push, so a status read immediately after pushing will show `queued`.
- **`uv.lock` is a universal lockfile.** Compiled packages list many wheels because it resolves for every supported platform. That is deliberate — the Phase 4 three-platform build matrix depends on it. Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
