# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

Setup is complete. Phase 0 has not started — no source code exists yet.

The specification (`PROPOSAL.md`), the UI design (`design/`), the data model (`design/data_models.py`), the task list (`feature_list.json`) and the working protocol (`CLAUDE.md`) are all written, committed and pushed. The next session writes the first Phase 0 deliverable.

## Repository

| | |
| --- | --- |
| Current branch | `dev`, in sync with `origin/dev`, working tree clean |
| `main` | 4 commits behind `dev`; holds `PROPOSAL.md`, `CLAUDE.md`, `.gitignore` |
| Open issues | 14, all Phase 0, none started |
| Feature statuses | 14 × `not_started` |

## Active feature

None. Nothing is `in_progress`.

## Next action

Start **`spec-metrics-cv`** — issue #5, priority 1, no dependencies.

```bash
git fetch origin
git checkout -b feature/5_metrics-cv-spec origin/dev
```

Deliverable is `docs/algorithms/metrics-and-validation.md`. Documentation only, no code. It blocks #7, which blocks the entire parity chain, and it is the decision that will explain most future "why does this not match Unscrambler" reports.

After it: #4 (`spec-pls`), #3 (`spec-pca`), then #1 (`scaffold`) — which is independent and can run in parallel whenever convenient.

## Waiting on the user

- **GitHub default branch is still `main`.** It should be `dev`, or pull requests will target the release line by default. Repository setting; cannot be changed from here with the current tools.
- **Dataset redistribution terms** (#6) need checking per dataset before raw files are committed.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Gotchas that would otherwise waste time

- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — it is a 2.4 MB generated file. Regenerate it with `design/canvas/build.py` plus the seeding step, never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues; `/new-branch` will ask for the issue number rather than looking up its title.
- **No virtualenv yet.** System Python is 3.13.9 with pydantic 2.12.4, which is enough to run `python3 design/data_models.py`. Issue #1 replaces this with `uv`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
