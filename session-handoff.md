# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-08-22

---

## Where things stand

**All three specifications are done.** Phase 0's documentation half is complete; everything remaining is code and data.

Issues #1 (`scaffold`), #2 (`ci`), #3 (`spec-pca`), #4 (`spec-pls`) and #5 (`spec-metrics-cv`) are merged into `dev`. `docs/algorithms/` now holds `pca.md`, `pls-regression.md` and `metrics-and-validation.md`. The schema lives at `src/chemometrics_workbench/models.py` with 20 tests in `tests/test_models.py`, `CONTRIBUTING.md` holds the canonical commands, and CI gates every pull request on Python 3.12 and 3.13.

**#6 (`reference-datasets`) is now the only unblocked feature.** It is also the last thing standing between here and #7 (`reference-values`), which gates the parity harness and therefore every kernel. The whole remaining chain is serial through it.

## Repository

| | |
| --- | --- |
| Current branch | `dev` |
| `main` | behind `dev`; receives a merge only at the end of Phase 0 |
| Open issues | 9 of 14 remaining |
| Feature statuses | 5 × `passing`, 9 × `not_started` |
| Verification | `uv run ruff check`, `ruff format --check`, `mypy`, `pytest` — all green on `dev`, and enforced by CI |

## Active feature

None. Nothing is `in_progress`.

## Next action

Start **`reference-datasets`** — issue #6, priority 6, no dependencies.

```bash
git fetch origin
git checkout -b feature/6_reference-datasets origin/dev
```

**Read the licence terms before committing a single raw file.** This is the one task in Phase 0 whose blocking risk is legal rather than technical: NIR corn, gasoline/octane and Tecator are all widely republished, but "widely republished" is not the same as "redistributable", and the terms differ per dataset. Check each one, record the finding and its source URL next to the data, and if a dataset cannot be redistributed, ship a download script and a checksum instead of the file. A dataset whose terms cannot be established is `blocked`, not committed hopefully.

Nothing downstream can start until this lands, so a dataset that has to become a download script is still forward progress — do not stall the whole chain waiting on one licence answer.

## Waiting on the user

- **GitHub default branch is still `main`.** Now that merges go through pull requests this matters: any pull request opened without an explicit base will target the release line. Change it under Settings → Branches; it cannot be changed from here with the current tools.
- **Dataset redistribution terms** (#6) — see the next action above. This is now live, not hypothetical.
- **Parity against a commercial package** — `PROPOSAL.md` §19 Q4 is unresolved. The EULA is not public; a licence would have to be confirmed and written permission sought before publishing a comparison. Tier 1 parity (R `mdatools`, `pls`, scikit-learn, published literature) is unaffected and is what Phase 0 builds.
- Remaining open questions are in `PROPOSAL.md` §19 — team and pace, funding intent, project name.

## Carried forward from the specifications

Findings that change downstream work, recorded here because they are easy to miss inside long documents.

From #4 (PLS):

- **The Jackson–Mudholkar SPE limit does not transfer from PCA to PLS.** PLS components are not eigenvectors of the covariance of X, so there is no residual eigenvalue sequence to sum. PLS uses a χ² moment match on the calibration residuals instead. See `pls-regression.md` §9.
- **SNV and MSC cannot be folded into exported coefficients**, because both depend on the sample being predicted. An exported model carries a residual preprocessing chain plus coefficients, not always a bare coefficient vector. This shapes the export format in #14 and `PROPOSAL.md` §9. See `pls-regression.md` §7.

From #5 (metrics and validation):

- **Our shuffle is not scikit-learn's.** We fix `numpy.random.default_rng` (PCG64); `check_random_state` gives a legacy `RandomState`. The same seed produces different folds. **The parity harness (#8) must pass our resolved fold indices to sklearn as an explicit `cv` iterable** — seeding both with 42 and comparing is an invalid test, and it will look like it works.
- **RMSEC divides by `n` here.** Degrees-of-freedom corrections live in SEC and SEP. Packages writing `n − A − 1` under the name RMSEC are reporting our SEC, and that is the expected third-significant-figure mismatch in #7's reference values — check the definition before recording a discrepancy as a defect.
- **Fold aggregation is pooled residuals, not the mean of per-fold RMSEs.** Any reference value taken from a paper needs its aggregation rule established before it is comparable.

## Gotchas that would otherwise waste time

- **Setup is `uv sync`.** Verification commands are in `CONTRIBUTING.md`; `clean-state-checklist.md` refers to them by role rather than by name, so they only need updating in one place.
- **`ruff format --check` also formats Python blocks inside markdown.** A fenced `python` snippet in `docs/` with cosmetic alignment fails CI. This bit #5.
- **The venv has no numpy.** `uv run --with numpy python -c ...` is the way to check a numerical claim before writing it into a document.
- **`design/canvas` is excluded from ruff** — it generates design artboards and is not shipped code. Roughly 200 of the initial 214 lint errors came from it.
- **`RUF001`–`RUF003` are ignored** because prose in docstrings uses real typography (en dashes, ×) that those rules flag as homoglyphs.
- **`.claude/` and `openspec/` are gitignored** by decision. The `new-branch` and `commit` skills are local-only and will not appear on a fresh clone.
- **`design/canvas/chemometrics-workbench-screens.html` is gitignored** — a 2.4 MB generated file. Regenerate with `design/canvas/build.py` plus the seeding step; never hand-edit it.
- **`gh` CLI is not installed.** Use the GitHub MCP tools for issues and pull requests; `/new-branch` will ask for the issue number rather than looking up its title.
- **Merging pull requests works** as of 2026-08-22, after the token was granted Pull requests: read and write. Merge queue permission is a different thing and does not grant this.
- **CI takes about 20 seconds** per matrix job. Check runs appear a few seconds after a push, so a status read immediately after pushing will show `queued` or `in_progress`.
- **`uv.lock` is a universal lockfile.** Compiled packages list many wheels because it resolves for every supported platform. That is deliberate — the Phase 4 three-platform build matrix depends on it. Do not restrict `[tool.uv] environments`.
- `main` is the release line. There is no `master`, despite it being mentioned in conversation.

## Published artifacts

- Proposal — https://claude.ai/code/artifact/3d5f2071-b55a-4197-aa97-409146fbb488
- Screen designs — https://claude.ai/code/artifact/faf2683c-ea9a-45b4-931e-0182ad236d62

Republish to the same URL when the underlying document or artboards change, so the two do not drift.
