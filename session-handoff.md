# Session handoff

Compact state for the next session. **Overwrite this file at the end of every session** — it is a snapshot, not a log. Read it first, then `feature_list.json`, `git log` on `dev`, and the open issues.

**Updated:** 2026-09-18

---

## Where things stand

**Phase 2 is complete and released.** `main` is tagged `v0.7.0`. Every entry on the phase's list
passed with evidence, and its exit criterion is recorded in `docs/phase-2/exit-run.md` rather than
argued: the whole workflow driven over HTTP against an independent PLS on the experiment's own
resolved folds. The completed list is archived at `docs/phase-2/feature_list.json`; `feature_list.json`
is now Phase 3's.

**What Phase 2 added**, all merged through green pull requests on 2026-09-18:

| Feature | Issue | PR |
| --- | --- | --- |
| One version source, an unrun node says so, no `stale` state the server never sends | #181 | #189 |
| Estimator and split nodes edited in the inspector; PLS models a real column | #182 | #190 |
| Phase 2 entered on the list, with the two decisions the review left open | — | #191 |
| Jackson–Mudholkar limit with `h0 <= 0` returned with its caveat | #71 | #192 |
| Train/test split executes, and the canvas offers it | #183 | #194 |
| VIP and the folded coefficient vector are drawn | #184 | #198 |
| The experiment record carries a regression's metrics | #188 | #199 |
| The exit criterion demonstrated on Tecator | #201 | #202 |
| Two-class PLS-DA: specification, kernel, parity claims, screen | #185 | #203 |
| Contribution plots, and the macOS tab-overflow flake | #186, #168 | #204 |
| The Bruker OPUS reader | #187 | #205 |
| A run holds its frontier, not every array it has computed | #176 | #206 |

**Three decisions taken during the phase**, recorded in the archived entries rather than only here:

- **#71.** A non-positive `h0` returns the limit *with a caveat* naming it. Clamping (what `mdatools`
  does) hides that the assumption failed; raising refuses a plot for a dataset that is otherwise fine.
- **The exit criterion.** "Matches reference software within stated tolerance" means the workflow over
  HTTP compared against an independent PLS *on the experiment's own resolved folds*, within the parity
  tolerances. Kernel parity alone is not it — the executor's fold handling is exactly what kernel
  parity cannot see, and #173 was that.
- **PLS-DA is two-class only.** One dummy column is PLS1, which the existing kernel fits and the
  existing parity claims cover. Three classes are PLS2, which `pls-regression.md` §10 defers, and are
  refused by name rather than reduced.

## Phase 3 so far

| Feature | Issue | PR | State |
| --- | --- | --- | --- |
| Every run kept, and one opened to its record | #209 | #210 | merged |
| The model artifact, one file readable without this application | #211 | #212 | merged |
| A plain JSON model and a standalone prediction snippet | #213 | #214 | merged |
| Two experiments compared step by step | #215 | #216 | merged |

**#217 is open against a mistake this session made.** `feature/215_lineage-comparison` was cut from
`feature/213_json-and-snippet-export` rather than from `dev`, so #216 carried the export commit into
`dev` and #214 then merged as a no-op returning the same sha. Both are on `dev` and the content is
unaffected; what the mistake cost was an hour of treating #214 as blocked when it was already going
to land through another pull request. **Cut every branch from a freshly pulled `dev`, and check with
`git merge-base --is-ancestor origin/dev <branch>` before opening the pull request.**

## Current work

**Nothing is `in_progress`.** `lineage-comparison` passed with evidence and merged through #216, all
six checks green. Three Phase 3 entries remain: `model-registry`, `html-report`, `phase-3-exit-run`.

**#217 is the one open issue** and it is a process correction, not code: this file and #216's body
both claimed an ancestry the history does not show. This file is fixed; the pull request body is not,
and is not worth rewriting.

**Untracked in the root, still not decided:** `AGENTS.md` (a Codex copy of `CLAUDE.md` that will
drift), `.codex/` and `tecator.csv`. Either gitignore them or commit them; leaving them is what makes
every `git status` lie a little.

**`./run.sh` is the way in.** It syncs, installs with **pnpm** — this project has no
`package-lock.json` and `npm ci` refuses it — builds the bundle if there is not one, and serves,
printing `http://127.0.0.1:<port>/?token=<token>`. `--build` forces the rebuild a changed
`frontend/src` needs.

**Two scripts worth knowing.** `uv run python -m tests.exit_run` drives the served application end to
end and rewrites `docs/phase-2/exit-run.md`. `uv run python -m tests.memory_probe 6000 1200` prints
the peak resident memory of a ten-fold branch, which is the number #176 is judged by.

## Next action

**Pick up `model-registry`.** It is next because it is what gives the artifact a home: `artifact.py`
takes a path and returns a hash, and recording that in the project's `model` table is deliberately
not its job. `docs/model-artifact.md` §9 says so by name, and `docs/model-export.md` §6 leaves
prediction on new data inside the application to a separate feature, so the registry is the whole of
what is next rather than the start of something wider.

## What Phase 3 has added worth knowing

- **The export splits the chain at the last unfoldable step.** Everything after it folds into the
  coefficient vector; everything up to and including it is carried as a residual chain. SNV, MSC and
  normalise are written out; a baseline is refused by name rather than approximated.
- **The export tolerance is `rtol = 1e-4` and the number is measured, not chosen.** The store is
  float32 and an export computes float64, so `docs/phase-2/exit-run.md`'s 7.09e-6 relative difference
  is the floor. `docs/model-export.md` §5 says so, and says what to do if the store ever goes float64.
- **An artifact is refused by schema version, the way `db.py` refuses a newer database.** A newer
  writer may have added a field whose absence this reader would take as a default.
- **The lineage diff matches nodes by id, so a renamed node reads as a remove plus an add.** That is
  the honest reading: nothing in the model says a rename is not a replacement.
- **A branch cut from another feature branch carries that feature into the pull request.** #216
  merged #213's export commit as a side effect, which nothing in the pull request said.
- **`count()` does not wait.** A Playwright assertion of the form `expect(await x.count())` compares a
  frame rather than a state, and that is what failed on the Windows runner in #210. Use
  `expect.poll(() => x.count())`.

## What Phase 2 left worth knowing

- **A train/test split is one `Fold` and not a partition.** `validate_partition` refuses it,
  `_State.display` returns the one array for a single fold, and `_pls` computes the cross-validated
  block only for more than one fold. `stratify_by` is refused by name until a class column exists.
- **A classification is the regression on a dummy response.** `_fit_pls1` is shared, `task` tells them
  apart, and every model quantity — scores, loadings, VIP, both diagnostics — means the same thing for
  both. The analysis tab keeps every regression panel and swaps one.
- **The seeded e2e project now holds sixteen nodes**, not fifteen: `plsda_d` sits beside `pls_d` below
  the split, and the seeded Tecator CSV carries a `fat_class` column. Five counts in `pipeline.spec.ts`
  and `inspector.spec.ts` moved with it.
- **The `runs` e2e project's last two tests rewrite its pipeline** through the API — one swaps the
  k-fold for a train/test split and drops the failing branch, the other adds a mean-centre-only branch
  so the raw-axis coefficients have a foldable chain to fold. Anything added to that file after them
  sees the rewritten graph.
- **`Run.displays` is a mapping over the store, not a dict of arrays.** A `Run` sits in the job table
  for the life of the process; one holding arrays kept a whole run resident after it finished.
- **An `id()`-based dedupe of a split's repeated array is wrong.** A freed array's address is reused,
  so folds pair with the wrong file. Tried, caught by `test_executor.py`, reverted.

## Older lessons still worth not rediscovering

- **`pnpm build` typechecks `src/__tests__`.** A "prove the test fails on the unfixed code" run that
  reverts the source but keeps the new unit tests fails to *build* and never runs Playwright, and the
  grep afterwards prints nothing — which looks like a pass of the wrong kind. Revert the tests too.
- **Outline buttons are named label plus dim.** `getByRole("button", { name: "SNV", exact: true })`
  matches nothing, because the row's accessible name is `SNV snv`. Use a regex anchored at the start.
- **A regression test is trusted only after it fails on the unfixed code.** Every test added in Phase 2
  was run that way first, and two of them failed for the wrong reason until the setup was fixed.
- **"It changed" is not the claim "it is what I set it to".** #162 shipped green and did nothing,
  because its test asserted a stored position was *not* the pre-drag value and a generated fallback
  satisfies that too.
- **Never `git stash -u` with a pathspec in this checkout.** On 2026-09-17 an untracked `.agents/`
  disappeared that way and the stash's untracked tree was empty.
- **A stray `pkill` takes Playwright's own web servers with it.** `fuser -k -n tcp 8765 8766 8767 8768`
  before a run, never during one.
- **This project uses pnpm**, and `run.sh` said `npm ci` until #160 and failed on any clean checkout.
  Delete the artefact before verifying the code that builds it.

## Three canvas traps, all of which cost time

- **A node's label is built from its parameters.** Editing `SG d1 w11` to a window of 9 renames it
  `SG d1 w9`, and `snv_savgol` — same window, same label — inherits the old name alone. Re-finding an
  edited node by the name it used to have finds a *different* node.
- **Making nodes draggable breaks two things silently.** A drag ends with a click on the node it moved,
  which opens its tab, so the drag has to consume that click; and any header button becomes a drag
  handle without React Flow's `nodrag` class.
- **`fitView` re-fits the viewport on every mount**, so comparing a node's screen box either side of a
  reload compares two zoom levels and fails on a change that did not happen.
