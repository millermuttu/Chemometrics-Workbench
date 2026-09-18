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

## Current work

**Nothing is `in_progress`.** No feature branch remains; `docs/close-phase-2` carries this file, the
archived list, the Phase 3 list and the version bump.

**Open issues:** none from Phase 2. #71, #168, #176 and #181 to #188 all closed with their pull
requests.

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

**Pick up `experiment-history`, the first Phase 3 entry.** It is first because the comparison view
(§8.3's "single feature most likely to make a researcher prefer this tool to a notebook") and the
model registry both hang from it, and because the data is already there: the `experiment` table keeps
every run and `read_experiment` returns only the most recently started, which nothing but
`/experiments/current` reads.

Then `model-artifact` and `json-and-snippet-export` — §9's constraint, that exported predictions match
in-application predictions within a stated tolerance verified in CI, is that entry's verification.
`lineage-comparison` can go in parallel with either.

**One thing to decide before the export work.** `docs/phase-2/exit-run.md` records that the float32
array store is visible at the prediction level against a float64 reference (7e-6 relative) and at the
coefficient level on a derivative chain. An exported snippet computes in float64, so it will differ
from the served numbers at that level, and §9's "stated tolerance" has to be stated with that in mind
rather than discovered by a failing test.

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
