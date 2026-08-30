# 0002 — Phase 1 runs as three sub-phases, and 1.1 is a walking skeleton rather than a frontend layer

**Date:** 2026-08-23
**Status:** accepted
**Issues:** [#40](https://github.com/millermuttu/Chemometrics-Workbench/issues/40) opens the list, [#53](https://github.com/millermuttu/Chemometrics-Workbench/issues/53) is the stub server the decision turns on.
**Decides:** the shape of `PROPOSAL.md` §16's Phase 1. It does not change what Phase 1 delivers or its exit criterion.

---

## Decision

**Phase 1 is split into three sub-phases:**

| | Scope |
| --- | --- |
| **1.1** | The React shell and the core screens, over a **stub server**: FastAPI on `127.0.0.1`, token-authenticated, serving fixtures generated from the real kernels at the endpoint paths 1.2 will implement. No executor, no database, no persistence. |
| **1.2** | Readers, the pipeline executor, real jobs and the HTTP surface — replacing the stub handlers one at a time, behind unchanged URLs. |
| **1.3** | SQLite in each project directory, and the two halves joined. |

**1.1 is a walking skeleton, not a frontend layer.** The frontend speaks HTTP from its first commit and never imports a fixture file.

## Why the sub-phases

Phase 1 is four engineering-months against Phase 0's two, and its parts have unequal risk. The kernels already exist and are parity-verified, so the backend is largely *known* work; the interface is the unknown, and it is also the part that is already designed — `design/canvas/` fixes the palettes, the type, the node geometry and every run state. Starting where the specification is densest and the risk is highest is the ordering that learns fastest.

Three sub-phases also give the solo maintainer a merge point every few weeks instead of one at four months.

## Why 1.1 is not simply "the frontend"

The first version of this plan was a horizontal slice: a complete frontend over imported fixture files, with nothing integrated until 1.2. That was rejected.

**Layer-first defers all integration to a single moment,** which is where the defects are. It is also at odds with the name `PROPOSAL.md` §16 already gives Phase 1 — *walking skeleton*, a thin slice through every layer that actually runs, thickened afterwards.

**Static fixtures cannot exercise the three things most likely to be wrong:**

1. **A job that takes time.** Fixtures return instantly and always succeed. Progress that advances, a run cancelled mid-flight and a status bar with something real to poll are the whole of `PROPOSAL.md` §11's *long-running work* requirement, and none of it is testable against a canned payload.
2. **A request that fails.** The failed state in [#49](https://github.com/millermuttu/Chemometrics-Workbench/issues/49) needs a server that can return an error. A state reachable only by editing code is not tested.
3. **A response with real bulk.** The decimated spectra payload is the largest thing crossing the wire, and its size is the constraint behind §13.

**The contract argument cuts both ways.** Generating fixtures from `models.py` fixes the *domain* shapes but not the *API* ones — pagination, error envelopes and the job lifecycle appear nowhere in the schema. Serving them over the real paths is what turns a guess into something 1.2 has to either meet or consciously change.

## What it costs

Roughly one module. The fixtures are generated either way ([#41](https://github.com/millermuttu/Chemometrics-Workbench/issues/41)); the stub server routes and delays them, and computes nothing. In-memory state only — building persistence or an executor here is exactly the scope creep the sub-phase split exists to prevent, and every handler names the 1.2 issue that replaces it.

**One thing is deliberately not a stub.** The token check is real from the first commit. §4.3 hardening is a Phase 4 deliverable, but an unauthenticated localhost server never gets authentication retrofitted, and the ephemeral-port-plus-token bootstrap is cheap only while nothing depends on its absence.

## What was considered and rejected

- **One undifferentiated Phase 1.** Four months to a single merge, with the riskiest work — the UI — reached last.
- **Backend first, frontend after.** Defers every design question to the end, wastes the fact that the artboards are already drawn, and produces an API shaped by what was convenient to implement rather than by what a screen needs.
- **Frontend only, fixtures imported directly.** The version this record rejects, for the reasons above.
- **An OpenAPI document as the contract instead of a running stub.** It would pin the shapes without exercising the behaviour, which is the half that matters here. A stub costs about the same and does both.

---

## Decisions taken alongside, carried into 1.2 and 1.3

Recorded here because they were taken in the same session and will otherwise be re-argued from preference. Neither rests on measurement yet; when one does, it earns its own numbered record.

**All four database decisions below were implemented as written in Phase 1.3** (#119–#124, tagged `v0.4.0`), and none of them was departed from — so there is no later record to read instead of this one. `src/chemometrics_workbench/db.py` is the database per project directory, its tables carry identity plus queried columns plus a `document`, there is no Alembic, and the float32 store boundary is where #77 put it. What 1.3 added rather than changed: a directory written before the database is read into one on the way past (#121), and the executor's `cache.json` — a map of references — joined the database rather than staying a file (#122).

**One SQLite database per project, inside the project directory** — `<project_dir>/project.db`. `PROPOSAL.md` §11 requires that a project directory can be zipped and sent to a colleague; a central application-level database breaks that, because the metadata would stay behind. Known project paths live in a small JSON registry in the user's config directory, not in a second database.

**Tables hold identity, the columns that are actually queried, and the Pydantic model as JSON.** `models.py` is the schema of record and its invariants are tested. Mirroring twenty Pydantic classes into SQLAlchemy columns creates two sources of truth that drift.

**No Alembic in Phase 1.** `create_all` plus a `PRAGMA user_version` check that refuses to open a newer schema. Migrations arrive when the first schema change ships to someone who has real projects on disk, which is Phase 2 at the earliest.

**Arrays are stored float32 on disk and upcast to float64 at the kernel boundary.** §13's envelope — ~20,000 × 4,000 at roughly 320 MB — is the *stored* form. `arrays.py` is a float64 contract, so compute peaks at about twice the stated figure, and the documentation should say so rather than quoting 320 MB as the whole cost.

**The canvas ships read-only in 1.1**, with pipelines built through a step list; direct manipulation is [#51](https://github.com/millermuttu/Chemometrics-Workbench/issues/51) against Phase 2. The Phase 1 exit criterion runs one linear path — SNV, Savitzky–Golay, PCA — and never needs a branch created by hand. Branch *editing* is what Phase 2 first exercises, when PLS variants are compared against each other.
