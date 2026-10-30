# 0007 — Arrays are stored as float64

**Date:** 2026-10-30
**Status:** accepted
**Issue:** [#402](https://github.com/millermuttu/Chemometrics-Workbench/issues/402)
**Supersedes:** #77's float32 store boundary (`PROPOSAL.md` §13, recorded as kept in decision 0002).

---

## Decision

**`write_array` stores float64.** Every array the executor writes - each node's output, each fold's, every source - is float64 on disk, as it already was at the kernel boundary. `read_array` widens whatever it finds, so a project written before v1.3, with float32 arrays, opens and runs unchanged; arrays written from now on are float64 and get new content hashes, so cached results recompute once.

`PROPOSAL.md` §13's envelope is kept at 20,000 spectra × 4,000 variables, which now costs about 640 MB rather than 320 MB.

## Why

Phase 7's exit run (#383) held each model's export to the application's own predictions. For PLS, OPLS and Elastic Net the float32 store was invisible: their predictions moved by less than one part in a million. For Ridge at α 3e-5 and MLR on a 22-variable VIP selection it was not. Their condition numbers are about 1.8e5, so one float32 rounding of an intermediate array (about 6e-8 relative) moved a prediction by about 10⁵ times that: the export, computing in float64, differed from the served prediction by up to 2.7e-5 and 4.3e-3 % fat. The export was right; the application's own number was the one carrying the store's rounding.

Two other fixes were considered and rejected: deriving a tolerance from the condition number (it came to 1.2 % fat for Ridge, a check that could not fail), and refitting only the final model in float64 from the source (served predictions would then disagree with the stored chain the screens show).

## What is kept

**The rank tolerance's data term stays at float32's epsilon** (decision 0004). It describes how far the matrix may be from the one the user meant, and imported text carries five to seven significant digits; float32's epsilon still describes that, and keeping it moved no PCA rank, no MLR refusal and no QDA refusal.

## Consequences

- The export tolerance (`docs/model-export.md` §5) came down from `rtol 1e-4, atol 1e-6` to the parity predictions class, `rtol 1e-6, atol 1e-9`, and `tests/test_export.py` holds it.
- Tests that rounded their references through float32 to match the store no longer need to; `_as_stored` in `tests/test_executor.py` is now the identity.
- The earlier phases' exit-run records (`docs/phase-*/exit-run.md`) were measured against the float32 store and are left as they were: they are records of what was run.
