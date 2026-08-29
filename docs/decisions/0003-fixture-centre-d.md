# 0003 — The Phase 1.1 fixture's `centre_d` stands; §9 is what the executor follows

**Date:** 2026-08-29
**Status:** accepted
**Issue:** [#97](https://github.com/millermuttu/Chemometrics-Workbench/issues/97)
**Decides:** the exception to "the fixtures are the contract and the numbers may not change",
stated in `tests/fixtures/contract/README.md`.

---

## Decision

**The fixture keeps the array it published, and the executor keeps disagreeing with it.**

`tests/fixtures/contract/spectra.json` serves `centre_d` — the `MeanCentre` node *below*
`split_d` — as a mean centring fitted on all 240 samples. `metrics-and-validation.md` §9 says
every node downstream of a split is refitted on the training fold alone, and the array such a
node displays is assembled out of fold, each sample taking the row from the fold that held it
out. The executor does that. The two differ, and they will keep differing.

`pca_d` is fitted on `centre_d` and diverges for the same reason. It is the second casualty and
is covered by this record.

## Why this way round

The fixture is the **Phase 1.1 record** — what the frontend was built against — and §9 is the
**specification**. Where they disagree, the specification is what the application follows and the
record stays a record. Regenerating the fixture would make a historical artefact agree with a rule
it predates, which is tidier and less true.

Two practical facts settled it as much as the principle:

- **The generator is gone.** `stub/generate_fixtures.py` was deleted with the rest of `stub/` in
  [#89](https://github.com/millermuttu/Chemometrics-Workbench/issues/89), and the fixtures became
  frozen regression inputs at that moment. Regenerating now means rebuilding a generator first.
- **The executor is not in doubt.** Every other array in the fixture pipeline — `source`, `snv`,
  `centre_a`, `msc`, `centre_b`, `savgol`, `autoscale_c`, `snv_savgol`, `split_d` — is reproduced
  to the fixture's own 6-decimal rounding. This is the one array where the rule bites, not a
  symptom of a shaky implementation.

## The measurement, which is the point

For sample 0 of `centre_d`, against the fixture's published row:

| computed as | max abs difference from the fixture |
| --- | --- |
| `MeanCentre` fitted on all 240 samples | 4.9e-07 — the fixture's own 6-decimal rounding |
| out-of-fold, §9 | 1.2e-03 |

Both are computed in
`tests/test_executor.py::test_a_node_below_a_split_is_refitted_per_fold_which_the_fixture_is_not`,
which asserts the fixture holds the fit-on-everything number *and* that the executor differs from
it by more than 1e-4, then reproduces the executor's own value from fold arithmetic done by hand.
`test_pca_d_diverges_from_the_fixture_for_the_reason_centre_d_does` does the same for `pca_d`.

The divergence is therefore a **measured fact that a test would notice changing**, not a
remembered one. That is what makes leaving it safe: nothing can drift quietly toward the fixture
or away from §9 without a test saying so.

## What this does not license

It is not a general permission for the executor to disagree with the fixtures. Every other
published number is reproduced, and a new disagreement is a finding to investigate, not a
precedent to cite. The exception is this node, for this reason, with this measurement.

## Rejected

**Regenerate `spectra.json` so `centre_d` carries the out-of-fold array.** It would keep "the
fixture is the contract" literally true, and it was the cheaper option when #97 was filed — before
#89 deleted the generator. It also means regenerating `pca.json`, since `pca_d` is fitted on this
array. Taken today it is more work than the thing it buys, and what it buys is a record that no
longer records what happened.
