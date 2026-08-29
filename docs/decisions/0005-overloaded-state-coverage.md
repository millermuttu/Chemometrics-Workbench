# 0005 — The beyond-the-envelope screen is covered as a component, and the end-to-end gap is recorded rather than closed

**Date:** 2026-08-29
**Status:** accepted
**Issue:** [#109](https://github.com/millermuttu/Chemometrics-Workbench/issues/109)
**Decides:** what replaces the stub's `?oversize` affordance, retired in #89.

---

## Decision

`frontend/src/__tests__/overloaded.test.tsx` renders `Overloaded.tsx` and the guard in
`SpectraView.tsx` through `react-dom/server`, and asserts the notice a user reads: which bounds
were crossed, what the dataset costs, what the limit is, the two ways back inside it, that it is
announced with `role="alert"`, and that the plot view is never entered.

Option A from the issue — importing a dataset that really is past the envelope — is **not**
implemented, and no substitute for it is either. The gap is recorded here: **nothing in the suite
proves that a real oversize import reaches the notice.**

## Why the state cannot be reached from a small input

PROPOSAL.md §13's envelope is **reported, not enforced** — #81. `frontend/src/states/envelope.ts`
computes it from the `n_samples` and `n_variables` a `DatasetVersion` records, and the server's
honest behaviour is to report the shape it read. So the only genuine way into this state is a
dataset of about 42,000 × 6,200: a gigabyte as float32, and several times that as the CSV that
becomes it. There is no small honest input, because the state is a function of size alone.

The stub had one because it was allowed to lie. `?oversize` fabricated the shape, which is exactly
the behaviour #89 removed — a server reporting a shape it did not read.

## Why not seed a `DatasetVersion` past the envelope

Option B — a record whose shape exceeds the envelope and whose `array_path` points at nothing —
would render the notice with no gigabyte anywhere. It is rejected because the application can
never produce such a record: every version the importer writes has an array behind it. A test that
seeds one proves the screen renders for a project state that cannot exist, which is the fixture
wearing a real project's clothes that #89 spent its length removing.

## Why `react-dom/server` and not jsdom

This state draws nothing and runs no effects — that is its entire purpose. Its output *is* static
markup, so `renderToStaticMarkup` sees all of it. `react-dom` is already a dependency; jsdom and a
testing library would be two new ones for a string the test already has.

The one stub is `plotly.js-gl2d-dist-min`, which reads `self` when it loads. `SpectraView.tsx`
imports it at module scope and, past the envelope, never calls it — the guard returns first. The
stub is what lets that be asserted outside a browser; it replaces a browser global, not a project
state.

## What is claimed, and what is not

| claim | where it lives |
| --- | --- |
| The thresholds and the megabyte arithmetic | `envelope.test.ts` |
| The notice's wording, its bounds and its `role="alert"` | `overloaded.test.tsx` |
| The plot screen routes to the notice instead of mounting the plot | `overloaded.test.tsx` |
| A real gigabyte import reaches the notice | **nothing — deliberately** |

The last row is the cost of the decision, stated so the next person does not read the absence of
an end-to-end test as an oversight. If a machine-sized import fixture ever becomes cheap — a
memory-mapped array written in place, say, rather than a CSV parsed — option A becomes worth
revisiting.
