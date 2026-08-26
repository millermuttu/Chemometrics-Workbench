# Chemometrics Workbench — Design Brief

For Claude Design. This brief is self-contained; do not read `PROPOSAL.md` for design purposes — it is a justification document aimed at reviewers, not a specification of the interface.

---

## 1. What we are designing

A desktop-class chemometrics application that runs in a browser window on the scientist's own machine. Spectroscopists load spectral datasets, build preprocessing and modelling pipelines, compare dozens of model variants, and export the winner.

It replaces expensive closed desktop software (The Unscrambler, SIMCA, OPUS). It must feel like serious instrument software — precise, dense, calm — not like a consumer web product.

**Design the application shell and its core working screens.** Not a marketing site, not a landing page.

## 2. Who uses it, and how

An analytical chemist or spectroscopist in a university or research lab. Comfortable with statistics, not necessarily with programming. Uses a large monitor, often two. Works in sessions of hours, not minutes. Frequently has fifteen model variants open across an afternoon and needs to remember which is which.

Consequences for the design:

- **Density is correct here.** Whitespace generous enough for a marketing page is wasted space that costs the user a scroll. Aim for the density of a professional analysis tool.
- **Desktop-first, large screens.** Minimum 1440px design width. No mobile layouts. Tablet is out of scope.
- **Long sessions mean low-fatigue color.** No high-saturation fills across large areas. Dark theme is not an afterthought — many labs run dim.
- **The user's data is the hero.** Chrome recedes; spectra, scores and residual plots dominate.

## 3. The mental model

```
Project  ──  the workspace, one per study
   │
   ├── Datasets        spectra + metadata, versioned, identified by content
   │
   ├── Pipeline        a node graph: dataset → preprocessing → model → validation
   │
   ├── Experiments     executed runs of a pipeline, with metrics
   │
   └── Models          fitted, saved, comparable, exportable
```

The pipeline graph is the central object in the application. Everything the user does happens by building and running one. This must be visually obvious.

## 4. Application shell

A single window, three regions, with a tabbed document area in the middle.

```
┌────────────────────┬─────────────────────────────────────────────┬──────────────────┐
│ ◈ Corn NIR study   │  ⊞ Pipeline │ SNV ×│ PCA scores ×│ Model C ×│  Inspector       │
│                    ├─────────────────────────────────────────────┤                  │
│ ▾ Datasets         │                                             │  Node: PLS       │
│    corn_raw   80×700│                                            │  ───────────     │
│    corn_v2    78×700│           main document area               │  Components  6   │
│                    │        (canvas / plots / tables)            │  Algorithm NIPALS│
│ ▾ Pipeline         │                                             │  Scaling  centre │
│    ├ corn_raw      │                                             │                  │
│    ├ SNV           │                                             │  Metrics         │
│    ├ SG d1 w11     │                                             │  RMSECV   0.389  │
│    └ PLS  6 LV     │                                             │  R²       0.981  │
│                    │                                             │  Bias    -0.004  │
│ ▾ Experiments  12  │                                             │                  │
│ ▾ Models        4  │                                             │  Provenance ▸    │
└────────────────────┴─────────────────────────────────────────────┴──────────────────┘
                              ▲ status bar: job progress, cancel
```

**Left sidebar — navigation and structure.** Project tree: datasets, the pipeline as a collapsible node outline, experiments, models. Resizable, collapsible to icons. This is the outline of the graph, not the graph itself.

**Main area — tabbed documents.** Each preprocessing step, analysis result, dataset view and model detail opens as a tab. Tabs are reorderable, closable, and can be split side by side for comparison. **Selecting a node — in the canvas or in the sidebar outline — focuses its tab, opening it if needed.** This is the mechanism that ties the graph to the pages.

**Right sidebar — inspector.** Context-sensitive to whatever is selected: node parameters, model metrics, dataset metadata, provenance. Editing a parameter here re-runs downstream and marks affected nodes stale. Collapsible.

**Status bar.** Running job name, progress, elapsed time, cancel. Cross-validation runs take minutes; this must never be a blocking modal.

### Tab proliferation

Fifteen preprocessing variants means many tabs. Design for it:

- Tabs derived from graph nodes carry the node's icon and short label (`SNV`, `PLS 6LV`, `PCA scores`).
- Unpinned result tabs are transient — single-click preview replaces the transient tab, double-click or edit pins it. (Same pattern as a code editor.)
- Show an overflow menu when tabs exceed the bar, with search.

## 5. The pipeline canvas

The signature screen. A pan/zoom node graph, opened as a document tab.

**Node types**, visually distinct by shape or header color, not by icon alone:

| Type | Examples | Notes |
| --- | --- | --- |
| Source | `corn_raw  80×700` | Always leftmost. Shows dimensions. |
| Preprocessing | `SNV`, `MSC`, `SG d1 w11`, `Range 1100–2500` | The most common node. Compact. |
| Split | `K-fold 10`, `Train/test 70:30 seed 42` | Shows the seed — reproducibility is visible. |
| Model | `PLS 6 LV`, `PCA 5 PC`, `PLS-DA` | Larger; shows headline metric when run. |
| Output | `Model C`, `Report` | Terminal nodes. |

**Node states** must read at a glance, encoded in form as well as color: *not yet run* (outlined, muted) · *queued* · *running* (progress in the node) · *complete* (headline metric shown) · *stale, upstream changed* (hatched or dimmed with a marker) · *failed* (error stripe, message on hover).

**Branching is the point.** The canvas exists so a user can fork one dataset into competing preprocessing paths and compare the results. Design the branch case as the primary case, not the exception:

```
                    ┌── SNV ──────── PLS 6LV ──► Model A   RMSECV 0.412
   corn_raw ────────┼── MSC ──────── PLS 6LV ──► Model B   RMSECV 0.418
                    ├── SG d1 ────── PLS 5LV ──► Model C   RMSECV 0.389
                    └── SNV + SG ─── PLS 5LV ──► Model D   RMSECV 0.381
```

Support: drag from a node's output port to create a branch; duplicate a subgraph; select two terminal nodes and open a comparison tab.

**Edges** are simple orthogonal or gentle bezier connectors, thin, low contrast — they are structure, not decoration. Animate flow only while a run is in progress, and respect `prefers-reduced-motion`.

## 6. Screens to design

Priority order. Design the first four fully; the rest as needed.

1. **Pipeline canvas** — as above, showing a four-branch comparison mid-run: one node complete, one running, one stale, one not yet run.
2. **Spectra view** — the preprocessing preview. Raw vs processed spectra overlaid, wavelength axis, sample selection and highlighting, a spectrum count, and the density-band treatment for large sets. This is the most-looked-at screen in the product.
3. **Analysis results** — PCA or PLS output: scores plot with Hotelling T² ellipse, loadings, explained variance, predicted-vs-actual with the 1:1 line, residuals. Multiple plots in one tab; the layout question is how they coexist without becoming a cramped dashboard.
4. **Experiment comparison** — a sortable table of runs with their differing steps called out, plus small-multiple plots. The feature that beats working in a notebook: *show exactly which steps differ between two models.*
5. **Model detail** — metrics, coefficients, VIP, full provenance record, and the export actions.
6. **Project / dataset view** — sample and variable table, metadata columns, exclusions, import summary.
7. **Empty project** — what a brand new user sees before importing anything.

## 7. Plot rules

Plots are the product. Treat them with the same care as type.

- **Colorblind-safe categorical palette.** Spectra get overlaid in groups; color is load-bearing information, not decoration. Never rely on red/green contrast alone.
- **Large sets get a density band, not 20,000 traces.** Show the distribution as a shaded envelope with selected or highlighted spectra drawn at full resolution on top. Decimation is a design constraint, not a technical detail to hide.
- **Tabular numerals everywhere numbers align.** Metrics tables, coefficients, sample lists.
- **Axes always labelled with units** — wavelength (nm) or wavenumber (cm⁻¹), absorbance. Scientists will notice their absence immediately.
- Faint grid, no chart junk, no drop shadows on data marks, no 3-D anything.
- Every plot needs a hover readout showing sample identity — clicking an outlier to find out which sample it is, is a core workflow.

## 8. States to design

- **Empty** — new project, no dataset. This is a first-run moment; make the next action obvious.
- **Importing** — file parsing, with a preview of what was detected (wavelength range, sample count, delimiter, metadata columns) before committing.
- **Running** — a ten-fold cross-validation taking two minutes. Non-blocking, cancellable, progress visible in the node, the tab and the status bar.
- **Stale** — a parameter changed upstream; downstream results are no longer valid but should not vanish. Show them dimmed with a re-run affordance.
- **Failed** — a reader that could not parse a file, a model that did not converge. Specific message, never a stack trace, with the offending file or setting named.
- **Overloaded** — a dataset beyond the supported envelope. State the limit honestly rather than freezing.

## 9. Visual direction

**Feel:** a precision instrument. Calm, dense, confident. Closer to a modern data or observability tool than to either a consumer SaaS app or a legacy scientific desktop application.

**Reference points to move toward:** the restraint of well-made professional audio and CAD software · the information density of a good trading or observability dashboard · the typographic care of a scientific journal.

**Move away from:** the dated Qt-widget look of existing open-source chemometrics tools · consumer-SaaS gradient heroes, oversized rounded cards, playful illustration · anything that makes a 700-variable spectrum feel like a marketing chart.

**Type:** a technical, highly legible sans for the interface, with real tabular numerals. Consider a distinct mono for data, parameters and identifiers, so a wavelength or a seed is visually separable from prose. Avoid Inter and Space Grotesk as defaults — they read as unconsidered.

**Color:** a restrained neutral ground with one accent, plus a categorical data palette that is separate from the accent and colorblind-safe. Semantic colors for run state (queued / running / complete / stale / failed) must be distinguishable from data colors — a failing node must not be mistaken for a red spectrum.

**Both themes required.** Light and dark, each designed rather than inverted, both with legible plots.

## 10. Non-goals

No mobile or tablet layouts · no marketing or landing page · no onboarding tour or gamification · no account, login, billing or sharing UI — the application is local and single-user · no cloud or sync affordances anywhere in the interface.

## 11. Open design questions

**§11.1 is settled.** *How do plots coexist within a single analysis tab?* **A titled panel grid, in one tab** — the answer `AnalysisResults.dc.html` drew and #48 built: scores, loadings, explained variance and diagnostics, each in a 24px-titled panel on `--surface`, two per row at 1440. Not a resizable split and not user-arranged panels: at the design width the grid fits without a scroll, and every panel keeps its title so a screenshot of one is self-describing. The row is laid out so Phase 2's predicted-vs-measured panel arrives beside the others rather than replacing them.

Remaining:

1. ~~How do plots coexist within a single analysis tab?~~ **Settled above.**
2. When two models are compared, is that a third tab or a split of two existing tabs?
3. Does the canvas need groups or subgraphs once a project has forty nodes, or is pan/zoom plus the sidebar outline enough?
4. Where do dataset *versions* appear — as nodes on the canvas, or only in the sidebar tree?
