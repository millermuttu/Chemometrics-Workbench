# Chemometrics Workbench — Project Proposal

**Version 2 (revised)** · Draft · MIT License · 2026-08-21

> Desktop-class chemometrics through a browser, installed as a single application, with your data never leaving your machine.

---

## 0. Decisions locked in this revision

Version 1 of this proposal left four architectural questions open. They are now decided and everything below follows from them.

| Question | Decision | Consequence |
| --- | --- | --- |
| How is it distributed? | **Single downloadable application.** Frozen Python backend + bundled web UI, launched by double-click. No Python, Node, or Docker install required by the end user. | Packaging and code signing become first-class engineering work, not an afterthought. Compiled scientific dependencies stay available (unlike a WASM approach). |
| License | **MIT** | Maximum adoption, including inside companies. No copyleft obligation, no hosted-service moat. Contributions accepted under the same terms (inbound = outbound). |
| Data scope | **2-D spectra only** (samples × variables) | Hyperspectral image cubes, 3-way data, PARAFAC/MCR are explicitly out of v1. Keeps everything in memory and keeps the data model simple. |
| Primary audience | **Research and academia** | Optimize for teaching, reproducibility, publication and zero-cost adoption. GxP/21 CFR Part 11 features are out of scope; the audit-log design leaves room for them later. |

### Assumptions still requiring confirmation

- **Team and pace.** The roadmap in §16 is expressed in *engineering-months*, not calendar dates, so it holds for any team size. A solo developer working part-time should read ~13 engineering-months as roughly 12–18 calendar months to 1.0.
- **Platforms.** Windows, macOS and Linux are all treated as release targets, with Windows assumed to be the largest share of the audience and the most expensive to package.

---

## 1. Vision

Build open-source chemometrics software with capabilities comparable to closed desktop tools such as The Unscrambler, SIMCA and OPUS, delivered through a modern web interface but installed and run like a desktop application.

Core principles, in priority order:

1. **Local-first** — data is read from and written to the user's own filesystem. No upload, no account, no network required.
2. **Reproducible** — every model carries the complete recipe that produced it.
3. **Open** — MIT source, open file formats, exportable models, no lock-in at any layer.
4. **Modern** — a UI and visualization layer that a 2026 researcher expects, on top of the Python scientific stack they already trust.
5. **Extensible** — algorithms are ordinary scikit-learn-compatible transformers and estimators, so adding one is ordinary Python.

The user experience target: download one file, double-click, the workbench opens, point it at a local spectra file, and complete a full PCA or PLS workflow without ever touching a terminal or a network.

---

## 2. Problem

Chemometrics tooling today splits into three groups, none of which serves the research user well.

**Closed desktop tools** (Unscrambler, SIMCA, OPUS, Analyze IQ) have strong, validated science and keep data local, but are expensive per seat, closed source, hard to extend, hard to script, awkward to integrate with modern ML tooling, and — critically for academia — often unavailable to students and unaffordable to small groups. Models built in them are hard to move anywhere else.

**Web-based analysis tools** offer accessibility and good UI but require uploading the dataset to a third-party server. For proprietary pharmaceutical, food, agricultural or manufacturing spectra this is frequently prohibited outright, and even in academia it creates data-governance friction that stops adoption before it starts.

**Open-source scripting libraries** (see §3) provide excellent algorithms but no workflow. The researcher assembles preprocessing, model fitting, validation and plotting by hand in a notebook, and the analysis context — which preprocessing, which split, which seed, which library version — lives in prose, in file names, or nowhere. Comparing fifteen preprocessing variants means fifteen notebook cells and a spreadsheet of results that nobody can reproduce six months later.

The gap this project fills sits between the second and third groups: **the workflow and visualization quality of a commercial desktop tool, on top of the open algorithms researchers already use, with reproducibility as a built-in property rather than a discipline.**

---

## 3. Prior art — and why this project rather than contributing to one of them

This section exists because it is the first question any reviewer or potential contributor will ask.

| Existing work | What it provides | What is missing for this use case |
| --- | --- | --- |
| Unscrambler / SIMCA / OPUS | Complete, mature, validated chemometrics workflows; local processing | Cost, closed source, no extension path, no model portability, weak reproducibility record-keeping |
| Orange + Spectroscopy add-on | Open source, visual workflow canvas, spectral preprocessing, local execution | Qt desktop app rather than web; workflow canvas is not a project/experiment/model registry; lineage and model export are not first-class; Python-plugin extension model is heavier |
| R: `mdatools`, `pls`, `prospectr` | Excellent, well-documented, reference-quality chemometrics implementations | Scripting only; no application layer; R adoption in industrial labs is narrower than Python |
| Python: `chemotools`, `nippy`, `pyChemometrics` | scikit-learn-compatible spectral preprocessing and PLS tooling | Libraries, not applications; no UI, no project model, no validation workflow, no reporting |
| Hosted web chemometrics services | Accessible, modern UI, zero install | Requires uploading private data — the disqualifying constraint |

**Position:** this project does not aim to reimplement chemometric algorithms. It aims to be the *application layer* — projects, datasets, pipelines, experiments, models, lineage, diagnostics, reporting and export — built on the existing Python scientific stack, delivered as a local application with a web UI. Where a good library exists, it is used. Where the work is an application concern, it is owned here.

Concretely, the combination that does not currently exist anywhere: **desktop-grade chemometrics workflow + web UI + open source + strictly local data + first-class experiment lineage + portable model export.**

---

## 4. Product shape and distribution

### 4.1 What the user downloads

A single application package per platform:

- **Windows** — installer or portable `.zip`
- **macOS** — `.dmg` (Apple Silicon and Intel)
- **Linux** — `.tar.gz` and/or AppImage

Inside: a frozen Python interpreter with the scientific stack, the FastAPI backend, and the pre-built static web UI. Launching it starts the local server and opens the UI in the user's default browser.

### 4.2 Packaging approach

**Start with PyInstaller (onedir) plus the system default browser.** The application binds a local server, then opens `http://127.0.0.1:<port>/?token=<token>` in the default browser. This is the smallest thing that fully delivers the promise: no user-installed runtime, no Rust toolchain, no webview packaging.

A native window shell (Tauri or pywebview) is deliberately deferred. It buys a nicer window — an application in the dock/taskbar rather than a browser tab, native file dialogs, no stray tab confusion — at the cost of a second toolchain and a more complex build matrix. Add it only if browser-tab UX proves to be an actual complaint, not in anticipation of one.

**Known costs to budget for, not discover later:**

- Package size will be large (NumPy, SciPy, scikit-learn, pandas ≈ 300–500 MB unpacked). Acceptable; state it plainly on the download page.
- **Code signing.** Unsigned binaries trigger Windows SmartScreen warnings and macOS Gatekeeper blocks. For an academic audience an unsigned release plus clear "how to open this" documentation is survivable at 1.0, but macOS notarization requires a paid Apple developer account and Windows signing requires a certificate. This is a funding question (§15), not a technical one.
- Build and release must be automated in CI across three platforms from day one of Phase 4; manual release builds do not survive contact with a second maintainer.

### 4.3 Localhost is a trust boundary, not a private room

A server bound on the user's machine is reachable by every other process on that machine and, without care, by any web page the user visits. This is not hypothetical — it is a well-known class of vulnerability in local-first desktop-web applications, and getting it wrong would undermine the project's central privacy claim.

Non-negotiable requirements for v1:

- Bind **127.0.0.1 only**, never `0.0.0.0`.
- **Ephemeral port**, chosen at startup.
- A **per-session bearer token** generated at launch and required on every API request. The token is passed once in the launch URL, then held by the frontend and sent as a header.
- **Strict `Origin` and `Host` header validation** on every request, to defeat DNS-rebinding attacks.
- **No cookie-based authentication**, so cross-site requests cannot ride an ambient session.
- **Filesystem access confined** to the user's chosen project directory and explicitly selected input files — the API must never accept an arbitrary server-side path from the client.

These are cheap to implement at the start and expensive to retrofit.

---

## 5. Scope for version 1.0

The purpose of v1 is to prove the thesis end to end for the most common chemometrics workflow, not to match a commercial feature matrix.

### In scope

- Local project workspace: projects, datasets, dataset versions, pipelines, experiments, models
- 2-D spectral and general multivariate data (samples × variables) with sample and variable metadata
- Data ingest: CSV, TXT, XLSX, JCAMP-DX, Bruker OPUS (§6)
- Preprocessing: mean centering, autoscaling, SNV, MSC, normalization, baseline correction, Savitzky–Golay smoothing, first and second derivative, spectral range selection, interpolation
- Missing-value handling, sample/variable exclusion, train/test splitting
- Analysis: PCA, PLS regression, PLS-DA
- Validation: train/test split, K-fold, repeated K-fold, LOOCV, external validation set
- Diagnostics: explained variance, Hotelling T², SPE/Q residuals, VIP, contribution plots
- Visualization: raw and processed spectra, scores, loadings, scree/explained variance, predicted vs. actual, residuals
- Experiment tracking, model lineage and comparison (§8)
- Model export and portable prediction (§9)
- Report export (HTML/PDF) for a model or an experiment comparison

### Deferred to post-1.0

PCR · SIMCA classification · permutation testing · bootstrap validation · MCR-ALS · PARAFAC · additional classifiers (LDA, kNN, SVM) · variable selection algorithms (GA, iPLS, CARS) · self-hosted server mode · PostgreSQL · object storage · multi-user and authentication · plugin API · Python scripting console

Every deferred item is *additive* against the v1 data model. None of them requires rewriting what v1 builds — this is the test each deferral was checked against.

### Out of scope, indefinitely

Hyperspectral image cubes and 3-way data · GxP validation packages, 21 CFR Part 11 audit trails and electronic signatures · instrument control and data acquisition · a hosted multi-tenant SaaS offering

---

## 6. Data ingest

File format support is the single most under-specified item in the original draft and one of the strongest reasons users stay locked into vendor software. It is treated here as a named deliverable.

| Format | Priority | Notes |
| --- | --- | --- |
| CSV / TXT (wide and long layouts) | v1, Phase 1 | Must handle European decimal commas, transposed layouts, wavelength headers, metadata columns |
| XLSX | v1, Phase 1 | Extremely common as the real-world interchange format |
| JCAMP-DX (`.jdx`, `.dx`) | v1, Phase 1 | Open standard, wide instrument support, good library availability |
| Bruker OPUS (`.0`, `.1`, …) | v1, Phase 2 | Highest-value proprietary format for FT-IR/NIR; readable via existing open readers |
| Thermo Galactic SPC (`.spc`) | Post-1.0 | Common legacy format |
| Thermo OMNIC SPA (`.spa`) | Post-1.0 | |
| ASD (`.asd`) | Post-1.0 | Field spectroscopy |
| MATLAB `.mat` | Post-1.0 | Migration path from PLS_Toolbox users |

Design rules: readers are independent, individually testable modules with a real sample file committed as a fixture; an unreadable file must produce a specific diagnostic message, never a stack trace; every import records the source file's content hash and the reader version.

---

## 7. Algorithms and dependency strategy

The original draft named **SpectroChemPy** and **process-improve** as core dependencies. Both are dropped from the core, for reasons worth recording:

- *SpectroChemPy* brings its own `NDDataset` object model plus traitlets and matplotlib. In a web application with a Pydantic/NumPy API boundary, its object model would be fought at every layer and its plotting stack shipped for nothing.
- *process-improve* is a small project with a narrow maintainer base. Core numerical results should not depend on it.

**What is used instead:**

- **NumPy / SciPy / pandas** — the numerical foundation the kernels are built on.
- **`scikit-learn`** — the **reference implementation the parity fixtures are generated against, and a development dependency only**. It is not on the application's code path. *This is a change from the draft, which said `PLSRegression`, `PCA`, cross-validation and metrics would come from scikit-learn.* Phase 0 wrote them instead, to the specifications in `docs/algorithms/`, because a kernel built on scikit-learn cannot then be compared against scikit-learn — and the parity report in §10 is the thing this project is asking to be trusted on. What was gained is not accuracy but evidence: coefficients, predictions, scores, loadings, eigenvalues, RMSECV and the rest now agree with an independent implementation to the last bits, and the disagreements that remain are documented conventions rather than accidents.
- **`chemotools`** — **evaluated in Phase 0 and rejected as a runtime dependency; adopted as a dev-only reference implementation** for SNV, MSC and the three baseline methods, which have no other external reference. It is not shipped: it requires scikit-learn, which is dev-only here, and installs 20 MB of which 17 MB is example datasets the application would never read. The full evidence, per transform, is in [`docs/decisions/0001-chemotools.md`](docs/decisions/0001-chemotools.md).
- **Purpose-written kernels** for everything scientifically load-bearing: preprocessing, PCA, PLS, the metrics, the cross-validation protocol, Hotelling T², SPE/Q and VIP scores. This is on the order of a few thousand lines of textbook mathematics. Owning it means owning its tests, its numerical conventions and its documentation — which for a project whose credibility rests on numerical trust is a feature, not a burden.
- **Format readers** — this is where third-party libraries genuinely earn their place (§6).

The dependency rule for the project: *take a dependency for the tedious and well-solved (file formats, numerics primitives); own the scientifically load-bearing and small.*

---

## 8. Reproducibility and lineage — the core data model

This is the project's central technical idea and therefore gets specified before any UI is designed. Everything else is built on it.

### 8.1 The pipeline is data

An analysis is a **serializable directed acyclic graph** of typed steps, stored as JSON:

```
Dataset (content hash)
  → Preprocessing steps (ordered, each with explicit parameters)
    → Variable/sample selection
      → Split definition (strategy, seed, resulting indices)
        → Algorithm + hyperparameters
          → Validation protocol
            → Metrics
              → Model artifact
```

Because the pipeline is data rather than code, it can be executed, stored, diffed, copied, edited into a variant, shared as a file, and reviewed by a third party. Everything the application does to a dataset happens by constructing and executing one of these graphs — there is no second, hidden path.

### 8.2 What every experiment must capture

- **Dataset identity by content hash**, not filename. Renaming a file must not break lineage; silently editing one must not go unnoticed.
- **Complete step parameters**, with defaults written out explicitly rather than implied.
- **Split reproducibility**: strategy, random seed, and the resulting index sets, stored — so a split survives a change in library version.
- **Environment**: application version, and versions of NumPy, SciPy, scikit-learn and any algorithm-providing dependency.
- **Timing and outcome**, including failures. A failed experiment is a result.

### 8.3 What this makes possible

Lineage becomes a query rather than a feature to be bolted on:

```
Dataset v1 ──┬── SNV        + PLS(6 LV)  → Model A   RMSECV 0.412
             ├── MSC        + PLS(6 LV)  → Model B   RMSECV 0.418
             ├── SG(d1,w11) + PLS(5 LV)  → Model C   RMSECV 0.389
             └── SNV+SG(d1) + PLS(5 LV)  → Model D   RMSECV 0.381
```

A model comparison view that shows *exactly which steps differ* between two models is the single feature most likely to make a researcher prefer this tool to a notebook — and it is free once the model above is correct.

### 8.4 The model artifact

One self-describing file containing the pipeline definition, the fitted parameters, the provenance record and the metrics. Openly documented format. Copyable between machines. Readable without this application.

---

## 9. Model export and portable prediction

The original draft included "predict" but not "take the model with you". Commercial tools gate this deliberately; making it free and easy is a genuine differentiator and costs very little.

A fitted PLS model is, at prediction time, a preprocessing recipe plus a coefficient vector. v1 will export:

1. **The native model artifact** (§8.4) — full fidelity, re-importable, carries provenance.
2. **A plain JSON model** — preprocessing steps with parameters, plus coefficients, intercept, and the scaling used. Documented schema.
3. **A standalone Python prediction snippet** — a self-contained function depending only on NumPy/SciPy that reproduces the model's predictions. Copy-pasteable into an instrument PC, a LIMS integration, or a colleague's notebook.

ONNX export is a post-1.0 consideration; the JSON-plus-snippet route covers the realistic deployment targets for spectroscopic models with far less machinery.

**Constraint this places on the codebase:** exported predictions must match in-application predictions to within a stated numerical tolerance, verified in CI. This is a test, not a promise.

---

## 10. Numerical correctness programme (Phase 0)

No laboratory will migrate an analysis to software that cannot demonstrate it produces the same numbers as the tool it replaces. The original draft did not address this at all. It is now **the first phase of work, before any user interface exists.**

### 10.1 Specify the algorithms, not just their names

"PLS" is not a specification. NIPALS and SIMPLS produce different scores and loadings; conventions for scaling, centering, sign, cross-validation folds and the definition of RMSECV all vary between packages, and users comparing against Unscrambler or SIMCA will notice. The project will publish, per algorithm: the variant implemented, the centering/scaling convention, the sign convention, and the exact definitions of every reported metric.

### 10.2 Reference datasets in the repository

Publicly available, widely published benchmark spectra committed as test fixtures — for example NIR corn, gasoline/octane and Tecator meat datasets — so that anyone can rerun the comparison.

### 10.3 Parity tests in CI

Assertions with explicit numerical tolerances against published reference values and against established implementations (R `mdatools` / `pls`, scikit-learn). These run on every commit. A change that moves a scientific number fails the build.

### 10.4 A published parity report

A page in the documentation showing, dataset by dataset and algorithm by algorithm, this project's numbers next to the reference numbers.

**This report is the project's single most valuable credibility asset.** No competing open-source chemometrics project publishes one, and it is the artifact that turns "an open-source tool" into "a tool I can put in a paper".

---

## 11. Architecture

```
┌──────────────────────────────────────────────┐
│  Application package (one download)          │
│                                              │
│   Browser tab ──► 127.0.0.1:<ephemeral port> │
│        │            (token-authenticated)     │
│        │                    │                 │
│   React UI            FastAPI backend         │
│   (bundled static)          │                 │
│                             ├── Pipeline executor
│                             ├── Algorithm kernels
│                             ├── SQLite (metadata, lineage)
│                             └── Local filesystem (project dir)
└──────────────────────────────────────────────┘
                No outbound network required
```

**Boundaries.** The frontend holds no scientific logic — it configures pipelines and renders results. The backend holds no presentation logic. The pipeline executor is the only path from a dataset to a result. Algorithm kernels are pure functions over arrays with no knowledge of the application, so they are testable in isolation and reusable as a library.

**Storage split.** SQLite holds metadata, pipeline definitions, experiments, metrics and lineage. The project directory on disk holds datasets, processed arrays, model artifacts and reports. The database stores references to files, never file contents. This split is what allows a project directory to be zipped and sent to a colleague.

**Long-running work.** Cross-validation over many preprocessing variants will exceed a request timeout. Experiments are submitted as jobs with progress reporting and cancellation, not as blocking HTTP calls. Designing this in from Phase 1 is far cheaper than retrofitting it.

---

## 12. Technology stack

### Frontend
React · TypeScript · Vite · Tailwind CSS · shadcn/ui · TanStack Query · Plotly.js (with WebGL rendering)

*Change from v1:* Zustand is dropped for now. For a single-user local application, TanStack Query covers server state and React's own state covers the rest. A dedicated client-state library can be added the moment there is state that genuinely needs it — shipping two state systems by default is a cost with no current benefit.

### Backend
Python · FastAPI · Pydantic · NumPy · SciPy · pandas · format readers per §6

*Not shipped:* scikit-learn and `chemotools` are development dependencies — reference implementations the parity fixtures are generated against, never on the application's code path (§7).

*Change from v1:* SpectroChemPy and process-improve removed (§7).

### Data
SQLite via SQLAlchemy · local filesystem

*Change from v1:* PostgreSQL, MinIO and S3 are removed from the v1 stack. SQLAlchemy already keeps the data model portable; the commitment for now is one sentence — *the schema stays engine-portable* — rather than infrastructure nobody has asked for yet. It returns when a real self-hosting user does.

### Build, test and release
Node.js · pnpm · uv · Ruff · mypy · pytest · Vitest · Playwright · PyInstaller · GitHub Actions (three-platform build matrix)

*Change from v1:* Docker and Docker Compose move from the product stack to the developer-environment stack. End users do not use containers.

---

## 13. Performance envelope

Stated as a target so it can be tested, and as a limit so it can be honestly documented.

- **v1 target:** up to ~20,000 spectra × ~4,000 variables, held in memory as float32 (≈ 320 MB). Beyond this, v1 documents the limit rather than pretending otherwise.
- **Interaction budgets:** preprocessing preview under 1 s; PCA on the target dataset under 5 s; a 10-fold cross-validated PLS under 30 s with progress reporting.
- **Plotting is the real constraint.** 20,000 spectra × 4,000 points is 80 million points — far past what Plotly renders. Required from the first plot, not as a later optimization: server-side decimation of each trace, a cap on individually drawn traces with the remainder shown as a density band, and WebGL (`scattergl`) rendering throughout. Selected or highlighted spectra are drawn at full resolution.

---

## 14. Deployment

**Local application — the only supported mode at 1.0.** Downloaded, installed, run on the user's machine, data on the user's disk, no network required.

**Self-hosted — post-1.0, when a real user asks.** The same backend, a shared database, an object store and authentication. The architecture in §11 does not prevent this; §12 deliberately does not build for it in advance.

**Hosted SaaS — not a goal.** It contradicts the project's central claim.

---

## 15. Licence, governance and sustainability

**Licence: MIT.** Contributions inbound under the same terms. Chosen for maximum adoption including commercial use — the goal is that this becomes standard teaching and research infrastructure, and copyleft would exclude a meaningful share of the industrial users whose feedback makes the science better.

**Governance.** Single-maintainer project initially; that is the honest description and should be stated rather than dressed up. Public roadmap, issues and design discussions from day one. A `CONTRIBUTING.md` and a documented "how to add an algorithm" path are Phase 4 deliverables, because the extension path is the project's growth mechanism.

**Sustainability.** Open source is a licence, not a funding model. Realistic options, to be chosen deliberately rather than by drift:

- Unfunded personal project — viable at this scope, with a corresponding bus-factor risk that must be stated publicly.
- Academic or open-science grant funding — a strong fit given the audience, the reproducibility angle and the published parity report.
- Institutional support or sponsored feature work from an industrial user.
- Paid support, training or validation packages, with the software itself remaining fully MIT.

The immediate concrete need is small: roughly €300–500 per year for code-signing certificates (§4.2), which unsigned academic releases can defer but not avoid forever.

---

## 16. Roadmap

Expressed in engineering-months (EM) so it is independent of team size. Each phase has an exit criterion — a phase is finished when the criterion is demonstrably met, not when it feels done.

| Phase | Focus | EM | Exit criterion |
| --- | --- | --- | --- |
| **0. Numerical foundation** | Repository, licence, CI. Algorithm specifications. Reference datasets committed. Preprocessing, PCA and PLS kernels. `chemotools` evaluation. Parity test harness. Pipeline DAG schema designed. **No UI.** | 2 | Parity report green in CI against published reference values |
| **1. Walking skeleton** | FastAPI + React shell. Project/dataset/pipeline data model on SQLite. CSV, XLSX, JCAMP ingest. Spectra plotting with decimation. Pipeline executor with job progress. PCA end to end with scores, loadings, explained variance. | 4 | Load a NIR dataset, apply SNV + Savitzky–Golay, run PCA, read the scores plot — entirely locally, no terminal |
| **2. Chemometrics depth** | PLS regression and PLS-DA. Full validation suite. Hotelling T², SPE, VIP, contribution plots. Predicted-vs-actual and residual plots. Train/test splitting UI. Bruker OPUS reader. | 3 | A complete calibration workflow on a real NIR dataset matches reference software within stated tolerance |
| **3. Reproducibility and export** | Experiment tracking UI. Model registry and lineage/comparison view. Model artifact format. JSON and Python-snippet export. HTML/PDF reporting. | 2 | Two models differing only in preprocessing can be compared step by step; an exported model reproduces application predictions within tolerance in a clean environment |
| **4. Package and release** | PyInstaller builds on three platforms in CI. Localhost security hardening (§4.3). Documentation site, published parity report, worked examples, `CONTRIBUTING.md`. | 2 | A non-developer on a clean machine downloads, installs and completes a PCA in under ten minutes |
| | **Total to 1.0** | **~13** | |

Post-1.0, ordered by expected demand rather than by ease: PCR and SIMCA · permutation and bootstrap validation · variable-selection algorithms · additional file formats · additional classifiers · plugin/scripting API · self-hosted mode.

---

## 17. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Numbers don't match established software; nobody trusts the tool | Fatal | Phase 0 exists entirely for this. Parity tests in CI, published parity report, explicit algorithm specifications |
| Packaging and code signing consume more time than expected | High | Budgeted as its own phase with CI automation from the start; unsigned academic release is an acceptable 1.0 fallback |
| Single maintainer; project stalls | High | MIT licence, plain code, algorithm kernels usable as a standalone library, documented extension path — so the work retains value even if the application does not continue |
| Plot performance collapses on real datasets | Medium | Decimation and WebGL designed in from the first plot; a stated and tested performance envelope (§13) |
| Scope creep toward commercial feature parity | Medium | §5 deferral list, with the explicit test that every deferred item is additive against the v1 data model |
| Adoption fails despite good software | Medium | Distribution decided up front (§4), install-to-first-result measured as a release criterion, worked examples and teaching material treated as product |
| Browser-tab UX confuses non-technical users | Low | Measured at Phase 4; native window shell added only if it proves to be a real complaint |

---

## 18. Success metrics

**At 1.0:** published parity report covering every implemented algorithm · three-platform signed-or-documented installers · install to first PCA under ten minutes for a non-developer on a clean machine · exported models reproduce application predictions within tolerance in CI.

**Twelve months after 1.0:** in genuine use by at least three research groups outside the author's own · at least five external contributors · cited or acknowledged in at least one publication · at least one instance of the tool being used in teaching.

Deliberately *not* a metric: GitHub stars.

---

## 19. Open questions

1. **Team and pace** — solo or collaborators, part-time or full-time? Converts §16's engineering-months into calendar dates.
2. **Funding intent** — unfunded, grant-seeking, or institutionally supported? Determines whether code signing, and eventually maintenance, are solved or deferred.
3. **Project name** — "Chemometrics Workbench" is a working title. Naming matters for adoption and should be settled before the repository is public.
4. **Reference software for parity** — R `mdatools` is free and reproducible by anyone; comparison against Unscrambler or PLS_Toolbox is more persuasive to industrial users but requires licence access. Which is available?
5. **First real dataset and first real user** — is there a specific research group whose workflow can drive Phase 1 and Phase 2 priorities? A single committed early user is worth more than any amount of feature planning.
