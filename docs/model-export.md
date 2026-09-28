# Model export — the JSON model and the prediction snippet

Status: **normative**. This document fixes what the two portable export forms contain and what they promise. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

`PROPOSAL.md` §9 names three export forms. The first is the native artifact, specified in [`model-artifact.md`](model-artifact.md). This document is the other two:

2. **A plain JSON model** — the preprocessing that has to be re-executed, plus coefficients, intercept and scaling. Documented schema.
3. **A standalone Python prediction snippet** — a self-contained function depending only on NumPy that reproduces the model's predictions.

§9 also places one constraint on the codebase, and it is a test rather than a promise: **exported predictions must match in-application predictions to within a stated numerical tolerance, verified in CI.** §5 states that tolerance and says where the number comes from.

---

## 1. What can be exported, and what cannot

`pls-regression.md` §7 folds a preprocessing step into the coefficient vector when it is a fixed linear map on X whose parameters were fixed at calibration time: mean centring, autoscaling, range selection, Savitzky–Golay. SNV, MSC and the baselines are not — each depends on the sample being predicted, so each must be **re-executed** at prediction time.

An export therefore **splits the chain at the last unfoldable step**:

```
raw  →  [ residual chain ]  →  [ foldable tail ]  →  b, intercept
         re-executed             folded into b
```

Everything after the last unfoldable step folds; everything up to and including it is carried as a residual chain with its fitted parameters. A chain with nothing unfoldable in it has an empty residual chain and exports as a bare coefficient vector, which is §9's ideal case; a chain whose *last* step is an SNV folds nothing, which is legal and says so by carrying the whole chain.

**Which residual steps are emitted.** `snv`, `msc` and `normalise`: each is a few lines of NumPy on one row, and each is written out in §4.

**A baseline is refused by name.** AsLS is a penalised sparse solve per spectrum, rubberband is a convex hull; putting either inside a file whose whole point is that it can be pasted into an instrument PC would make it neither short nor checkable. An export asked for a chain containing one fails with a sentence naming the step — the pattern `coefficients_original_units` already sets, and the honest answer rather than an approximation. The native artifact still carries such a model, and the application still predicts with it.

---

## 2. The JSON model

```json
{
  "schema_version": 1,
  "created_at": "2026-09-18T11:00:00+00:00",
  "application": {"name": "chemometrics-workbench", "version": "0.7.0"},
  "model": {
    "node_id": "pls_d",
    "task": "regression",
    "target": "fat",
    "classes": null,
    "n_components": 5,
    "threshold": null
  },
  "axis": {"kind": "wavelength_nm", "unit": "nm", "values": [850.0, "…"]},
  "preprocessing": [{"kind": "snv", "ddof": 1}],
  "coefficients": [0.0123, "…"],
  "intercept": 18.4271,
  "provenance": {
    "dataset_content_hash": "sha256:…",
    "pipeline_hash": "sha256:…",
    "experiment_id": null,
    "metrics": {"rmsec": 2.24, "rmsecv": 2.39}
  }
}
```

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer, currently **1**. A reader refuses a higher one by name, as [`model-artifact.md`](model-artifact.md) §2 has it |
| `model.task` | `regression` or `classification`. **A decomposition has no prediction to export** and is refused: PCA produces scores, not a response |
| `model.threshold` | `0.5` for a classification, `null` otherwise — `pls-da.md` §5's fixed cut, stated rather than assumed by the reader |
| `model.classes` | `[C_0, C_1]` for a classification; the prediction is `classes[1]` at or above the threshold |
| `axis.values` | The **raw** variable axis the model expects its input on, with as many entries as `coefficients` has when the residual chain preserves the variable count. It is the axis of the matrix the residual chain is applied to |
| `preprocessing` | The residual chain, in the order it is applied, each step as `models.py` serialises it plus whatever fitted parameters it needs (§4) |
| `coefficients` | `b` such that `y = x_after_chain · b + intercept` |
| `intercept` | The scalar the fold produced, carrying the estimator's own `y` centring and every folded step's offset |
| `provenance` | What this model came from. `metrics` is the estimator's own table |

**The JSON model is self-contained and lossy on purpose.** It holds what predicting needs. Scores, loadings, VIP and the diagnostics are in the artifact; a JSON model is what you send to whoever has to run the calibration, not the record of how it was built.

---

## 3. The snippet

One `.py` file, no imports but `numpy`, holding:

- `AXIS`, `COEFFICIENTS`, `INTERCEPT` and whatever the residual chain needs, as literals;
- one small function per residual step;
- `predict(X)` taking an `n × p` array of raw spectra on `AXIS` and returning `n` predictions — a float for a regression, a class label for a classification.

It is generated from the JSON model and computes the same arithmetic; anything true of one in §4 is true of the other. It prints its own provenance in a header comment: what it came from, the dataset's content hash, and the metrics the model scored, so a file found on an instrument PC in two years can be traced.

`predict` takes a 2-D array and returns a 1-D one. A single spectrum is `predict(x[None, :])[0]`, said in the file rather than assumed.

---

## 4. The arithmetic, written out

For a row `x` of the raw matrix, the residual chain applies in order, then:

$$\hat{y} = x_{\text{after chain}} \cdot b + \text{intercept}$$

and a classification assigns `classes[1]` when `ŷ ≥ threshold` (`pls-da.md` §5).

The residual steps, exactly as `preprocessing.py` computes them:

**`snv`** — each row centred and scaled by its own moments, `ddof` from the step:

```
x ← (x - mean(x)) / std(x, ddof)
```

**`msc`** — each row regressed against the stored `reference`, which is a fitted parameter and travels with the step:

```
r ← reference - mean(reference)
slope     ← ((x - mean(x)) · r) / (r · r)
intercept ← mean(x) - slope · mean(reference)
x ← (x - intercept) / slope
```

**`normalise`** — divided by the row's norm; `l1` the sum of absolute values, `l2` the Euclidean norm, `max` the largest absolute value, `area` the signed sum:

```
x ← x / norm(x)
```

A zero divisor in any of these is a dead spectrum, and both the application and the snippet raise rather than substituting one — `preprocessing.py`'s rule, carried across, because a prediction from a substituted scale is a plausible wrong number.

---

## 5. The tolerance, and why it is what it is

**An exported prediction agrees with the application's to `rtol = 1e-4`, `atol = 1e-6`.**

That is wider than the parity harness's prediction class (`rtol = 1e-6`) and the reason is measured rather than supposed. The executor stores every intermediate array as **float32** (`PROPOSAL.md` §13), so the application's own prediction is computed from float32-rounded inputs, while an export applied to the raw data computes in float64 throughout. `docs/phase-2/exit-run.md` measured exactly this on a Savitzky–Golay first-derivative chain: the served predictions differ from a float64 reference by **7.09e-6 relative**, and a first derivative is the worst case, because it amplifies the rounding of the array it differentiates.

The export tolerance is therefore set an order of magnitude above the largest difference the store can produce, not at a number that happened to pass. Two consequences worth stating:

- **The difference is the store's, not the export's.** Narrowing the exported coefficients to float32 would make the two agree more closely and make the export worse. The artifact and the JSON model both carry float64.
- **If the store ever becomes float64**, this tolerance should come down to the parity harness's prediction class, and the test that holds it should be the thing that notices.

---

## 6. Deliberately not specified here

- **ONNX.** §9 calls it a post-1.0 consideration; the JSON-plus-snippet route covers the realistic deployment targets with far less machinery.
- **Prediction on new data inside the application.** A screen that takes a new file and applies a saved model is a separate feature.
- **Baselines in the residual chain.** §1 refuses them by name. Emitting them means specifying which baseline implementations a snippet may carry, and that is a document of its own.
- **Multi-response PLS2.** `pls-regression.md` §10 defers it, so an export of one cannot exist yet.
