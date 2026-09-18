# The model artifact — format specification

Status: **normative**. This document fixes the format the implementation must write and read. Where an implementation and this document disagree, one of them is a bug; decide which before changing either.

`PROPOSAL.md` §8.4 states the requirement in one sentence: *one self-describing file containing the pipeline definition, the fitted parameters, the provenance record and the metrics; openly documented format; copyable between machines; readable without this application.* Everything below follows from that last clause.

---

## 1. The container

A **zip archive**, conventionally named `<something>.cwmodel`, holding:

```
manifest.json          the whole record except the arrays
arrays/<name>.npy      one file per fitted array, NumPy's own .npy format
```

**Why a zip of JSON and `.npy`.** `zipfile`, `json` and `.npy` are, respectively, two standard-library modules and a [documented format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) that any NumPy reads. Opening one needs nothing from this project:

```python
import json, zipfile, numpy as np

with zipfile.ZipFile("model.cwmodel") as archive:
    manifest = json.loads(archive.read("manifest.json"))
    with archive.open("arrays/coefficients.npy") as handle:
        b = np.load(handle)
```

**Why not a pickle.** A pickle fails *readable without this application* twice: it needs this package importable to reconstruct the objects, and unpickling a file someone sent you executes whatever it says to execute. An artifact is a thing people email to each other.

**Why not one JSON file with the numbers inline.** A coefficient vector at §13's envelope is 4,000 float64; as JSON text that is both large and lossy unless every value is printed at full precision. `.npy` carries the dtype and the exact bytes.

---

## 2. Versioning

`manifest.json` carries `schema_version`, an integer, currently **1**.

A reader **refuses a file stamped higher than the version it understands**, naming both numbers, exactly as `db.py` refuses a database written by a newer application: a newer writer may have added fields whose absence this reader would silently take as a default. A file stamped *lower* is read if the reader still supports that version, and refused by name otherwise.

The version is bumped when a field is removed, renamed, or changes meaning. Adding an optional field does not bump it, because a reader that ignores an unknown key loses nothing.

---

## 3. `manifest.json`

Every field is required unless it is marked optional. `null` means *this quantity does not exist for this model*, never zero and never a placeholder — the rule `metrics-and-validation.md` §11 sets for metrics, applied to the whole file.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | §2 |
| `created_at` | string | ISO-8601 UTC, when the artifact was written |
| `application` | object | `{name, version}` of the application that wrote it |
| `model` | object | §4 |
| `pipeline` | object | The `Pipeline` snapshot that produced the estimator's input matrix, as `models.py` serialises it. **The recipe, by value** — an artifact whose pipeline was a reference would lose its meaning the moment the pipeline was edited (`PROPOSAL.md` §8.2). |
| `dataset` | object | §5 |
| `split` | object or null | §6 |
| `metrics` | object | The estimator's own metrics table, flattened as `results/{id}` serves it. A metric that could not be computed is absent. |
| `environment` | object or null | What was installed when the model was fitted: `app_version`, `python_version`, `platform`, `packages`. `null` only for a model whose experiment recorded none. |
| `arrays` | object | §7 |

### 4. `model`

| Field | Type | Meaning |
| --- | --- | --- |
| `node_id` | string | Which estimator node in the pipeline produced this |
| `task` | string | `decomposition`, `regression` or `classification` |
| `n_components` | int | `A`, the components actually fitted |
| `n_variables` | int | The width of the matrix it was fitted on — the node's own, which is not the dataset's under a range selection |
| `n_samples` | int | The rows it was fitted on |
| `rank` | int | `pca.md` §9's effective rank; for PLS, `A` |
| `target` | string or null | The response modelled, for a regression; the class column, for a classification |
| `classes` | array of string or null | `pls-da.md` §3's `[C_0, C_1]`, for a classification |
| `y_mean` | number or null | The response mean the estimator subtracted before fitting and adds back to every prediction (`pls-regression.md` §3). `null` for a decomposition |
| `alpha` | number | The confidence level the limits are quoted at |
| `hotelling_t2_limit` | number | |
| `spe_limit` | number | |
| `spe_limit_caveat` | string or null | `pca.md` §8: why the limit is outside its approximation's domain, when it is |

### 5. `dataset`

| Field | Type | Meaning |
| --- | --- | --- |
| `version_id` | string | The `DatasetVersion` this was fitted from |
| `content_hash` | string | Its content hash. **A dataset identified by its contents, never its filename** (`PROPOSAL.md` §8) |
| `n_samples`, `n_variables` | int | The dataset's own, before any range selection |
| `axis` | object | `{kind, unit}` of the dataset's variable axis. The values are in `arrays/dataset_axis.npy` |

### 6. `split`

`null` when the model was fitted above a split. Otherwise the fold it was fitted on, as `ResolvedSplit` records it:

| Field | Type | Meaning |
| --- | --- | --- |
| `node_id` | string | The split node |
| `fold` | int | Which fold this model is — fold zero, as `executor.py` fits it |
| `n_folds` | int | How many the split resolved to |

The index sets themselves are arrays (§7), because a ten-fold split on 20,000 samples is 20,000 integers and JSON is the wrong place for them.

### 7. `arrays`

An object mapping a **name** to `{file, dtype, shape}`. `file` is the path inside the archive; `dtype` and `shape` are stated in the manifest as well as carried by the `.npy`, so a reader can see what it is about to load without loading it.

Every array is written **float64** except the index sets, which are **int64**. The application stores its intermediate arrays as float32 (`PROPOSAL.md` §13); an artifact is a record rather than a cache, and narrowing a coefficient vector to save 16 kB would be trading the thing the file exists for.

Which arrays are present depends on the task. A reader must not assume any of them beyond `dataset_axis`:

| Name | Shape | Task | Meaning |
| --- | --- | --- | --- |
| `dataset_axis` | `(n_variables_dataset,)` | all | The dataset's variable axis, in the unit `dataset.axis` names |
| `node_axis` | `(n_variables,)` | all | The axis the model's own matrix is on, which differs under a range selection (#134) |
| `loadings` | `(A, n_variables)` | all | `P'`, one row per component, as `results/{id}` serves it |
| `eigenvalues` | `(A,)` | all | `pca.md` §7's `λ`; for PLS, the score variances |
| `explained_variance_ratio` | `(A,)` | all | |
| `rotations` | `(A, n_variables)` | all | `R`, what a row is multiplied by to get its scores. PCA's are its loadings |
| `x_mean` | `(n_variables,)` | regression, classification | The column means the estimator subtracted before fitting (`pls-regression.md` §3). Not a pipeline node's centring — this is the estimator's own |
| `coefficients` | `(n_variables,)` | regression, classification | `b = Rq` on the node's matrix |
| `y_loadings` | `(A,)` | regression, classification | |
| `vip` | `(n_variables,)` | regression, classification | `pls-regression.md` §8 |
| `y_explained_variance_ratio` | `(A,)` | regression, classification | |
| `train_indices` | `(n_train,)` | split only | The rows the model was fitted on |
| `test_indices` | `(n_test,)` | split only | The rows held out from it |

**Scores, predictions and diagnostics per sample are deliberately absent.** They are properties of the samples the model happened to see, not of the model, and a reader with the model and the data can recompute every one of them. The artifact is what you need to *use* the model and to *explain* it, which is not the same as everything the run produced.

---

## 8. Reading the model back

The artifact carries enough to predict, and `json-and-snippet-export` is what turns that into a portable function. Written out, for one sample `x` already through the pipeline's preprocessing:

```
t     = (x - x_mean) @ rotations'
y_hat = (x - x_mean) @ coefficients + y_mean
```

and for a classification, `pls-da.md` §5 assigns `classes[1]` when `y_hat >= 0.5`.

**The preprocessing is not folded into the coefficients here.** `pls-regression.md` §7 folds it where the chain is a fixed linear map and says by name where it is not; that transformation belongs to the export, which states which form it produced. The artifact carries `b` on the node's own matrix, which is the number the model was fitted with, and the pipeline beside it.

---

## 9. Deliberately not specified here

- **Where the file lives and what refers to it.** The project's `model` table stores a path and a hash; that is `model-registry`.
- **The plain JSON model and the Python snippet.** `PROPOSAL.md` §9's other two export forms have their own schema and their own document.
- **Encryption, signing, compression level.** A zip is a zip.
- **ONNX.** §9 calls it a post-1.0 consideration.
