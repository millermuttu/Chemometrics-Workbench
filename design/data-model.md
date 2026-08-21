# Data model

Pydantic models live in [`data_models.py`](data_models.py). Run it directly (`python3 design/data_models.py`) to execute its self-check.

These models are the schema for the reproducibility guarantee in `PROPOSAL.md` §8. Three decisions carry most of the weight:

**An Experiment stores a snapshot of its pipeline, not a reference.** Pipelines are edited constantly. Holding a reference would mean that editing a pipeline silently rewrites the provenance of every experiment that ever used it. `Pipeline` is frozen, so "editing" produces a new object and every prior snapshot stays intact.

**A split's recipe and its result are stored separately.** `SplitSpec` holds strategy and seed — part of the recipe, reusable. `ResolvedSplit` holds the index sets the run actually produced, and lives on the Experiment. Storing the indices is what lets a split survive a change in scikit-learn's fold assignment.

**Node parameters are discriminated unions, not dicts.** An invalid pipeline fails at parse time rather than eleven minutes into a cross-validation. `SavitzkyGolay` refuses an even window; `Pipeline` refuses a cycle or a dangling input.

## Entities

```mermaid
erDiagram
    PROJECT ||--o{ DATASET : contains
    PROJECT ||--o{ PIPELINE : contains
    PROJECT ||--o{ EXPERIMENT : contains
    PROJECT ||--o{ MODEL : contains

    DATASET ||--o{ DATASET_VERSION : "versioned as"
    DATASET_VERSION ||--o| DATASET_VERSION : "derived from"
    DATASET_VERSION ||--o| SOURCE_FILE : "imported from"
    DATASET_VERSION ||--|| VARIABLE_AXIS : has

    PIPELINE ||--|{ PIPELINE_NODE : "graph of"
    PIPELINE_NODE }o--o{ PIPELINE_NODE : "inputs"

    EXPERIMENT ||--|| PIPELINE : "snapshots"
    EXPERIMENT }o--|| DATASET_VERSION : "runs against"
    EXPERIMENT ||--o{ RESOLVED_SPLIT : records
    EXPERIMENT ||--o| METRICS : produces
    EXPERIMENT ||--o| ENVIRONMENT : captures
    EXPERIMENT ||--o{ MODEL : yields

    MODEL ||--|| METRICS : carries

    PROJECT {
        UUID project_id PK
        string name
        string directory "on the user's disk"
    }
    DATASET {
        UUID dataset_id PK
        UUID project_id FK
        string name
    }
    DATASET_VERSION {
        UUID version_id PK
        UUID dataset_id FK
        int version
        string content_hash "identity, not the filename"
        int n_samples
        int n_variables
        dict targets "reference values by property"
        string array_path "file on disk, not in the DB"
    }
    VARIABLE_AXIS {
        enum kind "wavelength_nm, wavenumber_cm-1, raman_shift_cm-1, index"
        list values "strictly monotonic"
        string unit
    }
    SOURCE_FILE {
        string filename
        string file_hash
        string reader
        string reader_version
    }
    PIPELINE {
        UUID pipeline_id PK
        UUID project_id FK
        string name
        list nodes
        string content_hash "of the recipe alone"
    }
    PIPELINE_NODE {
        string id PK
        enum type "source, preprocess, split, estimator"
        tuple inputs FK
        object spec "discriminated by kind"
    }
    EXPERIMENT {
        UUID experiment_id PK
        Pipeline pipeline_snapshot "frozen copy"
        UUID dataset_version_id FK
        string dataset_content_hash
        enum status "pending, running, succeeded, failed, cancelled"
        string error "a failed run is still a result"
    }
    RESOLVED_SPLIT {
        string node_id FK
        list train_indices "one list per fold"
        list test_indices
    }
    METRICS {
        float rmsec
        float rmsecv
        float rmsep
        float r2
        float bias
        dict extra
    }
    ENVIRONMENT {
        string app_version
        string python_version
        string platform
        dict packages "everything that can move a number"
    }
    MODEL {
        UUID model_id PK
        UUID experiment_id FK
        string name
        enum task "regression, classification, decomposition"
        string node_id "which estimator produced it"
        string artifact_path
        string artifact_hash
    }
```

## Node and parameter types

Every node carries a typed spec, resolved by a discriminator field. This is what makes a pipeline safe to round-trip through JSON.

```mermaid
classDiagram
    class PipelineNode {
        <<union on type>>
        NodeId id
        tuple~NodeId~ inputs
    }
    class SourceNode {
        type = "source"
        UUID version_id
    }
    class PreprocessNode {
        type = "preprocess"
        PreprocessStep step
    }
    class SplitNode {
        type = "split"
        SplitSpec spec
    }
    class EstimatorNode {
        type = "estimator"
        EstimatorSpec spec
    }
    PipelineNode <|-- SourceNode
    PipelineNode <|-- PreprocessNode
    PipelineNode <|-- SplitNode
    PipelineNode <|-- EstimatorNode

    class PreprocessStep {
        <<union on kind>>
        snv, msc, savgol, mean_centre
        autoscale, normalise, baseline, range_select
    }
    class SplitSpec {
        <<union on kind>>
        train_test, kfold, repeated_kfold
        loo, external
    }
    class EstimatorSpec {
        <<union on kind>>
        pca, pls, plsda
    }
    PreprocessNode --> PreprocessStep
    SplitNode --> SplitSpec
    EstimatorNode --> EstimatorSpec
```

## A pipeline in practice

The branch-comparison case from the design brief, as it exists in the model — one `Pipeline` whose nodes fork from a single source:

```mermaid
flowchart LR
    SRC["source<br/>corn_raw 80x700"]

    SRC --> SNV["preprocess<br/>snv"]
    SRC --> MSC["preprocess<br/>msc"]
    SRC --> SG1["preprocess<br/>savgol d1 w11"]

    SNV --> SG2["preprocess<br/>savgol d1 w11"]

    SNV --> CV1["split<br/>kfold 10, seed 42"]
    MSC --> CV2["split<br/>kfold 10, seed 42"]
    SG1 --> CV3["split<br/>kfold 10, seed 42"]
    SG2 --> CV4["split<br/>kfold 10, seed 42"]

    CV1 --> A["estimator<br/>pls 6 LV"]
    CV2 --> B["estimator<br/>pls 6 LV"]
    CV3 --> C["estimator<br/>pls 5 LV"]
    CV4 --> D["estimator<br/>pls 5 LV"]

    A --> MA["Model A<br/>RMSECV 0.412"]
    B --> MB["Model B<br/>RMSECV 0.418"]
    C --> MC["Model C<br/>RMSECV 0.389"]
    D --> MD["Model D<br/>RMSECV 0.381"]
```

Because `Pipeline.content_hash()` covers the nodes and nothing else — not ids, not timestamps — two pipelines that would compute the same thing hash identically, and one changed parameter changes the hash. That is what makes *"these two models differ only in preprocessing"* a query rather than a feature to be built.

## Not yet modelled

Deliberate gaps, to be filled when the work reaches them:

- **Model artifact contents** — the fitted parameters themselves, and the JSON and Python-snippet export formats (`PROPOSAL.md` §9). Currently a path plus a hash.
- **Job and progress state** for long-running executions. The Experiment records the outcome; the job queue that produces it is a separate concern.
- **Report** entities.
- **Prediction runs** against a saved model.
- **Node layout** (canvas coordinates). Presentation state, deliberately kept out of the scientific record; it belongs alongside the pipeline, not inside its hash.
