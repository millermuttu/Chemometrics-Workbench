"""The classification tally, `classification.md`: an N-by-N confusion matrix
and the per-class table read from it, against scikit-learn's own.

scikit-learn is a development dependency only; it is the reference here, as in
the parity suite, and nothing in `src/` imports it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_confusion
from sklearn.metrics import precision_recall_fscore_support

from chemometrics_workbench.executor import (
    class_metrics,
    classification_metrics,
    confusion_matrix,
)

RNG = np.random.default_rng(268)
OBSERVED = RNG.integers(0, 4, size=200)
ASSIGNED = np.where(RNG.random(200) < 0.7, OBSERVED, RNG.integers(0, 4, size=200))


def test_the_matrix_is_scikit_learn_s_with_rows_observed() -> None:
    ours = confusion_matrix(OBSERVED, ASSIGNED, 4)
    assert ours == sk_confusion(OBSERVED, ASSIGNED, labels=range(4)).tolist()


def test_per_class_sensitivity_and_precision_are_scikit_learn_s() -> None:
    precision, recall, _, support = precision_recall_fscore_support(
        OBSERVED, ASSIGNED, labels=range(4), zero_division=np.nan
    )
    table = class_metrics(confusion_matrix(OBSERVED, ASSIGNED, 4))
    assert [entry["n"] for entry in table] == support.tolist()
    assert [entry["sensitivity"] for entry in table] == pytest.approx(recall.tolist())
    assert [entry["precision"] for entry in table] == pytest.approx(precision.tolist())


def test_specificity_is_the_other_classes_not_assigned_the_class() -> None:
    table = class_metrics(confusion_matrix(OBSERVED, ASSIGNED, 4))
    for j, entry in enumerate(table):
        others = j != OBSERVED
        assert entry["specificity"] == pytest.approx(np.mean(ASSIGNED[others] != j))


def test_two_classes_keep_pls_da_s_headline_metrics() -> None:
    """`[[TN, FP], [FN, TP]]` as before #269, and class 1's table agrees."""
    confusion = [[50, 5], [10, 35]]
    assert classification_metrics(confusion) == {
        "accuracy": 0.85,
        "sensitivity": 35 / 45,
        "specificity": 50 / 55,
    }
    positive = class_metrics(confusion)[1]
    assert positive["sensitivity"] == 35 / 45
    assert positive["specificity"] == 50 / 55


def test_more_than_two_classes_report_accuracy_alone_in_the_flat_table() -> None:
    metrics = classification_metrics([[3, 1, 0], [0, 4, 0], [1, 0, 5]], "_cv")
    assert metrics == {"accuracy_cv": 12 / 14}


def test_an_undefined_metric_is_absent_never_nan() -> None:
    """A class nobody belongs to, and nobody is assigned to, in this set."""
    table = class_metrics([[4, 0, 0], [1, 3, 0], [0, 0, 0]])
    assert table[2] == {"n": 0.0, "specificity": 1.0}
    assert classification_metrics([[0, 0], [0, 0]]) == {}


def test_the_results_payload_serves_each_set_s_per_class_table() -> None:
    """`classification.md` §4: derived from the matrices, one table per set."""
    from uuid import uuid4

    from chemometrics_workbench.api import results_payload
    from chemometrics_workbench.executor import EstimatorResult
    from chemometrics_workbench.models import DatasetVersion, VariableAxis

    confusion = {"calibration": [[5, 1], [0, 4]], "cross_validation": [[4, 2], [1, 3]]}
    result = EstimatorResult(
        node_id="plsda",
        key="k",
        task="classification",
        n_components=1,
        n_samples=10,
        n_variables=2,
        rank=1,
        fold=None,
        rows=list(range(10)),
        scores=[[0.0]] * 10,
        loadings=[[1.0, 0.0]],
        eigenvalues=[1.0],
        explained_variance_ratio=[1.0],
        cumulative_explained_variance=[1.0],
        hotelling_t2=[0.0] * 10,
        hotelling_t2_limit=1.0,
        spe=[0.0] * 10,
        spe_limit=1.0,
        target="grade",
        classes=["a", "b"],
        confusion=confusion,
    )
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash="sha256:" + "0" * 64,
        n_samples=10,
        n_variables=2,
        axis=VariableAxis(kind="index", values=[0.0, 1.0]),
        array_path="arrays/x.npy",
    )
    served = results_payload(result, version)["classification"]["class_metrics"]
    assert served == {name: class_metrics(matrix) for name, matrix in confusion.items()}
    assert served["calibration"][1] == {
        "n": 4.0,
        "sensitivity": 1.0,
        "specificity": 5 / 6,
        "precision": 0.8,
    }
