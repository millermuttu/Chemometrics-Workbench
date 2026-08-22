"""Regenerate `reference_values.json`, the parity fixture for issue #7.

Run with `uv run python tests/fixtures/generate_reference_values.py`. It
rewrites the fixture in place; commit the result and say in the message what
moved and why, per the "scientific numbers do not move silently" rule in
CONTRIBUTING.md.

**This script is the fixture's provenance.** Every entry it writes records the
preprocessing chain, the algorithm variant, the split, the software and its
version, and a citation. An entry that cannot record all of those is written
as `status: "unsourced"` with the reason, never filled in with a plausible
number.

Two things about the numbers here are easy to get wrong, and both are settled
in `docs/algorithms/`:

* **Nothing is fed raw data.** `pca.md` §2 and `pls-regression.md` §3 make
  centring an explicit pipeline step, while scikit-learn centres internally
  and unconditionally. Every matrix handed to scikit-learn here is already
  centred, so its own centring is a no-op and the comparison is valid.
* **Fold indices are resolved here and stored**, not reseeded downstream.
  `metrics-and-validation.md` §8.2: our shuffle is `default_rng` (PCG64) and
  scikit-learn's is a legacy `RandomState`, so the same seed gives different
  folds. The harness in #8 must read the indices out of the fixture rather
  than seed its own splitter.

Three reference implementations are read here and none of them ships: SciPy is
a runtime dependency but `scipy.signal` is not on any kernel's code path, and
scikit-learn and `chemotools` are development dependencies. See
`docs/decisions/0001-chemotools.md` for why the last of those is a reference
rather than a dependency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from chemotools.baseline import AsLs, PolynomialCorrection, RubberbandCorrection
from chemotools.scatter import MultiplicativeScatterCorrection, StandardNormalVariate
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from chemometrics_workbench.datasets import (
    ReferenceDataset,
    load_corn,
    load_gasoline,
    load_tecator,
)

FIXTURE = Path(__file__).parent / "reference_values.json"
SCHEMA_VERSION = 1

# Component counts. Five is enough to expose a sign, ordering or scaling
# disagreement in a kernel; the RMSECV curve runs further because its shape is
# the thing being compared, not one value.
N_COMPONENTS = 5
MAX_PLS_COMPONENTS = 10

# The split used for every generated RMSECV entry. Seed 42 is the SplitSpec
# default from metrics-and-validation.md §8.2.
N_FOLDS = 5
SEED = 42

SKLEARN_VERSION = sklearn.__version__
SCIPY_VERSION = scipy.__version__
CHEMOTOOLS_VERSION = version("chemotools")

# The Savitzky-Golay configuration the smoothing references are generated at.
# A window of 5 over a block of 8 variables leaves two edge positions at each
# end, which is the point: `mode="interp"` is a different formula there, and a
# reference computed only over the interior would not test it.
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2

# The target modelled for each dataset. One response per dataset keeps the
# fixture to a size a human will actually read; PLS1 models one response at a
# time anyway (pls-regression.md §10).
TARGETS = {"corn": "moisture", "gasoline": "octane", "tecator": "fat"}


# --------------------------------------------------------------------------
# fold assignment, per metrics-and-validation.md §8.2 and §8.3
# --------------------------------------------------------------------------


def kfold_indices(n: int, n_folds: int, seed: int) -> list[dict[str, list[int]]]:
    """Resolve a shuffled K-fold split into explicit index lists.

    The first `n % n_folds` folds get one extra member, which is
    scikit-learn's size rule kept deliberately so that only the permutation
    differs between us.
    """
    if n_folds > n:
        raise ValueError(f"{n_folds} folds requested for {n} samples")

    perm = np.random.default_rng(seed).permutation(n)
    quotient, remainder = divmod(n, n_folds)

    folds: list[dict[str, list[int]]] = []
    start = 0
    for k in range(n_folds):
        size = quotient + 1 if k < remainder else quotient
        validation = np.sort(perm[start : start + size])
        train = np.setdiff1d(np.arange(n), validation)
        folds.append({"train_indices": train.tolist(), "test_indices": validation.tolist()})
        start += size
    return folds


def _self_check() -> None:
    """The worked example from metrics-and-validation.md §8.3.

    If this fails, the permutation, the seeding or the size rule has moved,
    and every RMSECV in the fixture is against a different split than the
    document describes.
    """
    folds = kfold_indices(10, 3, 42)
    assert [f["test_indices"] for f in folds] == [[0, 5, 6, 7], [2, 3, 4], [1, 8, 9]], folds
    assert folds[0]["train_indices"] == [1, 2, 3, 4, 8, 9], folds[0]

    # Every sample is predicted exactly once — the disjoint-union assertion
    # metrics-and-validation.md §7 requires before pooling residuals.
    seen = sorted(i for fold in folds for i in fold["test_indices"])
    assert seen == list(range(10)), seen


# --------------------------------------------------------------------------
# metrics, per metrics-and-validation.md §4 and §6
# --------------------------------------------------------------------------


def rmse(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    """Divisor is n. Degrees-of-freedom corrections live in SEC and SEP (§4)."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def r2_score(y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
    """Residual form, not squared Pearson correlation (§6)."""
    return float(1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))


# --------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------


def entry(
    *,
    entry_id: str,
    dataset: str,
    dataset_content_hash: str | None,
    algorithm: str,
    quantity: str,
    value: Any,
    preprocessing: list[str],
    algorithm_variant: str,
    split: dict[str, Any] | None,
    software: str,
    software_version: str,
    citation: str,
    status: str = "sourced",
    comparable: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "dataset": dataset,
        "dataset_content_hash": dataset_content_hash,
        "algorithm": algorithm,
        "quantity": quantity,
        "status": status,
        "comparable": comparable,
        "preprocessing": preprocessing,
        "algorithm_variant": algorithm_variant,
        "split": split,
        "software": software,
        "software_version": software_version,
        "citation": citation,
        "notes": notes,
        "value": value,
    }


PCA_VARIANT = (
    "SVD via LAPACK, full_matrices=False. sklearn.decomposition.PCA with "
    "svd_solver='full' passed explicitly, on a pre-centred matrix so its unconditional "
    "internal centring is a no-op (pca.md §2). Randomised SVD is not used - and the "
    "default svd_solver='auto' would have used it for corn, unseeded."
)
PLS_VARIANT = (
    "sklearn.cross_decomposition.PLSRegression(scale=False) on pre-centred X and y. "
    "NIPALS with y-deflation, one response (PLS1). pls-regression.md §2 records that "
    "NIPALS and SIMPLS coincide for a single response in coefficients and predictions, "
    "so a SIMPLS reference would be valid here but not for weights and loadings."
)

SIGN_NOTE = (
    "Signs are scikit-learn's, decided from U (svd_flip u_based_decision=True). Ours "
    "are keyed on the largest-magnitude loading (pca.md §5). Align by inner product "
    "before comparing; never compare absolute values."
)


def sklearn_entries(name: str, dataset: ReferenceDataset) -> list[dict[str, Any]]:
    """PCA and PLS reference values from scikit-learn on one dataset."""
    content_hash = dataset.source.file_hash
    target = TARGETS[name]
    citation = (
        f"Generated by tests/fixtures/generate_reference_values.py against "
        f"scikit-learn {SKLEARN_VERSION}. Not a literature value: an open "
        f"implementation pinned by version, reproducible by anyone."
    )

    spectra = dataset.spectra
    centred = spectra - spectra.mean(axis=0)
    y = dataset.targets[target]
    y_centred = y - y.mean()

    # svd_solver="full" is passed deliberately and is not the default. With
    # svd_solver="auto" scikit-learn picks its randomised solver whenever the
    # matrix is wider than 500 and few components are asked for, which is true
    # of corn - and its random_state is unseeded, so the corn reference values
    # moved by around 1e-14 on every regeneration. pca.md §3 forbids randomised
    # SVD for our implementation and the reference has to be held to the same
    # rule, or the parity claim rests on a number that is not reproducible.
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full").fit(centred)
    scores = pca.transform(centred)

    # Diagnostics scikit-learn does not provide, computed here from its own
    # decomposition by the definitions in pca.md §7 and §8. Not an independent
    # formula - an independent decomposition, which is what the comparison is
    # actually testing.
    hotelling_t2 = ((scores**2) / pca.explained_variance_).sum(axis=1)
    residual = centred - scores @ pca.components_
    spe = (residual**2).sum(axis=1)
    pls = PLSRegression(n_components=N_COMPONENTS, scale=False).fit(centred, y_centred)

    # coef_ is (n_targets, n_features) in scikit-learn 1.9; the orientation has
    # changed across releases, so assert it rather than assume it
    # (pls-regression.md §14).
    assert pls.coef_.shape == (1, spectra.shape[1]), pls.coef_.shape
    coefficients = pls.coef_.ravel()

    # Predictions are un-centred back to the response's original units, because
    # every metric is computed there (metrics-and-validation.md §2).
    predictions = np.asarray(pls.predict(centred)).ravel() + y.mean()

    # VIP, computed here from scikit-learn's own weights, scores and y-loadings
    # by the Wold form in pls-regression.md §8. scikit-learn reports no VIP, so
    # this is our formula on an independent decomposition - the same standing
    # as the PCA T^2 and SPE entries, and it must not be presented in the
    # report as more than that.
    y_loadings = np.asarray(pls.y_loadings_).ravel()
    explained = y_loadings**2 * (np.asarray(pls.x_scores_) ** 2).sum(axis=0)
    unit_weights = np.asarray(pls.x_weights_) / np.linalg.norm(pls.x_weights_, axis=0)
    vip = np.sqrt(spectra.shape[1] * ((unit_weights**2) @ explained) / explained.sum())

    folds = kfold_indices(len(y), N_FOLDS, SEED)
    split = {
        "strategy": "k_fold",
        "n_folds": N_FOLDS,
        "shuffle": True,
        "seed": SEED,
        "generator": "numpy.random.default_rng (PCG64)",
        "folds": folds,
        "note": (
            "Resolved indices, not a seed. scikit-learn's legacy RandomState gives "
            "different folds from the same seed, so the harness must pass these to it "
            "as an explicit cv iterable (metrics-and-validation.md §8.2, §10)."
        ),
    }

    # RMSECV as a function of A, from one fold assignment for every component
    # count, pooled over folds (metrics-and-validation.md §7 and §9).
    rmsecv_curve: dict[str, float] = {}
    for n_components in range(1, MAX_PLS_COMPONENTS + 1):
        held_out = np.empty_like(y)
        for fold in folds:
            train = np.asarray(fold["train_indices"])
            test = np.asarray(fold["test_indices"])
            # Every node downstream of the split is refitted on the training
            # fold alone, centring included (metrics-and-validation.md §9).
            train_mean = spectra[train].mean(axis=0)
            train_y_mean = y[train].mean()
            fold_model = PLSRegression(n_components=n_components, scale=False).fit(
                spectra[train] - train_mean, y[train] - train_y_mean
            )
            held_out[test] = (
                np.asarray(fold_model.predict(spectra[test] - train_mean)).ravel() + train_y_mean
            )
        rmsecv_curve[str(n_components)] = rmse(y, held_out)

    common = {
        "dataset": name,
        "dataset_content_hash": content_hash,
        "software": "scikit-learn",
        "software_version": SKLEARN_VERSION,
        "citation": citation,
    }
    pca_common = {
        **common,
        "algorithm": "pca",
        "preprocessing": ["mean_centre_x"],
        "algorithm_variant": PCA_VARIANT,
        "split": None,
    }
    pls_common = {
        **common,
        "algorithm": "pls",
        "preprocessing": ["mean_centre_x", "mean_centre_y"],
        "algorithm_variant": PLS_VARIANT,
    }

    return [
        entry(
            entry_id=f"{name}.pca.eigenvalues.sklearn",
            quantity="eigenvalues",
            value=pca.explained_variance_.tolist(),
            notes="sigma^2/(n-1), the sample-variance convention (pca.md §4).",
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.explained_variance_ratio.sklearn",
            quantity="explained_variance_ratio",
            value=pca.explained_variance_ratio_.tolist(),
            notes=(
                "Denominator is the total variance over all r components, not the "
                f"{N_COMPONENTS} retained (pca.md §6). Verified against "
                "X.var(ddof=1).sum() when generated."
            ),
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.cumulative_explained_variance.sklearn",
            quantity="cumulative_explained_variance",
            value=np.cumsum(pca.explained_variance_ratio_).tolist(),
            notes=(
                f"The running total of the entry above, over {N_COMPONENTS} components "
                "(pca.md §6). Derived from that entry rather than independently "
                "sourced - scikit-learn reports no cumulative curve - so it adds no "
                "information about the decomposition. It is here because §6 names it "
                "as a reported quantity and the check that it is a running total of "
                "the right denominator is worth making explicitly: a curve that "
                "reaches 1.0 at the last retained component is the classic sign of "
                "normalising over the retained components instead of all r."
            ),
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.loadings.sklearn",
            quantity="loadings",
            value=pca.components_.T.tolist(),
            notes=f"Shape p x {N_COMPONENTS}; sklearn stores the transpose. {SIGN_NOTE}",
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.scores.sklearn",
            quantity="scores",
            value=scores.tolist(),
            notes=f"Shape n x {N_COMPONENTS}. {SIGN_NOTE}",
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.hotelling_t2.sklearn",
            quantity="hotelling_t2",
            value=hotelling_t2.tolist(),
            notes=(
                f"sum_k t_ik^2/lambda_k over the {N_COMPONENTS} retained components "
                "(pca.md §7), on the calibration samples. scikit-learn does not report "
                "T^2, so this is computed here from its scores and eigenvalues by the "
                "definition - an independent decomposition rather than an independent "
                "formula. Sign-invariant, since every score is squared."
            ),
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pca.spe.sklearn",
            quantity="spe",
            value=spe.tolist(),
            notes=(
                f"||x_i - t_i P^T||^2 with A={N_COMPONENTS} (pca.md §8), on the "
                "calibration samples. The sum of squares, not the mean and not the "
                "root - other packages report one of those. scikit-learn does not "
                "report SPE, so this is computed here from its decomposition by the "
                "definition. Sign-invariant: the reconstruction t_i P^T is unchanged "
                "by flipping a component."
            ),
            **pca_common,
        ),
        entry(
            entry_id=f"{name}.pls.coefficients.sklearn",
            quantity="coefficients",
            value=coefficients.tolist(),
            split=None,
            notes=(
                f"Target '{target}', A={N_COMPONENTS}. On the centred matrix, so no "
                "intercept term. Coefficients are sign-invariant (pls-regression.md "
                "§5) and need no alignment. coef_ orientation asserted as "
                "(n_targets, n_features) when generated."
            ),
            **pls_common,
        ),
        entry(
            entry_id=f"{name}.pls.predictions.sklearn",
            quantity="predictions",
            value=predictions.tolist(),
            split=None,
            notes=(
                f"Target '{target}', A={N_COMPONENTS}, calibration set. Un-centred "
                "back to the original units of the response (§2)."
            ),
            **pls_common,
        ),
        entry(
            entry_id=f"{name}.pls.rmsec.sklearn",
            quantity="rmsec",
            value=rmse(y, predictions),
            split=None,
            notes=(
                f"Target '{target}', A={N_COMPONENTS}. Divisor is n, not n-A-1 "
                "(§4). A package reporting n-A-1 under this name is reporting our SEC."
            ),
            **pls_common,
        ),
        entry(
            entry_id=f"{name}.pls.r2.sklearn",
            quantity="r2",
            value=r2_score(y, predictions),
            split=None,
            notes=(
                f"Target '{target}', A={N_COMPONENTS}. Residual form, not squared "
                "Pearson correlation (§6)."
            ),
            **pls_common,
        ),
        entry(
            entry_id=f"{name}.pls.vip.sklearn",
            quantity="vip",
            value=vip.tolist(),
            split=None,
            notes=(
                f"Target '{target}', A={N_COMPONENTS}. Wold's form, weighted by "
                "SS_a = q_a^2 (t_a't_a) and normalised so that sum_j VIP_j^2 = p "
                "(pls-regression.md §8). scikit-learn reports no VIP, so this is "
                "computed here from its weights, scores and y-loadings by the "
                "definition - an independent decomposition rather than an independent "
                "formula, exactly as for the PCA T^2 and SPE entries. Sign-invariant: "
                "every weight is squared. Several published VIP variants exist and "
                "this one is the standard Wold form."
            ),
            **pls_common,
        ),
        entry(
            entry_id=f"{name}.pls.rmsecv_curve.sklearn",
            quantity="rmsecv_curve",
            value=rmsecv_curve,
            split=split,
            notes=(
                f"Target '{target}', A=1..{MAX_PLS_COMPONENTS}, keyed by component "
                "count. Residuals pooled across folds then rooted once, not the mean "
                "of per-fold RMSEs (§7). One fold assignment for the whole curve (§9). "
                "Centring refitted on each training fold."
            ),
            **pls_common,
        ),
    ]


# The submatrix the preprocessing references are computed on. These transforms
# are row-wise or column-wise and independent per row or column, so a small
# block tests the arithmetic completely — and storing a full 80 x 700
# preprocessed matrix would add megabytes to a fixture a human is meant to be
# able to read.
PREPROCESS_ROWS = 5
PREPROCESS_COLUMNS = 8


# The block the baseline references are computed on. Baselines are shape
# sensitive and a five-column block would not be a baseline at all, so this one
# is 120 variables wide - a real spectral segment - and only three rows deep,
# because a baseline is estimated per spectrum and a fourth row would add bytes
# rather than evidence. Tecator is only 100 variables wide, so it takes all of
# them.
BASELINE_ROWS = 3
BASELINE_COLUMNS = 120

# Eilers and Boelens's own values, and the ones our kernel defaults to.
ASLS_LAM = 1e5
ASLS_P = 0.01
# chemotools runs a fixed iteration count with no convergence test; ours stops
# at an exact fixed point of the reweighting, capped at this. They are matched
# here so that neither side is compared at a different number of iterations -
# and once ours reaches a fixed point, further iterations change nothing, so
# the two agree whether ours stopped early or not.
ASLS_ITERATIONS = 20
POLYNOMIAL_ORDER = 2


def chemotools_entries(name: str, dataset: ReferenceDataset) -> list[dict[str, Any]]:
    """SNV, MSC and the three baselines, from chemotools.

    These five quantities had no external reference at all until #13 evaluated
    `chemotools` and #27 wired it in: neither scikit-learn nor SciPy implements
    scatter correction or baseline estimation, so they were checked against
    their defining identities alone. Those identity tests stay - they are exact
    where this is a second opinion - and this adds the second opinion.

    `docs/decisions/0001-chemotools.md` is why this is a development dependency
    and not a runtime one, and it records the measured agreement per transform.
    """
    block = dataset.spectra[:PREPROCESS_ROWS, :PREPROCESS_COLUMNS]
    baseline_block = dataset.spectra[:BASELINE_ROWS, :BASELINE_COLUMNS]
    citation = (
        f"Generated by tests/fixtures/generate_reference_values.py against "
        f"chemotools {CHEMOTOOLS_VERSION}. Not a literature value: an open "
        f"implementation pinned by version, reproducible by anyone."
    )
    common: dict[str, Any] = {
        "dataset": name,
        "dataset_content_hash": dataset.source.file_hash,
        "algorithm": "preprocess",
        "split": None,
        "software": "chemotools",
        "software_version": CHEMOTOOLS_VERSION,
        "citation": citation,
    }
    where = (
        f"Computed on the first {PREPROCESS_ROWS} samples and first "
        f"{PREPROCESS_COLUMNS} variables of the dataset - the same block as the "
        "scikit-learn preprocessing entries."
    )
    baseline_where = (
        f"Computed on the first {BASELINE_ROWS} samples and first "
        f"{BASELINE_COLUMNS} variables, a wider block than the other preprocessing "
        "entries because a baseline over eight variables is not a baseline. Tecator "
        "has only 100 variables and therefore contributes all of them."
    )

    return [
        entry(
            entry_id=f"{name}.preprocess.snv.chemotools",
            quantity="snv_corrected",
            value=StandardNormalVariate().fit_transform(block).tolist(),
            preprocessing=["snv"],
            algorithm_variant=(
                "chemotools.scatter.StandardNormalVariate, which centres each row and "
                "divides by its population standard deviation."
            ),
            notes=(
                f"{where} **ddof=0**, because chemotools uses the population standard "
                "deviation and offers no choice - the same relationship our autoscale "
                "entry has with StandardScaler. Our SNVTransformer defaults to ddof=1, "
                "the sample convention used for eigenvalues and for SEC and SEP, so "
                "parity against this entry must pass ddof=0 explicitly. That "
                "difference is a convention, not a defect, and at ddof=0 the two "
                "implementations are bit-identical."
            ),
            **common,
        ),
        entry(
            entry_id=f"{name}.preprocess.msc.chemotools",
            quantity="msc_corrected",
            value=MultiplicativeScatterCorrection(method="mean").fit_transform(block).tolist(),
            preprocessing=["msc"],
            algorithm_variant=(
                "chemotools.scatter.MultiplicativeScatterCorrection(method='mean'). "
                "Same estimator as ours and a different arrangement of it: it forms "
                "the normal equations for the two-column design [reference, 1] and "
                "inverts A'A, where our kernel centres the reference and projects onto "
                "it. Same regression, and the normal-equation route squares the "
                "condition number."
            ),
            notes=(
                f"{where} The reference is the column mean of the block, estimated at "
                "fit and reused at transform on both sides. The agreement is 3.5e-17 on "
                "gasoline and 2.9e-10 on tecator, and the spread between those is the "
                "conditioning difference above rather than a disagreement about the "
                "formula - which is why this quantity has its own tolerance class."
            ),
            **common,
        ),
        entry(
            entry_id=f"{name}.preprocess.baseline_asls.chemotools",
            quantity="baseline_asls",
            value=AsLs(lam=ASLS_LAM, penalty=ASLS_P, nr_iterations=ASLS_ITERATIONS)
            .fit_transform(baseline_block)
            .tolist(),
            preprocessing=["baseline_asls"],
            algorithm_variant=(
                f"chemotools.baseline.AsLs(lam={ASLS_LAM:g}, penalty={ASLS_P}, "
                f"nr_iterations={ASLS_ITERATIONS}), which solves the penalised system "
                "with a banded Cholesky factorisation (scipy.linalg.solveh_banded) and "
                "runs a fixed iteration count. Ours solves the sparse system with "
                "spsolve and stops at an exact fixed point of the reweighting, capped "
                "at the same 20. Two solvers, one formulation."
            ),
            notes=(
                f"{baseline_where} Eilers and Boelens's own lam and p. chemotools "
                "performs no convergence test at all, so the iteration counts are "
                "matched deliberately; once our reweighting reaches a fixed point the "
                "remaining iterations change nothing, so the two agree whether ours "
                "stopped early or ran to the cap. This is the entry the fixture most "
                "lacked - two independent implementations of the same paper."
            ),
            **common,
        ),
        entry(
            entry_id=f"{name}.preprocess.baseline_rubberband.chemotools",
            quantity="baseline_rubberband",
            value=RubberbandCorrection().fit_transform(baseline_block).tolist(),
            preprocessing=["baseline_rubberband"],
            algorithm_variant=(
                "chemotools.baseline.RubberbandCorrection, the lower convex hull "
                "interpolated linearly. Ours walks the hull with Andrew's monotone "
                "chain in integer index arithmetic."
            ),
            notes=(
                f"{baseline_where} Bit-identical to ours on all three datasets, which "
                "is what an exact geometric construction should be: the hull is "
                "decided by comparisons rather than by arithmetic, so two correct "
                "implementations cannot differ by rounding."
            ),
            **common,
        ),
        entry(
            entry_id=f"{name}.preprocess.baseline_polynomial.chemotools",
            quantity="baseline_polynomial",
            value=PolynomialCorrection(order=POLYNOMIAL_ORDER)
            .fit_transform(baseline_block)
            .tolist(),
            preprocessing=[f"baseline_polynomial_{POLYNOMIAL_ORDER}"],
            algorithm_variant=(
                f"chemotools.baseline.PolynomialCorrection(order={POLYNOMIAL_ORDER}), "
                "fitted against the raw variable index over the whole spectrum. Ours "
                "maps the index onto [-1, 1] before building the Vandermonde matrix, "
                "which spans the same polynomial space and conditions the fit."
            ),
            notes=(
                f"{baseline_where} Degree {POLYNOMIAL_ORDER}. The two agree to 4e-15 "
                "despite fitting against differently scaled abscissae, which is the "
                "evidence that the [-1, 1] mapping is exactly the affine change of "
                "variable smoothing-and-baselines.md claims it is."
            ),
            **common,
        ),
    ]


def preprocessing_entries(name: str, dataset: ReferenceDataset) -> list[dict[str, Any]]:
    """Reference values for the scaling kernels, from scikit-learn.

    scikit-learn has no SNV or MSC, so those two have no external reference
    here and are checked against their defining identities instead. That is
    recorded as a gap in `unsourced_entries()` rather than left implicit.
    """
    block = dataset.spectra[:PREPROCESS_ROWS, :PREPROCESS_COLUMNS]
    content_hash = dataset.source.file_hash
    citation = (
        f"Generated by tests/fixtures/generate_reference_values.py against "
        f"scikit-learn {SKLEARN_VERSION}. Not a literature value: an open "
        f"implementation pinned by version, reproducible by anyone."
    )
    common = {
        "dataset": name,
        "dataset_content_hash": content_hash,
        "algorithm": "preprocess",
        "split": None,
        "software": "scikit-learn",
        "software_version": SKLEARN_VERSION,
        "citation": citation,
    }
    where = (
        f"Computed on the first {PREPROCESS_ROWS} samples and first "
        f"{PREPROCESS_COLUMNS} variables of the dataset. The transform is "
        "independent per row or per column, so the block tests it completely."
    )

    entries = [
        entry(
            entry_id=f"{name}.preprocess.mean_centred.sklearn",
            quantity="mean_centred",
            value=StandardScaler(with_std=False).fit_transform(block).tolist(),
            preprocessing=["mean_centre"],
            algorithm_variant="sklearn.preprocessing.StandardScaler(with_std=False).",
            notes=f"{where} Column means subtracted; no scaling.",
            **common,
        ),
        entry(
            entry_id=f"{name}.preprocess.autoscaled_ddof0.sklearn",
            quantity="autoscaled",
            value=StandardScaler().fit_transform(block).tolist(),
            preprocessing=["autoscale"],
            algorithm_variant="sklearn.preprocessing.StandardScaler(), which is fixed at ddof=0.",
            notes=(
                f"{where} **ddof=0**, because StandardScaler offers no choice. Our "
                "AutoscaleTransformer defaults to ddof=1, the sample convention used "
                "for eigenvalues and for SEC and SEP, so parity against this entry "
                "must pass ddof=0 explicitly. That difference is a convention, not a "
                "defect."
            ),
            **common,
        ),
    ]
    scipy_citation = (
        f"Generated by tests/fixtures/generate_reference_values.py against "
        f"SciPy {SCIPY_VERSION}. Not a literature value: an open implementation "
        f"pinned by version, reproducible by anyone."
    )
    savgol_common = {
        **common,
        "software": "SciPy",
        "software_version": SCIPY_VERSION,
        "citation": scipy_citation,
    }
    savgol_variant = (
        f"scipy.signal.savgol_filter(window_length={SAVGOL_WINDOW}, "
        f"polyorder={SAVGOL_POLYORDER}, mode='interp', delta=1.0), applied along the "
        "variable axis. SciPy solves a least-squares system per output position; our "
        "kernel builds one convolution matrix from the pseudo-inverse of the window's "
        "Vandermonde matrix. Same filter, independent code paths - scipy.signal is "
        "never called by the kernel."
    )
    savgol_note = (
        f"{where} Window {SAVGOL_WINDOW}, polyorder {SAVGOL_POLYORDER}, edge mode "
        f"**interp**: with a half-window of {SAVGOL_WINDOW // 2}, the first and last "
        f"{SAVGOL_WINDOW // 2} of the {PREPROCESS_COLUMNS} columns are evaluated off "
        "the centre of the end window rather than padded, so this block tests the "
        "edge convention and not only the interior. Derivatives are per variable "
        "index (delta=1), not per axis unit."
    )
    entries += [
        entry(
            entry_id=f"{name}.preprocess.savgol_deriv{deriv}.scipy",
            quantity=f"savgol_deriv{deriv}",
            value=savgol_filter(
                block, SAVGOL_WINDOW, SAVGOL_POLYORDER, deriv=deriv, mode="interp", axis=-1
            ).tolist(),
            preprocessing=[f"savgol_{SAVGOL_WINDOW}_{SAVGOL_POLYORDER}_{deriv}"],
            algorithm_variant=savgol_variant,
            notes=savgol_note,
            **savgol_common,
        )
        for deriv in (0, 1, 2)
    ]
    entries += [
        entry(
            entry_id=f"{name}.preprocess.normalised_{norm}.sklearn",
            quantity=f"normalised_{norm}",
            value=normalize(block, norm=norm).tolist(),
            preprocessing=[f"normalise_{norm}"],
            algorithm_variant=f"sklearn.preprocessing.normalize(norm={norm!r}), row-wise.",
            notes=f"{where} Each row divided by its {norm} norm.",
            **common,
        )
        for norm in ("l1", "l2", "max")
    ]
    return entries


def literature_entries() -> list[dict[str, Any]]:
    """Values taken from published documents rather than generated here.

    These are the entries that make the fixture worth more than a snapshot of
    our own tooling: an independent implementation, in a published document,
    with its configuration stated.
    """
    pls_citation = (
        "Mevik, B.-H., Wehrens, R. and Liland, K. H., 'Introduction to the pls "
        "Package', vignette for R package pls, "
        "https://cran.r-project.org/web/packages/pls/vignettes/pls-manual.html "
        "(retrieved 2026-08-22). Underlying data: Kalivas, J. H. (1997), "
        "'Two data sets of near infrared spectra', Chemometrics and Intelligent "
        "Laboratory Systems 37, 255-259."
    )
    pls_split = {
        "strategy": "leave_one_out",
        "note": (
            "LOO over the first 50 rows. Deterministic, so no seed and no shuffle "
            "stream to reconcile (metrics-and-validation.md §8.4) - which is why this "
            "is the strongest reference in the fixture."
        ),
        "calibration_indices": list(range(50)),
        "held_out_indices": list(range(50, 60)),
    }
    pls_variant = (
        "R pls::plsr, defaults read from the pls 2.8-5 source rather than assumed: "
        "method='kernelpls' (pls.options.R), center=TRUE, scale=FALSE (mvr.R). "
        "kernelpls and NIPALS agree on coefficients and predictions for a single "
        "response (pls-regression.md §2)."
    )

    return [
        entry(
            entry_id="gasoline.pls.rmsecv_curve.r_pls_vignette",
            dataset="gasoline",
            dataset_content_hash=None,
            algorithm="pls",
            quantity="rmsecv_curve",
            value={
                "0": 1.545,
                "1": 1.357,
                "2": 0.2966,
                "3": 0.2524,
                "4": 0.2476,
                "5": 0.2398,
                "6": 0.2319,
                "7": 0.2386,
                "8": 0.2316,
                "9": 0.2449,
                "10": 0.2673,
            },
            preprocessing=["mean_centre_x", "mean_centre_y"],
            algorithm_variant=pls_variant,
            split=pls_split,
            software="R pls",
            software_version="2.8-5",
            citation=pls_citation,
            notes=(
                "The vignette's 'CV' row, printed to four significant figures. Key "
                "'0' is the intercept-only model. R pls computes MSEP as SSE/nobj "
                "(mvrVal.R), so its divisor is n and the values are directly "
                "comparable with our RMSECV with no definitional correction. The "
                "vignette's 'adjCV' row is a bias-corrected estimate we do not report "
                "and is deliberately not recorded here. Do not read this curve against "
                "gasoline.pls.rmsecv_curve.sklearn as if the two were the same "
                "experiment: that one is 5-fold over all 60 samples, this one is LOO "
                "over the first 50. They agree closely from about five components "
                "(0.2398 here against 0.2396 there) and differ at two, where the "
                "estimate is far more sensitive to the split."
            ),
        ),
        entry(
            entry_id="gasoline.pls.explained_variance.r_pls_vignette",
            dataset="gasoline",
            dataset_content_hash=None,
            algorithm="pls",
            quantity="cumulative_explained_variance_at_2_components",
            value={"x": 0.8558, "y": 0.9685},
            preprocessing=["mean_centre_x", "mean_centre_y"],
            algorithm_variant=pls_variant,
            split={"strategy": "none", "calibration_indices": list(range(50))},
            software="R pls",
            software_version="2.8-5",
            citation=pls_citation,
            notes=(
                "Cumulative percentages from the vignette's summary(), converted to "
                "fractions: X 85.58%, octane 96.85%. Calibration rows 0-49 only."
            ),
        ),
        entry(
            entry_id="tecator.pls.sep.thodberg",
            dataset="tecator",
            dataset_content_hash=None,
            algorithm="pls",
            quantity="sep",
            value=2.78,
            preprocessing=["principal_components_supplied_with_the_dataset"],
            algorithm_variant="Linear model on 10 inputs.",
            split={
                "strategy": "external_set",
                "note": "Trained on C+M (samples 0-171), evaluated on T (samples 172-214).",
            },
            software="unstated",
            software_version="unstated",
            citation=(
                "tecator.txt section 4, distributed with the dataset; the surrounding "
                "results are Borggaard, C. and Thodberg, H. H. (1992), 'Optimal "
                "Minimal Neural Interpretation of Spectra', Analytical Chemistry 64, "
                "545-551."
            ),
            comparable=False,
            notes=(
                "Recorded because it is the only published number that travels with "
                "this dataset, and #14 will want it for context. NOT a parity target: "
                "the ten inputs are the first ten of the 22 principal components "
                "supplied in the file, which load_tecator() discards, and the file "
                "does not define its SEP denominator. Reproducing it would mean "
                "modelling the authors' 1992 preprocessing rather than ours."
            ),
        ),
    ]


def unsourced_entries() -> list[dict[str, Any]]:
    """Values that were looked for and not found.

    Recorded rather than omitted, because a gap that is written down is a task
    and a gap that is not is a false impression of coverage.

    SNV, MSC and the three baseline methods used to be here, on the grounds
    that neither scikit-learn nor SciPy implements them. #13 evaluated
    `chemotools`, which does, and #27 sourced all five from it - see
    `chemotools_entries()`. What is left is the R `mdatools` limits, which need
    an R installation (#24), and a corn PCA loading vector, which needs a paper
    that states its preprocessing chain precisely enough to reproduce.
    """
    no_r = (
        "R is not installed on the development machine, so no mdatools value could "
        "be generated. Install R and the mdatools package, run the configuration "
        "named above, and replace this entry with the result."
    )
    common: dict[str, Any] = {
        "value": None,
        "status": "unsourced",
        "comparable": False,
        "dataset_content_hash": None,
        "software": "R mdatools",
        "software_version": "not installed",
        "citation": "none - not sourced",
        "split": None,
    }
    entries = [
        entry(
            entry_id=f"{name}.{algorithm}.{quantity}.r_mdatools",
            dataset=name,
            algorithm=algorithm,
            quantity=quantity,
            preprocessing=["mean_centre_x"]
            if algorithm == "pca"
            else ["mean_centre_x", "mean_centre_y"],
            algorithm_variant=(
                "mdatools::pca, for the T2 and SPE limits scikit-learn does not "
                "provide (pca.md §14)."
                if algorithm == "pca"
                else "mdatools::pls, SIMPLS. Valid for coefficients and predictions "
                "only, not weights and loadings (pls-regression.md §2)."
            ),
            notes=no_r,
            **common,
        )
        for name in TARGETS
        for algorithm, quantity in (("pca", "spe_limit"), ("pls", "coefficients"))
    ]

    entries.append(
        entry(
            entry_id="corn.pca.loadings.literature",
            dataset="corn",
            dataset_content_hash=None,
            algorithm="pca",
            quantity="loadings",
            value=None,
            status="unsourced",
            comparable=False,
            preprocessing=["mean_centre_x"],
            algorithm_variant="unknown",
            split=None,
            software="published literature",
            software_version="n/a",
            citation="none - not sourced",
            notes=(
                "The corn dataset is used widely in the calibration-transfer "
                "literature, but the papers report transfer performance rather than "
                "the decomposition itself, and none found states a preprocessing "
                "chain precisely enough to reproduce a loading vector. Recorded as a "
                "gap rather than filled with a number from a plot."
            ),
        )
    )
    return entries


def main() -> None:
    _self_check()

    datasets = {
        "corn": load_corn(),
        "gasoline": load_gasoline(),
        "tecator": load_tecator(),
    }

    entries: list[dict[str, Any]] = []
    for name, dataset in datasets.items():
        entries.extend(sklearn_entries(name, dataset))
        entries.extend(preprocessing_entries(name, dataset))
        entries.extend(chemotools_entries(name, dataset))
    entries.extend(literature_entries())
    entries.extend(unsourced_entries())

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "tests/fixtures/generate_reference_values.py",
        "conventions": (
            "Every value follows the definitions in docs/algorithms/. Metrics are in "
            "the original units of the response; RMSE denominators are n; fold "
            "aggregation is pooled residuals; signs follow the generating software "
            "and must be aligned by inner product before comparison."
        ),
        "targets": TARGETS,
        "entries": entries,
    }

    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    sourced = sum(1 for e in entries if e["status"] == "sourced")
    print(f"wrote {FIXTURE} - {len(entries)} entries, {sourced} sourced")


if __name__ == "__main__":
    main()
