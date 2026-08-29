"""Metrics and fold assignment, per `docs/algorithms/metrics-and-validation.md`.

The document is normative and this module implements it; every function names
the section it comes from. Where the two disagree, one of them is a bug —
decide which before changing either.

## Why the metrics are here rather than written out where they are used

There are four places a root-mean-square error could be spelled out — the
kernel, the fixture generator, the parity suite and the report — and a project
whose whole claim is *our numbers are defined and reproducible* cannot afford
four spellings of one definition. §4 fixes the divisor at `n`; a package
dividing by `n - A - 1` under the name RMSEC is reporting what §5 calls SEC,
and that is the single most common cause of a third-significant-figure
mismatch against another tool.

## Folds are data, not a seed

`k_fold()` returns realised index arrays, and every function that
cross-validates takes those arrays rather than a seed. `metrics-and-validation.md`
§10 is the reason: a seed only reproduces a split against a fixed generator
implementation, and our stream is `numpy.random.default_rng` (PCG64) where
scikit-learn's is a legacy `RandomState` — the same seed gives different
folds. A comparison that seeds both sides and compares is not a comparison,
and it passes.

That also makes `Fold` the seam `ResolvedSplit` is stored through: a run
recorded in the schema is replayed by handing its stored index lists back to
`folds_from_indices()`. The schema itself is deliberately not imported here —
these are arrays in, arrays out, as in `preprocessing.py` and
`decomposition.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.arrays import as_float64_vector

__all__ = [
    "Fold",
    "bias",
    "folds_from_indices",
    "k_fold",
    "leave_one_out",
    "q2",
    "r2",
    "rmse",
    "sec",
    "sep",
    "validate_partition",
]


# --------------------------------------------------------------------------
# metrics, §4 to §6
# --------------------------------------------------------------------------


def rmse(y: object, y_hat: object) -> float:
    """RMSEC, RMSECV and RMSEP are one formula over three sets (§4).

    **The divisor is `n`.** Degrees-of-freedom corrections live in SEC and SEP
    (§5) and are not applied here under another name.
    """
    residual = _residual(y, y_hat)
    return float(np.sqrt(np.mean(residual**2)))


def bias(y: object, y_hat: object) -> float:
    """The mean signed residual (§5), essentially zero on a calibration set."""
    return float(np.mean(_residual(y, y_hat)))


def r2(y: object, y_hat: object) -> float:
    """The residual form, not the squared Pearson correlation (§6).

    The two coincide for a least-squares fit on the data it was fitted to, and
    diverge on a prediction set, where the correlation is blind to bias and
    slope error and is therefore flattering. Where the correlation is wanted it
    is reported separately as `r2_pearson`, never as `r2`.
    """
    values = as_float64_vector(y, "y")
    residual = _residual(y, y_hat)
    total = float(np.sum((values - values.mean()) ** 2))
    if total == 0.0:
        raise ValueError(
            "R^2 is undefined: the response has no variance about its mean, so the "
            "denominator is zero. Report it as absent rather than as a number."
        )
    return float(1.0 - np.sum(residual**2) / total)


def sec(y: object, y_hat: object, *, n_components: int) -> float:
    """The standard error of calibration, §5: bias-corrected, `n - A - 1`.

    SEC and SEP describe the *scatter* of the residuals about their own mean
    where the RMSEs of §4 describe total error including any offset. The
    denominators differ deliberately and that difference is the whole reason
    these are separate functions from `rmse`: the calibration residuals come
    from a model that spent `A` latent variables plus an intercept fitting
    those same samples, so the naive variance is optimistic.

    **A package reporting `n - A - 1` under the name RMSEC is reporting this.**

    With `n - A - 1 <= 0` there is nothing to report: §5 says to report SEC as
    absent and say why, and never to fall back to another denominator, which
    would produce a number that is not SEC under a label that says it is. So
    this raises with both counts named, and the caller records the absence.
    """
    if n_components < 0:
        raise ValueError(f"n_components must not be negative, got {n_components}")
    residual = _residual(y, y_hat)
    degrees = residual.size - n_components - 1
    if degrees <= 0:
        raise ValueError(
            f"SEC is undefined: {residual.size} samples less {n_components} components "
            "less 1 for the intercept leaves no degrees of freedom. Report it as absent "
            "rather than dividing by something else and calling the result SEC."
        )
    centred = residual - residual.mean()
    return float(np.sqrt(np.sum(centred**2) / degrees))


def sep(y: object, y_hat: object) -> float:
    """The standard error of prediction, §5: bias-corrected, `n - 1`.

    The prediction samples took no part in the fit, so the only parameter
    estimated from them is the mean subtracted in the bias correction, and one
    degree of freedom is all that is lost. That is why this takes no component
    count and SEC does.

    §5 ties it to RMSEP exactly — `RMSEP^2 = bias^2 + (n-1)/n * SEP^2` — which
    is the cheap unit test, and `tests/test_validation.py` makes it.
    """
    residual = _residual(y, y_hat)
    if residual.size < 2:
        raise ValueError(
            f"SEP is undefined for {residual.size} sample(s): the bias correction "
            "leaves no degrees of freedom. Report it as absent."
        )
    centred = residual - residual.mean()
    return float(np.sqrt(np.sum(centred**2) / (residual.size - 1)))


def q2(y: object, y_cross_validated: object) -> float:
    """§6: `1 - PRESS / TSS`, over held-out predictions.

    The same formula as `r2` over different predictions, and that is the point
    rather than an accident. §6 puts both denominators on the total sum of
    squares of the calibration response about the **full calibration mean**, so
    the two are directly comparable and `Q^2 <= R^2` keeps its usual meaning.
    Recomputing the mean inside each fold changes the number, and packages
    differ on it.

    It is a separate name because the argument is different — one held-out
    prediction per sample, pooled across folds (§7), never a per-fold average.

    **Q² can be negative and is not clipped.** A negative Q² means the model
    predicts held-out samples worse than the calibration mean does, which is a
    real finding; reporting it as zero hides a failed model.
    """
    return r2(y, y_cross_validated)


# --------------------------------------------------------------------------
# fold assignment, §8
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One fold's realised index sets — what `ResolvedSplit` stores (§10)."""

    train: NDArray[np.intp]
    test: NDArray[np.intp]


def k_fold(n_samples: int, n_splits: int, *, shuffle: bool = True, seed: int = 42) -> list[Fold]:
    """K-fold assignment, reproducible by hand from §8.2 and §8.3.

    The permutation is `numpy.random.default_rng(seed).permutation(n)`, and the
    first `n % K` folds take one extra member — scikit-learn's size rule, kept
    deliberately so that the *only* difference between us is the permutation.
    With `shuffle=False` the permutation is the identity and the seed is
    ignored rather than silently accepted.
    """
    if n_splits < 2:
        raise ValueError(f"K-fold needs at least 2 folds, got {n_splits}")
    if n_splits > n_samples:
        raise ValueError(
            f"{n_splits} folds were asked of {n_samples} samples. K may not exceed n; "
            "K = n is leave-one-out and is expressed as such (§8.3)."
        )

    perm = (
        np.random.default_rng(seed).permutation(n_samples)
        if shuffle
        else np.arange(n_samples, dtype=np.intp)
    )
    quotient, remainder = divmod(n_samples, n_splits)

    folds: list[Fold] = []
    start = 0
    for k in range(n_splits):
        size = quotient + 1 if k < remainder else quotient
        test = np.sort(perm[start : start + size])
        folds.append(Fold(train=np.setdiff1d(np.arange(n_samples), test), test=test))
        start += size
    return folds


def leave_one_out(n_samples: int) -> list[Fold]:
    """`n` folds, fold `i` holding out sample `i` alone (§8.4).

    Deterministic: no shuffle and no seed, because its result cannot depend on
    one. It is also the most optimistic scheme for spectral data — replicate
    scans of one sample stay in the training set when their twin is held out —
    which the application says where it offers it.
    """
    if n_samples < 2:
        raise ValueError(f"leave-one-out needs at least 2 samples, got {n_samples}")
    every = np.arange(n_samples)
    return [Fold(train=np.delete(every, i), test=np.array([i], dtype=np.intp)) for i in every]


def folds_from_indices(
    train_indices: Sequence[Sequence[int]], test_indices: Sequence[Sequence[int]]
) -> list[Fold]:
    """Rebuild folds from stored index lists — a `ResolvedSplit` replayed (§10).

    This is the path a recorded experiment is rerun through, and the reason
    `ResolvedSplit` stores indices rather than a seed: they survive a change of
    random number generator, a library upgrade, and a switch of splitter.
    """
    if len(train_indices) != len(test_indices):
        raise ValueError(
            f"{len(train_indices)} training index sets against {len(test_indices)} test "
            "sets. A stored split has one of each per fold."
        )
    return [
        Fold(train=np.asarray(train, dtype=np.intp), test=np.asarray(test, dtype=np.intp))
        for train, test in zip(train_indices, test_indices, strict=True)
    ]


def validate_partition(folds: Sequence[Fold], n_samples: int) -> None:
    """§7: every sample is predicted exactly once, and no fold trains on its own test set.

    Asserted before residuals are pooled, because both failures produce a
    plausible RMSECV: a sample left out of every validation set silently
    shrinks the sum, and a sample in two of them is counted twice at a
    denominator that does not know it.
    """
    if not folds:
        raise ValueError("a split with no folds cross-validates nothing")

    held_out = np.concatenate([fold.test for fold in folds])
    if held_out.size != n_samples or not np.array_equal(np.sort(held_out), np.arange(n_samples)):
        raise ValueError(
            f"the validation sets do not partition the {n_samples} samples: they hold "
            f"{held_out.size} indices, {np.unique(held_out).size} of them distinct. Every "
            "sample must be predicted exactly once before residuals are pooled (§7)."
        )
    for k, fold in enumerate(folds):
        if np.intersect1d(fold.train, fold.test).size:
            raise ValueError(f"fold {k} trains on samples it also validates on")


# --------------------------------------------------------------------------
# shared checks
# --------------------------------------------------------------------------


def _residual(y: object, y_hat: object) -> NDArray[np.float64]:
    values = as_float64_vector(y, "y")
    predicted = as_float64_vector(y_hat, "y_hat")
    if values.shape != predicted.shape:
        raise ValueError(
            f"{values.size} reference values against {predicted.size} predictions. A "
            "metric is computed over one set, and mixing calibration with "
            "cross-validation in one number is never right (§2)."
        )
    return values - predicted
