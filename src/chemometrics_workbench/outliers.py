"""Outlier diagnostics, per `docs/algorithms/outliers.md`.

Three measures on a fitted model's scores, beside the `T^2` and `Q` that
`decomposition.py` already computes: leverage, the studentised residual of a
regression, and a robust Mahalanobis distance through a hand-written FastMCD.
Each is a pure function of arrays the executor already stores, so the
diagnostics of a result fitted before they existed are the same numbers.

They **flag**; nothing here removes a sample. Excluding one is the user's
decision, made as a derived dataset version (`PROPOSAL.md` §16, #279).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from chemometrics_workbench.arrays import as_float64, as_float64_vector

__all__ = [
    "LEVERAGE_FACTOR",
    "MCD_SEED",
    "RESIDUAL_LIMIT",
    "ROBUST_QUANTILE",
    "RobustCovariance",
    "leverage",
    "leverage_limit",
    "min_cov_det",
    "robust_distance_limit",
    "studentised_residuals",
]

#: §2: a sample is flagged at three times the average leverage, `(A + 1) / n`.
LEVERAGE_FACTOR = 3.0

#: §3: a studentised residual beyond three in either direction.
RESIDUAL_LIMIT = 3.0

#: §4: the chi-squared quantile both the reweighting and the flag use.
ROBUST_QUANTILE = 0.975

#: §4: FastMCD draws random starting subsets from this fixed seed, so the same
#: scores always give the same distances.
MCD_SEED = 0

#: §4, Rousseeuw and Van Driessen (1999): starting subsets, and how many of the
#: best after two C-steps are iterated to convergence.
_N_TRIALS = 500
_N_BEST = 10
_MAX_STEPS = 100


def leverage(scores: object) -> NDArray[np.float64]:
    """§2: the diagonal of the hat matrix of `[1, T]`.

    Computed from an orthonormal basis of that design, so it holds whether or
    not the scores are centred or mutually orthogonal: `h_i = sum_j Q_ij^2`.
    For a centred model with orthogonal scores it is `1/n + sum_a t_ia^2 / t_a't_a`.
    """
    values = as_float64(scores, "scores")
    design = np.column_stack([np.ones(values.shape[0]), values])
    basis, triangle = np.linalg.qr(design)
    if np.min(np.abs(np.diag(triangle))) <= 1e-12 * np.max(np.abs(np.diag(triangle))):
        raise ValueError("the scores and an intercept are linearly dependent")
    hat: NDArray[np.float64] = (basis**2).sum(axis=1)
    return hat


def leverage_limit(n_samples: int, n_components: int) -> float:
    """§2: `3 (A + 1) / n`, three times the mean of the hat diagonal."""
    return LEVERAGE_FACTOR * (n_components + 1) / n_samples


def studentised_residuals(residuals: object, hat: object, n_components: int) -> NDArray[np.float64]:
    """§3: internally studentised, `e_i / (s sqrt(1 - h_i))`, `s^2 = SSE / (n - A - 1)`.

    `residuals` are observed less fitted on the calibration rows. Raises when
    there are no residual degrees of freedom; a row with `h_i = 1` is fitted
    exactly and has no studentised residual, so it is `NaN`.
    """
    e = as_float64_vector(residuals, "residuals")
    h = as_float64_vector(hat, "leverage")
    if e.shape != h.shape:
        raise ValueError(f"{e.size} residuals and {h.size} leverages")
    dof = e.size - n_components - 1
    if dof <= 0:
        raise ValueError(
            f"{e.size} samples leave no residual degrees of freedom for {n_components} "
            "components and an intercept"
        )
    s = float(np.sqrt((e**2).sum() / dof))
    spread = 1.0 - h
    with np.errstate(divide="ignore", invalid="ignore"):
        studentised: NDArray[np.float64] = np.where(
            spread > 1e-12, e / (s * np.sqrt(spread)), np.nan
        )
    return studentised


@dataclass(frozen=True)
class RobustCovariance:
    """§4: the reweighted MCD location and covariance, and every row's distance."""

    location: NDArray[np.float64]
    covariance: NDArray[np.float64]
    support: NDArray[np.bool_]
    distances: NDArray[np.float64]
    """Squared robust Mahalanobis distances, `d_i^2`."""


def robust_distance_limit(n_components: int) -> float:
    """§4: `chi^2_{0.975}(A)`, the cut the reweighting also uses."""
    return float(chi2.ppf(ROBUST_QUANTILE, n_components))


def min_cov_det(X: object, seed: int = MCD_SEED) -> RobustCovariance:
    """§4: FastMCD with the consistency correction and one reweighting step."""
    values = as_float64(X, "X")
    n, p = values.shape
    h = min(int(np.ceil(0.5 * (n + p + 1))), n)
    if n <= p + 1:
        raise ValueError(f"{n} samples cannot support a robust covariance in {p} dimensions")
    if p == 1:
        location, covariance = _univariate(values, h)
    else:
        location, covariance = _fast_mcd(values, h, np.random.default_rng(seed))

    # Consistency correction (Croux and Haesbroeck 1999): scale the raw
    # distances so they are chi-squared at the normal model, then reweight.
    raw = _squared_distances(values, location, covariance) / _consistency(p, h / n)
    keep = raw < chi2.ppf(ROBUST_QUANTILE, p)
    location = values[keep].mean(axis=0)
    covariance = _covariance(values[keep]) * _consistency(p, ROBUST_QUANTILE)
    return RobustCovariance(
        location=location,
        covariance=covariance,
        support=keep,
        distances=_squared_distances(values, location, covariance),
    )


def _fast_mcd(
    values: NDArray[np.float64], h: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rousseeuw and Van Driessen's FastMCD: random `(p + 1)`-subsets, two
    C-steps each, the best ten iterated until the determinant stops falling.

    The starts are drawn one at a time, so a seed always gives the same ones,
    and then stepped together as stacked arrays (#314): 500 starts as 500
    loops of small linear algebra cost 1.5 s at 3,000 samples."""
    n, p = values.shape
    locations: list[NDArray[np.float64]] = []
    covariances: list[NDArray[np.float64]] = []
    for _ in range(_N_TRIALS):
        start = rng.choice(n, size=p + 1, replace=False)
        covariance = _covariance(values[start])
        # A singular start grows by one random row until it is not (§4).
        while np.linalg.matrix_rank(covariance) < p and start.size < n:
            rest = np.setdiff1d(np.arange(n), start)
            start = np.append(start, rng.choice(rest))
            covariance = _covariance(values[start])
        if np.linalg.matrix_rank(covariance) < p:
            continue
        locations.append(values[start].mean(axis=0))
        covariances.append(covariance)
    if not locations:
        raise ValueError("every starting subset was singular; the scores have no full-rank spread")

    supports = _nearest_many(values, np.asarray(locations), np.asarray(covariances), h)
    log_dets = _log_dets(values, supports)
    # Two C-steps for every start at once; a start stops as soon as one fails
    # to lower its determinant, exactly as `_c_steps` would stop it.
    active = np.ones(len(supports), dtype=bool)
    for _ in range(2):
        rows = values[supports[active]]
        following = _nearest_many(values, rows.mean(axis=1), _covariances(rows), h)
        following_dets = _log_dets(values, following)
        better = following_dets < log_dets[active]
        indices = np.flatnonzero(active)
        supports[indices[better]] = following[better]
        log_dets[indices[better]] = following_dets[better]
        active[indices[~better]] = False
        if not active.any():
            break

    order = np.argsort(log_dets, kind="stable")[:_N_BEST]
    best = min(
        (_c_steps(values, supports[i], h, _MAX_STEPS) for i in order),
        key=lambda item: item[0],
    )
    rows = values[best[1]]
    return rows.mean(axis=0), _covariance(rows)


def _nearest_many(
    values: NDArray[np.float64],
    locations: NDArray[np.float64],
    covariances: NDArray[np.float64],
    h: int,
) -> NDArray[np.intp]:
    """`_nearest` for a stack of estimates: each row of the result is one
    estimate's `h` nearest rows, sorted. Taken in blocks of estimates, so the
    `estimates x n x p` intermediate stays near 4 million numbers."""
    n, p = values.shape
    block = max(1, 4_000_000 // (n * p))
    precisions = np.linalg.pinv(covariances, hermitian=True)
    out = np.empty((len(locations), h), dtype=np.intp)
    for first in range(0, len(locations), block):
        last = first + block
        centred = values[None, :, :] - locations[first:last, None, :]
        distances = ((centred @ precisions[first:last]) * centred).sum(axis=2)
        out[first:last] = np.sort(np.argsort(distances, axis=1, kind="stable")[:, :h], axis=1)
    return out


def _covariances(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """`_covariance` for a stack of row sets, `estimates x h x p`."""
    centred = rows - rows.mean(axis=1, keepdims=True)
    covariances: NDArray[np.float64] = centred.transpose(0, 2, 1) @ centred / rows.shape[1]
    return covariances


def _log_dets(values: NDArray[np.float64], supports: NDArray[np.intp]) -> NDArray[np.float64]:
    """`_log_det` for every support at once."""
    signs, log_dets = np.linalg.slogdet(_covariances(values[supports]))
    out: NDArray[np.float64] = np.where(signs > 0, log_dets, -np.inf)
    return out


def _c_steps(
    values: NDArray[np.float64], support: NDArray[np.intp], h: int, steps: int
) -> tuple[float, NDArray[np.intp]]:
    """C-steps from `support`: refit on it, keep the `h` nearest, repeat until
    the support stops changing or `steps` run out. The determinant never rises."""
    log_det = _log_det(values[support])
    for _ in range(steps):
        rows = values[support]
        following = _nearest(values, rows.mean(axis=0), _covariance(rows), h)
        following_det = _log_det(values[following])
        if following_det >= log_det:
            break
        support, log_det = following, following_det
    return log_det, support


def _univariate(
    values: NDArray[np.float64], h: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One dimension is exact: the shortest half, the window of `h` sorted
    values with the smallest range, and the mean of its two ends."""
    x = values[:, 0]
    n = x.size
    if h == n:
        return np.array([x.mean()]), np.array([[x.var()]])
    ordered = np.sort(x)
    widths = ordered[h:] - ordered[: n - h]
    starts = np.flatnonzero(widths == widths.min())
    centre = 0.5 * float((ordered[h + starts] + ordered[starts]).mean())
    support = np.argsort(np.abs(x - centre), kind="stable")[:h]
    return np.array([centre]), np.array([[x[support].var()]])


def _nearest(
    values: NDArray[np.float64],
    location: NDArray[np.float64],
    covariance: NDArray[np.float64],
    h: int,
) -> NDArray[np.intp]:
    order = np.argsort(_squared_distances(values, location, covariance), kind="stable")
    return np.sort(order[:h])


def _squared_distances(
    values: NDArray[np.float64], location: NDArray[np.float64], covariance: NDArray[np.float64]
) -> NDArray[np.float64]:
    centred = values - location
    precision = np.linalg.pinv(covariance, hermitian=True)
    distances: NDArray[np.float64] = (centred @ precision * centred).sum(axis=1)
    return distances


def _covariance(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Maximum likelihood, divided by the row count, as the MCD defines it."""
    centred = rows - rows.mean(axis=0)
    covariance: NDArray[np.float64] = centred.T @ centred / rows.shape[0]
    return covariance


def _log_det(rows: NDArray[np.float64]) -> float:
    sign, log_det = np.linalg.slogdet(_covariance(rows))
    return float(log_det) if sign > 0 else -np.inf


def _consistency(p: int, fraction: float) -> float:
    """Makes a covariance from the inner `fraction` of a normal sample consistent
    (Croux and Haesbroeck 1999, as Pison et al. 2002 write it)."""
    quantile = chi2.ppf(fraction, p)
    return float(fraction / chi2.cdf(quantile, p + 2))
