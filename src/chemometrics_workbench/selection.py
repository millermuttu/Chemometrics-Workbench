"""Variable selection kernels, per `docs/algorithms/variable-selection.md`.

iPLS (#282) and CARS (#283) are here. A selection is never applied by the kernel: it returns
the positions, and the user applies them as a `select_variables` step (#280),
so what was selected is part of the recipe and its lineage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.arrays import as_float64_vector
from chemometrics_workbench.regression import PLS, _fold_matrices, rmsecv_curve
from chemometrics_workbench.validation import Fold

__all__ = ["CARSResult", "CARSRun", "IPLSResult", "Interval", "cars", "interval_bounds", "ipls"]


@dataclass(frozen=True)
class Interval:
    """One interval's own model (§2): columns `start` to `stop - 1`."""

    start: int
    stop: int
    rmsecv: float
    n_components: int


@dataclass(frozen=True)
class IPLSResult:
    """§2 and §3: every interval alone, the full spectrum, and the forward path."""

    intervals: list[Interval]
    full_rmsecv: float
    full_components: int
    steps: list[tuple[int, float]]
    """Each forward step: the interval added and the RMSECV of the union so far."""

    @property
    def selected(self) -> list[int]:
        """The column positions of the intervals forward selection kept, sorted."""
        chosen = sorted(index for index, _ in self.steps)
        return [
            column
            for index in chosen
            for column in range(self.intervals[index].start, self.intervals[index].stop)
        ]


def interval_bounds(n_variables: int, n_intervals: int) -> list[tuple[int, int]]:
    """§1: `n_intervals` contiguous intervals, as equal as the count allows,
    the first `n_variables % n_intervals` one variable wider."""
    if not 2 <= n_intervals <= n_variables:
        raise ValueError(
            f"{n_intervals} intervals cannot be cut from {n_variables} variables; "
            f"choose between 2 and {n_variables}"
        )
    quotient, remainder = divmod(n_variables, n_intervals)
    bounds: list[tuple[int, int]] = []
    start = 0
    for k in range(n_intervals):
        stop = start + quotient + (1 if k < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def ipls(
    X: object,
    y: object,
    folds: list[Fold],
    n_intervals: int,
    max_components: int,
) -> IPLSResult:
    """§2 and §3: interval PLS and forward interval selection.

    `X` is one matrix, or one per fold as `rmsecv_curve` takes it, so the
    cross-validation below a split uses each fold's own preprocessing.
    """
    if max_components < 1:
        raise ValueError(f"max_components must be at least 1, got {max_components}")
    if len(folds) < 2:
        raise ValueError("iPLS compares cross-validated errors, and needs at least two folds")
    matrices = _fold_matrices(X, folds)
    width = matrices[0].shape[1]
    bounds = interval_bounds(width, n_intervals)

    def best(columns: NDArray[np.intp]) -> tuple[float, int]:
        # §2: the curve's first minimum, over as many components as the
        # columns can carry.
        components = min(max_components, columns.size)
        curve = rmsecv_curve([m[:, columns] for m in matrices], y, folds, components)
        a = int(np.argmin(curve))
        return float(curve[a]), a + 1

    intervals = []
    for start, stop in bounds:
        rmsecv, a = best(np.arange(start, stop))
        intervals.append(Interval(start, stop, rmsecv, a))
    full_rmsecv, full_components = best(np.arange(width))

    # §3: add whichever interval lowers the union's RMSECV most; stop when
    # none lowers it. A tie goes to the earlier interval.
    steps: list[tuple[int, float]] = []
    chosen: set[int] = set()
    current = np.inf
    while len(chosen) < len(bounds):
        trial = [
            (best(_columns(bounds, chosen | {k}))[0], k)
            for k in range(len(bounds))
            if k not in chosen
        ]
        rmsecv, k = min(trial)
        if rmsecv >= current:
            break
        chosen.add(k)
        steps.append((k, rmsecv))
        current = rmsecv
    return IPLSResult(intervals, full_rmsecv, full_components, steps)


def _columns(bounds: list[tuple[int, int]], chosen: set[int]) -> NDArray[np.intp]:
    return np.concatenate([np.arange(*bounds[k]) for k in sorted(chosen)])


# --------------------------------------------------------------------------
# CARS, section 6
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CARSRun:
    """One sampling run (§6): the variables it kept and their RMSECV."""

    variables: list[int]
    rmsecv: float
    n_components: int


@dataclass(frozen=True)
class CARSResult:
    runs: list[CARSRun]
    best: int
    """The run with the lowest RMSECV; the first, on a tie."""

    @property
    def selected(self) -> list[int]:
        return self.runs[self.best].variables


def cars(
    X: object,
    y: object,
    folds: list[Fold],
    max_components: int,
    *,
    n_runs: int = 50,
    fraction: float = 0.8,
    seed: int = 0,
) -> CARSResult:
    """§6: competitive adaptive reweighted sampling (Li et al. 2009).

    Each run fits a PLS on a random `fraction` of the samples, keeps the
    variables with the largest |b| on an exponentially decreasing schedule,
    then resamples those by weight. Every run's subset is cross-validated as
    an iPLS interval is (§2), and the subset with the lowest RMSECV is the
    selection. The sampling draws from one seeded generator, so a seed always
    gives the same runs.
    """
    if max_components < 1:
        raise ValueError(f"max_components must be at least 1, got {max_components}")
    if n_runs < 2:
        raise ValueError(f"CARS needs at least two sampling runs, got {n_runs}")
    if not 0 < fraction < 1:
        raise ValueError(f"the sampled fraction must be between 0 and 1, got {fraction}")
    if len(folds) < 2:
        raise ValueError("CARS compares cross-validated errors, and needs at least two folds")
    matrices = _fold_matrices(X, folds)
    response = as_float64_vector(y, "y")
    # The sampling fits use fold zero's matrix, every row of it
    # (`variable-selection.md` §6; #331 revisits it).
    values = matrices[0]
    n, p = values.shape
    if p < 3:
        raise ValueError(f"CARS selects from at least three variables, and there are {p}")
    rng = np.random.default_rng(seed)
    # §6: r_i = a exp(-k i), so that r_1 = 1 and r_N = 2 / p.
    k = np.log(p / 2) / (n_runs - 1)
    a = (p / 2) ** (1 / (n_runs - 1))
    take = max(2, round(fraction * n))

    retained = np.arange(p)
    runs: list[CARSRun] = []
    for i in range(1, n_runs + 1):
        rows = rng.choice(n, size=take, replace=False)
        block = values[np.ix_(rows, retained)]
        target = response[rows]
        model = PLS(min(max_components, retained.size, take - 1)).fit(
            block - block.mean(axis=0), target - target.mean()
        )
        weights = np.abs(np.asarray(model.coefficients_, dtype=np.float64).ravel())
        if not weights.sum() > 0:
            weights = np.ones_like(weights)
        keep = max(2, round(a * np.exp(-k * i) * p))
        # Forced removal: the `keep` heaviest, ties to the earlier position.
        order = np.argsort(-weights, kind="stable")[:keep]
        survivors, chances = retained[order], weights[order] / weights[order].sum()
        # Adaptive reweighted sampling: `keep` draws by weight, with replacement.
        drawn = rng.choice(survivors.size, size=keep, replace=True, p=chances)
        retained = np.sort(np.unique(survivors[drawn]))

        components = min(max_components, retained.size)
        curve = rmsecv_curve([m[:, retained] for m in matrices], response, folds, components)
        best_a = int(np.argmin(curve))
        runs.append(CARSRun([int(v) for v in retained], float(curve[best_a]), best_a + 1))
    best = int(np.argmin([run.rmsecv for run in runs]))
    return CARSResult(runs, best)
