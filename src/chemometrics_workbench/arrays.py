"""The array contract every kernel enforces, in one place.

Shape, dtype and missing values are decided identically for preprocessing and
for the estimators, so the check lives here rather than being written twice
and drifting. `pca.md` §10 and §11 are the normative statements of it:
computation is float64 whatever the caller stores, the caller's array is never
modified, and a missing value is an error that names its position rather than
something quietly imputed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["as_float64"]


def as_float64(array: object, name: str) -> NDArray[np.float64]:
    """Promote to float64 and reject anything that is not a finite 2-D matrix."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D, n_samples x n_variables; got shape {values.shape}. "
            "A single spectrum is a 1 x n_variables matrix, not a 1-D array."
        )
    if not np.isfinite(values).all():
        rows, columns = np.nonzero(~np.isfinite(values))
        raise ValueError(
            f"{name} holds {rows.size} non-finite values, first at "
            f"row {rows[0]}, column {columns[0]}. Missing values are handled upstream "
            "and visibly: exclude the sample, exclude the variable, or add an "
            "imputation step to the pipeline."
        )
    return values
