"""MATLAB MAT-files (#284).

A MAT-file is a bag of named arrays, not a table. Which array is the spectra,
which way round it is stored, and which vector is the axis are decisions the
file does not state, so each is a `Choice` the import screen offers:

- `matrix`: the two-dimensional numeric arrays, the largest first;
- `orientation`: whether a sample is a row or a column of it. MATLAB code
  often stores variables down the rows - MLNIRdata, the fixture, holds 2635 x
  208 for 208 spectra - so the detection takes whichever way an axis-length
  vector or a sample-length vector agrees with;
- `axis_variable`: the vectors as long as the variable count, or none.

Every other vector as long as the sample count is a target. Formats 4 to 7.2
are read with SciPy's `loadmat`. Version 7.3 is HDF5 underneath, needs a
library this application does not carry, and is refused by name with what to
do about it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from numpy.typing import NDArray

from chemometrics_workbench.readers import (
    HEAD_ROWS,
    Choice,
    Detection,
    Imported,
    ReaderError,
    index_axis,
    source_file,
)
from chemometrics_workbench.readers.grid import axis_from_values

NAME = "matlab_mat"
VERSION = "1"
SUFFIXES: tuple[str, ...] = (".mat",)

NOT_APPLICABLE = "n/a"
NO_AXIS = "none"
ORIENTATIONS = ("samples_in_rows", "samples_in_columns")


def sniff(path: Path) -> Detection:
    arrays = _load(path)
    matrices = _matrices(arrays, path)
    return _detect(arrays, matrices[0], others=tuple(matrices[1:]))


def resniff(path: Path, detection: Detection) -> Detection:
    """Another matrix or the other orientation is a different table: its
    shape, its axis candidates and its targets all follow."""
    assert detection.matrix is not None and detection.axis_variable is not None
    arrays = _load(path)
    redone = _detect(
        arrays,
        detection.matrix.value,
        others=detection.matrix.alternatives,
        orientation=detection.orientation.value,
    )
    # The user's axis, if it still fits the table they now have.
    assert redone.axis_variable is not None
    wanted = detection.axis_variable.value
    offered = (redone.axis_variable.value, *redone.axis_variable.alternatives)
    if wanted in offered and wanted != redone.axis_variable.value:
        redone = _with_axis(arrays, redone, wanted)
    return redone


def read(path: Path, detection: Detection) -> Imported:
    arrays = _load(path)
    values = _values(arrays, detection)
    axis = _axis(arrays, detection)
    return Imported(
        values=values,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        targets={name: [float(v) for v in _vector(arrays[name])] for name in detection.targets},
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    values = _values(_load(path), detection)[:HEAD_ROWS]
    return {
        "sample_ids": [f"row {i}" for i in range(values.shape[0])],
        "rows": [[float(v) for v in row[:HEAD_ROWS]] for row in values],
    }


# --- detection ----------------------------------------------------------------


def _detect(
    arrays: dict[str, NDArray[np.float64]],
    matrix: str,
    *,
    others: tuple[str, ...],
    orientation: str | None = None,
) -> Detection:
    rows, columns = arrays[matrix].shape
    vectors = {name: _vector(value).size for name, value in arrays.items() if _is_vector(value)}
    if orientation is None:
        # Columns are samples when a vector as long as the rows exists and
        # none as long as the columns would make them variables.
        down = any(size == rows for size in vectors.values())
        across = any(size == columns for size in vectors.values())
        orientation = "samples_in_columns" if down and not across else "samples_in_rows"
        if rows != columns and down and across:
            # Both lengths have a vector: the axis is the one that is ordered.
            ordered = [n for n, s in vectors.items() if s == rows and _monotonic(arrays[n])]
            orientation = "samples_in_columns" if ordered else "samples_in_rows"
    n_samples, n_variables = (
        (rows, columns) if orientation == "samples_in_rows" else (columns, rows)
    )
    axes = sorted(
        (
            name
            for name, size in vectors.items()
            if size == n_variables and _monotonic(arrays[name])
        ),
        key=lambda name: ("axis" not in name.lower(), name),
    )
    detection = Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice(orientation, tuple(o for o in ORIENTATIONS if o != orientation)),
        n_samples=n_samples,
        n_variables=n_variables,
        axis=index_axis(n_variables),
        matrix=Choice(matrix, tuple(name for name in others if name != matrix)),
        axis_variable=Choice(NO_AXIS, ()),
        correctable=("matrix", "orientation", "axis_variable"),
    )
    detection = replace(
        detection,
        axis_variable=Choice(*_first_and_rest([*axes, NO_AXIS])),
        targets=tuple(sorted(n for n, size in vectors.items() if size == n_samples)),
    )
    return _with_axis(arrays, detection, detection.axis_variable.value)  # type: ignore[union-attr]


def _with_axis(
    arrays: dict[str, NDArray[np.float64]], detection: Detection, chosen: str
) -> Detection:
    assert detection.axis_variable is not None
    offered = (detection.axis_variable.value, *detection.axis_variable.alternatives)
    rest = tuple(name for name in offered if name != chosen)
    note: str | None
    if chosen == NO_AXIS:
        axis, note = index_axis(detection.n_variables), "No vector is chosen as the axis."
    else:
        axis, _, note = axis_from_values(
            [float(v) for v in _vector(arrays[chosen])], detection.n_variables
        )
    return replace(
        detection,
        axis=axis,
        axis_note=note,
        axis_variable=Choice(chosen, rest),
        # The axis is not also a response.
        targets=tuple(name for name in detection.targets if name != chosen),
    )


def _axis(arrays: dict[str, NDArray[np.float64]], detection: Detection) -> Any:
    assert detection.axis_variable is not None
    return _with_axis(arrays, detection, detection.axis_variable.value).axis


def _values(arrays: dict[str, NDArray[np.float64]], detection: Detection) -> NDArray[np.float64]:
    assert detection.matrix is not None
    values = arrays[detection.matrix.value]
    return np.ascontiguousarray(
        values if detection.orientation.value == "samples_in_rows" else values.T
    )


# --- the file -------------------------------------------------------------------


def _load(path: Path) -> dict[str, NDArray[np.float64]]:
    """Every real numeric array in the file, by name, as float64."""
    try:
        raw = scipy.io.loadmat(str(path), squeeze_me=False)
    except NotImplementedError as error:
        raise ReaderError(
            f"{path.name} is a MATLAB v7.3 MAT-file, which is HDF5 underneath and needs a "
            "library this application does not carry. Save it again from MATLAB with "
            "save(..., '-v7') and import that."
        ) from error
    except (ValueError, TypeError, OSError) as error:
        raise ReaderError(f"{path.name} is not a MAT-file this reader can open: {error}") from error
    arrays: dict[str, NDArray[np.float64]] = {}
    for name, value in raw.items():
        if name.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.dtype.kind not in "iuf" or value.ndim != 2 or value.size == 0:
            continue
        arrays[name] = value.astype(np.float64)
    return arrays


def _matrices(arrays: dict[str, NDArray[np.float64]], path: Path) -> list[str]:
    found = sorted(
        (name for name, value in arrays.items() if min(value.shape) > 1),
        key=lambda name: (-arrays[name].size, name),
    )
    if not found:
        raise ReaderError(
            f"{path.name} holds no two-dimensional numeric array, so there are no spectra in it. "
            "A MAT-file is read for a samples-by-variables matrix of real numbers."
        )
    return found


def _is_vector(value: NDArray[np.float64]) -> bool:
    return bool(min(value.shape) == 1 and max(value.shape) > 1)


def _vector(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return value.ravel()


def _monotonic(value: NDArray[np.float64]) -> bool:
    steps = np.diff(_vector(value))
    return bool(np.all(steps > 0) or np.all(steps < 0))


def _first_and_rest(options: list[str]) -> tuple[str, tuple[str, ...]]:
    return options[0], tuple(options[1:])
