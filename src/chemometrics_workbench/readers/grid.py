"""A grid of cells becomes a dataset — the half of a reader no format owns.

`delimited.py` has bytes to decode and a delimiter to guess; `xlsx.py` has a
workbook to open and a sheet to pick. Once either has produced a rectangle of
strings, everything left is the same question in both: which column holds the
identifiers, which hold the spectra, which are targets and which are notes, and
what — if anything — the header row says about the axis.

That question is answered here, once. A second reader answering it a second way
is how two formats end up disagreeing about the same file exported twice.

Cells arrive as strings even from a spreadsheet, where they were typed. That is
deliberate: a workbook stores a number as text often enough that a reader which
trusts the cell type imports a column of empty values and calls it success.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from chemometrics_workbench.models import AxisKind, SourceFile, VariableAxis
from chemometrics_workbench.readers import (
    HEAD_ROWS,
    Detection,
    Imported,
    ReaderError,
    index_axis,
)

#: Above this, an ascending axis is read as wavenumbers rather than nanometres.
#: 2500 nm is the far end of NIR and 4000 cm-1 the near end of mid-IR, so the
#: gap between them is where the ambiguity actually lives.
NM_CEILING = 3000.0

__all__ = [
    "NM_CEILING",
    "all_text",
    "as_float",
    "axis_from",
    "axis_from_values",
    "classify",
    "detect_orientation",
    "head_payload",
    "is_number",
    "monotonic",
    "no_spectra_message",
    "parse_number",
    "read_rows",
    "read_transposed",
    "split_header",
]


def split_header(rows: list[list[str]], decimal: str) -> tuple[list[str] | None, list[list[str]]]:
    """A first row that is not all numbers is a header; otherwise there is none."""
    first = rows[0]
    if all(is_number(cell, decimal) or not cell for cell in first):
        return None, rows
    return first, rows[1:]


def detect_orientation(header: list[str] | None, body: list[list[str]], decimal: str) -> str:
    """Transposed when an axis runs down the first column under a row of labels.

    Requires both halves of that sentence. A numeric first column on its own is
    a sample number; labels on their own are ordinary column names. Together
    they are a spectrum per column, which is how instrument software exports
    and how spreadsheets get built by hand.
    """
    if header is None:
        return "samples_in_rows"
    labels = [cell for cell in header[1:] if cell]
    if not labels or any(is_number(cell, decimal) for cell in labels):
        return "samples_in_rows"
    first_column = [row[0] for row in body if row]
    if len(first_column) < 3 or not all(is_number(cell, decimal) for cell in first_column):
        return "samples_in_rows"
    values = [as_float(cell, decimal) for cell in first_column]
    if monotonic(values):
        return "samples_in_columns"
    return "samples_in_rows"


def classify(header: list[str] | None, body: list[list[str]], decimal: str) -> dict[str, Any]:
    """Sort the columns into identifiers, spectra, targets, metadata and rubbish.

    **A numeric header name settles it.** A column headed `850` is a variable
    whatever its cells hold, so a single `n/a` in a wavelength column fails at
    the parse, naming its line, rather than quietly reclassifying the whole
    column as metadata and importing a dataset one variable short.

    When no header name is a number there are no wavelengths to recognise, so a
    file of `V1, V2, V3…` is read as spectra on an index axis rather than as a
    hundred separate targets.
    """
    width = max(len(row) for row in body)
    column = {
        index: [row[index] if index < len(row) else "" for row in body] for index in range(width)
    }
    names = header if header is not None else [""] * width
    if len(names) < width:
        names = [*names, *[""] * (width - len(names))]

    ids: int | None = None
    if header is not None and looks_like_identifiers(column[0], names[0], decimal):
        ids = 0

    spectra: list[int] = []
    targets: list[int] = []
    metadata: list[int] = []
    discarded: list[dict[str, str]] = []

    named_axis = header is not None and any(is_number(cell, decimal) for cell in names)

    for index in range(width):
        if index == ids:
            continue
        cells = column[index]
        name = names[index] if index < len(names) else ""
        if all(not cell for cell in cells):
            discarded.append({"what": name or f"column {index + 1}", "why": "the column is empty"})
            continue
        if header is not None and is_number(name, decimal):
            spectra.append(index)
            continue
        numeric = all(is_number(cell, decimal) for cell in cells if cell)
        if not numeric:
            metadata.append(index)
        elif header is None or not named_axis:
            spectra.append(index)
        else:
            targets.append(index)

    return {
        "ids": ids,
        "spectra": spectra,
        "targets": targets,
        "metadata": metadata,
        "target_names": [names[i] or f"column {i + 1}" for i in targets],
        "metadata_names": [names[i] or f"column {i + 1}" for i in metadata],
        "discarded": discarded,
    }


def axis_from(
    header: list[str] | None, spectra: list[int]
) -> tuple[VariableAxis, bool, str | None]:
    if header is None:
        return (
            index_axis(len(spectra)),
            False,
            "The file carries no header, so the variables are numbered rather than "
            "given wavelengths.",
        )
    if not all(is_number(header[index], ",") for index in spectra):
        return (
            index_axis(len(spectra)),
            False,
            "The columns are named rather than numbered, so the variables carry no wavelengths.",
        )
    values = [float(header[index].replace(",", ".")) for index in spectra]
    return axis_from_values(values, len(values))


def axis_from_values(values: list[float], n: int) -> tuple[VariableAxis, bool, str | None]:
    """Name the axis from its own numbers, or decline to.

    Descending is the FT-IR convention and reads as wavenumbers. Ascending
    below `NM_CEILING` reads as nanometres. Everything else is left as an index
    with a note, because a wrong unit on a plot axis is a claim about the
    instrument.
    """
    if not monotonic(values):
        return (
            index_axis(n),
            False,
            "The header's numbers are not ordered, so they are not an axis.",
        )
    ascending = values[-1] > values[0]
    if not ascending:
        return (
            VariableAxis(kind=AxisKind.WAVENUMBER_CM1, values=values, unit="cm-1"),
            False,
            None,
        )
    if max(values) <= NM_CEILING:
        return VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=values, unit="nm"), False, None
    return (
        VariableAxis(kind=AxisKind.WAVENUMBER_CM1, values=values, unit="cm-1"),
        False,
        f"Ascending and above {NM_CEILING:.0f}, so read as wavenumbers rather than nanometres.",
    )


def read_rows(
    provenance: SourceFile,
    header: list[str] | None,
    body: list[list[str]],
    decimal: str,
    detection: Detection,
) -> Imported:
    """A grid with samples in rows becomes a dataset."""
    source = provenance.filename
    columns = classify(header, body, decimal)
    spectra = columns["spectra"]
    if not spectra:
        raise ReaderError(no_spectra_message(source, header, decimal))

    values = np.array(
        [
            [parse_number(row[i], decimal, source, line) for i in spectra]
            for line, row in enumerate(body)
        ],
        dtype=np.float64,
    )
    axis, _, _ = axis_from(header, spectra)
    if axis.kind is AxisKind.INDEX and detection.axis.kind is not AxisKind.INDEX:
        axis = detection.axis

    ids = columns["ids"]
    sample_ids = tuple(row[ids] for row in body) if ids is not None else ()
    targets = {
        name: [parse_number(row[i], decimal, source, line) for line, row in enumerate(body)]
        for name, i in zip(columns["target_names"], columns["targets"], strict=True)
    }
    metadata = {
        name: [row[i] for row in body]
        for name, i in zip(columns["metadata_names"], columns["metadata"], strict=True)
    }
    return Imported(
        values=values,
        axis=axis,
        source=provenance,
        sample_ids=sample_ids,
        targets=targets,
        metadata_columns=metadata,
        discarded=tuple(columns["discarded"]),
    )


def read_transposed(
    provenance: SourceFile,
    header: list[str] | None,
    body: list[list[str]],
    decimal: str,
    detection: Detection,
) -> Imported:
    """One spectrum per column, the axis down the first."""
    source = provenance.filename
    if header is None:
        raise ReaderError(
            f"{source} was read as samples in columns, but it has no header row to "
            "take the sample names from. Correct the orientation, or add a header."
        )
    axis_values = [parse_number(row[0], decimal, source, line) for line, row in enumerate(body)]
    values = np.array(
        [
            [parse_number(row[column], decimal, source, line) for line, row in enumerate(body)]
            for column in range(1, len(header))
        ],
        dtype=np.float64,
    )
    axis, _, _ = axis_from_values(axis_values, len(axis_values))
    return Imported(
        values=values,
        axis=axis,
        source=provenance,
        sample_ids=tuple(header[1:]),
    )


def no_spectra_message(source: str, header: list[str] | None, decimal: str) -> str:
    """Say *why* there are no spectra, which is usually the decimal separator.

    A file whose columns are headed with wavelengths but whose cells will not
    parse is a file being read with the wrong decimal separator, and saying
    "no column of numbers" to someone who has just corrected that separator is
    an answer to a question they did not ask.
    """
    if header is not None and any(is_number(cell, decimal) for cell in header):
        return (
            f"{source} has wavelength columns, but their values are not numbers when "
            f"read with {decimal!r} as the decimal separator. Correct the decimal separator."
        )
    return (
        f"{source} has no column of numbers to read as a spectrum. "
        "Every column is either text or empty."
    )


def is_number(cell: str, decimal: str) -> bool:
    if not cell:
        return False
    try:
        float(cell.replace(",", ".") if decimal == "," else cell)
    except ValueError:
        return False
    return True


def as_float(cell: str, decimal: str) -> float:
    return float(cell.replace(",", ".") if decimal == "," else cell)


def parse_number(cell: str, decimal: str, source: str, line: int) -> float:
    """Parse a cell that has to be a number, naming it if it is not."""
    try:
        return as_float(cell, decimal)
    except ValueError:
        raise ReaderError(
            f"{source} line {line + 1}: {cell!r} is not a number. "
            f"The file is being read with {decimal!r} as its decimal separator."
        ) from None


def all_text(cells: list[str]) -> bool:
    return bool(cells) and all(cell and not is_number(cell, ".") for cell in cells)


def looks_like_identifiers(cells: list[str], name: str, decimal: str) -> bool:
    """Whether a leading column names the samples rather than measuring them.

    **Type is not the test; uniqueness is.** This used to be `all_text`, which
    made a column of `S001, S002` identifiers and a column of `1, 2, 3` a
    response variable — so a file whose ids happen to be integers imported a
    row number as a target a user could select for PLS, and lost every sample
    name (#135). Most files number their samples.

    A measurement repeats: two meat samples have the same fat content often
    enough that a real target is almost never distinct across every row, while
    an identifier is distinct by definition. So the test is that no value
    appears twice, whatever the values look like.

    The header still settles it first. A numeric header name is a wavelength
    and `classify` has already taken it as one, so this is only asked about a
    column headed with a word.
    """
    if not cells or any(not cell for cell in cells):
        return False
    if is_number(name, decimal):
        return False
    if all_text(cells):
        return True
    return len(set(cells)) == len(cells)


def monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    ascending = all(b > a for a, b in pairwise(values))
    descending = all(b < a for a, b in pairwise(values))
    return ascending or descending


def head_payload(
    source: str,
    header: list[str] | None,
    body: list[list[str]],
    decimal: str,
    orientation: str,
) -> dict[str, Any]:
    """The first rows, for the preview's table.

    Parsed with the same `parse_number` the full read uses, so a correction that
    makes the file unreadable is reported as a diagnostic in the preview rather
    than as a table of blanks the user has to interpret.
    """
    rows: list[list[float]] = []
    ids: list[str] = []

    if orientation == "samples_in_columns":
        names = header or []
        for column in range(1, min(len(names), HEAD_ROWS + 1)):
            ids.append(names[column])
            rows.append(
                [
                    parse_number(row[column], decimal, source, line)
                    for line, row in enumerate(body[:HEAD_ROWS])
                    if len(row) > column
                ]
            )
        return {"sample_ids": ids, "rows": rows}

    columns = classify(header, body, decimal)
    spectra = columns["spectra"][:HEAD_ROWS]
    for index, row in enumerate(body[:HEAD_ROWS]):
        if columns["ids"] is not None:
            ids.append(row[columns["ids"]])
        else:
            # A placeholder, and it has to look like one. This was
            # `f"{index + 1}"`, which is indistinguishable from a column of
            # numbered samples - on the file that prompted #135 the preview
            # showed "1", "2", "3" and the import stored nothing, and the two
            # were impossible to tell apart. `api.py` already says `row N`
            # where it has no id, so say the same thing here.
            ids.append(f"row {index + 1}")
        rows.append([parse_number(row[i], decimal, source, index) for i in spectra if i < len(row)])
    return {"sample_ids": ids, "rows": rows}
