"""CSV and TXT: the format that carries every detection problem there is.

`PROPOSAL.md` §6 lists what this reader has to survive — "European decimal
commas, transposed layouts, wavelength headers, metadata columns" — and that
list is why the reader interface is designed against this format first and then
proven by XLSX (#79) and JCAMP-DX (#80) rather than the other way round. A
spreadsheet arrives with its numbers already typed and its columns already
separated; a text file arrives as bytes and every one of those decisions is a
guess.

**Every guess is offered with its alternatives.** A comma-delimited file of
European numbers and a semicolon-delimited file of American ones are the same
bytes under different readings often enough that no heuristic settles it. What
this reader owes the user is its best reading plus the means to overrule it,
which is what `Choice` is for.

The detections, and what decides each:

- **Delimiter** — the candidate that splits the sample lines into the same
  number of fields on every line, and into the most fields. Consistency first,
  because a file split on the wrong character is usually ragged, and a ragged
  split is the signal.
- **Decimal comma** — decided after the delimiter, and only when the delimiter
  is not itself a comma: it is the presence of `123,45`-shaped tokens in a file
  whose fields are otherwise numeric.
- **Orientation** — samples in rows unless the file looks transposed: a header
  row of labels with a numeric, monotonic first column is an axis running down
  the side, which is a spectrum per column.
- **The axis** — numeric header cells are wavelengths. Monotonic and
  descending reads as wavenumbers, the FT-IR convention; ascending and below
  3000 reads as nanometres. Anything else stays `index`, because a reader that
  invents an axis produces numbers that look read rather than guessed. See
  `index_axis`.
- **Columns** — a leading column of text is the sample identifiers; a numeric
  column under a non-numeric header is a target; a text column is metadata; an
  empty column is discarded with its reason.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from chemometrics_workbench.models import AxisKind, VariableAxis
from chemometrics_workbench.readers import (
    HEAD_ROWS,
    Choice,
    Detection,
    Imported,
    ReaderError,
    index_axis,
    source_file,
)

NAME = "delimited"
VERSION = "1"
SUFFIXES = (".csv", ".txt", ".tsv", ".dat")

#: Named rather than written as a character because a run of spaces and a run
#: of tabs are one delimiter to a human and two to `str.split`.
WHITESPACE = "whitespace"

DELIMITERS = (",", ";", "\t", "|", WHITESPACE)
DECIMALS = (".", ",")
ORIENTATIONS = ("samples_in_rows", "samples_in_columns")

#: How many lines a sniff reads. Enough to see a ragged file and a header, few
#: enough that sniffing a file at §13's envelope costs nothing.
SNIFF_LINES = 50

#: Above this, an ascending axis is read as wavenumbers rather than nanometres.
#: 2500 nm is the far end of NIR and 4000 cm-1 the near end of mid-IR, so the
#: gap between them is where the ambiguity actually lives.
NM_CEILING = 3000.0

_EU_NUMBER = re.compile(r"^[+-]?\d+,\d+$")


# --- Detection ------------------------------------------------------------


def sniff(path: Path) -> Detection:
    """Read the head of a file and report what it appears to be."""
    lines = _sample_lines(path)
    delimiter = _detect_delimiter(path, lines)
    rows = [_split(line, delimiter) for line in lines]
    decimal = _detect_decimal(rows, delimiter)

    header, body = _split_header(rows, decimal)
    if not body:
        raise ReaderError(f"{path.name} has a header but no data rows.")

    orientation = _detect_orientation(header, body, decimal)
    if orientation == "samples_in_columns":
        return _transposed_detection(path, header, body, delimiter, decimal)
    return _row_detection(path, header, body, delimiter, decimal)


def _row_detection(
    path: Path,
    header: list[str] | None,
    body: list[list[str]],
    delimiter: str,
    decimal: str,
) -> Detection:
    columns = _classify(header, body, decimal)
    if not columns["spectra"]:
        raise ReaderError(_no_spectra_message(path, header, decimal))

    axis, reconstructed, note = _axis_from(header, columns["spectra"])
    n_samples = _count_rows(path, delimiter, header is not None)
    return Detection(
        delimiter=_choice(delimiter, DELIMITERS),
        decimal=_choice(decimal, DECIMALS),
        orientation=_choice("samples_in_rows", ORIENTATIONS),
        n_samples=n_samples,
        n_variables=len(columns["spectra"]),
        axis=axis,
        axis_reconstructed=reconstructed,
        axis_note=note,
        metadata_columns=tuple(columns["metadata_names"]),
        targets=tuple(columns["target_names"]),
        discarded=tuple(columns["discarded"]),
        private={
            "header": header,
            "spectra": columns["spectra"],
            "targets": columns["targets"],
            "metadata": columns["metadata"],
            "ids": columns["ids"],
            "n_columns": len(body[0]),
        },
    )


def _transposed_detection(
    path: Path,
    header: list[str] | None,
    body: list[list[str]],
    delimiter: str,
    decimal: str,
) -> Detection:
    """Samples in columns: the first column is the axis, each other column a spectrum."""
    n_variables = _count_rows(path, delimiter, header is not None)
    sample_names = (header or [])[1:]
    axis_values = [_number(row[0], decimal, path, 0) for row in body]
    axis, reconstructed, note = _axis_from_values(axis_values, len(axis_values))
    return Detection(
        delimiter=_choice(delimiter, DELIMITERS),
        decimal=_choice(decimal, DECIMALS),
        orientation=_choice("samples_in_columns", ORIENTATIONS),
        n_samples=len(sample_names),
        n_variables=n_variables,
        axis=axis,
        axis_reconstructed=reconstructed,
        axis_note=note,
        private={"header": header, "sample_names": sample_names},
    )


def _sample_lines(path: Path) -> list[str]:
    text = _read_text(path)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ReaderError(f"{path.name} is empty.")
    return lines[:SNIFF_LINES]


def _read_text(path: Path) -> str:
    """Decode, tolerating the encodings instruments and spreadsheets emit.

    `utf-8-sig` first because Excel writes a byte-order mark and a stray U+FEFF
    turns the first header cell into a name that matches nothing. Latin-1 as
    the fallback because it cannot fail, and a mangled degree sign in a
    metadata column is a better outcome than a refusal to open the file.
    """
    try:
        blob = path.read_bytes()
    except OSError as error:
        raise ReaderError(f"cannot read {path.name}: {error.strerror}") from error
    if not blob.strip():
        raise ReaderError(f"{path.name} is empty.")
    try:
        return blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        return blob.decode("latin-1")


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == WHITESPACE:
        return line.split()
    return [cell.strip().strip('"') for cell in line.split(delimiter)]


def _detect_delimiter(path: Path, lines: list[str]) -> str:
    """The candidate that splits every sample line into the same, largest count.

    Consistency is worth more than field count: a file split on the wrong
    character is usually ragged, and the raggedness is what gives it away.
    """
    best: tuple[int, int, str] | None = None
    for candidate in DELIMITERS:
        counts = [len(_split(line, candidate)) for line in lines]
        if min(counts) < 2:
            continue
        consistent = len(set(counts)) == 1
        score = (1 if consistent else 0, min(counts), candidate)
        if best is None or score[:2] > best[:2]:
            best = score
    if best is None:
        raise ReaderError(
            f"{path.name} has no consistent delimiter. Its first line splits into one "
            f"field on every candidate ({', '.join(DELIMITERS)}). "
            "A one-column file has no spectra in it."
        )
    return best[2]


def _detect_decimal(rows: list[list[str]], delimiter: str) -> str:
    """A decimal comma is `123,45` in a file whose fields are otherwise numbers.

    Only askable when the delimiter is not a comma. When it is, `1,5` has
    already been split into two fields and the question cannot be posed from
    the text alone.
    """
    if delimiter == ",":
        return "."
    european = sum(1 for row in rows for cell in row if _EU_NUMBER.match(cell))
    plain = sum(1 for row in rows for cell in row if _is_number(cell, "."))
    return "," if european > plain else "."


def _split_header(rows: list[list[str]], decimal: str) -> tuple[list[str] | None, list[list[str]]]:
    """A first row that is not all numbers is a header; otherwise there is none."""
    first = rows[0]
    if all(_is_number(cell, decimal) or not cell for cell in first):
        return None, rows
    return first, rows[1:]


def _detect_orientation(header: list[str] | None, body: list[list[str]], decimal: str) -> str:
    """Transposed when an axis runs down the first column under a row of labels.

    Requires both halves of that sentence. A numeric first column on its own is
    a sample number; labels on their own are ordinary column names. Together
    they are a spectrum per column, which is how instrument software exports
    and how spreadsheets get built by hand.
    """
    if header is None:
        return "samples_in_rows"
    labels = [cell for cell in header[1:] if cell]
    if not labels or any(_is_number(cell, decimal) for cell in labels):
        return "samples_in_rows"
    first_column = [row[0] for row in body if row]
    if len(first_column) < 3 or not all(_is_number(cell, decimal) for cell in first_column):
        return "samples_in_rows"
    values = [_to_float(cell, decimal) for cell in first_column]
    if _monotonic(values):
        return "samples_in_columns"
    return "samples_in_rows"


def _classify(header: list[str] | None, body: list[list[str]], decimal: str) -> dict[str, Any]:
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

    ids: int | None = None
    if _all_text(column[0]) and header is not None:
        ids = 0

    spectra: list[int] = []
    targets: list[int] = []
    metadata: list[int] = []
    discarded: list[dict[str, str]] = []

    named_axis = header is not None and any(_is_number(cell, decimal) for cell in names)

    for index in range(width):
        if index == ids:
            continue
        cells = column[index]
        name = names[index] if index < len(names) else ""
        if all(not cell for cell in cells):
            discarded.append({"what": name or f"column {index + 1}", "why": "the column is empty"})
            continue
        if header is not None and _is_number(name, decimal):
            spectra.append(index)
            continue
        numeric = all(_is_number(cell, decimal) for cell in cells if cell)
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


def _axis_from(
    header: list[str] | None, spectra: list[int]
) -> tuple[VariableAxis, bool, str | None]:
    if header is None:
        return (
            index_axis(len(spectra)),
            False,
            "The file carries no header, so the variables are numbered rather than "
            "given wavelengths.",
        )
    if not all(_is_number(header[index], ",") for index in spectra):
        return (
            index_axis(len(spectra)),
            False,
            "The columns are named rather than numbered, so the variables carry no wavelengths.",
        )
    values = [float(header[index].replace(",", ".")) for index in spectra]
    return _axis_from_values(values, len(values))


def _axis_from_values(values: list[float], n: int) -> tuple[VariableAxis, bool, str | None]:
    """Name the axis from its own numbers, or decline to.

    Descending is the FT-IR convention and reads as wavenumbers. Ascending
    below `NM_CEILING` reads as nanometres. Everything else is left as an index
    with a note, because a wrong unit on a plot axis is a claim about the
    instrument.
    """
    if not _monotonic(values):
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


def _count_rows(path: Path, delimiter: str, has_header: bool) -> int:
    """Count data rows without holding the file, so a preview stays cheap."""
    total = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total - 1 if has_header else total


# --- Reading --------------------------------------------------------------


def read(path: Path, detection: Detection) -> Imported:
    """Parse the whole file according to `detection`.

    The detection may be the one `sniff` returned or that one with the user's
    corrections folded in, and this function cannot tell the difference — which
    is the property that makes a correction real rather than decorative.
    """
    delimiter = detection.delimiter.value
    decimal = detection.decimal.value
    rows = [_split(line, delimiter) for line in _all_lines(path)]
    header, body = _split_header(rows, decimal)
    if not body:
        raise ReaderError(f"{path.name} has a header but no data rows.")

    _check_rectangular(path, header, body)

    if detection.orientation.value == "samples_in_columns":
        return _read_transposed(path, header, body, decimal, detection)
    return _read_rows(path, header, body, decimal, detection)


def _read_rows(
    path: Path,
    header: list[str] | None,
    body: list[list[str]],
    decimal: str,
    detection: Detection,
) -> Imported:
    columns = _classify(header, body, decimal)
    spectra = columns["spectra"]
    if not spectra:
        raise ReaderError(_no_spectra_message(path, header, decimal))

    values = np.array(
        [[_number(row[i], decimal, path, line) for i in spectra] for line, row in enumerate(body)],
        dtype=np.float64,
    )
    axis, _, _ = _axis_from(header, spectra)
    if axis.kind is AxisKind.INDEX and detection.axis.kind is not AxisKind.INDEX:
        axis = detection.axis

    ids = columns["ids"]
    sample_ids = tuple(row[ids] for row in body) if ids is not None else ()
    targets = {
        name: [_number(row[i], decimal, path, line) for line, row in enumerate(body)]
        for name, i in zip(columns["target_names"], columns["targets"], strict=True)
    }
    metadata = {
        name: [row[i] for row in body]
        for name, i in zip(columns["metadata_names"], columns["metadata"], strict=True)
    }
    return Imported(
        values=values,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=sample_ids,
        targets=targets,
        metadata_columns=metadata,
        discarded=tuple(columns["discarded"]),
    )


def _read_transposed(
    path: Path,
    header: list[str] | None,
    body: list[list[str]],
    decimal: str,
    detection: Detection,
) -> Imported:
    """One spectrum per column, the axis down the first."""
    if header is None:
        raise ReaderError(
            f"{path.name} was read as samples in columns, but it has no header row to "
            "take the sample names from. Correct the orientation, or add a header."
        )
    axis_values = [_number(row[0], decimal, path, line) for line, row in enumerate(body)]
    values = np.array(
        [
            [_number(row[column], decimal, path, line) for line, row in enumerate(body)]
            for column in range(1, len(header))
        ],
        dtype=np.float64,
    )
    axis, _, _ = _axis_from_values(axis_values, len(axis_values))
    return Imported(
        values=values,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=tuple(header[1:]),
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    """The first rows, for the preview's table.

    Parsed with the same `_number` the full read uses, so a correction that
    makes the file unreadable is reported as a diagnostic in the preview rather
    than as a table of blanks the user has to interpret.
    """
    imported_rows: list[list[float]] = []
    ids: list[str] = []
    delimiter = detection.delimiter.value
    decimal = detection.decimal.value
    rows = [_split(line, delimiter) for line in _sample_lines(path)]
    header, body = _split_header(rows, decimal)

    if detection.orientation.value == "samples_in_columns":
        columns = range(1, min(len(header or []), HEAD_ROWS + 1))
        for column in columns:
            ids.append((header or [])[column])
            imported_rows.append(
                [
                    _number(row[column], decimal, path, line)
                    for line, row in enumerate(body[:HEAD_ROWS])
                    if len(row) > column
                ]
            )
        return {"sample_ids": ids, "rows": imported_rows}

    classified = _classify(header, body, decimal)
    spectra = classified["spectra"][:HEAD_ROWS]
    for index, row in enumerate(body[:HEAD_ROWS]):
        if classified["ids"] is not None:
            ids.append(row[classified["ids"]])
        else:
            ids.append(f"{index + 1}")
        imported_rows.append(
            [_number(row[i], decimal, path, index) for i in spectra if i < len(row)]
        )
    return {"sample_ids": ids, "rows": imported_rows}


def _all_lines(path: Path) -> list[str]:
    return [line for line in _read_text(path).splitlines() if line.strip()]


def _check_rectangular(path: Path, header: list[str] | None, body: list[list[str]]) -> None:
    """A ragged file is a wrong delimiter far more often than it is a broken file."""
    expected = len(header) if header is not None else len(body[0])
    for offset, row in enumerate(body):
        if len(row) != expected:
            line = offset + (2 if header is not None else 1)
            raise ReaderError(
                f"{path.name} line {line} has {len(row)} fields where the rest of the file "
                f"has {expected}. Check the delimiter, or the line itself."
            )


# --- Small helpers --------------------------------------------------------


def _no_spectra_message(path: Path, header: list[str] | None, decimal: str) -> str:
    """Say *why* there are no spectra, which is usually the decimal separator.

    A file whose columns are headed with wavelengths but whose cells will not
    parse is a file being read with the wrong decimal separator, and saying
    "no column of numbers" to someone who has just corrected that separator is
    an answer to a question they did not ask.
    """
    if header is not None and any(_is_number(cell, decimal) for cell in header):
        return (
            f"{path.name} has wavelength columns, but their values are not numbers when "
            f"read with {decimal!r} as the decimal separator. Correct the decimal separator."
        )
    return (
        f"{path.name} has no column of numbers to read as a spectrum. "
        "Every column is either text or empty."
    )


def _choice(value: str, options: tuple[str, ...]) -> Choice:
    return Choice(value, tuple(option for option in options if option != value))


def _is_number(cell: str, decimal: str) -> bool:
    if not cell:
        return False
    try:
        float(cell.replace(",", ".") if decimal == "," else cell)
    except ValueError:
        return False
    return True


def _to_float(cell: str, decimal: str) -> float:
    return float(cell.replace(",", ".") if decimal == "," else cell)


def _number(cell: str, decimal: str, path: Path, line: int) -> float:
    """Parse a cell that has to be a number, naming it if it is not."""
    try:
        return _to_float(cell, decimal)
    except ValueError:
        raise ReaderError(
            f"{path.name} line {line + 1}: {cell!r} is not a number. "
            f"The file is being read with {decimal!r} as its decimal separator."
        ) from None


def _all_text(cells: list[str]) -> bool:
    return bool(cells) and all(cell and not _is_number(cell, ".") for cell in cells)


def _monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    ascending = all(b > a for a, b in pairwise(values))
    descending = all(b < a for a, b in pairwise(values))
    return ascending or descending
