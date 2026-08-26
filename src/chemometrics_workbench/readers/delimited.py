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
from pathlib import Path
from typing import Any

from chemometrics_workbench.readers import (
    Choice,
    Detection,
    Imported,
    ReaderError,
    grid,
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

_EU_NUMBER = re.compile(r"^[+-]?\d+,\d+$")


# --- Detection ------------------------------------------------------------


def sniff(path: Path) -> Detection:
    """Read the head of a file and report what it appears to be."""
    lines = _sample_lines(path)
    delimiter = _detect_delimiter(path, lines)
    rows = [_split(line, delimiter) for line in lines]
    decimal = _detect_decimal(rows, delimiter)

    header, body = grid.split_header(rows, decimal)
    if not body:
        raise ReaderError(f"{path.name} has a header but no data rows.")

    orientation = grid.detect_orientation(header, body, decimal)
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
    columns = grid.classify(header, body, decimal)
    if not columns["spectra"]:
        raise ReaderError(grid.no_spectra_message(path.name, header, decimal))

    axis, reconstructed, note = grid.axis_from(header, columns["spectra"])
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
    axis_values = [grid.parse_number(row[0], decimal, path.name, 0) for row in body]
    axis, reconstructed, note = grid.axis_from_values(axis_values, len(axis_values))
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
    plain = sum(1 for row in rows for cell in row if grid.is_number(cell, "."))
    return "," if european > plain else "."


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
    header, body = grid.split_header(rows, decimal)
    if not body:
        raise ReaderError(f"{path.name} has a header but no data rows.")

    _check_rectangular(path, header, body)

    provenance = source_file(path, NAME, VERSION)
    if detection.orientation.value == "samples_in_columns":
        return grid.read_transposed(provenance, header, body, decimal, detection)
    return grid.read_rows(provenance, header, body, decimal, detection)


def head(path: Path, detection: Detection) -> dict[str, Any]:
    """The first rows of the file, read as the detection says to read them."""
    rows = [_split(line, detection.delimiter.value) for line in _sample_lines(path)]
    header, body = grid.split_header(rows, detection.decimal.value)
    return grid.head_payload(
        path.name, header, body, detection.decimal.value, detection.orientation.value
    )


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


def _choice(value: str, options: tuple[str, ...]) -> Choice:
    return Choice(value, tuple(option for option in options if option != value))
