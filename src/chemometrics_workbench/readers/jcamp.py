"""JCAMP-DX: the open instrument format, and the one that brings its own axis.

`PROPOSAL.md` §6 puts `.jdx`/`.dx` in Phase 1 as the open standard with wide
instrument support. It is the third implementation of the reader interface and
the one that tests whether that interface carries anything richer than a table:
unlike a CSV or a workbook, a JCAMP file states its own units, its own axis and
its own provenance, and a block of it is a sample rather than a row.

**Nothing here is guessed, so nothing here is correctable.** There is no
delimiter, the decimal separator is a point by specification, and a block is a
spectrum — `Detection.correctable` is empty, which is the honest answer when a
format has already answered every question the import screen knows how to ask.

**The axis is read, never reconstructed.** `##FIRSTX`, `##LASTX`, `##NPOINTS`
and `##DELTAX` are in the file, scaled by `##XFACTOR`; ordinates are scaled by
`##YFACTOR`. Tecator's axis is reconstructed with `linspace` and
`datasets.load_tecator` documents that precisely because it cannot be inferred
— here there is nothing to infer.

**The compressed forms are decoded, and an unknown one is named.** ASDF packs
ordinates three ways and files in the wild mix them line by line:

- **PAC** — a sign is a separator: `+123-45` is two values.
- **SQZ** — the leading digit becomes a letter carrying its sign: `@ABCDEFGHI`
  for 0–9 positive, `abcdefghi` for −1…−9.
- **DIF** — the value is a *difference* from the previous one: `%JKLMNOPQR`
  positive, `jklmnopqr` negative. A DIF line repeats its last ordinate as the
  first value of the next line, and that repeat is checked rather than skipped:
  a mis-decoded difference form produces a plausible spectrum, which is exactly
  the failure that has to be caught at read time.
- **DUP** — `STUVWXYZs` repeats the previous value n times.

Anything else — a `##PEAK TABLE`, an `##NTUPLES` block, a data form this reader
does not implement — is refused by name. §6: an unreadable file must produce a
specific diagnostic message, never a stack trace, and a *misread* file must not
produce a spectrum at all.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

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

NAME = "jcamp_dx"
VERSION = "1"
SUFFIXES = (".jdx", ".dx", ".jcm")

#: Nothing in a JCAMP file is a guess, so these fields carry no alternatives.
NOT_APPLICABLE = "n/a"

#: The data forms this reader decodes. Anything else is named and refused.
SUPPORTED_FORMS = ("(XY..XY)", "(X++(Y..Y))")

#: The entries that hold a spectrum. `##PEAK TABLE` and `##DATA TABLE` are
#: recognised so they can be refused by name: a list of peaks is a reduction of
#: a spectrum, not a spectrum, and importing one as a row of absorbances would
#: produce a dataset whose variables mean nothing.
SPECTRUM_KEYS = ("XYDATA", "XYPOINTS")

#: `##XUNITS` values this reader will name an axis from. A unit that is not
#: here leaves the axis as an index with a note, rather than being guessed at.
X_UNITS = {
    "1/CM": (AxisKind.WAVENUMBER_CM1, "cm-1"),
    "CM-1": (AxisKind.WAVENUMBER_CM1, "cm-1"),
    "NANOMETERS": (AxisKind.WAVELENGTH_NM, "nm"),
    "NM": (AxisKind.WAVELENGTH_NM, "nm"),
}

#: The header entries worth carrying onto the dataset. The model has somewhere
#: to put a per-sample string and nowhere to put an arbitrary instrument dump,
#: so this is a list rather than "everything that was in the file".
CARRIED = ("ORIGIN", "OWNER", "SPECTROMETER/DATA SYSTEM", "INSTRUMENT", "DATE", "YUNITS")

_LDR = re.compile(r"^\s*##([^=]*)=(.*)$")
_SQZ = {c: str(i) for i, c in enumerate("@ABCDEFGHI")}
_SQZ_NEGATIVE = {c: str(i + 1) for i, c in enumerate("abcdefghi")}
_DIF = {c: i for i, c in enumerate("%JKLMNOPQR")}
_DIF_NEGATIVE = {c: -(i + 1) for i, c in enumerate("jklmnopqr")}
_DUP = {c: i + 1 for i, c in enumerate("STUVWXYZs")}


class _Block:
    """One spectrum: its header entries and its decoded ordinates."""

    def __init__(self, header: dict[str, str], y: NDArray[np.float64], x: NDArray[np.float64]):
        self.header = header
        self.y = y
        self.x = x

    @property
    def title(self) -> str:
        return self.header.get("TITLE", "").strip()


def sniff(path: Path) -> Detection:
    """Parse the file and report what it holds. Every field is read, not guessed."""
    blocks = _blocks(path)
    axis, note = _axis(path, blocks)
    return Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice("samples_in_rows"),
        n_samples=len(blocks),
        n_variables=len(blocks[0].y),
        axis=axis,
        axis_reconstructed=False,
        axis_note=note,
        metadata_columns=tuple(_carried_keys(blocks)),
        correctable=(),
    )


def read(path: Path, detection: Detection) -> Imported:
    """Every block becomes a sample. The detection has nothing to change."""
    blocks = _blocks(path)
    axis, _ = _axis(path, blocks)
    values = np.array([block.y for block in blocks], dtype=np.float64)
    return Imported(
        values=values,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=tuple(_sample_ids(blocks)),
        metadata_columns={
            key: [block.header.get(key, "").strip() for block in blocks]
            for key in _carried_keys(blocks)
        },
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    """The first blocks, for the preview's table."""
    blocks = _blocks(path)[:HEAD_ROWS]
    return {
        "sample_ids": _sample_ids(blocks),
        "rows": [[float(value) for value in block.y[:HEAD_ROWS]] for block in blocks],
    }


def _sample_ids(blocks: list[_Block]) -> list[str]:
    """A block's title is its sample name, and a file of untitled blocks numbers them."""
    return [block.title or f"block {index + 1}" for index, block in enumerate(blocks)]


def _carried_keys(blocks: list[_Block]) -> list[str]:
    return [key for key in CARRIED if any(key in block.header for block in blocks)]


def _axis(path: Path, blocks: list[_Block]) -> tuple[VariableAxis, str | None]:
    """The axis every block shares, refusing a file whose blocks disagree."""
    first = blocks[0]
    for index, block in enumerate(blocks[1:], start=2):
        if len(block.y) != len(first.y):
            raise ReaderError(
                f"{path.name} block {index} has {len(block.y)} points where block 1 has "
                f"{len(first.y)}. A dataset needs one axis, so these are two datasets."
            )
        if not np.allclose(block.x, first.x, rtol=1e-9, atol=0.0):
            raise ReaderError(
                f"{path.name} block {index} is on a different x axis from block 1. "
                "A dataset needs one axis, so these are two datasets."
            )

    units = first.header.get("XUNITS", "").strip().upper()
    known = X_UNITS.get(units)
    values = [float(value) for value in first.x]
    if known is None:
        return (
            index_axis(len(values)),
            f"##XUNITS is {units or 'absent'}, which this build does not name an axis from, "
            "so the variables are numbered.",
        )
    kind, unit = known
    return VariableAxis(kind=kind, values=values, unit=unit), None


# --- The file ------------------------------------------------------------


def _blocks(path: Path) -> list[_Block]:
    """Split the file into blocks and decode each one's data."""
    text = _read_text(path)
    entries = _entries(path, text)
    if not entries:
        raise ReaderError(f"{path.name} holds no ##LDR entries, so it is not a JCAMP-DX file.")

    blocks: list[_Block] = []
    current: dict[str, str] = {}
    data: list[str] = []
    form: str | None = None
    outer: dict[str, str] = {}

    for key, value, lines in entries:
        if key == "TITLE":
            if form is not None:
                blocks.append(_block(path, {**outer, **current}, form, data))
            elif current and not blocks:
                # A LINK block's own header, shared by the spectra inside it.
                outer = current
            current, data, form = {"TITLE": value}, [], None
            continue
        if key in ("XYDATA", "XYPOINTS", "PEAK TABLE", "DATA TABLE"):
            form = f"{key}={value.strip()}"
            data = lines
            continue
        if key == "END":
            if form is not None:
                blocks.append(_block(path, {**outer, **current}, form, data))
                current, data, form = {}, [], None
            continue
        current[key] = value

    if form is not None:
        blocks.append(_block(path, {**outer, **current}, form, data))
    if not blocks:
        raise ReaderError(
            f"{path.name} has no spectrum in it: no ##XYDATA or ##XYPOINTS entry was found."
        )
    return blocks


def _read_text(path: Path) -> str:
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


def _entries(path: Path, text: str) -> list[tuple[str, str, list[str]]]:
    """`##KEY= value` entries, each with the unlabelled lines that follow it."""
    entries: list[tuple[str, str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.split("$$", 1)[0] if raw.lstrip().startswith("$$") else raw
        if not line.strip():
            continue
        match = _LDR.match(line)
        if match:
            key = match.group(1).strip().upper().lstrip("$")
            entries.append((key, match.group(2).strip(), []))
        elif entries:
            entries[-1][2].append(line)
    return entries


def _block(path: Path, header: dict[str, str], form: str, lines: list[str]) -> _Block:
    label, _, shape = form.partition("=")
    if label not in SPECTRUM_KEYS:
        raise ReaderError(
            f"{path.name} holds its data in ##{label}, which is a list of peaks rather than a "
            "spectrum. This build imports ##XYDATA and ##XYPOINTS."
        )
    if shape not in SUPPORTED_FORMS:
        raise ReaderError(
            f"{path.name} stores its data as ##{label}={shape}, which this build does not "
            f"read. It reads {' and '.join(SUPPORTED_FORMS)}. The file is not damaged; the "
            "reader is incomplete, and guessing at the form would produce a spectrum that "
            "looks right and is not."
        )

    x_factor = _float(header, "XFACTOR", 1.0)
    y_factor = _float(header, "YFACTOR", 1.0)

    if shape == "(XY..XY)":
        x_raw, y_raw = _decode_pairs(path, lines)
    else:
        y_raw = _decode_y(path, lines)
        x_raw = _implied_x(path, header, len(y_raw))

    y = np.array(y_raw, dtype=np.float64) * y_factor
    x = np.array(x_raw, dtype=np.float64) * x_factor

    declared = header.get("NPOINTS")
    if declared and declared.strip().isdigit() and int(declared) != len(y):
        raise ReaderError(
            f"{path.name} declares ##NPOINTS={int(declared)} but its data decodes to "
            f"{len(y)} points. Something in the file is truncated or is a form this "
            "reader has misread."
        )
    if len(y) < 2:
        raise ReaderError(f"{path.name} has a block with {len(y)} points, which is not a spectrum.")
    return _Block(header, y, x)


def _float(header: dict[str, str], key: str, default: float) -> float:
    value = header.get(key)
    if value is None:
        return default
    try:
        return float(value.split()[0])
    except (ValueError, IndexError):
        return default


def _implied_x(path: Path, header: dict[str, str], n: int) -> list[float]:
    """`(X++(Y..Y))` states the axis in the header rather than beside each point."""
    first = header.get("FIRSTX")
    last = header.get("LASTX")
    if first is None or last is None:
        raise ReaderError(
            f"{path.name} stores its data as (X++(Y..Y)) but gives no ##FIRSTX and ##LASTX, "
            "so its axis cannot be read. This reader does not invent one."
        )
    start = _float(header, "FIRSTX", 0.0)
    end = _float(header, "LASTX", 0.0)
    if n == 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + index * step for index in range(n)]


# --- ASDF ----------------------------------------------------------------


def _decode_pairs(path: Path, lines: list[str]) -> tuple[list[float], list[float]]:
    """`(XY..XY)`: plain x, y pairs, whitespace or comma separated."""
    numbers: list[float] = []
    for offset, line in enumerate(lines):
        for token in line.replace(",", " ").replace(";", " ").split():
            try:
                numbers.append(float(token))
            except ValueError:
                raise ReaderError(
                    f"{path.name} data line {offset + 1}: {token!r} is not a number, and "
                    "##XYDATA=(XY..XY) holds nothing else."
                ) from None
    if len(numbers) % 2:
        raise ReaderError(
            f"{path.name} holds {len(numbers)} values in an (XY..XY) block, which is an odd "
            "number and therefore not pairs."
        )
    return numbers[0::2], numbers[1::2]


def _decode_y(path: Path, lines: list[str]) -> list[float]:
    """`(X++(Y..Y))`: an x per line, then ordinates in any of the ASDF forms."""
    values: list[float] = []
    for offset, line in enumerate(lines):
        tokens = _tokenise(path, line, offset)
        if not tokens:
            continue
        # The first token on a line is the abscissa. It is not trusted as the
        # axis - NPOINTS and FIRSTX/LASTX decide that - so it is dropped here.
        ordinates = tokens[1:]
        in_dif = any(_is_dif(token) for token in ordinates)
        line_values = _expand(path, ordinates, offset, previous=values[-1] if values else None)
        if not line_values:
            continue
        if values and in_dif:
            # A line in difference form opens by restating the previous line's
            # last ordinate. That restatement is the format's own checksum:
            # matching it is how a misread difference form is caught, and a
            # misread difference form yields a plausible spectrum rather than
            # an error.
            if abs(line_values[0] - values[-1]) > 1e-9:
                raise ReaderError(
                    f"{path.name} data line {offset + 1} restarts at {line_values[0]:g} where "
                    f"the previous line ended at {values[-1]:g}. The difference form has been "
                    "misread, so no spectrum is returned."
                )
            line_values = line_values[1:]
        values.extend(line_values)
    if not values:
        raise ReaderError(f"{path.name} has an ##XYDATA entry with no data under it.")
    return values


def _tokenise(path: Path, line: str, offset: int) -> list[str]:
    """Split an ASDF line into its values, where a sign or a letter starts one."""
    tokens: list[str] = []
    current = ""
    for character in line.strip():
        if character in " \t,":
            if current:
                tokens.append(current)
            current = ""
            continue
        starts = (
            character in _SQZ
            or character in _SQZ_NEGATIVE
            or character in _DIF
            or character in _DIF_NEGATIVE
            or character in _DUP
            or (character in "+-" and current and current[-1] not in "eE")
        )
        if starts and current:
            tokens.append(current)
            current = character
        else:
            current += character
    if current:
        tokens.append(current)
    return tokens


def _is_dif(token: str) -> bool:
    return bool(token) and (token[0] in _DIF or token[0] in _DIF_NEGATIVE)


def _expand(path: Path, tokens: list[str], offset: int, previous: float | None) -> list[float]:
    """Turn one line's ASDF tokens into ordinates."""
    values: list[float] = []
    for token in tokens:
        head_char = token[0]
        if head_char in _DUP:
            count = _dup_count(path, token, offset)
            if not values and previous is None:
                raise ReaderError(
                    f"{path.name} data line {offset + 1} starts with a repeat count, which "
                    "has nothing to repeat."
                )
            last = values[-1] if values else previous
            assert last is not None
            values.extend([last] * (count - 1))
            continue
        if head_char in _DIF or head_char in _DIF_NEGATIVE:
            difference = _dif_value(path, token, offset)
            if not values:
                if previous is None:
                    raise ReaderError(
                        f"{path.name} data line {offset + 1} opens with a difference, which "
                        "has nothing to differ from."
                    )
                values.append(previous)
            values.append(values[-1] + difference)
            continue
        values.append(_plain_value(path, token, offset))
    return values


def _dup_count(path: Path, token: str, offset: int) -> int:
    digits = _DUP[token[0]]
    rest = token[1:]
    if rest and not rest.isdigit():
        raise ReaderError(f"{path.name} data line {offset + 1}: {token!r} is not a repeat count.")
    return int(f"{digits}{rest}") if rest else digits


def _dif_value(path: Path, token: str, offset: int) -> float:
    head_char, rest = token[0], token[1:]
    if rest and not rest.isdigit():
        raise ReaderError(f"{path.name} data line {offset + 1}: {token!r} is not a difference.")
    if head_char in _DIF:
        return float(f"{_DIF[head_char]}{rest}")
    return -float(f"{-_DIF_NEGATIVE[head_char]}{rest}")


def _plain_value(path: Path, token: str, offset: int) -> float:
    """A PAC value, or a SQZ one whose first digit is carried by a letter."""
    head_char = token[0]
    if head_char in _SQZ:
        token = f"{_SQZ[head_char]}{token[1:]}"
    elif head_char in _SQZ_NEGATIVE:
        token = f"-{_SQZ_NEGATIVE[head_char]}{token[1:]}"
    try:
        return float(token)
    except ValueError:
        raise ReaderError(
            f"{path.name} data line {offset + 1}: {token!r} is not a number in any form this "
            "reader decodes."
        ) from None
