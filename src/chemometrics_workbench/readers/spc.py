"""Thermo Galactic SPC files (#285).

SPC is one binary file holding one spectrum or many ("multifile"), written
by GRAMS and by most instrument software that exports to it. Two layouts
exist and both are read:

- **New format** (version byte `0x4B`, little-endian): a 512-byte header, an
  optional x-array of `float32` when the `TXVALS` flag is set, then per
  subfile a 32-byte subheader and its y values.
- **Old format** (`0x4D`): a 224-byte header and each subfile's 32-byte
  subheader, with 32-bit y values stored high word first.

**Y scaling.** Y is `float32` when the exponent is `0x80`, and otherwise a
scaled integer: `y = n * 2^exp / 2^32`, or `/ 2^16` with the 16-bit flag.
Each subfile is scaled by its own exponent, read as a **signed** byte
(`char subexp` in GRAMS's `SPC.H`), except a single old-format spectrum,
which is scaled by the main header's and leaves its subheader's at zero.
`rohanisaac/spc` reads a negative subfile exponent as zero, which halves or
quarters those subfiles; this reader does not follow it. The fixture tests
compare against its text conversions only where the exponent is not
negative, and say so.

Refused by name, each with what it would need: the big-endian new format
(`0x4C`), and the `xyxy` layout, in which every subfile carries its own x
values and the subfiles do not form a matrix.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
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

NAME = "galactic_spc"
VERSION = "1"
SUFFIXES: tuple[str, ...] = (".spc",)

NOT_APPLICABLE = "n/a"

NEW, NEW_BIG_ENDIAN, OLD = 0x4B, 0x4C, 0x4D
TSPREC, TMULTI, TXYXYS, TXVALS = 0x01, 0x04, 0x40, 0x80
FLOAT_EXPONENT = -128  # 0x80, read as a signed byte
SUBHEADER = struct.Struct("<BbhfffiifI")

#: `fxtype` codes that are a spectroscopic axis this application names.
AXES: dict[int, tuple[AxisKind, str]] = {
    1: (AxisKind.WAVENUMBER_CM1, "cm-1"),
    3: (AxisKind.WAVELENGTH_NM, "nm"),
    13: (AxisKind.RAMAN_SHIFT_CM1, "cm-1"),
}
#: Every other code, named for the note that says why the axis is an index.
UNITS = {
    0: "arbitrary", 2: "micrometres", 4: "seconds", 5: "minutes", 6: "Hz", 7: "kHz",
    8: "MHz", 9: "m/z", 10: "ppm", 11: "days", 12: "years", 14: "eV", 16: "diode number",
    17: "channel", 18: "degrees", 22: "data points", 23: "ms", 24: "us", 25: "ns", 26: "GHz",
    27: "cm", 28: "m", 29: "mm", 30: "hours",
}  # fmt: skip


@dataclass(frozen=True)
class _Spectra:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    """`n_subfiles x n_points`."""
    xtype: int
    old: bool


def sniff(path: Path) -> Detection:
    spectra = _parse(path)
    axis, note = _axis(spectra)
    return Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice("samples_in_rows"),
        n_samples=spectra.y.shape[0],
        n_variables=spectra.y.shape[1],
        axis=axis,
        axis_note=note,
        correctable=(),
    )


def read(path: Path, detection: Detection) -> Imported:
    spectra = _parse(path)
    axis, _ = _axis(spectra)
    return Imported(
        values=spectra.y,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=_ids(path, spectra.y.shape[0]),
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    spectra = _parse(path)
    return {
        "sample_ids": list(_ids(path, spectra.y.shape[0]))[:HEAD_ROWS],
        "rows": [[float(v) for v in row[:HEAD_ROWS]] for row in spectra.y[:HEAD_ROWS]],
    }


def _ids(path: Path, n: int) -> tuple[str, ...]:
    return (path.stem,) if n == 1 else tuple(f"{path.stem} #{i + 1}" for i in range(n))


def _axis(spectra: _Spectra) -> tuple[VariableAxis, str | None]:
    values = [float(v) for v in spectra.x]
    if spectra.xtype in AXES:
        kind, unit = AXES[spectra.xtype]
        return VariableAxis(kind=kind, values=values, unit=unit), None
    unit = UNITS.get(spectra.xtype, f"code {spectra.xtype}")
    return index_axis(len(values)), (
        f"The file's x axis is in {unit}, from {values[0]:g} to {values[-1]:g}, which is not "
        "a spectroscopic axis this application names, so the variables are numbered."
    )


# --- the file ---------------------------------------------------------------------


def _parse(path: Path) -> _Spectra:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ReaderError(f"cannot read {path.name}: {error.strerror}") from error
    if len(data) < 2:
        raise ReaderError(f"{path.name} is empty, so it is not an SPC file.")
    flags, version = data[0], data[1]
    try:
        if version == NEW:
            return _new(data, flags, path)
        if version == OLD:
            return _old(data, flags, path)
    except struct.error as error:
        raise ReaderError(f"{path.name} is truncated: it ends inside a header.") from error
    if version == NEW_BIG_ENDIAN:
        raise ReaderError(
            f"{path.name} is a big-endian SPC file (version 0x4C), written on a Motorola-era "
            "machine. This reader reads the little-endian new format and the old format."
        )
    raise ReaderError(
        f"{path.name} does not carry an SPC version byte (0x4B or 0x4D) at offset 1, so it is "
        "not an SPC file."
    )


def _new(data: bytes, flags: int, path: Path) -> _Spectra:
    if flags & TXYXYS:
        raise ReaderError(
            f"{path.name} is an xyxy SPC file: every subfile carries its own x values, so the "
            "spectra do not share an axis and cannot form a samples-by-variables matrix. "
            "Resample them onto one axis and save them again."
        )
    (npts,) = struct.unpack_from("<i", data, 4)
    first, last = struct.unpack_from("<dd", data, 8)
    (nsub,) = struct.unpack_from("<i", data, 24)
    xtype = data[28]
    if not flags & TMULTI:
        nsub = 1
    if npts < 1 or nsub < 1:
        raise ReaderError(f"{path.name} declares {npts} points and {nsub} spectra.")
    offset = 512
    if flags & TXVALS:
        x = _take(data, "<f4", npts, offset, path).astype(np.float64)
        offset += 4 * npts
    else:
        x = np.linspace(first, last, npts)
    rows = []
    for _ in range(nsub):
        exponent = SUBHEADER.unpack_from(data, offset)[1]
        offset += SUBHEADER.size
        y, offset = _values(data, offset, npts, exponent, flags & TSPREC, path, old=False)
        rows.append(y)
    return _Spectra(x=x, y=np.vstack(rows), xtype=xtype, old=False)


def _old(data: bytes, flags: int, path: Path) -> _Spectra:
    (exponent,) = struct.unpack_from("<h", data, 2)
    npts_f, first, last = struct.unpack_from("<fff", data, 4)
    npts = int(npts_f)
    xtype = data[16]
    if npts < 1:
        raise ReaderError(f"{path.name} declares {npts} points.")
    # The first subfile's header is the main header's last 32 bytes, and the
    # file holds as many subfiles as fit: the old format has no count.
    offset = 224
    rows: list[NDArray[np.float64]] = []
    while offset + SUBHEADER.size <= len(data) and (rows == [] or flags & TMULTI):
        own = SUBHEADER.unpack_from(data, offset)[1]
        offset += SUBHEADER.size
        # A multifile scales each subfile by its own exponent; a single
        # spectrum by the main header's, its subheader's being left at zero.
        scale = own if flags & TMULTI else exponent
        y, offset = _values(data, offset, npts, scale, flags & TSPREC, path, old=True)
        rows.append(y)
        if len(data) - offset < SUBHEADER.size + 4 * npts:
            break
    return _Spectra(x=np.linspace(first, last, npts), y=np.vstack(rows), xtype=xtype, old=True)


def _values(
    data: bytes, offset: int, npts: int, exponent: int, short: int, path: Path, *, old: bool
) -> tuple[NDArray[np.float64], int]:
    if exponent == FLOAT_EXPONENT and not old:
        return _take(data, "<f4", npts, offset, path).astype(np.float64), offset + 4 * npts
    if short:
        n = _take(data, "<i2", npts, offset, path).astype(np.float64)
        return n * 2.0**exponent / 2**16, offset + 2 * npts
    if old:
        # High word first: two little-endian 16-bit halves in that order.
        words = _take(data, "<u2", 2 * npts, offset, path).astype(np.int64).reshape(-1, 2)
        n64 = (words[:, 0] << 16) | words[:, 1]
        n = np.where(n64 >= 2**31, n64 - 2**32, n64).astype(np.float64)
    else:
        n = _take(data, "<i4", npts, offset, path).astype(np.float64)
    return n * 2.0**exponent / 2**32, offset + 4 * npts


def _take(data: bytes, dtype: str, count: int, offset: int, path: Path) -> NDArray[Any]:
    size = np.dtype(dtype).itemsize * count
    if offset + size > len(data):
        raise ReaderError(f"{path.name} is truncated: it ends inside its spectra.")
    return np.frombuffer(data, dtype=dtype, count=count, offset=offset)
