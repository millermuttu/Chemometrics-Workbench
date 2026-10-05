"""Thermo OMNIC SPA files (#286).

One SPA file holds one spectrum. Its layout, as read here and checked against
the instrument's own CSV export of the same spectra:

- the title, null-terminated, at offset 30;
- a directory of 16-byte entries from offset 304, each a key byte, then the
  block's position and size as little-endian `uint32` at +2 and +6, ending at
  key 0;
- key 2, the header block: the point count at +4, the x unit at +8, the y
  unit at +12, and the first and last x as `float32` at +16 and +20;
- key 3, the intensities: `float32`, as many as the point count.

The x axis is even, from first to last, in the file's order: descending
wavenumbers for an FTIR. OMNIC's CSV export lists it ascending and pads one
point below the range with a zero; neither is in the file.

Because a file is one spectrum, a dataset is a zip of them: every member an
SPA file, on one axis and in one y unit, or refused naming the one that is
not. `readers.reader_for` sends a zip here when its members are SPA files;
`readers.spectrum_files` opens it.
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
    spectrum_files,
)

NAME = "omnic_spa"
VERSION = "1"
SUFFIXES: tuple[str, ...] = (".spa",)

NOT_APPLICABLE = "n/a"
DIRECTORY = 304
HEADER, INTENSITIES = 2, 3
WAVENUMBERS = 1
#: OMNIC's y unit codes, for the message that refuses a mix of them.
Y_UNITS = {16: "%T", 17: "absorbance"}


@dataclass(frozen=True)
class _Spectrum:
    name: str
    title: str
    first: float
    last: float
    xunit: int
    yunit: int
    y: NDArray[np.float64]


def sniff(path: Path) -> Detection:
    spectra = _spectra(path)
    axis, note = _shared_axis(spectra)
    titled = all(spectrum.title for spectrum in spectra)
    return Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice("samples_in_rows"),
        n_samples=len(spectra),
        n_variables=spectra[0].y.size,
        axis=axis,
        axis_note=note,
        metadata_columns=("title",) if titled else (),
        correctable=(),
    )


def read(path: Path, detection: Detection) -> Imported:
    spectra = _spectra(path)
    axis, _ = _shared_axis(spectra)
    metadata: dict[str, list[str]] = {}
    if all(spectrum.title for spectrum in spectra):
        metadata["title"] = [spectrum.title for spectrum in spectra]
    return Imported(
        values=np.vstack([spectrum.y for spectrum in spectra]),
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=tuple(Path(spectrum.name).stem for spectrum in spectra),
        metadata_columns=metadata,
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    spectra = _spectra(path)[:HEAD_ROWS]
    return {
        "sample_ids": [Path(spectrum.name).stem for spectrum in spectra],
        "rows": [[float(v) for v in spectrum.y[:HEAD_ROWS]] for spectrum in spectra],
    }


# --- the files ---------------------------------------------------------------------


def _spectra(path: Path) -> list[_Spectrum]:
    return [
        _spectrum(name, data, where=where)
        for name, data, where in spectrum_files(path, SUFFIXES, "SPA")
    ]


def _spectrum(name: str, data: bytes, *, where: str) -> _Spectrum:
    blocks: dict[int, tuple[int, int]] = {}
    for entry in range(DIRECTORY, len(data) - 10, 16):
        key = data[entry]
        if key == 0:
            break
        position, size = struct.unpack_from("<II", data, entry + 2)
        blocks.setdefault(key, (position, size))
    if HEADER not in blocks or INTENSITIES not in blocks:
        raise ReaderError(
            f"{where} has no spectrum header or no intensities in its directory, so it is not "
            "an SPA file this reader can read."
        )
    header, _ = blocks[HEADER]
    if header + 24 > len(data):
        raise ReaderError(f"{where} is truncated: it ends inside the spectrum header.")
    (npts,) = struct.unpack_from("<I", data, header + 4)
    xunit, yunit = data[header + 8], data[header + 12]
    first, last = struct.unpack_from("<ff", data, header + 16)
    position, size = blocks[INTENSITIES]
    if npts < 2 or size < 4 * npts or position + 4 * npts > len(data):
        raise ReaderError(
            f"{where} declares {npts} points and holds {min(size, len(data) - position) // 4}: "
            "it is truncated or not an SPA file."
        )
    y = np.frombuffer(data, dtype="<f4", count=npts, offset=position).astype(np.float64)
    title = data[30 : 30 + 256].split(b"\0", 1)[0].decode("latin-1").strip()
    return _Spectrum(name, title, float(first), float(last), xunit, yunit, y)


def _shared_axis(spectra: list[_Spectrum]) -> tuple[VariableAxis, str | None]:
    lead = spectra[0]
    for spectrum in spectra[1:]:
        if spectrum.y.size != lead.y.size:
            raise ReaderError(
                f"{spectrum.name} has {spectrum.y.size} points where {lead.name} has "
                f"{lead.y.size}. One dataset is one axis."
            )
        if spectrum.yunit != lead.yunit:
            raise ReaderError(
                f"{spectrum.name} is in {_y(spectrum.yunit)} where {lead.name} is in "
                f"{_y(lead.yunit)}. Convert one of them in OMNIC before importing them together."
            )
    spacing = abs(lead.last - lead.first) / (lead.y.size - 1)
    drift = max(
        (max(abs(s.first - lead.first), abs(s.last - lead.last)) for s in spectra[1:]),
        default=0.0,
    )
    if drift > spacing:
        raise ReaderError(
            f"the files' axes differ by {drift:.4g}, more than one point spacing "
            f"({spacing:.4g}), so they are not one measurement's axis."
        )
    if lead.xunit != WAVENUMBERS:
        return index_axis(lead.y.size), (
            f"the x unit code {lead.xunit} is not one this reader names; the axis is the point "
            "index."
        )
    axis = VariableAxis(
        kind=AxisKind.WAVENUMBER_CM1,
        values=[float(v) for v in np.linspace(lead.first, lead.last, lead.y.size)],
        unit="cm-1",
    )
    note = (
        f"the {len(spectra)} files' axes differ by up to {drift:.4g} cm-1; {lead.name}'s is used."
        if drift > 0.0
        else None
    )
    return axis, note


def _y(code: int) -> str:
    return Y_UNITS.get(code, f"y unit code {code}")
