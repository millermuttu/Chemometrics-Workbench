"""ASD FieldSpec files (#287).

One ASD file holds one spectrum. Its layout, as read here and checked against
the Eaton Fire record's published reflectance of the same files:

- the version, three bytes at offset 0: `ASD` for version 1, `as2` to `as8`
  after it;
- the header, 484 bytes: the data type at 186 (raw, reflectance, radiance...),
  the first wavelength and the step as `float32` at 191 and 195, the data
  format at 199 (`float32`, `int32` or `float64`) and the channel count as
  `uint16` at 204;
- the spectrum from 484, as many values as channels;
- from version 2, the reference header after it - a flag, two times, and a
  description whose `uint16` length is at +18 - and the white reference's
  values after that, in the same format.

A file whose data type is reflectance stores the target's counts and the white
reference's, and reflectance is their ratio: what ViewSpec shows. Nothing more
is applied. The published spectra this was checked against are that ratio
times the panel's own calibration curve, the same for every sample, and with
the VNIR detector spliced to the first SWIR one, a constant per sample; this
reader does neither, because the panel's curve is not in the file and a splice
is a correction the user chooses. Any other data type is read as stored.

Because a file is one spectrum, a dataset is a zip of them, every member an
ASD file on one axis and of one data type, or refused naming the one that is
not.
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
    source_file,
    spectrum_files,
)

NAME = "asd_fieldspec"
VERSION = "1"
SUFFIXES: tuple[str, ...] = (".asd",)

NOT_APPLICABLE = "n/a"
SPECTRUM = 484
REFLECTANCE = 1
DATA_TYPES = {
    0: "raw counts",
    1: "reflectance",
    2: "radiance",
    3: "no units",
    4: "irradiance",
    5: "QI",
    6: "transmittance",
    7: "an unknown data type",
    8: "absorbance",
}
FORMATS = {0: "<f4", 1: "<i4", 2: "<f8"}


@dataclass(frozen=True)
class _Spectrum:
    name: str
    data_type: int
    first: float
    step: float
    y: NDArray[np.float64]


def sniff(path: Path) -> Detection:
    spectra = _spectra(path)
    return Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice("samples_in_rows"),
        n_samples=len(spectra),
        n_variables=spectra[0].y.size,
        axis=_shared_axis(spectra),
        correctable=(),
    )


def read(path: Path, detection: Detection) -> Imported:
    spectra = _spectra(path)
    return Imported(
        values=np.vstack([spectrum.y for spectrum in spectra]),
        axis=_shared_axis(spectra),
        source=source_file(path, NAME, VERSION),
        sample_ids=tuple(Path(spectrum.name).stem for spectrum in spectra),
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
        for name, data, where in spectrum_files(path, SUFFIXES, "ASD")
    ]


def _spectrum(name: str, data: bytes, *, where: str) -> _Spectrum:
    signature = data[:3]
    if signature != b"ASD" and not (signature[:2] == b"as" and signature[2:3].isdigit()):
        raise ReaderError(
            f"{where} does not start with an ASD version signature, so it is not an ASD file."
        )
    if len(data) < SPECTRUM:
        raise ReaderError(f"{where} is truncated: it ends inside the 484-byte header.")
    data_type, data_format = data[186], data[199]
    first, step = struct.unpack_from("<ff", data, 191)
    (channels,) = struct.unpack_from("<H", data, 204)
    if data_format not in FORMATS:
        raise ReaderError(
            f"{where} declares data format {data_format}, which is not a number type."
        )
    dtype = np.dtype(FORMATS[data_format])
    end = SPECTRUM + channels * dtype.itemsize
    if channels < 2 or end > len(data):
        raise ReaderError(
            f"{where} declares {channels} channels and holds "
            f"{max(len(data) - SPECTRUM, 0) // dtype.itemsize}: it is truncated."
        )
    y = np.frombuffer(data, dtype=dtype, count=channels, offset=SPECTRUM).astype(np.float64)
    if data_type == REFLECTANCE and signature != b"ASD":
        y = y / _reference(data, end, channels, dtype, where=where)
    return _Spectrum(name, data_type, float(first), float(step), y)


def _reference(
    data: bytes, header: int, channels: int, dtype: np.dtype[Any], *, where: str
) -> NDArray[np.float64]:
    if header + 20 > len(data):
        raise ReaderError(f"{where} is a reflectance file truncated before its white reference.")
    (length,) = struct.unpack_from("<H", data, header + 18)
    start = header + 20 + length
    if start + channels * dtype.itemsize > len(data):
        raise ReaderError(f"{where} is a reflectance file truncated inside its white reference.")
    reference = np.frombuffer(data, dtype=dtype, count=channels, offset=start)
    if not np.all(reference > 0):
        raise ReaderError(
            f"{where} is a reflectance file whose white reference has a zero or negative "
            "channel, so reflectance cannot be computed from it."
        )
    return reference.astype(np.float64)


def _shared_axis(spectra: list[_Spectrum]) -> VariableAxis:
    lead = spectra[0]
    for spectrum in spectra[1:]:
        if (spectrum.y.size, spectrum.first, spectrum.step) != (lead.y.size, lead.first, lead.step):
            raise ReaderError(
                f"{spectrum.name} has {spectrum.y.size} channels from {spectrum.first:g} nm in "
                f"steps of {spectrum.step:g} where {lead.name} has {lead.y.size} from "
                f"{lead.first:g} in steps of {lead.step:g}. One dataset is one axis."
            )
        if spectrum.data_type != lead.data_type:
            raise ReaderError(
                f"{spectrum.name} is {_kind(spectrum.data_type)} where {lead.name} is "
                f"{_kind(lead.data_type)}. One dataset is one kind of measurement."
            )
    return VariableAxis(
        kind=AxisKind.WAVELENGTH_NM,
        values=[lead.first + lead.step * i for i in range(lead.y.size)],
        unit="nm",
    )


def _kind(code: int) -> str:
    return DATA_TYPES.get(code, f"data type {code}")
