"""Bruker OPUS files: one spectrum per file, a folder of them as a `.zip` (#187).

`PROPOSAL.md` §6 calls file-format support the strongest reason users stay
locked into vendor software, and OPUS is the format a Bruker FT-IR writes.
It is binary and undocumented by Bruker; what is known of it was reverse
engineered by open projects, and this reader was written from two of them -
`brukeropus` (MIT, Josh Duran) and `opusreader2` (MIT, spectral-cockpit) -
and checked against the sample files the second ships. Nothing from either is
imported: the block table is a few dozen lines, and §7's dependency rule
applies to readers too.

## The format, as much of it as this reader needs

- Four magic bytes, `0A 0A FE FE`. Then, little-endian: a float64 version at
  byte 4 and three int32 at 12, 16 and 20 - the directory's offset, its
  capacity in entries, and how many entries it holds.
- The directory is 12-byte entries: an int32 type code, the block's size in
  4-byte words, and its offset. An entry with a zero offset ends the list.
- The type code's low byte says what kind of block it is: 7 sample data, 11
  reference data, 15 result spectrum, and each of those plus 16 (23, 27, 31)
  the *data status* parameter block that describes it. The second byte is the
  channel: 4 single-channel spectrum, 8 interferogram, 16 absorbance, 48
  reflectance. The fourth byte is 64 for the plain block and 0 for one OPUS
  has post-processed (atmospheric compensation); the third byte is non-zero
  for reports and text, which are not spectra.
- A parameter block is records of `XXX`, a pad byte, an int16 type (0 int32,
  1 float64, anything else a string), an int16 length in 2-byte words, and the
  value; `END` terminates. A data status block carries `NPT`, `FXV`, `LXV`
  (points, first and last x), `DXU` (the x unit) and `CSF` (a y scale).
- A data block is float32 values, `CSF` times which is the spectrum; the
  block can be padded past `NPT`.

## What is offered, and what is decided

**The block is a choice** (`Detection.block`): `AB` absorbance, `Refl`
reflectance, `ScSm` the sample's single channel, `ScRf` the reference's, in
that order of preference and only those present in every file. Where OPUS
stored both a plain and a post-processed absorbance the post-processed one is
read, as the OPUS viewer shows it; a reader that silently took the other would
disagree with the screen the user compared it to.

**Every file must hold the same block with the same number of points**, and
their axes must agree to within one point spacing - instruments drift by a
fraction of a wavenumber between measurements, and refusing a real folder for
0.27 cm⁻¹ would be refusing every folder. The first file's axis is used and
the spread is stated in `axis_note`; a spread past one spacing is refused by
name, because that is a different measurement.

**Only `WN` is mapped** to an axis kind (wavenumber, cm⁻¹). Any other `DXU`
gives an index axis with a note saying which unit was not mapped, which is
`index_axis`'s rule: say so rather than invent plausible numbers.

Sample ids are the file names; the `SNM` sample name, when every file carries
one, becomes a metadata column.
"""

from __future__ import annotations

import re
import struct
import zipfile
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

NAME = "bruker_opus"
VERSION = "1"
#: A `.zip` of OPUS files, or one file. OPUS names files by a counter - `.0`,
#: `.1`, `.001` - which `reader_for` matches by pattern rather than by list.
SUFFIXES: tuple[str, ...] = (".zip",)
NUMERIC_SUFFIX = re.compile(r"^\.\d+$")
NOT_APPLICABLE = "n/a"

MAGIC = b"\n\n\xfe\xfe"

#: The blocks a user can choose, in the order they are preferred. Low byte of
#: the type code, and the channel byte.
BLOCKS: dict[str, tuple[int, int]] = {
    "AB": (15, 16),
    "Refl": (15, 48),
    "ScSm": (7, 4),
    "ScRf": (11, 4),
}
_STATUS_OFFSET = 16
_MAX_MEMBER_BYTES = 64 << 20


@dataclass(frozen=True)
class _Entry:
    kind: int
    channel: int
    text: int
    additional: int
    offset: int
    size: int


@dataclass(frozen=True)
class _Spectrum:
    """One block of one file: its values and its status parameters."""

    values: NDArray[np.float64]
    npt: int
    first: float
    last: float
    unit: str


@dataclass(frozen=True)
class _Member:
    name: str
    data: bytes
    entries: tuple[_Entry, ...]
    sample_name: str | None


# --- the reader protocol --------------------------------------------------


def sniff(path: Path) -> Detection:
    members = _members(path)
    offered = _offered(members)
    block = offered[0]
    spectra = [_spectrum(member, block) for member in members]
    axis, note = _shared_axis(spectra, members)
    named = [member.sample_name for member in members]
    return Detection(
        delimiter=Choice(NOT_APPLICABLE),
        decimal=Choice("."),
        orientation=Choice("samples_in_rows"),
        n_samples=len(members),
        n_variables=spectra[0].npt,
        axis=axis,
        axis_note=note,
        metadata_columns=("sample_name",) if all(named) else (),
        block=Choice(block, tuple(offered[1:])),
        correctable=("block",),
        private={"members": [member.name for member in members]},
    )


def resniff(path: Path, detection: Detection) -> Detection:
    """A different block is a different spectrum: points, axis, everything."""
    assert detection.block is not None
    members = _members(path)
    block = detection.block.value
    spectra = [_spectrum(member, block) for member in members]
    axis, note = _shared_axis(spectra, members)
    from dataclasses import replace

    return replace(detection, n_variables=spectra[0].npt, axis=axis, axis_note=note)


def read(path: Path, detection: Detection) -> Imported:
    assert detection.block is not None
    members = _members(path)
    spectra = [_spectrum(member, detection.block.value) for member in members]
    axis, _ = _shared_axis(spectra, members)
    values = np.vstack([spectrum.values for spectrum in spectra])
    metadata: dict[str, list[str]] = {}
    if all(member.sample_name for member in members):
        metadata["sample_name"] = [str(member.sample_name) for member in members]
    return Imported(
        values=values,
        axis=axis,
        source=source_file(path, NAME, VERSION),
        sample_ids=tuple(member.name for member in members),
        metadata_columns=metadata,
    )


def head(path: Path, detection: Detection) -> dict[str, Any]:
    assert detection.block is not None
    members = _members(path)[:HEAD_ROWS]
    spectra = [_spectrum(member, detection.block.value) for member in members]
    return {
        "sample_ids": [member.name for member in members],
        "rows": [[float(v) for v in spectrum.values[:HEAD_ROWS]] for spectrum in spectra],
    }


# --- the files --------------------------------------------------------------


def _members(path: Path) -> list[_Member]:
    """Every OPUS file in the upload: the members of a zip, or the file itself."""
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                infos = sorted(
                    (info for info in archive.infolist() if not info.is_dir()),
                    key=lambda info: info.filename,
                )
                if not infos:
                    raise ReaderError(f"{path.name} is an empty archive.")
                members = []
                for info in infos:
                    if info.file_size > _MAX_MEMBER_BYTES:
                        raise ReaderError(
                            f"{info.filename} in {path.name} is {info.file_size} bytes, which is "
                            "larger than one OPUS file can be."
                        )
                    name = Path(info.filename).name
                    members.append(
                        _member(name, archive.read(info), where=f"{name} in {path.name}")
                    )
                return members
        except zipfile.BadZipFile as error:
            raise ReaderError(f"{path.name} is not a zip archive: {error}") from error
        except OSError as error:
            raise ReaderError(f"cannot read {path.name}: {error.strerror}") from error
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ReaderError(f"cannot read {path.name}: {error.strerror}") from error
    return [_member(path.name, data, where=path.name)]


def _member(name: str, data: bytes, *, where: str) -> _Member:
    if data[:4] != MAGIC:
        raise ReaderError(
            f"{where} does not start with the OPUS signature, so it is not an OPUS file. "
            "A zip of OPUS files holds nothing else."
        )
    if len(data) < 24:
        raise ReaderError(f"{where} is truncated: it ends inside the header.")
    start, _capacity, count = struct.unpack_from("<3i", data, 12)
    if start < 24 or start + 12 * max(count, 0) > len(data):
        raise ReaderError(f"{where} points its directory outside the file.")
    entries: list[_Entry] = []
    for index in range(max(count, 0)):
        code, words, offset = struct.unpack_from("<3i", data, start + 12 * index)
        if offset <= 0:
            break
        size = words * 4
        if offset + size > len(data):
            raise ReaderError(
                f"{where} is truncated: block {index} claims {size} bytes at offset {offset} "
                f"in a file of {len(data)}."
            )
        entries.append(
            _Entry(
                kind=code & 0xFF,
                channel=(code >> 8) & 0xFF,
                text=(code >> 16) & 0xFF,
                additional=(code >> 24) & 0xFF,
                offset=offset,
                size=size,
            )
        )
    sample_name = None
    for entry in entries:
        if entry.kind == 160 and entry.channel == 0 and entry.text == 0:
            value = _params(data, entry).get("SNM")
            sample_name = str(value) if isinstance(value, str) and value.strip() else None
    return _Member(name=name, data=data, entries=tuple(entries), sample_name=sample_name)


def _params(data: bytes, entry: _Entry) -> dict[str, Any]:
    block = data[entry.offset : entry.offset + entry.size]
    params: dict[str, Any] = {}
    loc = 0
    while loc + 8 <= len(block):
        key = block[loc : loc + 3].decode("latin-1")
        if key == "END":
            break
        code, words = struct.unpack_from("<2h", block, loc + 4)
        size = words * 2
        if loc + 8 + size > len(block):
            break
        if code == 0:
            params[key] = struct.unpack_from("<i", block, loc + 8)[0]
        elif code == 1:
            params[key] = struct.unpack_from("<d", block, loc + 8)[0]
        else:
            # OPUS pads and separates string fields with control bytes; a sample
            # name is what a person typed, so those are dropped.
            text = block[loc + 8 : loc + 8 + size].split(b"\x00")[0].decode("latin-1")
            params[key] = "".join(ch for ch in text if ch.isprintable())
        loc += 8 + size
    return params


# --- the blocks -------------------------------------------------------------


def _find(member: _Member, block: str) -> tuple[_Entry, _Entry] | None:
    """The data block and its status block, post-processed one preferred."""
    kind, channel = BLOCKS[block]
    data = [e for e in member.entries if e.kind == kind and e.channel == channel and e.text == 0]
    status = [
        e
        for e in member.entries
        if e.kind == kind + _STATUS_OFFSET and e.channel == channel and e.text == 0
    ]
    if not data or not status:
        return None
    for additional in (0, 64):
        d = [e for e in data if e.additional == additional]
        s = [e for e in status if e.additional == additional]
        if d and s:
            return d[-1], s[-1]
    return data[-1], status[-1]


def _offered(members: list[_Member]) -> list[str]:
    offered = [block for block in BLOCKS if all(_find(m, block) is not None for m in members)]
    if not offered:
        held = sorted({block for block in BLOCKS for m in members if _find(m, block) is not None})
        raise ReaderError(
            "no spectrum block is present in every file. "
            + (
                f"Some files hold {', '.join(held)}; a zip is read as one dataset and needs "
                "one block in all of them."
                if held
                else "None of AB, Refl, ScSm or ScRf was found; interferograms and reports "
                "are not spectra."
            )
        )
    return offered


def _spectrum(member: _Member, block: str) -> _Spectrum:
    found = _find(member, block)
    if found is None:
        raise ReaderError(f"{member.name} holds no {block} block.")
    data_entry, status_entry = found
    status = _params(member.data, status_entry)
    try:
        npt = int(status["NPT"])
        first = float(status["FXV"])
        last = float(status["LXV"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReaderError(
            f"{member.name}'s {block} status block does not state NPT, FXV and LXV: {error}"
        ) from error
    if npt < 2:
        raise ReaderError(
            f"{member.name}'s {block} block has {npt} points, which is not a spectrum."
        )
    raw = member.data[data_entry.offset : data_entry.offset + data_entry.size]
    values = np.frombuffer(raw, dtype="<f4")
    if values.size < npt:
        raise ReaderError(
            f"{member.name}'s {block} block holds {values.size} values and its status block "
            f"says {npt}."
        )
    scale = float(status.get("CSF", 1.0) or 1.0)
    spectrum = values[:npt].astype(np.float64) * scale
    if not np.all(np.isfinite(spectrum)):
        raise ReaderError(f"{member.name}'s {block} block holds values that are not numbers.")
    unit = str(status.get("DXU", "")).strip().upper()
    return _Spectrum(values=spectrum, npt=npt, first=first, last=last, unit=unit)


def _shared_axis(
    spectra: list[_Spectrum], members: list[_Member]
) -> tuple[VariableAxis, str | None]:
    """One axis for the dataset, the first file's, with the others' drift stated."""
    lead = spectra[0]
    for member, spectrum in zip(members[1:], spectra[1:], strict=True):
        if spectrum.npt != lead.npt:
            raise ReaderError(
                f"{member.name} has {spectrum.npt} points where {members[0].name} has "
                f"{lead.npt}. A zip is read as one dataset on one axis."
            )
        if spectrum.unit != lead.unit:
            raise ReaderError(
                f"{member.name}'s axis is in {spectrum.unit!r} where {members[0].name}'s is in "
                f"{lead.unit!r}."
            )
    spacing = abs(lead.last - lead.first) / (lead.npt - 1)
    drift = max(
        (max(abs(s.first - lead.first), abs(s.last - lead.last)) for s in spectra[1:]),
        default=0.0,
    )
    if drift > spacing:
        worst = max(
            zip(members[1:], spectra[1:], strict=True),
            key=lambda pair: max(abs(pair[1].first - lead.first), abs(pair[1].last - lead.last)),
        )[0]
        raise ReaderError(
            f"{worst.name}'s axis differs from {members[0].name}'s by {drift:.4g}, more than "
            f"one point spacing ({spacing:.4g}). These are not the same measurement."
        )

    note: str | None = None
    if lead.unit == "WN":
        axis = VariableAxis(
            kind=AxisKind.WAVENUMBER_CM1,
            values=[float(v) for v in np.linspace(lead.first, lead.last, lead.npt)],
            unit="cm-1",
        )
        if drift > 0.0:
            note = (
                f"the {len(spectra)} files' axes differ by up to {drift:.4g} cm-1; "
                f"{members[0].name}'s is used for all of them."
            )
    else:
        axis = index_axis(lead.npt)
        note = (
            f"the x unit {lead.unit!r} is not one this reader maps to an axis; the axis is the "
            "point index."
        )
    return axis, note


def is_opus_name(path: str | Path) -> bool:
    """Whether a file name is one OPUS writes: a numeric suffix such as `.0`."""
    return bool(NUMERIC_SUFFIX.match(Path(path).suffix))


__all__ = [
    "BLOCKS",
    "MAGIC",
    "NAME",
    "SUFFIXES",
    "VERSION",
    "head",
    "is_opus_name",
    "read",
    "resniff",
    "sniff",
]
