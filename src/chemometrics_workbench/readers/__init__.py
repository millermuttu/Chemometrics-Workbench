"""Readers: how a file on the user's disk becomes a dataset.

`PROPOSAL.md` §6 calls file format support the single most under-specified item
in the original draft and one of the strongest reasons users stay locked into
vendor software, and treats it as a named deliverable. Its design rules are the
shape of this package: readers are independent, individually testable modules
with a real sample file committed as a fixture; an unreadable file produces a
specific diagnostic rather than a stack trace; every import records the source
file's content hash and the reader version.

**A reader is two functions, not one.** `sniff` looks at the head of a file and
reports what it thinks the file is, with alternatives for every guess. `read`
then parses the whole file according to a `Detection` — the one `sniff`
returned, or that one with the user's corrections applied. Splitting them is
what makes the import preview honest: the preview shows a decision that has not
been taken yet, and the correction the user makes is the same object the parse
is driven by, so it cannot be displayed and then ignored.

A guess presented as a fact is the failure mode this design exists to prevent.
Every ambiguous decision is a `Choice`, carrying what was picked *and* what else
it could have been, and the alternatives are what the import screen offers.

The three readers Phase 1.2 ships — CSV/TXT (#78), XLSX (#79) and JCAMP-DX
(#80) — implement the same two functions, so the import endpoints in #81 call
one interface and never learn which format they are holding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.models import AxisKind, SourceFile, VariableAxis

__all__ = [
    "Choice",
    "Detection",
    "Imported",
    "Reader",
    "ReaderError",
    "apply_corrections",
    "file_hash",
    "index_axis",
    "preview",
    "read",
    "reader_for",
    "source_file",
]

#: How many rows a preview shows. Six is what the import screen's table holds.
HEAD_ROWS = 6


class ReaderError(Exception):
    """A file could not be read, and this says why in words the user can act on.

    §6: "an unreadable file must produce a specific diagnostic message, never a
    stack trace". Everything a reader rejects is rejected with one of these,
    naming the file, the line if there is one, and what was expected.
    """


@dataclass(frozen=True)
class Choice:
    """One detection, and what else it might have been.

    `alternatives` never repeats `value`. The import screen renders the pair as
    a select whose first option is what was detected, and marks the field as
    corrected when the user picks another — so an empty `alternatives` is a
    claim that there was nothing to decide, not merely that nothing came to
    mind.
    """

    value: str
    alternatives: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return {"value": self.value, "alternatives": list(self.alternatives)}


@dataclass(frozen=True)
class Detection:
    """What a reader believes about a file, before it parses all of it.

    This is both the preview's content and the parse's instructions. The user's
    corrections are applied to it with `apply_corrections`, and the corrected
    `Detection` is what `read` obeys.
    """

    delimiter: Choice
    decimal: Choice
    orientation: Choice
    n_samples: int
    n_variables: int
    axis: VariableAxis
    axis_reconstructed: bool = False
    axis_note: str | None = None
    metadata_columns: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    discarded: tuple[dict[str, str], ...] = ()
    #: Which sheet a workbook is being read from, and the others it could be.
    #: `None` for formats that hold one table and have nothing to choose.
    sheet: Choice | None = None
    #: The fields this reader will accept a correction to. Per-reader because a
    #: delimiter means nothing to a spreadsheet and a sheet means nothing to a
    #: text file, and offering a correction that cannot be applied is the same
    #: lie as applying one that was never offered.
    correctable: tuple[str, ...] = ("delimiter", "decimal", "orientation")
    #: Whatever the reader needs to remember between `sniff` and `read` and has
    #: no place for in the published shape — the header row's index, a sheet
    #: name, a block offset. Never rendered, never corrected.
    private: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """The `detected` half of the import preview the frontend already renders."""
        axis: dict[str, Any] = {
            "kind": self.axis.kind.value,
            "unit": self.axis.unit,
            "start": self.axis.values[0],
            "end": self.axis.values[-1],
            "reconstructed": self.axis_reconstructed,
        }
        if self.axis_note:
            axis["note"] = self.axis_note
        payload: dict[str, Any] = {
            "delimiter": self.delimiter.payload(),
            "decimal": self.decimal.payload(),
            "orientation": self.orientation.payload(),
            "n_samples": self.n_samples,
            "n_variables": self.n_variables,
            "axis": axis,
            "metadata_columns": list(self.metadata_columns),
            "targets": list(self.targets),
            "discarded": [dict(entry) for entry in self.discarded],
        }
        if self.sheet is not None:
            payload["sheet"] = self.sheet.payload()
        return payload


@dataclass(eq=False)
class Imported:
    """A file, read.

    `eq=False` for the same reason `ReferenceDataset` has it: the fields are
    arrays, and a generated `__eq__` would raise on the first ambiguous truth
    value rather than compare anything.
    """

    values: NDArray[np.float64]
    """Absorbance or whatever the file holds, n_samples x n_variables."""

    axis: VariableAxis
    source: SourceFile
    sample_ids: tuple[str, ...] = ()
    targets: dict[str, list[float]] = field(default_factory=dict)
    metadata_columns: dict[str, list[str]] = field(default_factory=dict)
    discarded: tuple[dict[str, str], ...] = ()


class Reader(Protocol):
    """What every reader module provides. #79 and #80 satisfy this too."""

    NAME: str
    VERSION: str
    SUFFIXES: tuple[str, ...]

    def sniff(self, path: Path) -> Detection: ...

    def read(self, path: Path, detection: Detection) -> Imported: ...

    def head(self, path: Path, detection: Detection) -> dict[str, Any]: ...


def reader_for(path: str | Path) -> Any:
    """The reader module for a file, chosen by suffix.

    Chosen by suffix rather than by content because the user picked this file
    in a dialog and knows what it is; a reader that guesses past a `.xlsx`
    suffix is a reader that will one day parse a spreadsheet as text and
    produce a diagnostic about line 1.
    """
    from chemometrics_workbench.readers import delimited, xlsx

    modules = [delimited, xlsx]
    suffix = Path(path).suffix.lower()
    for module in modules:
        if suffix in module.SUFFIXES:
            return module
    known = sorted({s for module in modules for s in module.SUFFIXES})
    raise ReaderError(
        f"there is no reader for {suffix or 'a file with no suffix'}. "
        f"This build reads {', '.join(known)}."
    )


def apply_corrections(detection: Detection, corrections: dict[str, str]) -> Detection:
    """Fold the user's corrections into a detection.

    Only the fields this reader offers can be corrected, and a correction to
    anything else is refused rather than dropped: a correction silently ignored
    is worse than one that never existed, because the user watched themselves
    make it.
    """
    from dataclasses import replace

    correctable = set(detection.correctable)
    unknown = sorted(set(corrections) - correctable)
    if unknown:
        raise ReaderError(
            f"{', '.join(unknown)} cannot be corrected. "
            f"This reader takes corrections to {', '.join(sorted(correctable))}."
        )

    changes: dict[str, Any] = {}
    for name in correctable:
        chosen = corrections.get(name)
        if chosen is None:
            continue
        current: Choice = getattr(detection, name)
        allowed = (current.value, *current.alternatives)
        if chosen not in allowed:
            raise ReaderError(
                f"{chosen!r} is not one of the {name} options for this file: {', '.join(allowed)}."
            )
        changes[name] = Choice(chosen, tuple(option for option in allowed if option != chosen))
    return replace(detection, **changes)


def preview(path: str | Path, corrections: dict[str, str] | None = None) -> dict[str, Any]:
    """The import preview: provenance, what was detected, and the first rows.

    The shape is the one the Phase 1.1 frontend already renders, and it is a
    contract rather than a convenience — #81 serves this from an endpoint and
    the screen is not edited to match it.
    """
    file = Path(path)
    module = reader_for(file)
    detection = _detect_with(module, file, corrections)

    return {
        "source": {
            "filename": file.name,
            "file_hash": file_hash(file),
            "reader": module.NAME,
            "reader_version": module.VERSION,
            "size_bytes": file.stat().st_size,
        },
        "detected": detection.payload(),
        "head": module.head(file, detection),
    }


def read(path: str | Path, corrections: dict[str, str] | None = None) -> Imported:
    """Read a file whole, with the user's corrections applied."""
    file = Path(path)
    module = reader_for(file)
    detection = _detect_with(module, file, corrections)
    imported: Imported = module.read(file, detection)
    return imported


def _detect_with(module: Any, file: Path, corrections: dict[str, str] | None) -> Detection:
    """Sniff, apply the corrections, and let the reader look again if it needs to.

    Most corrections re-read the same table differently. A few — a sheet in a
    workbook, one day a block in a JCAMP file — change which table is being read,
    and everything derived from it goes stale. A reader that has such a
    correction offers `resniff`, and the second `apply_corrections` puts the
    user's other choices back on top of the fresh detection.
    """
    detection: Detection = module.sniff(file)
    if not corrections:
        return detection
    detection = apply_corrections(detection, corrections)
    resniff = getattr(module, "resniff", None)
    if resniff is not None:
        detection = apply_corrections(resniff(file, detection), corrections)
    return detection


def file_hash(path: Path) -> str:
    """`sha256:…`, the shape `SourceFile.file_hash` takes.

    §6: every import records the source file's content hash. Read in blocks,
    because §13's envelope allows a file far larger than memory is comfortable
    holding twice.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise ReaderError(f"cannot read {path}: {error.strerror}") from error
    return f"sha256:{digest.hexdigest()}"


def source_file(path: Path, reader: str, version: str) -> SourceFile:
    """The provenance record every import carries."""
    return SourceFile(
        filename=path.name,
        file_hash=file_hash(path),
        reader=reader,
        reader_version=version,
    )


def index_axis(n: int) -> VariableAxis:
    """The axis a file that states none falls back to.

    Deliberately `index` rather than a reconstructed wavelength range: Tecator
    taught us that a reconstructed axis is indistinguishable from a read one
    three months later, and `datasets.load_tecator` documents its own
    reconstruction precisely because it cannot be inferred. A reader that does
    not know the axis says so instead of inventing plausible numbers.
    """
    return VariableAxis(kind=AxisKind.INDEX, values=[float(i) for i in range(n)], unit=None)
