"""Reference datasets for parity testing.

Three benchmark NIR datasets, each with a loader returning a spectra matrix,
its wavelength axis in real units, reference values, and a `SourceFile`
recording where the bytes came from and what read them.

Only one of the three is committed to this repository. Tecator carries an
explicit redistribution permission; corn and gasoline do not, so they are
downloaded on first use and verified against a pinned SHA-256. Each dataset's
directory under `data/` records the terms that decided it, the source URL and
the checksum. Read those before changing anything here.

The download cache lives at `$CHEMOMETRICS_DATA_HOME`, else
`$XDG_CACHE_HOME/chemometrics-workbench/datasets`, else
`~/.cache/chemometrics-workbench/datasets`.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import warnings
import zipfile
from dataclasses import dataclass
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from chemometrics_workbench.models import AxisKind, SourceFile, VariableAxis

__all__ = [
    "ReferenceDataset",
    "load_corn",
    "load_gasoline",
    "load_tecator",
]

DATA_DIR = Path(__file__).parent / "data"

# Network reads are a courtesy to the host, not an interactive operation.
_DOWNLOAD_TIMEOUT_S = 120


@dataclass(frozen=True, eq=False)
class ReferenceDataset:
    """One benchmark dataset, loaded.

    `eq=False` because the fields are arrays: the generated `__eq__` would
    raise on the first ambiguous truth value rather than compare anything.
    """

    spectra: NDArray[np.float64]
    """Absorbance or log(1/R), shaped n_samples x n_variables. Never transposed."""

    axis: VariableAxis
    """The wavelength axis in real units, shared by every spectrum."""

    targets: dict[str, NDArray[np.float64]]
    """Reference values by property name, each of length n_samples."""

    source: SourceFile
    """Provenance: the file read, its hash, and the reader that read it."""

    sample_ids: tuple[str, ...] = ()

    @property
    def n_samples(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_variables(self) -> int:
        return int(self.spectra.shape[1])


# --------------------------------------------------------------------------
# fetching and verifying
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Remote:
    """A file this repository does not carry, and the hash it must have."""

    filename: str
    url: str
    sha256: str


CORN_ARCHIVE = _Remote(
    filename="corn.mat_.zip",
    url="https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip",
    sha256="8a2d1a03648b6ad334caaafa5d8377bf945ba1477a5082c4963d705d07cca795",
)
CORN_MEMBER = "corn.mat"
CORN_MEMBER_SHA256 = "e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5"

# CRAN's Archive/ rather than src/contrib/: src/contrib holds only the current
# release, so a pinned hash there breaks on the next pls release.
GASOLINE_ARCHIVE = _Remote(
    filename="pls_2.8-5.tar.gz",
    url="https://cran.r-project.org/src/contrib/Archive/pls/pls_2.8-5.tar.gz",
    sha256="8029018d4c8921fa4c7ec5081551afdcc55d53271d9920db828483b442a033cf",
)
GASOLINE_MEMBER = "pls/data/gasoline.RData"
GASOLINE_MEMBER_SHA256 = "fdfd17ff9a407d9ee04d0c966aab3e4f2e0390c39724cab99334c74b842acbdd"

TECATOR_FILE = DATA_DIR / "tecator" / "tecator.txt"
TECATOR_SHA256 = "e435c0538cd706473d87525da595d75dbff43a58bf2694104bd948081c3790d7"


def cache_dir() -> Path:
    """Where downloaded datasets are kept between runs."""
    override = os.environ.get("CHEMOMETRICS_DATA_HOME")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "chemometrics-workbench" / "datasets"


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _check(blob: bytes, expected: str, what: str) -> bytes:
    actual = _digest(blob)
    if actual != expected:
        raise ValueError(
            f"{what} does not match its pinned checksum.\n"
            f"  expected sha256:{expected}\n"
            f"  actual   sha256:{actual}\n"
            "Either the file was edited or the source changed what it serves. "
            "Do not update the pin without establishing which."
        )
    return blob


def is_cached(remote: _Remote) -> bool:
    """True when `remote` is already downloaded, without touching the network."""
    return (cache_dir() / remote.filename).exists()


def fetch(remote: _Remote) -> Path:
    """Return a local path to `remote`, downloading it if the cache is empty.

    The checksum is verified on download and on every subsequent read, so a
    corrupted or tampered cache entry fails loudly rather than quietly
    producing different science.
    """
    path = cache_dir() / remote.filename
    if path.exists():
        _check(path.read_bytes(), remote.sha256, str(path))
        return path

    with urlopen(remote.url, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        blob = response.read()
    _check(blob, remote.sha256, f"{remote.url} ({len(blob)} bytes)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _source_file(filename: str, blob: bytes, reader: str, reader_version: str) -> SourceFile:
    return SourceFile(
        filename=filename,
        file_hash=f"sha256:{_digest(blob)}",
        reader=reader,
        reader_version=reader_version,
    )


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

Instrument = Literal["m5", "mp5", "mp6"]

_CORN_TARGET_ORDER = ("moisture", "oil", "protein", "starch")


def load_corn(instrument: Instrument = "m5") -> ReferenceDataset:
    """80 corn samples on one of three NIR spectrometers, 1100-2498 nm.

    Downloads `corn.mat` on first use; see `data/corn/README.md` for why it is
    not committed. Targets are moisture, oil, protein and starch in percent,
    and are identical across the three instruments — the same samples were
    measured on each, which is what makes this the calibration-transfer
    benchmark.
    """
    from scipy.io import loadmat  # heavy import, and only this loader needs it

    archive = fetch(CORN_ARCHIVE)
    with zipfile.ZipFile(archive) as zf:
        blob = _check(zf.read(CORN_MEMBER), CORN_MEMBER_SHA256, f"{CORN_MEMBER} in {archive}")

    # A MATLAB struct array of PLS_Toolbox DataSet objects; each field arrives
    # wrapped in an object array, hence the [0, 0] unwrapping throughout.
    mat = loadmat(io.BytesIO(blob))
    spectra_obj = mat[f"{instrument}spec"][0, 0]
    spectra = np.asarray(spectra_obj["data"], dtype=np.float64)

    # axisscale is a 2 x 2 cell grid: [mode, set]. The wavelengths are the
    # column-mode (mode 1) scale of the first set.
    wavelengths = np.ravel(spectra_obj["axisscale"][1, 0]).astype(np.float64)

    properties = mat["propvals"][0, 0]
    values = np.asarray(properties["data"], dtype=np.float64)
    # label is the same 2 x 2 [mode, set] cell grid as axisscale; the property
    # names are the column-mode labels of the first set, and are space-padded.
    labels = [str(label).strip().lower() for label in np.ravel(properties["label"][1, 0])]
    if tuple(labels) != _CORN_TARGET_ORDER:
        raise ValueError(f"unexpected property labels in corn.mat: {labels}")

    return ReferenceDataset(
        spectra=spectra,
        axis=VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=wavelengths.tolist(), unit="nm"),
        targets={name: values[:, i] for i, name in enumerate(labels)},
        source=_source_file(
            filename=f"{CORN_ARCHIVE.filename}::{CORN_MEMBER}",
            blob=blob,
            reader="scipy.io.loadmat",
            reader_version=_package_version("scipy"),
        ),
    )


def load_gasoline() -> ReferenceDataset:
    """60 gasoline samples, log(1/R) from 900 to 1700 nm, with octane numbers.

    Downloads the CRAN source tarball of the R package `pls` on first use and
    reads `gasoline.RData` out of it; see `data/gasoline/README.md` for why it
    is not committed.
    """
    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "load_gasoline() reads an R .RData file and needs the 'rdata' package, "
            "which is in the dev dependency group rather than the runtime "
            "dependencies. Install it with `uv sync` or `pip install rdata`."
        ) from exc

    archive = fetch(GASOLINE_ARCHIVE)
    with tarfile.open(archive) as tf:
        member = tf.extractfile(GASOLINE_MEMBER)
        if member is None:
            raise ValueError(f"{GASOLINE_MEMBER} missing from {archive}")
        blob = _check(member.read(), GASOLINE_MEMBER_SHA256, f"{GASOLINE_MEMBER} in {archive}")

    # An empty constructor_dict skips rdata's data.frame constructor, which
    # cannot represent the 60 x 401 AsIs matrix column as a pandas column. It
    # warns about every class it then has no constructor for; that is the
    # intent here, not a problem, so the warnings are suppressed rather than
    # left to look like a defect.
    parsed = rdata.parser.parse_data(blob)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        warnings.filterwarnings("ignore", message="Unknown encoding")
        converted: dict[str, Any] = rdata.conversion.SimpleConverter(constructor_dict={}).convert(
            parsed
        )
    frame = converted["gasoline"]

    nir = frame["NIR"]
    spectra = np.asarray(nir, dtype=np.float64)
    # Dimnames carry the real axis, e.g. "900 nm" — read it rather than
    # reconstructing a linspace and hoping the step is uniform.
    wavelengths = [float(str(name).split()[0]) for name in np.ravel(nir.coords["dim_1"].values)]

    return ReferenceDataset(
        spectra=spectra,
        axis=VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=wavelengths, unit="nm"),
        targets={"octane": np.asarray(frame["octane"], dtype=np.float64)},
        source=_source_file(
            filename=f"{GASOLINE_ARCHIVE.filename}::{GASOLINE_MEMBER}",
            blob=blob,
            reader="rdata",
            reader_version=_package_version("rdata"),
        ),
    )


# The 240 samples appear in five contiguous groups, in this order. The file
# states the sizes; the loader encodes them into sample ids so the published
# split survives any later reordering of rows.
TECATOR_GROUPS = (("C", 129), ("M", 43), ("T", 43), ("E1", 8), ("E2", 17))

# 100 absorbances, then 22 principal components computed by the original
# authors, then moisture, fat and protein.
_TECATOR_N_ABSORBANCES = 100
_TECATOR_N_COMPONENTS = 22
_TECATOR_TARGETS = ("moisture", "fat", "protein")
_TECATOR_ROW_WIDTH = _TECATOR_N_ABSORBANCES + _TECATOR_N_COMPONENTS + len(_TECATOR_TARGETS)

# The file gives a range and a channel count but no wavelength vector.
TECATOR_RANGE_NM = (850.0, 1050.0)


def load_tecator() -> ReferenceDataset:
    """240 meat samples, 100 NIT absorbance channels over 850-1050 nm.

    Committed to this repository under the permission note reproduced in
    `data/tecator/README.md`. **If you publish a result from this dataset you
    must name the instrument and company (Tecator).**

    The wavelength axis is reconstructed as `linspace(850, 1050, 100)`; the
    file itself carries no axis. The 22 principal components the file also
    supplies are preprocessing, not raw data, and are discarded.
    """
    blob = _check(TECATOR_FILE.read_bytes(), TECATOR_SHA256, str(TECATOR_FILE))
    text = blob.decode("ascii")

    # The header block is printed twice, so split on the last occurrence. The
    # first token after the marker is the count on the marker line itself.
    tail = text.rsplit("extrapolation_examples=", 1)[1]
    values = np.fromiter(tail.split()[1:], dtype=np.float64)
    if values.size % _TECATOR_ROW_WIDTH:
        raise ValueError(
            f"tecator.txt holds {values.size} values, not a multiple of {_TECATOR_ROW_WIDTH}"
        )
    rows = values.reshape(-1, _TECATOR_ROW_WIDTH)

    spectra = rows[:, :_TECATOR_N_ABSORBANCES]
    reference = rows[:, _TECATOR_N_ABSORBANCES + _TECATOR_N_COMPONENTS :]

    sample_ids = tuple(
        f"{group}{i:03d}" for group, count in TECATOR_GROUPS for i in range(1, count + 1)
    )
    if len(sample_ids) != rows.shape[0]:
        raise ValueError(f"{len(sample_ids)} group slots for {rows.shape[0]} samples")

    low, high = TECATOR_RANGE_NM
    return ReferenceDataset(
        spectra=spectra,
        axis=VariableAxis(
            kind=AxisKind.WAVELENGTH_NM,
            values=np.linspace(low, high, _TECATOR_N_ABSORBANCES).tolist(),
            unit="nm",
        ),
        targets={name: reference[:, i] for i, name in enumerate(_TECATOR_TARGETS)},
        source=_source_file(
            filename=TECATOR_FILE.name,
            blob=blob,
            reader="tecator_txt",
            reader_version="1",
        ),
        sample_ids=sample_ids,
    )
