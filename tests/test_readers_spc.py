"""The Galactic SPC reader (#285), on real files with their licences.

New and old format, single spectra and multifiles, even and explicit axes are
each read and compared against the text conversion `rohanisaac/spc` publishes
beside the file. The bone file (CC-BY-4.0) has no such twin, and is checked
for what its header states.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, preview, read

HERE = Path(__file__).parent / "fixtures" / "readers" / "spc"
OTHER = HERE / "rohanisaac"


def _twin(name: str) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(OTHER / f"{name}.txt")
    return table[:, 0], table[:, 1:].T


@pytest.mark.parametrize(
    ("name", "shape", "kind"),
    [
        ("nir.spc", (20, 700), AxisKind.WAVELENGTH_NM),  # new format, multifile, even
        ("s_xy.spc", (1, 512), AxisKind.INDEX),  # new format, explicit x in minutes
        ("DOERNER.spc", (1, 1602), AxisKind.RAMAN_SHIFT_CM1),  # old format, single
        ("m_ordz.spc", (10, 857), AxisKind.WAVENUMBER_CM1),  # old format, multifile
    ],
)
def test_each_layout_matches_its_text_conversion(
    name: str, shape: tuple[int, int], kind: AxisKind
) -> None:
    imported = read(OTHER / name)
    x, y = _twin(name)
    assert imported.values.shape == shape
    assert imported.axis.kind is kind
    np.testing.assert_allclose(imported.values, y, rtol=1e-9, atol=1e-12 * np.abs(y).max())
    if kind is not AxisKind.INDEX:
        np.testing.assert_allclose(imported.axis.values, x, rtol=1e-9)


def test_an_explicit_x_array_is_read_not_reconstructed() -> None:
    """`s_xy` is in minutes, so its axis is numbered, and the note says what it was."""
    detected = preview(OTHER / "s_xy.spc")["detected"]
    assert "minutes" in detected["axis"]["note"]
    x, _ = _twin("s_xy.spc")
    # Not an even grid: the file's own x values, which the note quotes.
    assert not np.allclose(np.diff(x), np.diff(x)[0])
    assert f"{x[0]:g}" in detected["axis"]["note"]


def test_a_negative_subfile_exponent_is_signed() -> None:
    """`char subexp` is signed. Where it is not negative, the text conversion
    agrees; where it is, ours is the conversion's times 2^exp."""
    imported = read(OTHER / "m_evenz.spc")
    _, y = _twin("m_evenz.spc")
    data = (OTHER / "m_evenz.spc").read_bytes()
    npts = struct.unpack_from("<i", data, 4)[0]
    for k in range(imported.values.shape[0]):
        exponent = struct.unpack_from("<b", data, 512 + k * (32 + 4 * npts) + 1)[0]
        expected = y[k] * (2.0**exponent if exponent < 0 else 1.0)
        np.testing.assert_allclose(imported.values[k], expected, rtol=1e-9, atol=1e-15)
    assert any(
        struct.unpack_from("<b", data, 512 + k * (32 + 4 * npts) + 1)[0] < 0 for k in range(32)
    ), "the fixture exercises a negative exponent"


def test_the_bone_spectrum_reads_as_its_header_states() -> None:
    imported = read(HERE / "bone_sample_1.spc")
    assert imported.values.shape == (1, 14935)
    assert imported.axis.kind is AxisKind.WAVENUMBER_CM1
    axis = np.asarray(imported.axis.values)
    assert axis[0] == pytest.approx(3999.8811, abs=1e-3)
    assert axis[-1] == pytest.approx(399.9158, abs=1e-3)
    # Absorbance of a bone powder: finite, and within what an FTIR reports.
    assert np.all(np.isfinite(imported.values)) and np.abs(imported.values).max() < 5
    assert imported.sample_ids == ("bone_sample_1",)
    assert imported.source.reader == "galactic_spc"


def test_a_multifile_names_each_subfile() -> None:
    imported = read(OTHER / "nir.spc")
    assert imported.sample_ids[:2] == ("nir #1", "nir #2")


def test_the_xyxy_layout_is_refused_by_name() -> None:
    with pytest.raises(ReaderError, match="xyxy SPC file"):
        read(OTHER / "ms.spc")


def test_a_file_that_is_not_spc_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "text.spc"
    file.write_text("wavelength,absorbance\n")
    with pytest.raises(ReaderError, match="does not carry an SPC version byte"):
        read(file)


def test_a_truncated_file_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "cut.spc"
    file.write_bytes((HERE / "bone_sample_1.spc").read_bytes()[:4000])
    with pytest.raises(ReaderError, match="truncated"):
        read(file)
