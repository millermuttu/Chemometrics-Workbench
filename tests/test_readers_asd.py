"""The ASD FieldSpec reader (#287), on real files with their licence.

The record's published reflectance is not the stored ratio itself: it is that
ratio times the white panel's calibration curve - one curve, the same for every
sample - with the VNIR detector (350 to 1000 nm) spliced to the SWIR by one
factor per sample. So the claim is that this reader's reflectance, divided out
of the published one, leaves exactly that: one curve over the SWIR, and that
curve times a constant over the VNIR.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, preview, read, reader_for

HERE = Path(__file__).parent / "fixtures" / "readers" / "asd"
FILES = sorted(HERE.glob("*.asd"))
VNIR = slice(0, 651)  # 350 to 1000 nm
SWIR = slice(651, None)


def published() -> dict[str, np.ndarray]:
    with (HERE / "measurements.csv").open(encoding="utf-8") as file:
        rows = list(csv.reader(file))
    return {row[1]: np.array(row[2:], dtype=float) for row in rows[1:]}


def bundle(tmp_path: Path, files: dict[str, bytes]) -> Path:
    archive = tmp_path / "spectra.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for name, data in files.items():
            out.writestr(name, data)
    return archive


def test_reflectance_is_the_published_one_less_the_panel_and_the_splice() -> None:
    expected = published()
    ratios = np.array([expected[f.name] / read(f).values[0] for f in FILES])
    panel = ratios[0]
    # SWIR: one curve for every sample, the panel's calibration and nothing else.
    np.testing.assert_allclose(ratios[:, SWIR], np.tile(panel[SWIR], (len(FILES), 1)), rtol=1e-8)
    assert 0.9 < panel[SWIR].min() < panel[SWIR].max() < 1.0
    # VNIR: that curve times one splice factor per sample.
    splice = ratios[:, VNIR] / panel[VNIR]
    np.testing.assert_allclose(splice, splice[:, :1] * np.ones_like(splice), rtol=1e-8)


def test_the_axis_is_the_header_wavelengths() -> None:
    imported = read(FILES[0])
    assert imported.values.shape == (1, 2151)
    assert imported.axis.kind is AxisKind.WAVELENGTH_NM
    assert (imported.axis.values[0], imported.axis.values[-1]) == (350.0, 2500.0)
    assert imported.source.reader == "asd_fieldspec"
    detected = preview(FILES[0])["detected"]
    assert (detected["n_samples"], detected["n_variables"]) == (1, 2151)


def test_a_zip_of_asd_files_is_one_dataset(tmp_path: Path) -> None:
    archive = bundle(tmp_path, {f.name: f.read_bytes() for f in FILES})
    assert reader_for(archive).NAME == "asd_fieldspec"
    imported = read(archive)
    assert imported.values.shape == (len(FILES), 2151)
    assert imported.sample_ids == tuple(f.stem for f in FILES)
    np.testing.assert_array_equal(imported.values[1], read(FILES[1]).values[0])


def test_a_zip_mixing_data_types_is_refused(tmp_path: Path) -> None:
    raw = bytearray(FILES[1].read_bytes())
    raw[186] = 0
    archive = bundle(tmp_path, {FILES[0].name: FILES[0].read_bytes(), "raw.asd": bytes(raw)})
    with pytest.raises(ReaderError, match=r"raw counts where .* is reflectance"):
        read(archive)


def test_a_raw_file_is_read_as_stored(tmp_path: Path) -> None:
    raw = bytearray(FILES[0].read_bytes())
    raw[186] = 0
    file = tmp_path / "raw.asd"
    file.write_bytes(bytes(raw))
    counts = np.frombuffer(bytes(raw), dtype="<f8", count=2151, offset=484)
    np.testing.assert_array_equal(read(file).values[0], counts)


def test_a_file_that_is_not_asd_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "text.asd"
    file.write_bytes(b"\0" * 600)
    with pytest.raises(ReaderError, match="not an ASD file"):
        read(file)


@pytest.mark.parametrize(
    ("cut", "where"), [(300, "header"), (5000, "truncated"), (18000, "white reference")]
)
def test_a_truncated_file_is_refused_by_name(tmp_path: Path, cut: int, where: str) -> None:
    file = tmp_path / "cut.asd"
    file.write_bytes(FILES[0].read_bytes()[:cut])
    with pytest.raises(ReaderError, match=where):
        read(file)
