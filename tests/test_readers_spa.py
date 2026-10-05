"""The OMNIC SPA reader (#286), on real files with their licences.

The claim decision 0006 makes for every new format: a real file imports and
matches its source's own export - here OMNIC's CSV of the same spectrum,
which lists it ascending and pads one zero point below the range.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, preview, read, reader_for

HERE = Path(__file__).parent / "fixtures" / "readers" / "spa"
ECOFLEX = HERE / "ecoflex-1.SPA"
DUST = sorted((HERE / "dust").glob("*.SPA"))


def test_an_spa_file_matches_omnics_own_csv_export() -> None:
    imported = read(ECOFLEX)
    twin = np.loadtxt(HERE / "ecoflex-1.CSV", delimiter=",")
    assert imported.values.shape == (1, 7468)
    assert imported.axis.kind is AxisKind.WAVENUMBER_CM1
    # The file runs 4000 to 400 cm-1; the export ascends and adds one zero
    # point below 400, which is not in the file.
    assert twin[0, 1] == 0.0 and twin.shape[0] == 7469
    np.testing.assert_allclose(imported.values[0][::-1], twin[1:, 1], rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(np.asarray(imported.axis.values)[::-1], twin[1:, 0], atol=1e-3)
    assert imported.metadata_columns == {"title": ["ecoflex-1"]}
    assert imported.source.reader == "omnic_spa"


def test_a_zip_of_spa_files_is_one_dataset_on_their_shared_axis(tmp_path: Path) -> None:
    archive = tmp_path / "dust.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file in DUST:
            bundle.write(file, file.name)
    assert reader_for(archive).NAME == "omnic_spa"
    imported = read(archive)
    assert imported.values.shape == (len(DUST), 29868)
    assert imported.sample_ids == tuple(file.stem for file in DUST)
    single = read(DUST[0])
    np.testing.assert_array_equal(imported.values[0], single.values[0])


def test_a_zip_mixing_transmittance_and_absorbance_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(ECOFLEX, ECOFLEX.name)  # %T
        bundle.write(HERE / "Eco (1-1) - 1.SPA", "Eco (1-1) - 1.SPA")  # absorbance
    with pytest.raises(ReaderError, match=r"%T.*absorbance|absorbance.*%T"):
        read(archive)


def test_a_zip_of_other_files_is_still_opus(tmp_path: Path) -> None:
    archive = tmp_path / "not_spa.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.0", b"x")
    assert reader_for(archive).NAME == "bruker_opus"


def test_the_preview_says_what_it_read() -> None:
    detected = preview(ECOFLEX)["detected"]
    assert (detected["n_samples"], detected["n_variables"]) == (1, 7468)
    assert detected["axis"]["start"] == pytest.approx(4000.1875)
    assert detected["axis"]["end"] == pytest.approx(400.1635, abs=1e-3)


def test_a_file_that_is_not_spa_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "text.spa"
    file.write_bytes(b"\0" * 400)
    with pytest.raises(ReaderError, match="not an SPA file"):
        read(file)


def test_a_truncated_file_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "cut.spa"
    file.write_bytes(ECOFLEX.read_bytes()[:5000])
    with pytest.raises(ReaderError, match="truncated"):
        read(file)
