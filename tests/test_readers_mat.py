"""The MATLAB MAT-file reader (#284), on a slice of MLNIRdata (CC-BY-4.0).

The claim is the one `docs/decisions/0006-phase-5-data-sources.md` makes for
every new format: a real file imports, and matches its source's own export -
here the publisher's CSV twin of the same data, which is a separate file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, preview, read

HERE = Path(__file__).parent / "fixtures" / "readers" / "mat"
SLICE = HERE / "mlnir_slice.mat"
FULL = Path(__file__).parent.parent / "dataset" / "mat"
FULL_MAT = FULL / "MLNIRdata_matrixXY_NirSpectrum_DensityNormalized.mat"
DENSITY = "matrixYNirPropertyDensityNormalized"


def _csv(name: str) -> np.ndarray:
    return np.loadtxt(HERE / name, delimiter=",")


def test_the_spectra_are_found_turned_the_right_way_and_match_the_csv_twin() -> None:
    """MLNIRdata stores variables down the rows; a sample must come out as a row."""
    imported = read(SLICE)
    data = _csv("mlnir_slice_data.csv")
    assert imported.values.shape == (12, 53)
    np.testing.assert_allclose(imported.values, data.T, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(imported.axis.values, _csv("mlnir_slice_axis.csv"), rtol=1e-12)
    np.testing.assert_allclose(
        imported.targets[DENSITY], _csv("mlnir_slice_density.csv"), rtol=1e-12
    )
    assert imported.source.reader == "matlab_mat"


def test_the_preview_offers_every_choice_it_made() -> None:
    detected = preview(SLICE)["detected"]
    assert detected["matrix"] == {
        "value": "matrixXNirSpectrumData",
        "alternatives": ["matrixXNirSpectrumDerivative"],
    }
    assert detected["orientation"]["value"] == "samples_in_columns"
    assert detected["axis_variable"] == {
        "value": "matrixXNirSpectrumDataAxis",
        "alternatives": ["none"],
    }
    assert detected["targets"] == [DENSITY]
    # Ascending and above 3000: wavenumbers, and said so.
    assert detected["axis"]["kind"] == AxisKind.WAVENUMBER_CM1.value
    assert "wavenumbers" in detected["axis"]["note"]


def test_choosing_the_other_matrix_takes_its_own_axis() -> None:
    detected = preview(SLICE, {"matrix": "matrixXNirSpectrumDerivative"})["detected"]
    assert detected["n_variables"] == 52
    assert detected["axis_variable"]["value"] == "matrixXNirSpectrumDerivativeAxis"
    imported = read(SLICE, {"matrix": "matrixXNirSpectrumDerivative"})
    assert imported.values.shape == (12, 52)


def test_no_axis_falls_back_to_an_index_and_says_so() -> None:
    imported = read(SLICE, {"axis_variable": "none"})
    assert imported.axis.kind is AxisKind.INDEX


def test_a_matrix_stored_the_usual_way_reads_without_turning(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    file = tmp_path / "plain.mat"
    x = rng.standard_normal((10, 30))
    scipy.io.savemat(file, {"X": x, "wl": np.arange(900.0, 1800.0, 30.0), "y": rng.random(10)})
    imported = read(file)
    np.testing.assert_array_equal(imported.values, x)
    assert imported.axis.kind is AxisKind.WAVELENGTH_NM
    assert list(imported.targets) == ["y"]


def test_a_v7_3_file_is_refused_by_name(tmp_path: Path) -> None:
    """v7.3 is HDF5 underneath; SciPy reads its header and declines."""
    file = tmp_path / "large.mat"
    header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, HDF5 schema 1.00 ."
    file.write_bytes(header.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM" + b"\x00" * 512)
    with pytest.raises(ReaderError, match=r"v7\.3 MAT-file.*-v7"):
        read(file)


def test_a_file_with_no_matrix_is_refused_by_name(tmp_path: Path) -> None:
    file = tmp_path / "scalars.mat"
    scipy.io.savemat(file, {"a": 1.0, "b": np.arange(5.0)})
    with pytest.raises(ReaderError, match="no two-dimensional numeric array"):
        read(file)


@pytest.mark.skipif(not FULL_MAT.exists(), reason="MLNIRdata is not downloaded into dataset/mat/")
def test_the_whole_mlnirdata_file_matches_its_csv_export() -> None:
    imported = read(FULL_MAT)
    assert imported.values.shape == (208, 2635)
    data = np.loadtxt(FULL / "MLNIR_matrixX_NirSpectrumData.csv", delimiter=",")
    np.testing.assert_allclose(imported.values, data.T, rtol=1e-12, atol=1e-13)
