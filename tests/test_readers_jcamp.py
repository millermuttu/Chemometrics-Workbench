"""The JCAMP-DX reader (#80).

Two fixtures. `tecator_subset.jdx` is the same eight samples and twelve channels
as every other fixture here, written as a LINK block of eight spectra — so the
reader can be checked against the CSV and the workbook rather than only against
itself. `compressed_forms.jdx` is hand-written and small enough that its five
ordinates can be stated in the test: it is the only way to assert that SQZ, DIF
and DUP decode to the numbers they are supposed to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, apply_corrections, jcamp, preview, read

FIXTURES = Path(__file__).parent / "fixtures" / "readers"
BOOK = FIXTURES / "tecator_subset.jdx"
FORMS = FIXTURES / "compressed_forms.jdx"
CSV = FIXTURES / "tecator_subset.csv"

HEADER = """##TITLE=synthetic
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=1000
##LASTX=997
##NPOINTS=4
##XFACTOR=1.0
##YFACTOR=1.0
"""


def _write(path: Path, body: str, header: str = HEADER) -> Path:
    path.write_text(header + body + "##END=\n", encoding="ascii")
    return path


@pytest.fixture(scope="module")
def expected() -> np.ndarray:
    return load_tecator().spectra[:8, :12]


# --- Multi-block files ----------------------------------------------------


def test_every_block_is_a_sample(expected: np.ndarray) -> None:
    detection = jcamp.sniff(BOOK)

    assert (detection.n_samples, detection.n_variables) == expected.shape


def test_a_block_title_is_the_sample_name() -> None:
    imported = read(BOOK)

    assert imported.sample_ids[:2] == ("C001", "C002")


def test_the_link_header_is_carried_onto_every_sample() -> None:
    """A LINK block's own header belongs to the spectra inside it."""
    imported = read(BOOK)

    assert imported.metadata_columns["ORIGIN"][0].startswith("Tecator")
    assert len(imported.metadata_columns["ORIGIN"]) == 8
    assert imported.metadata_columns["YUNITS"][0] == "ABSORBANCE"


def test_blocks_on_different_axes_are_two_datasets(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jdx"
    path.write_text(
        HEADER
        + "##XYDATA=(X++(Y..Y))\n1000 1 2 3 4\n##END=\n"
        + HEADER.replace("##NPOINTS=4", "##NPOINTS=3").replace("##TITLE=synthetic", "##TITLE=other")
        + "##XYDATA=(X++(Y..Y))\n1000 1 2 3\n##END=\n",
        encoding="ascii",
    )

    with pytest.raises(ReaderError, match="A dataset needs one axis"):
        jcamp.sniff(path)


# --- The axis is read, not reconstructed ----------------------------------


def test_the_axis_comes_from_the_file() -> None:
    detection = jcamp.sniff(BOOK)

    assert detection.axis.kind is AxisKind.WAVELENGTH_NM
    assert detection.axis.unit == "nm"
    assert not detection.axis_reconstructed
    assert detection.axis.values[0] == pytest.approx(850.0)
    assert detection.axis.values[-1] == pytest.approx(872.2222, abs=1e-3)


def test_wavenumbers_are_named_as_such() -> None:
    imported = read(FORMS)

    assert imported.axis.kind is AxisKind.WAVENUMBER_CM1
    assert imported.axis.unit == "cm-1"
    assert imported.axis.values == [1000.0, 999.0, 998.0, 997.0, 996.0]


def test_xfactor_and_yfactor_are_applied(tmp_path: Path) -> None:
    header = HEADER.replace("##XFACTOR=1.0", "##XFACTOR=2.0").replace(
        "##YFACTOR=1.0", "##YFACTOR=0.001"
    )
    header = header.replace("##NPOINTS=4", "##NPOINTS=2")
    path = _write(tmp_path / "scaled.jdx", "##XYDATA=(XY..XY)\n500 1000 501 2000\n", header)

    imported = read(path)
    assert imported.values[0].tolist() == [1.0, 2.0]
    assert imported.axis.values == [1000.0, 1002.0]


def test_an_unknown_unit_leaves_the_axis_numbered(tmp_path: Path) -> None:
    """Declining to name an axis beats naming it wrongly on a plot."""
    header = HEADER.replace("##XUNITS=1/CM", "##XUNITS=MICROMETERS")
    path = _write(tmp_path / "micro.jdx", "##XYDATA=(X++(Y..Y))\n1000 1 2 3 4\n", header)

    detection = jcamp.sniff(path)
    assert detection.axis.kind is AxisKind.INDEX
    assert "MICROMETERS" in (detection.axis_note or "")


def test_a_missing_axis_is_refused_rather_than_invented(tmp_path: Path) -> None:
    header = HEADER.replace("##FIRSTX=1000\n", "").replace("##LASTX=997\n", "")
    path = _write(tmp_path / "axisless.jdx", "##XYDATA=(X++(Y..Y))\n1000 1 2 3 4\n", header)

    with pytest.raises(ReaderError, match="does not invent one"):
        jcamp.sniff(path)


# --- The compressed forms -------------------------------------------------


def test_sqz_dif_and_dup_decode_to_their_numbers() -> None:
    """`A00 J K` then `A03 J T`: 100, +1, +2, restate 103, +1, repeat."""
    imported = read(FORMS)

    assert imported.values[0].tolist() == [100.0, 101.0, 103.0, 104.0, 104.0]


def test_plain_pac_values_decode(tmp_path: Path) -> None:
    path = _write(tmp_path / "pac.jdx", "##XYDATA=(X++(Y..Y))\n1000 +10-20+30-40\n")

    assert read(path).values[0].tolist() == [10.0, -20.0, 30.0, -40.0]


def test_negative_sqz_and_dif_decode(tmp_path: Path) -> None:
    path = _write(tmp_path / "negatives.jdx", "##XYDATA=(X++(Y..Y))\n1000 a0 j J %\n")

    # a0 is -10; j is -1 from it; J is +1; % is +0.
    assert read(path).values[0].tolist() == [-10.0, -11.0, -10.0, -10.0]


def test_a_difference_form_that_does_not_check_out_is_refused(tmp_path: Path) -> None:
    """The format's own checksum. A misread DIF line yields a plausible spectrum."""
    header = HEADER.replace("##NPOINTS=4", "##NPOINTS=5")
    path = _write(
        tmp_path / "broken.jdx", "##XYDATA=(X++(Y..Y))\n1000 A00 J K\n997 A09 J\n", header
    )

    with pytest.raises(ReaderError, match="difference form has been misread"):
        read(path)


def test_a_peak_table_is_refused_rather_than_read_as_a_spectrum(tmp_path: Path) -> None:
    path = _write(tmp_path / "peaks.jdx", "##PEAK TABLE=(XY..XY)\n1000,1 999,2\n")

    with pytest.raises(ReaderError, match="list of peaks"):
        jcamp.sniff(path)


def test_a_point_count_that_disagrees_with_the_data_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "short.jdx", "##XYDATA=(X++(Y..Y))\n1000 1 2\n")

    with pytest.raises(ReaderError, match="##NPOINTS=4 but its data decodes to 2"):
        jcamp.sniff(path)


def test_an_unsupported_data_form_is_named_not_guessed_at(tmp_path: Path) -> None:
    """Guessing at a form produces a spectrum that looks right and is not."""
    path = _write(tmp_path / "ntuples.jdx", "##XYDATA=(T2..T2)\n1000 1 2 3 4\n")

    with pytest.raises(ReaderError, match="does not read"):
        jcamp.sniff(path)


# --- Diagnostics, not stack traces ----------------------------------------


def test_a_file_that_is_not_jcamp_is_named(tmp_path: Path) -> None:
    path = tmp_path / "nope.jdx"
    path.write_text("id,850,860\nA,0.1,0.2\n", encoding="ascii")

    with pytest.raises(ReaderError, match="not a JCAMP-DX file"):
        jcamp.sniff(path)


def test_a_file_with_headers_but_no_data_is_named(tmp_path: Path) -> None:
    path = tmp_path / "headers.jdx"
    path.write_text(HEADER + "##END=\n", encoding="ascii")

    with pytest.raises(ReaderError, match="no spectrum in it"):
        jcamp.sniff(path)


def test_an_empty_file_is_named(tmp_path: Path) -> None:
    path = tmp_path / "empty.jdx"
    path.write_text("", encoding="ascii")

    with pytest.raises(ReaderError, match="is empty"):
        jcamp.sniff(path)


def test_a_token_in_no_form_this_reader_knows_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path / "junk.jdx", "##XYDATA=(X++(Y..Y))\n1000 1 2 ~3 4\n")

    with pytest.raises(ReaderError, match="not a number in any form"):
        read(path)


# --- Nothing to correct ---------------------------------------------------


def test_a_jcamp_file_offers_no_corrections() -> None:
    """The honest answer when the format has already answered every question."""
    detection = jcamp.sniff(BOOK)

    assert detection.correctable == ()
    assert detection.delimiter.alternatives == ()
    assert detection.orientation.alternatives == ()


def test_any_correction_is_refused() -> None:
    detection = jcamp.sniff(BOOK)

    with pytest.raises(ReaderError, match="cannot be corrected"):
        apply_corrections(detection, {"orientation": "samples_in_columns"})


# --- Agreement with the other readers -------------------------------------


def test_jcamp_and_csv_of_the_same_data_agree(expected: np.ndarray) -> None:
    """Three formats, one dataset. That is what the interface is for."""
    from_jcamp = read(BOOK)
    from_csv = read(CSV)

    assert from_jcamp.values == pytest.approx(from_csv.values, abs=1e-4)
    assert from_jcamp.values == pytest.approx(expected, abs=1e-4)
    assert from_jcamp.sample_ids == from_csv.sample_ids


def test_the_import_records_its_reader_and_the_file_hash() -> None:
    imported = read(BOOK)

    assert imported.source.reader == jcamp.NAME
    assert imported.source.reader_version == jcamp.VERSION
    assert imported.source.file_hash.startswith("sha256:")


def test_the_preview_keeps_the_shape_the_frontend_renders() -> None:
    payload = preview(BOOK)

    assert set(payload) == {"source", "detected", "head"}
    assert payload["source"]["reader"] == jcamp.NAME
    assert payload["detected"]["axis"]["kind"] == "wavelength_nm"
    assert payload["detected"]["axis"]["reconstructed"] is False
    assert payload["head"]["sample_ids"][0] == "C001"
    assert payload["head"]["rows"][0][0] == pytest.approx(2.6178, abs=1e-4)
