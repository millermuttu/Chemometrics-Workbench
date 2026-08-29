"""The XLSX reader (#79).

`tecator_subset.xlsx` holds the same eight samples and twelve channels as the
CSV fixtures, so the sharpest test here is not that the reader works but that
it agrees: a workbook and a comma-separated file of the same data come back as
the same numbers, which is what makes `grid.py` one implementation rather than
two that happen to look alike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, apply_corrections, preview, read, xlsx

FIXTURES = Path(__file__).parent / "fixtures" / "readers"
BOOK = FIXTURES / "tecator_subset.xlsx"
CSV = FIXTURES / "tecator_subset.csv"


@pytest.fixture(scope="module")
def expected() -> np.ndarray:
    return load_tecator().spectra[:8, :12]


def _write(path: Path, rows: list[list[object]], sheet: str = "Sheet1") -> Path:
    book = Workbook()
    book.active.title = sheet
    for row in rows:
        book.active.append(row)
    book.save(path)
    return path


# --- Detection ------------------------------------------------------------


def test_a_workbook_is_detected(expected: np.ndarray) -> None:
    detection = xlsx.sniff(BOOK)

    assert (detection.n_samples, detection.n_variables) == expected.shape
    assert detection.orientation.value == "samples_in_rows"
    assert detection.axis.kind is AxisKind.WAVELENGTH_NM


def test_a_workbook_has_no_delimiter_to_offer() -> None:
    """The field stays in the shape; it just has nothing to decide."""
    detection = xlsx.sniff(BOOK)

    assert detection.delimiter.value == xlsx.NOT_APPLICABLE
    assert detection.delimiter.alternatives == ()
    assert "delimiter" not in detection.correctable


def test_the_sheets_are_offered_with_the_first_chosen() -> None:
    detection = xlsx.sniff(BOOK)

    assert detection.sheet is not None
    assert detection.sheet.value == "Spectra"
    assert detection.sheet.alternatives == ("Notes",)
    assert "sheet" in detection.correctable


def test_targets_and_metadata_are_separated_as_in_a_csv() -> None:
    detection = xlsx.sniff(BOOK)

    assert list(detection.targets) == ["fat", "moisture"]
    assert list(detection.metadata_columns) == ["batch"]


# --- The packaging a spreadsheet arrives in -------------------------------


def test_a_merged_title_row_does_not_shift_the_data(expected: np.ndarray) -> None:
    """The fixture's first row is a merged title over a blank row. Both go."""
    imported = read(BOOK)

    assert imported.values.shape == expected.shape
    assert imported.sample_ids[0] == "C001"


def test_trailing_blank_rows_and_columns_are_not_samples(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "gaps.xlsx",
        [
            ["sample", 850, 860, None],
            ["A", 0.1, 0.2, None],
            ["B", 0.2, 0.3, None],
            [None, None, None, None],
        ],
    )

    detection = xlsx.sniff(path)
    assert (detection.n_samples, detection.n_variables) == (2, 2)


def test_a_sheet_with_no_table_says_so(tmp_path: Path) -> None:
    path = _write(tmp_path / "banner.xlsx", [["just a note"], ["another one"]])

    with pytest.raises(ReaderError, match="no table in it"):
        xlsx.sniff(path)


def test_an_empty_sheet_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path / "blank.xlsx", [])

    with pytest.raises(ReaderError, match="is empty"):
        xlsx.sniff(path)


def test_a_file_that_is_not_a_workbook_is_named(tmp_path: Path) -> None:
    path = tmp_path / "pretend.xlsx"
    path.write_text("id,850,860\nA,0.1,0.2\n")

    with pytest.raises(ReaderError, match=r"not a workbook|cannot open"):
        xlsx.sniff(path)


def test_a_date_column_is_metadata_not_absorbance(tmp_path: Path) -> None:
    """A serial number underneath a date is a plausible-looking measurement."""
    from datetime import date

    path = _write(
        tmp_path / "dated.xlsx",
        [
            ["sample", "measured", 850, 860],
            ["A", date(2026, 3, 1), 0.1, 0.2],
            ["B", date(2026, 3, 2), 0.2, 0.3],
        ],
    )

    detection = xlsx.sniff(path)
    assert list(detection.metadata_columns) == ["measured"]
    assert detection.n_variables == 2
    assert read(path).metadata_columns["measured"][0].startswith("2026-03-01")


def test_numbers_stored_as_text_are_still_numbers(tmp_path: Path) -> None:
    """The reason every cell goes through the same detection a text file gets."""
    path = _write(
        tmp_path / "as_text.xlsx",
        [["sample", "850", "860"], ["A", "1,5", "2,5"], ["B", "3,5", "4,5"]],
    )

    detection = xlsx.sniff(path)
    assert detection.decimal.value == ","
    assert read(path).values[0, 0] == pytest.approx(1.5)


# --- Agreement with the CSV reader ----------------------------------------


def test_a_workbook_and_a_csv_of_the_same_data_agree(expected: np.ndarray) -> None:
    """One interface, two formats. This is what #79 exists to prove."""
    from_book = read(BOOK)
    from_csv = read(CSV)

    assert from_book.values == pytest.approx(from_csv.values)
    assert from_book.values == pytest.approx(expected, abs=1e-4)
    assert from_book.sample_ids == from_csv.sample_ids
    assert from_book.targets["fat"] == from_csv.targets["fat"]
    # The CSV fixture prints its wavelength headers to six significant digits;
    # the workbook holds the full float. Same axis, different rounding.
    assert from_book.axis.values == pytest.approx(from_csv.axis.values, abs=0.05)


def test_the_workbook_records_its_own_reader(expected: np.ndarray) -> None:
    imported = read(BOOK)

    assert imported.source.reader == xlsx.NAME
    assert imported.source.reader_version == xlsx.VERSION
    assert imported.source.file_hash.startswith("sha256:")


# --- Corrections ----------------------------------------------------------


def test_the_sheet_can_be_corrected() -> None:
    detection = xlsx.sniff(BOOK)
    corrected = apply_corrections(detection, {"sheet": "Notes"})

    assert corrected.sheet is not None
    assert corrected.sheet.value == "Notes"


def test_correcting_to_a_sheet_that_is_not_there_is_refused() -> None:
    detection = xlsx.sniff(BOOK)

    with pytest.raises(ReaderError, match="not one of the sheet options"):
        apply_corrections(detection, {"sheet": "Elsewhere"})


def test_a_delimiter_correction_is_refused_for_a_workbook() -> None:
    """Offering a correction that cannot be applied is the same lie as ignoring one."""
    detection = xlsx.sniff(BOOK)

    with pytest.raises(ReaderError, match="cannot be corrected"):
        apply_corrections(detection, {"delimiter": ";"})


# --- The preview shape ----------------------------------------------------


def test_the_preview_keeps_the_shape_the_frontend_renders() -> None:
    payload = preview(BOOK)

    assert set(payload) == {"source", "detected", "head"}
    assert payload["source"]["reader"] == xlsx.NAME
    detected = payload["detected"]
    assert set(detected) >= {"delimiter", "decimal", "orientation", "axis", "targets"}
    assert detected["sheet"] == {"value": "Spectra", "alternatives": ["Notes"]}
    assert payload["head"]["sample_ids"][0] == "C001"
    assert payload["head"]["rows"][0][0] == pytest.approx(2.6178)


def test_the_preview_follows_a_sheet_correction() -> None:
    with pytest.raises(ReaderError, match=r"no column of numbers|wavelength columns"):
        preview(BOOK, {"sheet": "Notes"})
