"""The reader interface and the CSV/TXT reader (#78).

The fixtures under `tests/fixtures/readers/` are real Tecator data written out
in the layouts `PROPOSAL.md` §6 names — their provenance is in the README beside
them. Because all four hold the same eight samples and twelve channels, the
strongest test here is that a European-decimal-comma semicolon file and a plain
comma file come back as the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import (
    Choice,
    ReaderError,
    apply_corrections,
    delimited,
    preview,
    read,
    reader_for,
)

FIXTURES = Path(__file__).parent / "fixtures" / "readers"
PLAIN = FIXTURES / "tecator_subset.csv"
EUROPEAN = FIXTURES / "tecator_subset_eu.csv"
TRANSPOSED = FIXTURES / "tecator_transposed.csv"
WHITESPACE = FIXTURES / "spectra_whitespace.txt"


@pytest.fixture(scope="module")
def expected() -> np.ndarray:
    """What the fixtures were cut from: eight samples, twelve channels."""
    return load_tecator().spectra[:8, :12]


# --- Detection ------------------------------------------------------------


def test_a_comma_file_is_detected(expected: np.ndarray) -> None:
    detection = delimited.sniff(PLAIN)

    assert detection.delimiter.value == ","
    assert detection.decimal.value == "."
    assert detection.orientation.value == "samples_in_rows"
    assert (detection.n_samples, detection.n_variables) == expected.shape


def test_a_european_file_is_detected_as_semicolons_and_decimal_commas() -> None:
    """§6's first named hazard, and the reason decimal is decided after delimiter."""
    detection = delimited.sniff(EUROPEAN)

    assert detection.delimiter.value == ";"
    assert detection.decimal.value == ","


def test_a_transposed_file_is_detected() -> None:
    detection = delimited.sniff(TRANSPOSED)

    assert detection.orientation.value == "samples_in_columns"
    assert detection.n_samples == 8
    assert detection.n_variables == 12


def test_a_headerless_whitespace_file_is_detected(expected: np.ndarray) -> None:
    detection = delimited.sniff(WHITESPACE)

    assert detection.delimiter.value == delimited.WHITESPACE
    assert (detection.n_samples, detection.n_variables) == expected.shape
    assert detection.axis.kind is AxisKind.INDEX
    assert "no header" in (detection.axis_note or "")


def test_every_guess_carries_its_alternatives() -> None:
    """A guess presented as a fact is the failure mode the Choice type exists for."""
    detection = delimited.sniff(EUROPEAN)

    for choice in (detection.delimiter, detection.decimal, detection.orientation):
        assert choice.alternatives
        assert choice.value not in choice.alternatives


def test_wavelength_headers_become_the_axis() -> None:
    detection = delimited.sniff(PLAIN)

    assert detection.axis.kind is AxisKind.WAVELENGTH_NM
    assert detection.axis.unit == "nm"
    assert detection.axis.values[0] == pytest.approx(850.0)
    assert not detection.axis_reconstructed


def test_a_descending_axis_reads_as_wavenumbers(tmp_path: Path) -> None:
    """The FT-IR convention. Ascending and above the nm ceiling reads the same way."""
    path = tmp_path / "ftir.csv"
    path.write_text("id,4000,3000,2000,1000\nA,0.1,0.2,0.3,0.4\nB,0.2,0.3,0.4,0.5\n")

    detection = delimited.sniff(path)
    assert detection.axis.kind is AxisKind.WAVENUMBER_CM1
    assert detection.axis.unit == "cm-1"


def test_an_unordered_header_is_not_an_axis(tmp_path: Path) -> None:
    path = tmp_path / "shuffled.csv"
    path.write_text("id,900,850,1000,875\nA,0.1,0.2,0.3,0.4\nB,0.2,0.3,0.4,0.5\n")

    detection = delimited.sniff(path)
    assert detection.axis.kind is AxisKind.INDEX
    assert "not ordered" in (detection.axis_note or "")


def test_targets_and_metadata_columns_are_separated() -> None:
    detection = delimited.sniff(EUROPEAN)

    assert list(detection.targets) == ["fat", "moisture"]
    assert list(detection.metadata_columns) == ["batch"]


def test_an_empty_column_is_discarded_with_its_reason(tmp_path: Path) -> None:
    path = tmp_path / "gappy.csv"
    path.write_text("id,850,860,notes\nA,0.1,0.2,\nB,0.2,0.3,\n")

    detection = delimited.sniff(path)
    assert [entry["what"] for entry in detection.discarded] == ["notes"]
    assert "empty" in detection.discarded[0]["why"]


# --- Reading --------------------------------------------------------------


def test_a_comma_file_reads_its_numbers(expected: np.ndarray) -> None:
    imported = read(PLAIN)

    assert imported.values.shape == expected.shape
    assert imported.values == pytest.approx(expected, abs=1e-4)
    assert imported.sample_ids[0] == "C001"


def test_a_european_file_reads_the_same_numbers(expected: np.ndarray) -> None:
    """The point of the whole detection apparatus: two layouts, one dataset."""
    plain = read(PLAIN)
    european = read(EUROPEAN)

    assert european.values == pytest.approx(plain.values)
    assert european.values == pytest.approx(expected, abs=1e-4)
    assert european.targets["fat"] == plain.targets["fat"]
    assert european.metadata_columns["batch"][0] == "morning"


def test_a_transposed_file_reads_the_same_numbers(expected: np.ndarray) -> None:
    imported = read(TRANSPOSED)

    assert imported.values.shape == expected.shape
    assert imported.values == pytest.approx(expected, abs=1e-4)
    assert imported.sample_ids[:2] == ("C001", "C002")
    assert imported.axis.kind is AxisKind.WAVELENGTH_NM


def test_a_headerless_file_reads_with_an_index_axis(expected: np.ndarray) -> None:
    imported = read(WHITESPACE)

    assert imported.values == pytest.approx(expected, abs=1e-4)
    assert imported.axis.kind is AxisKind.INDEX
    assert imported.sample_ids == ()


def test_targets_come_back_as_numbers_not_variables() -> None:
    imported = read(PLAIN)

    assert list(imported.targets) == ["fat"]
    assert imported.targets["fat"][0] == pytest.approx(22.5)
    assert imported.values.shape[1] == 12


# --- Corrections ----------------------------------------------------------


def test_a_correction_changes_what_is_parsed(tmp_path: Path) -> None:
    """The detection the user corrects is the object the parse obeys."""
    path = tmp_path / "ambiguous.csv"
    path.write_text("a;b;c\n1;2;3\n4;5;6\n")

    detected = delimited.sniff(path)
    assert detected.orientation.value == "samples_in_rows"

    corrected = apply_corrections(detected, {"orientation": "samples_in_columns"})
    assert corrected.orientation.value == "samples_in_columns"
    assert "samples_in_rows" in corrected.orientation.alternatives


def test_correcting_the_decimal_separator_changes_the_numbers(tmp_path: Path) -> None:
    path = tmp_path / "thousands.csv"
    path.write_text("id;850;860\nA;1,5;2,5\nB;3,5;4,5\n")

    as_detected = read(path)
    assert as_detected.values[0, 0] == pytest.approx(1.5)

    with pytest.raises(ReaderError, match="not a number"):
        read(path, {"decimal": "."})


def test_a_correction_the_reader_does_not_offer_is_refused(tmp_path: Path) -> None:
    detection = delimited.sniff(PLAIN)

    with pytest.raises(ReaderError, match="not one of the delimiter options"):
        apply_corrections(detection, {"delimiter": "~"})


def test_a_correction_to_something_uncorrectable_is_refused() -> None:
    """Silently dropping it is worse: the user watched themselves make it."""
    detection = delimited.sniff(PLAIN)

    with pytest.raises(ReaderError, match="cannot be corrected"):
        apply_corrections(detection, {"axis": "wavenumber_cm-1"})


def test_corrections_leave_the_original_detection_alone() -> None:
    detection = delimited.sniff(EUROPEAN)
    apply_corrections(detection, {"decimal": "."})

    assert detection.decimal == Choice(",", (".",))


# --- Diagnostics, not stack traces ----------------------------------------


def test_an_empty_file_is_named(tmp_path: Path) -> None:
    path = tmp_path / "nothing.csv"
    path.write_text("")

    with pytest.raises(ReaderError, match="is empty"):
        delimited.sniff(path)


def test_a_one_column_file_says_it_has_no_spectra(tmp_path: Path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("value\n1\n2\n3\n")

    with pytest.raises(ReaderError, match="no consistent delimiter"):
        delimited.sniff(path)


def test_a_ragged_line_is_named_by_number(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text("id,850,860\nA,0.1,0.2\nB,0.2\nC,0.3,0.4\n")

    with pytest.raises(ReaderError, match="line 3 has 2 fields"):
        read(path)


def test_a_text_cell_in_the_spectra_block_is_named(tmp_path: Path) -> None:
    path = tmp_path / "typo.csv"
    path.write_text("id,850,860\nA,0.1,0.2\nB,0.2,n/a\n")

    with pytest.raises(ReaderError, match="not a number"):
        read(path)


def test_a_file_of_only_text_says_there_are_no_spectra(tmp_path: Path) -> None:
    path = tmp_path / "words.csv"
    path.write_text("id,site,operator\nA,north,jo\nB,south,sam\n")

    with pytest.raises(ReaderError, match="no column of numbers"):
        delimited.sniff(path)


def test_an_unknown_suffix_names_what_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "spectra.opus"
    path.write_text("anything")

    with pytest.raises(ReaderError, match=r"no reader for \.opus"):
        reader_for(path)


def test_no_diagnostic_is_a_stack_trace(tmp_path: Path) -> None:
    """§6's rule, asserted rather than assumed."""
    path = tmp_path / "ragged.csv"
    path.write_text("id,850,860\nA,0.1,0.2\nB,0.2\n")

    with pytest.raises(ReaderError) as raised:
        read(path)
    assert "Traceback" not in str(raised.value)
    assert str(raised.value).endswith(".")


# --- Provenance and the preview shape -------------------------------------


def test_every_import_records_the_file_hash_and_the_reader_version() -> None:
    """§6: every import records the source file's content hash and reader version."""
    imported = read(PLAIN)

    assert imported.source.filename == PLAIN.name
    assert imported.source.file_hash.startswith("sha256:")
    assert imported.source.reader == delimited.NAME
    assert imported.source.reader_version == delimited.VERSION


def test_the_preview_has_the_shape_the_frontend_already_renders() -> None:
    """The Phase 1.1 contract. If a field is missing, that is a finding, not an edit."""
    payload = preview(EUROPEAN)

    assert set(payload) == {"source", "detected", "head"}
    assert set(payload["source"]) == {
        "filename",
        "file_hash",
        "reader",
        "reader_version",
        "size_bytes",
    }
    detected = payload["detected"]
    assert set(detected) >= {
        "delimiter",
        "decimal",
        "orientation",
        "n_samples",
        "n_variables",
        "axis",
        "metadata_columns",
        "targets",
        "discarded",
    }
    for name in ("delimiter", "decimal", "orientation"):
        assert set(detected[name]) == {"value", "alternatives"}
    assert set(detected["axis"]) >= {"kind", "unit", "start", "end", "reconstructed"}
    assert set(payload["head"]) == {"sample_ids", "rows"}


def test_the_preview_matches_the_fixture_the_frontend_was_built_against() -> None:
    """Field for field against the Phase 1.1 contract in `tests/fixtures/contract/`."""
    import json

    contract = Path(__file__).resolve().parent / "fixtures" / "contract" / "import_preview.json"
    fixture = json.loads(contract.read_text(encoding="utf-8"))
    payload = preview(PLAIN)

    assert set(payload) == set(fixture)
    assert set(payload["source"]) == set(fixture["source"])
    assert set(payload["detected"]) == set(fixture["detected"])
    assert set(payload["detected"]["axis"]) <= set(fixture["detected"]["axis"])
    assert set(payload["head"]) == set(fixture["head"])


def test_the_preview_shows_the_corrected_reading(tmp_path: Path) -> None:
    """Both readings parse this file, so the correction is visible rather than fatal."""
    path = tmp_path / "either.csv"
    path.write_text("id;850;860\nA;1.5;2.5\nB;3.5;4.5\n")

    payload = preview(path, {"decimal": ","})

    assert payload["detected"]["decimal"]["value"] == ","
    assert "." in payload["detected"]["decimal"]["alternatives"]
    assert payload["head"]["rows"][0] == [1.5, 2.5]


def test_a_correction_that_makes_the_file_unreadable_says_so() -> None:
    """The preview reports the damage rather than rendering a table of blanks."""
    with pytest.raises(ReaderError, match=r"not a number|decimal separator"):
        preview(EUROPEAN, {"decimal": "."})


def test_the_preview_head_names_its_samples() -> None:
    payload = preview(PLAIN)

    assert payload["head"]["sample_ids"][0] == "C001"
    assert payload["head"]["rows"][0][0] == pytest.approx(2.6178)
    assert len(payload["head"]["rows"]) <= 6


def test_a_preview_does_not_read_the_whole_file(tmp_path: Path) -> None:
    """Sniffing is cheap by construction: a file far past the head still previews."""
    path = tmp_path / "long.csv"
    header = "id," + ",".join(str(850 + i) for i in range(12))
    rows = [f"S{i:04d}," + ",".join(f"{i * 0.001 + j:.4f}" for j in range(12)) for i in range(5000)]
    path.write_text("\n".join([header, *rows]) + "\n")

    payload = preview(path)
    assert payload["detected"]["n_samples"] == 5000
    assert len(payload["head"]["rows"]) == 6


# --------------------------------------------------------------------------
# Identifiers, #135
# --------------------------------------------------------------------------


def test_a_numbered_id_column_names_the_samples_rather_than_measuring_them(
    tmp_path: Path,
) -> None:
    """The case that prompted #135: ids that happen to be integers.

    They used to fail `all_text` and be classified as a target, so a row number
    became a response a user could pick for PLS and every sample lost its name.
    """
    path = tmp_path / "numbered.csv"
    path.write_text(
        "sample_id,850.0,900.0,950.0,fat\n"
        "1,0.1,0.2,0.3,22.5\n"
        "2,0.2,0.3,0.4,18.1\n"
        "3,0.3,0.4,0.5,22.5\n",
        encoding="utf-8",
    )
    imported = read(path)

    assert list(imported.sample_ids) == ["1", "2", "3"]
    assert list(imported.targets) == ["fat"]
    assert "sample_id" not in imported.targets
    assert imported.values.shape == (3, 3)


def test_the_preview_and_the_import_name_the_same_samples(tmp_path: Path) -> None:
    """Two answers to one question was the sharper half of #135."""
    path = tmp_path / "numbered.csv"
    path.write_text(
        "sample_id,850.0,900.0\n1,0.1,0.2\n2,0.2,0.3\n",
        encoding="utf-8",
    )
    assert preview(path)["head"]["sample_ids"] == list(read(path).sample_ids)


def test_a_first_column_that_repeats_is_a_measurement_not_an_identifier(
    tmp_path: Path,
) -> None:
    """The guard on the rule. Uniqueness is the test, and a target repeats."""
    path = tmp_path / "batch.csv"
    path.write_text(
        "moisture,850.0,900.0\n60.5,0.1,0.2\n60.5,0.2,0.3\n72.1,0.3,0.4\n",
        encoding="utf-8",
    )
    imported = read(path)

    assert imported.sample_ids == ()
    assert list(imported.targets) == ["moisture"]
    assert imported.values.shape == (3, 2)


def test_text_identifiers_still_work_and_need_not_be_unique_to_be_text(
    tmp_path: Path,
) -> None:
    """A text first column was already an identifier and stays one."""
    path = tmp_path / "named.csv"
    path.write_text(
        "name,850.0,900.0\nC001,0.1,0.2\nC002,0.2,0.3\n",
        encoding="utf-8",
    )
    assert list(read(path).sample_ids) == ["C001", "C002"]


def test_a_file_with_no_identifiers_previews_a_visible_placeholder(tmp_path: Path) -> None:
    """`1` and `2` were indistinguishable from a column of numbered samples.

    An all-numeric first line is data rather than a header, so this file has
    three rows and no names for any of them.
    """
    path = tmp_path / "bare.csv"
    path.write_text(
        "850.0,900.0\n0.1,0.2\n0.2,0.3\n",
        encoding="utf-8",
    )
    assert preview(path)["head"]["sample_ids"] == ["row 1", "row 2", "row 3"]
    assert read(path).sample_ids == ()
