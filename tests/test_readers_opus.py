"""The Bruker OPUS reader (#187), on the sample files `opusreader2` ships.

Every number asserted here was read off the files with a twelve-line parser
written from the format description in `readers/opus.py`, and two of them are
the file's own: a data status block records the minimum and maximum of the
spectrum it describes (`MNY`, `MXY`), so a reader that decoded the wrong
block, the wrong scale or the wrong byte order lands on different numbers.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.models import AxisKind
from chemometrics_workbench.readers import ReaderError, opus, preview, read, reader_for

OPUS = Path(__file__).resolve().parent / "fixtures" / "readers" / "opus"
SINGLE = OPUS / "test_spectra.0"
SOIL_A = OPUS / "617262_1TP_C-1_A5.0"
SOIL_B = OPUS / "629266_1TP_A-1_C1.0"
OTHER = OPUS / "BF_lo_01_soil_cal.1"
PAIR = OPUS / "soil_pair.zip"


def test_a_numeric_suffix_is_an_opus_file_and_a_zip_is_a_folder_of_them() -> None:
    assert reader_for("sample.0") is opus
    assert reader_for("sample.001") is opus
    assert reader_for("folder.zip") is opus
    with pytest.raises(ReaderError, match="numeric suffix"):
        reader_for("sample.xyz")


def test_one_file_is_one_spectrum_on_the_axis_its_status_block_states() -> None:
    """`test_spectra.0` holds Refl, ScSm and ScRf and no AB, so Refl leads."""
    detection = opus.sniff(SINGLE)
    assert detection.block is not None
    assert detection.block.value == "Refl"
    assert detection.block.alternatives == ("ScSm", "ScRf")
    assert detection.correctable == ("block",)
    assert (detection.n_samples, detection.n_variables) == (1, 4819)
    assert detection.axis.kind is AxisKind.WAVENUMBER_CM1
    assert detection.axis.values[0] == pytest.approx(7498.2916914224625)
    assert detection.axis.values[-1] == pytest.approx(599.920606970787)
    assert detection.axis_note is None

    imported = opus.read(SINGLE, detection)
    assert imported.values.shape == (1, 4819)
    # The status block's own min and max, which is the file checking the read.
    assert imported.values.min() == pytest.approx(0.020948348566889763)
    assert imported.values.max() == pytest.approx(0.5861161947250366)
    assert imported.sample_ids == ("test_spectra.0",)
    assert imported.source.reader == "bruker_opus"


def test_choosing_another_block_reads_that_block() -> None:
    imported = read(SINGLE, {"block": "ScSm"})
    assert imported.values.max() == pytest.approx(0.17240329086780548)
    with pytest.raises(ReaderError, match="not one of the block options"):
        read(SINGLE, {"block": "AB"})


def test_a_zip_is_one_dataset_with_the_first_files_axis_and_the_drift_stated() -> None:
    """Two soil spectra from one instrument, 0.27 cm-1 apart: a real folder."""
    detection = opus.sniff(PAIR)
    assert detection.block is not None and detection.block.value == "AB"
    assert (detection.n_samples, detection.n_variables) == (2, 3578)
    assert detection.axis.values[0] == pytest.approx(7497.697861283203)
    assert detection.axis_note is not None and "0.27" in detection.axis_note
    assert detection.metadata_columns == ("sample_name",)

    imported = opus.read(PAIR, detection)
    assert imported.values.shape == (2, 3578)
    assert imported.sample_ids == ("617262_1TP_C-1_A5.0", "629266_1TP_A-1_C1.0")
    assert imported.metadata_columns["sample_name"][0] == "6172621TP C-1;;;soil;soil"
    # Each row is its own file's AB block, checked against that file's MXY.
    assert imported.values[1].max() == pytest.approx(
        opus.read(SOIL_B, opus.sniff(SOIL_B)).values.max()
    )
    # The post-processed absorbance is read where OPUS stored both, as the
    # viewer shows it.
    assert imported.values[1].max() == pytest.approx(
        _status(SOIL_B, kind=31, channel=16, additional=0)["MXY"]
    )


def test_the_preview_carries_the_block_choice_and_the_head() -> None:
    payload = preview(PAIR)
    assert payload["source"]["reader"] == "bruker_opus"
    assert payload["detected"]["block"] == {"value": "AB", "alternatives": ["ScSm", "ScRf"]}
    assert payload["detected"]["axis"]["kind"] == "wavenumber_cm-1"
    assert len(payload["head"]["rows"]) == 2 and len(payload["head"]["rows"][0]) == 6


def test_files_on_different_axes_are_refused_by_name(tmp_path: Path) -> None:
    archive = tmp_path / "mixed.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(SOIL_A, SOIL_A.name)
        zipped.write(OTHER, OTHER.name)
    with pytest.raises(ReaderError, match=r"has 1716 points where 617262_1TP_C-1_A5.0 has 3578"):
        opus.sniff(archive)


def test_a_member_that_is_not_opus_is_refused_by_name(tmp_path: Path) -> None:
    archive = tmp_path / "notes.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(SOIL_A, SOIL_A.name)
        zipped.writestr("readme.txt", "not a spectrum")
    with pytest.raises(
        ReaderError, match=r"readme\.txt in notes\.zip does not start with the OPUS signature"
    ):
        opus.sniff(archive)


def test_a_truncated_file_is_a_sentence_not_a_stack_trace(tmp_path: Path) -> None:
    cut = tmp_path / "cut.0"
    cut.write_bytes(SINGLE.read_bytes()[:20000])
    with pytest.raises(ReaderError, match="truncated"):
        opus.sniff(cut)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(ReaderError, match="empty archive"):
        opus.sniff(empty)


def test_a_block_missing_from_one_file_is_not_offered(tmp_path: Path) -> None:
    """`test_spectra.0` has no AB; zipped with a soil file that does, AB is
    not on offer and the common blocks are."""
    archive = tmp_path / "mixed_blocks.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(SINGLE, SINGLE.name)
        zipped.write(SOIL_A, SOIL_A.name)
    members = opus._members(archive)
    assert opus._offered(members) == ["ScSm", "ScRf"]
    # Different axes too, so a sniff refuses on the points before anything else.
    with pytest.raises(ReaderError, match="has 4819 points"):
        opus.sniff(archive)


def _status(path: Path, *, kind: int, channel: int, additional: int) -> dict[str, object]:
    [member] = opus._members(path)
    for entry in member.entries:
        if (entry.kind, entry.channel, entry.additional, entry.text) == (
            kind,
            channel,
            additional,
            0,
        ):
            return opus._params(member.data, entry)
    raise AssertionError("no such block")


def test_the_two_soil_files_read_alone_agree_with_the_zip() -> None:
    zipped = opus.read(PAIR, opus.sniff(PAIR))
    for row, path in enumerate((SOIL_A, SOIL_B)):
        alone = opus.read(path, opus.sniff(path))
        np.testing.assert_array_equal(zipped.values[row], alone.values[0])
