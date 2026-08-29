"""Tests for the reference dataset loaders.

Tecator is committed, so its tests always run and its checksum is a real CI
gate: a silent edit to `tecator.txt` fails the build.

Corn and gasoline are downloaded rather than committed. Their tests run only
when the file is already in the cache, or when `CHEMOMETRICS_DOWNLOAD_DATASETS`
is set, so the default test run does not depend on the network. What CI does
gate unconditionally is that the pinned checksums have not been edited — a
loosened pin is exactly as dangerous as an edited dataset, and considerably
easier to do by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from chemometrics_workbench.datasets import (
    CORN_ARCHIVE,
    CORN_MEMBER_SHA256,
    GASOLINE_ARCHIVE,
    GASOLINE_MEMBER_SHA256,
    TECATOR_FILE,
    TECATOR_GROUPS,
    TECATOR_SHA256,
    ReferenceDataset,
    _Remote,
    cache_dir,
    fetch,
    is_cached,
    load_corn,
    load_gasoline,
    load_tecator,
)

DATA_DIR = TECATOR_FILE.parent.parent


def _downloadable(remote: _Remote) -> bool:
    return is_cached(remote) or bool(os.environ.get("CHEMOMETRICS_DOWNLOAD_DATASETS"))


def _needs(remote: _Remote) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not _downloadable(remote),
        reason=(
            f"{remote.filename} is not in {cache_dir()}; "
            "set CHEMOMETRICS_DOWNLOAD_DATASETS=1 to fetch it"
        ),
    )


def _check_shape(dataset: ReferenceDataset, n_samples: int, n_variables: int) -> None:
    assert dataset.spectra.shape == (n_samples, n_variables)
    assert dataset.n_samples == n_samples
    assert dataset.n_variables == n_variables
    assert len(dataset.axis.values) == n_variables
    for name, values in dataset.targets.items():
        assert values.shape == (n_samples,), name
    assert np.isfinite(dataset.spectra).all()
    assert dataset.source.file_hash.startswith("sha256:")


# --------------------------------------------------------------------------
# every dataset directory documents itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["corn", "gasoline", "tecator"])
def test_each_dataset_directory_records_source_terms_and_checksum(name: str) -> None:
    directory = DATA_DIR / name
    readme = (directory / "README.md").read_text(encoding="utf-8")
    sums = (directory / "SHA256SUMS").read_text(encoding="utf-8")

    assert "http" in readme, "no source URL recorded"
    assert "SHA-256" in readme, "no checksum recorded"
    assert sums.strip(), "SHA256SUMS is empty"
    for line in sums.strip().splitlines():
        digest, _, filename = line.partition("  ")
        assert len(digest) == 64, line
        assert filename, line


def test_pinned_checksums_are_unchanged() -> None:
    """A loosened pin is as dangerous as an edited dataset. Freeze the pins."""
    assert TECATOR_SHA256 == "e435c0538cd706473d87525da595d75dbff43a58bf2694104bd948081c3790d7"
    assert CORN_ARCHIVE.sha256 == "8a2d1a03648b6ad334caaafa5d8377bf945ba1477a5082c4963d705d07cca795"
    assert CORN_MEMBER_SHA256 == "e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5"
    assert (
        GASOLINE_ARCHIVE.sha256
        == "8029018d4c8921fa4c7ec5081551afdcc55d53271d9920db828483b442a033cf"
    )
    assert (
        GASOLINE_MEMBER_SHA256 == "fdfd17ff9a407d9ee04d0c966aab3e4f2e0390c39724cab99334c74b842acbdd"
    )


# --------------------------------------------------------------------------
# tecator: committed, so always tested
# --------------------------------------------------------------------------


def test_tecator_shape_and_targets() -> None:
    tecator = load_tecator()
    _check_shape(tecator, 240, 100)
    assert set(tecator.targets) == {"moisture", "fat", "protein"}

    # Absorbance is -log10(transmittance); these ranges come from the file.
    assert tecator.spectra.min() == pytest.approx(2.063, abs=1e-3)
    assert tecator.spectra.max() == pytest.approx(5.4737, abs=1e-3)
    assert tecator.targets["fat"].min() == pytest.approx(0.9)
    assert tecator.targets["fat"].max() == pytest.approx(58.5)
    assert tecator.targets["moisture"][0] == pytest.approx(60.5)


def test_the_committed_data_is_checked_out_byte_for_byte() -> None:
    """No CRLF in a file whose digest is its identity.

    Git rewrites LF as CRLF on checkout on Windows unless told not to, which
    changes every line of `tecator.txt` and therefore its SHA-256 - so
    `load_tecator` raised "does not match its pinned checksum" there and
    nowhere else. `.gitattributes` marks these files `-text`; this asserts the
    outcome, on every platform, in a second, rather than waiting for a Windows
    runner to notice.

    The reader fixtures are included because their exact bytes are the input
    under test: a line ending is precisely the sort of thing those tests are
    about.
    """
    root = Path(__file__).resolve().parents[1]
    byte_sensitive = [
        root / "src" / "chemometrics_workbench" / "data" / "tecator" / "tecator.txt",
        *sorted((root / "tests" / "fixtures" / "readers").glob("*.csv")),
        *sorted((root / "tests" / "fixtures" / "readers").glob("*.txt")),
        *sorted((root / "tests" / "fixtures" / "readers").glob("*.jdx")),
    ]
    assert byte_sensitive, "the fixtures moved; this test is pointing at nothing"

    carriage_returns = [
        str(path.relative_to(root)) for path in byte_sensitive if b"\r" in path.read_bytes()
    ]
    assert carriage_returns == [], (
        "these files carry CRLF, which changes their digest and breaks the "
        "pinned checksum: " + ", ".join(carriage_returns)
    )


def test_tecator_content_hash_is_pinned() -> None:
    tecator = load_tecator()
    assert tecator.source.file_hash == f"sha256:{TECATOR_SHA256}"
    assert tecator.source.reader == "tecator_txt"


def test_tecator_axis_spans_the_documented_range() -> None:
    axis = load_tecator().axis
    assert axis.unit == "nm"
    assert axis.values[0] == pytest.approx(850.0)
    assert axis.values[-1] == pytest.approx(1050.0)


def test_tecator_sample_ids_carry_the_published_split() -> None:
    tecator = load_tecator()
    assert len(tecator.sample_ids) == 240
    assert tecator.sample_ids[0] == "C001"
    assert tecator.sample_ids[128] == "C129"
    assert tecator.sample_ids[129] == "M001"
    assert tecator.sample_ids[-1] == "E2017"
    assert sum(count for _, count in TECATOR_GROUPS) == 240


def test_tecator_rejects_an_edited_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip one byte and the loader must refuse it."""
    original = TECATOR_FILE.read_bytes()
    corrupted = tmp_path / "tecator.txt"
    # The last character of the file is a digit of the final protein value.
    corrupted.write_bytes(original[:-2] + b"9" + original[-1:])

    monkeypatch.setattr("chemometrics_workbench.datasets.TECATOR_FILE", corrupted)
    with pytest.raises(ValueError, match="does not match its pinned checksum"):
        load_tecator()


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def test_cache_dir_honours_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEMOMETRICS_DATA_HOME", "/somewhere/else")
    assert cache_dir() == Path("/somewhere/else")


def test_cache_dir_falls_back_to_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHEMOMETRICS_DATA_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert cache_dir() == Path("/xdg/chemometrics-workbench/datasets")


def test_fetch_rejects_a_corrupted_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached file is re-verified on every read, not trusted because it exists."""
    monkeypatch.setenv("CHEMOMETRICS_DATA_HOME", str(tmp_path))
    (tmp_path / CORN_ARCHIVE.filename).write_bytes(b"not the corn dataset")
    with pytest.raises(ValueError, match="does not match its pinned checksum"):
        fetch(CORN_ARCHIVE)


# --------------------------------------------------------------------------
# corn and gasoline: downloaded, so skipped unless available
# --------------------------------------------------------------------------


@_needs(CORN_ARCHIVE)
def test_corn_shape_and_targets() -> None:
    corn = load_corn()
    _check_shape(corn, 80, 700)
    assert set(corn.targets) == {"moisture", "oil", "protein", "starch"}
    assert corn.source.file_hash == f"sha256:{CORN_MEMBER_SHA256}"

    assert corn.axis.values[0] == pytest.approx(1100.0)
    assert corn.axis.values[-1] == pytest.approx(2498.0)
    assert corn.axis.values[1] - corn.axis.values[0] == pytest.approx(2.0)


@_needs(CORN_ARCHIVE)
@pytest.mark.parametrize("instrument", ["m5", "mp5", "mp6"])
def test_corn_instruments_share_the_reference_values(instrument: str) -> None:
    """The same 80 samples on three spectrometers — that is the whole point."""
    reference = load_corn("m5")
    other = load_corn(instrument)  # type: ignore[arg-type]
    _check_shape(other, 80, 700)
    for name, values in reference.targets.items():
        np.testing.assert_array_equal(other.targets[name], values)


@_needs(GASOLINE_ARCHIVE)
def test_gasoline_shape_and_octane() -> None:
    gasoline = load_gasoline()
    _check_shape(gasoline, 60, 401)
    assert set(gasoline.targets) == {"octane"}
    assert gasoline.source.file_hash == f"sha256:{GASOLINE_MEMBER_SHA256}"

    octane = gasoline.targets["octane"]
    assert octane.min() == pytest.approx(83.4)
    assert octane.max() == pytest.approx(89.6)
    assert octane[0] == pytest.approx(85.3)


@_needs(GASOLINE_ARCHIVE)
def test_gasoline_axis_is_read_from_the_file() -> None:
    axis = load_gasoline().axis
    assert axis.unit == "nm"
    assert axis.values[0] == pytest.approx(900.0)
    assert axis.values[-1] == pytest.approx(1700.0)
    steps = np.diff(np.asarray(axis.values))
    np.testing.assert_allclose(steps, 2.0)
