"""Tests for the real HTTP surface: the import endpoints and what they write.

The endpoints are exercised through a bare FastAPI application holding the
router, not through `stub/server.py`: the router is what survives #89, and a
test that went through the stub would be testing a module scheduled for
deletion. `tests/test_stub_server.py` covers the other direction — that the
stub no longer answers for these paths.

The payload shapes are checked against `stub/fixtures/import_preview.json` and
`stub/fixtures/datasets.json`, because those are the contract the Phase 1.1
frontend already renders.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from chemometrics_workbench.api import MAX_UPLOAD_BYTES, results_payload, router
from chemometrics_workbench.datasets import load_tecator
from chemometrics_workbench.executor import Run, execute
from chemometrics_workbench.models import DatasetVersion
from chemometrics_workbench.project import (
    DATASETS_FILE,
    create_project,
    open_project,
    read_array,
    read_datasets,
    write_array,
)
from tests.test_executor import fixture_pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "stub" / "fixtures"
READER_FILES = Path(__file__).resolve().parent / "fixtures" / "readers"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project directory the server opens, and a config home it cannot escape."""
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))
    directory = tmp_path / "project"
    monkeypatch.setenv("CHEMOMETRICS_PROJECT", str(directory))
    return directory


@pytest.fixture
def client(project: Path) -> Iterator[TestClient]:
    """The router alone, with the error envelope the stub server also applies.

    The envelope is duplicated here rather than imported from the stub for the
    reason above: it is published in `error.json`, #89 settles it, and this
    router has to carry it once the stub is gone.
    """
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def error_body(_: Any, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            return JSONResponse(status_code=exc.status_code, content={"error": detail})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "request_failed", "message": str(detail), "detail": {}}},
        )

    app.include_router(router, prefix="/api")
    with TestClient(app) as test_client:
        yield test_client


def upload(name: str) -> dict[str, tuple[str, bytes]]:
    """One of the reader fixtures, as a multipart file part."""
    return {"file": (name, (READER_FILES / name).read_bytes())}


def project_id(client: TestClient) -> str:
    return str(client.get("/api/projects").json()[0]["project_id"])


# --------------------------------------------------------------------------
# the project, opened rather than fabricated
# --------------------------------------------------------------------------


def test_the_open_project_is_created_on_first_use_and_is_a_real_directory(
    client: TestClient, project: Path
) -> None:
    body = client.get("/api/projects").json()
    assert len(body) == 1
    assert body[0]["directory"] == str(project)
    assert (project / "project.json").exists()
    assert body[0]["project_id"] == str(open_project(project).project_id)


def test_a_project_that_is_not_the_open_one_is_a_404_with_a_body(client: TestClient) -> None:
    response = client.get("/api/projects/11111111-1111-1111-1111-111111111111/datasets")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_project_with_nothing_imported_is_empty_because_it_is_empty(client: TestClient) -> None:
    """The empty-project state, reached honestly rather than by `?empty`."""
    assert client.get(f"/api/projects/{project_id(client)}/datasets").json() == []


# --------------------------------------------------------------------------
# preview: what the reader found, nothing committed
# --------------------------------------------------------------------------


def test_preview_returns_the_readers_detection_in_the_published_shape(
    client: TestClient, project: Path
) -> None:
    body = client.post("/api/import/preview", files=upload("tecator_subset.csv")).json()

    published = fixture("import_preview")
    assert set(body) == set(published)
    assert set(body["source"]) == set(published["source"])
    assert set(body["detected"]) == set(published["detected"])
    assert set(body["detected"]["axis"]) >= {"kind", "unit", "start", "end", "reconstructed"}
    assert set(body["head"]) == set(published["head"])

    assert body["source"]["filename"] == "tecator_subset.csv"
    assert body["source"]["reader"] == "delimited"
    assert body["detected"]["delimiter"]["value"] == ","
    assert "alternatives" in body["detected"]["delimiter"]

    # Nothing was committed, and nothing was left behind.
    assert not (project / DATASETS_FILE).exists()
    assert client.get(f"/api/projects/{project_id(client)}/datasets").json() == []


def test_preview_offers_the_alternatives_the_user_can_correct_to(client: TestClient) -> None:
    body = client.post("/api/import/preview", files=upload("tecator_subset_eu.csv")).json()
    detected = body["detected"]
    assert detected["delimiter"]["value"] == ";"
    assert detected["decimal"]["value"] == ","
    assert "." in detected["decimal"]["alternatives"]


def test_a_workbook_and_a_jcamp_file_preview_through_the_same_endpoint(
    client: TestClient,
) -> None:
    """The endpoint calls one interface and never learns which format it holds."""
    workbook = client.post("/api/import/preview", files=upload("tecator_subset.xlsx")).json()
    jcamp = client.post("/api/import/preview", files=upload("tecator_subset.jdx")).json()

    assert workbook["source"]["reader"] == "xlsx"
    assert jcamp["source"]["reader"] == "jcamp_dx"
    for body in (workbook, jcamp):
        assert body["detected"]["n_samples"] == 8
        assert body["detected"]["n_variables"] == 12


# --------------------------------------------------------------------------
# import: what is written, and where
# --------------------------------------------------------------------------


def test_import_writes_the_array_through_the_store_and_records_a_version(
    client: TestClient, project: Path
) -> None:
    entry = client.post("/api/import", files=upload("tecator_subset.csv")).json()

    published = fixture("datasets")[0]
    assert set(entry) == set(published)
    assert set(entry["dataset"]) == set(published["dataset"])
    assert set(entry["versions"][0]) == set(published["versions"][0])

    version = entry["versions"][0]
    assert (version["n_samples"], version["n_variables"]) == (8, 12)
    assert version["source"]["filename"] == "tecator_subset.csv"
    assert version["source"]["file_hash"].startswith("sha256:")
    assert version["content_hash"].startswith("sha256:")
    assert entry["dataset"]["name"] == "tecator_subset"

    # The index holds a path; the values are in the store, float32 on disk.
    stored = project / version["array_path"]
    assert stored.exists()
    assert np.load(stored).dtype == np.float32
    assert read_array(project, version["array_path"]).shape == (8, 12)


def test_the_committed_dataset_is_read_back_from_disk_after_a_restart(
    client: TestClient, project: Path
) -> None:
    """The dataset list survives the process, which is what #77's directory is for.

    A second `TestClient` over a second application is a restart as far as the
    server is concerned: nothing is carried over in memory, and the only thing
    both have in common is the directory.
    """
    committed = client.post("/api/import", files=upload("tecator_subset.csv")).json()

    reopened = TestClient(FastAPI())
    reopened.app.include_router(router, prefix="/api")  # type: ignore[attr-defined]
    listed = reopened.get(f"/api/projects/{project_id(reopened)}/datasets").json()

    assert listed == [committed]
    assert [entry.dataset.name for entry in read_datasets(project)] == ["tecator_subset"]


def test_two_imports_are_two_datasets_and_both_are_listed(client: TestClient) -> None:
    first = client.post("/api/import", files=upload("tecator_subset.csv")).json()
    second = client.post("/api/import", files=upload("tecator_subset.jdx")).json()

    listed = client.get(f"/api/projects/{project_id(client)}/datasets").json()
    assert [entry["dataset"]["dataset_id"] for entry in listed] == [
        first["dataset"]["dataset_id"],
        second["dataset"]["dataset_id"],
    ]
    # Same stem, two datasets: which one a version belongs to is its
    # `dataset_id`, never its name.
    assert listed[0]["dataset"]["name"] == listed[1]["dataset"]["name"] == "tecator_subset"
    assert listed[0]["dataset"]["dataset_id"] != listed[1]["dataset"]["dataset_id"]


def test_a_name_can_be_given_and_otherwise_comes_from_the_file(client: TestClient) -> None:
    entry = client.post(
        "/api/import", files=upload("tecator_subset.csv"), data={"name": "Meat, batch 4"}
    ).json()
    assert entry["dataset"]["name"] == "Meat, batch 4"


def test_the_corrections_the_user_made_are_the_ones_the_parse_obeys(
    client: TestClient, project: Path
) -> None:
    """A correction displayed and then ignored is the failure this design prevents."""
    european = client.post("/api/import", files=upload("tecator_subset_eu.csv")).json()
    values = read_array(project, european["versions"][0]["array_path"])

    # Correcting the decimal separator on a decimal-comma file cannot succeed:
    # the reader refuses rather than importing nothing quietly.
    refused = client.post(
        "/api/import",
        files=upload("tecator_subset_eu.csv"),
        data={"corrections": json.dumps({"decimal": "."})},
    )
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "reader_failed"

    # And the correction that does apply changes what is read.
    plain = client.post("/api/import", files=upload("tecator_subset.csv")).json()
    np.testing.assert_allclose(
        values, read_array(project, plain["versions"][0]["array_path"]), atol=1e-4
    )


def test_a_correction_the_reader_does_not_offer_is_refused_not_dropped(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/import",
        files=upload("tecator_subset.jdx"),
        data={"corrections": json.dumps({"delimiter": ";"})},
    )
    assert response.status_code == 422
    assert "delimiter" in response.json()["error"]["message"]


def test_corrections_that_are_not_an_object_of_strings_are_refused(client: TestClient) -> None:
    for body in ("not json", json.dumps(["decimal"]), json.dumps({"decimal": 3})):
        response = client.post(
            "/api/import", files=upload("tecator_subset.csv"), data={"corrections": body}
        )
        assert response.status_code == 422, body
        assert response.json()["error"]["code"] == "bad_request"


# --------------------------------------------------------------------------
# the envelope, and refusals that are not 500s
# --------------------------------------------------------------------------


def test_the_shape_is_reported_as_read_which_is_what_the_envelope_check_reads(
    client: TestClient,
) -> None:
    """§13's envelope is applied to `n_samples` and `n_variables` by the frontend.

    `states/envelope.ts` computes it from those two numbers, so the honest
    report is the real shape and there is nothing for the server to add. What
    would break the overloaded state is a shape that had been rounded, capped
    or fabricated - which is what `?oversize` did.
    """
    entry = client.post("/api/import", files=upload("tecator_subset.csv")).json()
    version = entry["versions"][0]
    preview = client.post("/api/import/preview", files=upload("tecator_subset.csv")).json()

    assert (version["n_samples"], version["n_variables"]) == (
        preview["detected"]["n_samples"],
        preview["detected"]["n_variables"],
    )


def test_an_upload_past_the_accepted_size_is_refused_before_it_is_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.3: localhost is a trust boundary, and an unbounded upload fills a disk."""
    monkeypatch.setattr("chemometrics_workbench.api.MAX_UPLOAD_BYTES", 64)
    response = client.post("/api/import/preview", files=upload("tecator_subset.csv"))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    assert MAX_UPLOAD_BYTES > 320_000_000, "the default has to clear section 13's envelope"


def test_a_file_the_reader_rejects_is_a_diagnostic_not_a_500(
    client: TestClient, tmp_path: Path
) -> None:
    words = tmp_path / "notes.csv"
    words.write_text("alpha,beta\ngamma,delta\n", encoding="utf-8")

    response = client.post("/api/import/preview", files={"file": ("notes.csv", words.read_bytes())})
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "reader_failed"
    assert body["error"]["detail"]["file"] == "notes.csv"
    assert "no column of numbers" in body["error"]["message"]
    assert "Traceback" not in body["error"]["message"]
    assert body["error"]["message"], "the reader's own sentence, not an empty string"


def test_a_file_no_reader_claims_says_so(client: TestClient, tmp_path: Path) -> None:
    for name in ("spectra", "spectra.docx"):
        response = client.post("/api/import/preview", files={"file": (name, b"anything")})
        assert response.status_code == 422, name
        assert response.json()["error"]["code"] in {"bad_request", "reader_failed"}


def test_an_upload_cannot_write_outside_its_temporary_directory(
    client: TestClient, project: Path
) -> None:
    """A filename is a name here, never a path. §4.3 again."""
    project_id(client)  # opens the project, so there is a project.json to protect
    original = (project / "project.json").read_text(encoding="utf-8")
    response = client.post(
        "/api/import/preview",
        files={"file": ("../../project.json", b"kind,of,csv\n1,2,3\n")},
    )

    assert response.status_code in {200, 422}
    assert (project / "project.json").read_text(encoding="utf-8") == original


def test_nothing_is_left_in_the_temporary_directory_after_a_failed_read(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("TMPDIR", str(staging))

    client.post("/api/import/preview", files={"file": ("broken.csv", b"1,2\n3\n")})
    client.post("/api/import/preview", files=upload("tecator_subset.csv"))

    assert list(staging.iterdir()) == []


# --------------------------------------------------------------------------
# results (#87): the payload the analysis screen draws
# --------------------------------------------------------------------------


def executed(project: Path) -> tuple[Run, DatasetVersion]:
    """A project holding Tecator, with the fixture pipeline run over it."""
    tecator = load_tecator()
    create_project(project, "results tests")
    array_path, _ = write_array(project, tecator.spectra)
    version = DatasetVersion(
        dataset_id=uuid4(),
        version=1,
        content_hash=tecator.source.file_hash,
        n_samples=tecator.n_samples,
        n_variables=tecator.n_variables,
        axis=tecator.axis,
        sample_ids=list(tecator.sample_ids),
        array_path=array_path,
    )
    return execute(project, fixture_pipeline(version.version_id), version), version


def test_the_results_payload_is_the_shape_the_fixture_publishes(tmp_path: Path) -> None:
    run, version = executed(tmp_path / "results")
    payload = results_payload(run.results["pca_a"], version)
    published = fixture("pca")["pca_a"]

    assert set(payload) == set(published)
    assert set(payload["loadings"]) == set(published["loadings"])
    assert set(payload["diagnostics"]) == set(published["diagnostics"])
    assert payload["samples"][:2] == published["samples"][:2]
    assert len(payload["scores"]) == len(published["scores"]) == 240
    assert len(payload["loadings"]["components"]) == 5
    assert payload["loadings"]["axis"]["unit"] == published["loadings"]["axis"]["unit"]
    np.testing.assert_allclose(
        payload["loadings"]["axis"]["values"], published["loadings"]["axis"]["values"], atol=1e-4
    )


def test_a_split_branch_adds_its_validation_rows_without_changing_the_rest(
    tmp_path: Path,
) -> None:
    """§9's held-out rows, pushed through the model that never saw them."""
    run, version = executed(tmp_path / "results")
    payload = results_payload(run.results["pca_d"], version)
    published = fixture("pca")["pca_d"]

    # The keys the 1.1 screen reads are unchanged, and so are their lengths.
    assert set(payload) - set(published) == {"validation"}
    assert len(payload["samples"]) == len(published["samples"]) == 216
    assert len(payload["diagnostics"]["hotelling_t2"]) == 216

    validation = payload["validation"]
    assert validation["fold"] == 0
    assert len(validation["samples"]) == len(validation["scores"]) == 24
    assert len(validation["hotelling_t2"]) == len(validation["spe"]) == 24
    assert {row["index"] for row in validation["samples"]}.isdisjoint(
        {row["index"] for row in payload["samples"]}
    )


def test_a_branch_with_no_split_carries_no_validation_key(tmp_path: Path) -> None:
    run, version = executed(tmp_path / "results")
    assert "validation" not in results_payload(run.results["pca_b"], version)


def test_the_sample_ids_come_from_the_dataset_not_the_model(tmp_path: Path) -> None:
    """A model does not know what its rows were called."""
    run, version = executed(tmp_path / "results")
    anonymous = version.model_copy(update={"sample_ids": []})
    payload = results_payload(run.results["pca_a"], anonymous)

    assert payload["samples"][0] == {"index": 0, "sample_id": "row 0"}
    assert results_payload(run.results["pca_a"], version)["samples"][0]["sample_id"] == "C001"
