"""Tests for the assembled application: the whole surface, behind its token.

`tests/test_api.py` covers the payload builders on their own. What is checked
here is the server a browser talks to — that every URL the frontend has used
since its first commit answers, that the token is a real check, and that every
failure has the documented body.

The URLs are the assertion. Not one of them changed when the handlers behind
them became real, which is what building the frontend against them in Phase 1.1
was for.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

READER_FILES = Path(__file__).resolve().parent / "fixtures" / "readers"
AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A server over an empty project directory, reloaded per test.

    The module is re-imported because the token and the bundle path are read at
    import time, as they are in a launched application — reading them lazily to
    suit a test would be testing something the user never runs.
    """
    monkeypatch.setenv("CHEMOMETRICS_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("CHEMOMETRICS_PROJECT", str(tmp_path / "project"))
    monkeypatch.setenv("WORKBENCH_TOKEN", "test-token")
    monkeypatch.setenv("WORKBENCH_BUNDLE", str(tmp_path / "no-bundle"))

    import chemometrics_workbench.api as api_module
    import chemometrics_workbench.server as server_module

    importlib.reload(api_module)
    server = importlib.reload(server_module)
    with TestClient(server.app) as test_client:
        yield test_client
    api_module.JOBS.shutdown(wait=False)


def upload(name: str) -> dict[str, tuple[str, bytes]]:
    return {"file": (name, (READER_FILES / name).read_bytes())}


def imported(client: TestClient, name: str = "tecator_subset.csv") -> Any:
    response = client.post("/api/import", files=upload(name), headers=AUTH)
    assert response.status_code == 200, response.text
    return response.json()


def wait_for(client: TestClient, job_id: str, seconds: float = 30.0) -> Any:
    deadline = time.monotonic() + seconds
    body: Any = None
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}", headers=AUTH).json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.005)
    raise AssertionError(f"job did not finish; last saw {body}")


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------


def test_every_api_request_needs_the_token(client: TestClient) -> None:
    """§4.3: localhost is a trust boundary, not a private room."""
    for path in ("/api/projects", "/api/schema/steps", "/api/pipelines/current"):
        response = client.get(path)
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "unauthorized"

    assert client.get("/api/projects", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_a_failure_from_the_router_itself_carries_the_same_body(client: TestClient) -> None:
    """A client never has to tell two error shapes apart."""
    response = client.get("/api/jobs/nope", headers=AUTH)
    assert response.status_code == 404
    assert set(response.json()["error"]) == {"code", "message", "detail"}
    assert response.json()["error"]["code"] == "not_found"


def test_the_dev_server_may_call_the_api_across_origins(client: TestClient) -> None:
    response = client.options(
        "/api/projects",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


# --------------------------------------------------------------------------
# every URL the frontend uses
# --------------------------------------------------------------------------


def test_an_empty_project_answers_every_url_it_can_and_says_so_for_the_rest(
    client: TestClient,
) -> None:
    """The empty-project state, reached because the project is empty."""
    project = client.get("/api/projects", headers=AUTH).json()[0]
    assert client.get(f"/api/projects/{project['project_id']}/datasets", headers=AUTH).json() == []

    for path in ("/api/pipelines/current", "/api/experiments/current"):
        response = client.get(path, headers=AUTH)
        assert response.status_code == 404, path
        assert response.json()["error"]["message"], "a sentence, not an empty body"


def test_the_step_schema_comes_from_the_models_rather_than_a_file(client: TestClient) -> None:
    schema = client.get("/api/schema/steps", headers=AUTH).json()
    kinds = {entry["properties"]["kind"]["const"] for entry in schema["$defs"].values()}

    assert {"snv", "msc", "savgol", "mean_centre", "autoscale", "normalise"} <= kinds
    assert schema["$defs"]["SavitzkyGolay"]["properties"]["deriv"]["maximum"] == 2
    assert schema["$defs"]["MSC"]["properties"]["reference"]["enum"] == ["mean", "median"]


def test_a_step_is_validated_against_the_model_that_will_enforce_it(client: TestClient) -> None:
    good = {"kind": "savgol", "window_length": 11, "polyorder": 2, "deriv": 1}
    assert client.post("/api/steps/validate", json=good, headers=AUTH).json() == {
        "valid": True,
        "errors": [],
    }

    even = client.post(
        "/api/steps/validate", json={**good, "window_length": 10}, headers=AUTH
    ).json()
    assert even["valid"] is False
    assert even["errors"][0]["message"] == "window_length must be odd"


# --------------------------------------------------------------------------
# import, and the pipeline it starts
# --------------------------------------------------------------------------


def test_importing_a_file_starts_a_pipeline_on_it(client: TestClient) -> None:
    entry = imported(client)
    version_id = entry["versions"][0]["version_id"]

    pipeline = client.get("/api/pipelines/current", headers=AUTH).json()
    assert [node["id"] for node in pipeline["nodes"]] == ["source"]
    assert pipeline["nodes"][0]["version_id"] == version_id
    assert "tecator_subset.csv" in pipeline["name"]


def test_the_pipeline_is_read_back_after_a_restart(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported(client)
    first = client.get("/api/pipelines/current", headers=AUTH).json()

    import chemometrics_workbench.server as server_module

    restarted = importlib.reload(server_module)
    with TestClient(restarted.app) as second_client:
        again = second_client.get(
            "/api/pipelines/current",
            headers={"Authorization": f"Bearer {restarted.TOKEN}"},
        ).json()

    assert again == first


def test_a_second_import_does_not_replace_the_pipeline(client: TestClient) -> None:
    """A recipe someone is building is not something an import may throw away."""
    imported(client)
    first = client.get("/api/pipelines/current", headers=AUTH).json()
    imported(client, "tecator_subset.jdx")

    assert client.get("/api/pipelines/current", headers=AUTH).json() == first


def test_the_pipeline_answers_to_its_own_id_as_well_as_to_current(client: TestClient) -> None:
    imported(client)
    pipeline = client.get("/api/pipelines/current", headers=AUTH).json()

    assert client.get(f"/api/pipelines/{pipeline['pipeline_id']}", headers=AUTH).json() == pipeline
    assert client.get("/api/pipelines/other", headers=AUTH).status_code == 404


# --------------------------------------------------------------------------
# state, runs and results
# --------------------------------------------------------------------------


def test_node_state_is_derived_from_what_is_on_disk(client: TestClient) -> None:
    """`complete` because the output is there, not because a flag says so."""
    imported(client)
    before = client.get("/api/pipelines/current/state", headers=AUTH).json()
    assert before["nodes"]["source"]["state"] == "not_run"
    assert before["layout"]["source"] == {"x": 40.0, "y": 40.0}

    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    assert (job["status"], job["progress"]) == ("queued", 0.0)
    assert wait_for(client, job["job_id"])["status"] == "succeeded"

    after = client.get("/api/pipelines/current/state", headers=AUTH).json()
    assert after["nodes"]["source"]["state"] == "complete"


def test_a_run_is_submitted_and_answers_before_it_has_finished(client: TestClient) -> None:
    imported(client)
    job = client.post("/api/experiments/current/run", headers=AUTH).json()

    assert set(job) == {"job_id", "experiment_id", "status", "progress", "message", "node_id"}
    assert job["status"] == "queued"
    wait_for(client, job["job_id"])


def test_a_cancelled_run_reports_cancelled(client: TestClient) -> None:
    imported(client)
    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    cancelled = client.post(f"/api/jobs/{job['job_id']}/cancel", headers=AUTH).json()

    # The status flips when the user asks, not when the arithmetic catches up.
    assert cancelled["status"] in ("cancelled", "succeeded")
    assert wait_for(client, job["job_id"])["status"] == cancelled["status"]


def test_spectra_are_served_from_the_run_and_404_before_it(client: TestClient) -> None:
    imported(client)
    missing = client.get("/api/spectra/source", headers=AUTH)
    assert missing.status_code == 404
    assert "no result yet" in missing.json()["error"]["message"]

    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    wait_for(client, job["job_id"])

    payload = client.get("/api/spectra/source", headers=AUTH).json()
    assert payload["node_id"] == "source"
    assert payload["n_spectra"] == 8
    assert len(payload["traces"]) == 8
    assert payload["decimation"]["variables_kept"] == 12


def test_a_highlighted_spectrum_comes_back_at_full_resolution(client: TestClient) -> None:
    imported(client)
    wait_for(client, client.post("/api/experiments/current/run", headers=AUTH).json()["job_id"])

    payload = client.get("/api/spectra/source?highlight=1,3", headers=AUTH).json()
    assert [trace["index"] for trace in payload["highlighted"]["traces"]] == [1, 3]
    assert len(payload["highlighted"]["axis"]["values"]) == 12


def test_an_unknown_node_is_a_404_with_a_body(client: TestClient) -> None:
    imported(client)
    for path in ("/api/spectra/nope", "/api/results/nope"):
        response = client.get(path, headers=AUTH)
        assert response.status_code == 404, path
        assert response.json()["error"]["detail"]["node_id"] == "nope"


def test_the_validator_runs_against_the_stored_pipeline(client: TestClient) -> None:
    imported(client)
    body = client.post("/api/pipelines/current/validate", headers=AUTH).json()

    assert set(body) == {"pipeline_id", "valid", "problems", "warnings"}
    assert body["valid"] is True, "a source node alone has nothing wrong with it"


# --------------------------------------------------------------------------
# what a run leaves behind
# --------------------------------------------------------------------------


def _failing_pipeline(client: TestClient, tmp_path: Path) -> None:
    """Append a branch that cannot be fitted: twelve components out of a few channels.

    `tecator_subset.csv` is 8 x 12 over 850-872 nm, so six nanometres is about
    four channels and twelve components cannot come out of four.
    """
    from chemometrics_workbench.models import (
        EstimatorNode,
        MeanCentre,
        PCASpec,
        PreprocessNode,
        RangeSelect,
    )
    from chemometrics_workbench.project import read_pipeline, write_pipeline

    directory = tmp_path / "project"
    pipeline = read_pipeline(directory)
    assert pipeline is not None
    write_pipeline(
        directory,
        pipeline.model_copy(
            update={
                "nodes": [
                    *pipeline.nodes,
                    PreprocessNode(
                        id="narrow", inputs=("source",), step=RangeSelect(start=850.0, end=856.0)
                    ),
                    PreprocessNode(id="centre_x", inputs=("narrow",), step=MeanCentre()),
                    EstimatorNode(id="pca_x", inputs=("centre_x",), spec=PCASpec(n_components=12)),
                ]
            }
        ),
    )


def test_a_run_writes_the_experiment_it_produced(client: TestClient) -> None:
    """`experiments/current` is in the published contract; it must stop 404ing.

    Nothing wrote `experiment.json` before #89, so the endpoint answered 404
    for the life of the project however many runs had happened.
    """
    assert client.get("/api/experiments/current", headers=AUTH).status_code == 404

    imported(client)
    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    assert wait_for(client, job["job_id"])["status"] == "succeeded"

    experiment = client.get("/api/experiments/current", headers=AUTH)
    assert experiment.status_code == 200
    body = experiment.json()
    assert body["status"] == "succeeded"
    assert body["environment"] is not None, "a succeeded experiment records its environment"
    assert body["finished_at"] is not None
    assert body["pipeline_snapshot"]["nodes"], "the pipeline is snapshot by value, not referenced"


def test_a_failed_run_is_recorded_and_the_node_that_raised_is_the_one_blamed(
    client: TestClient, tmp_path: Path
) -> None:
    """The canvas marks a node red; it must be the node that actually failed.

    The last node to *report progress* is the last one that finished, which is
    precisely not the one that raised. `ExecutorError` carries the id as a
    field so the id is read rather than parsed back out of English.
    """
    imported(client)
    _failing_pipeline(client, tmp_path)

    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    finished = wait_for(client, job["job_id"])
    assert finished["status"] == "failed"
    assert finished["node_id"] == "pca_x", finished
    assert "components were asked of a matrix of rank" in finished["message"]
    assert "Traceback" not in finished["message"], "§6: a diagnostic, never a stack trace"

    state = client.get("/api/pipelines/current/state", headers=AUTH).json()
    assert state["nodes"]["pca_x"]["state"] == "failed"
    assert "rank" in state["nodes"]["pca_x"]["message"]

    experiment = client.get("/api/experiments/current", headers=AUTH).json()
    assert experiment["status"] == "failed"
    assert experiment["error"], "a failed experiment is a result; it keeps its cause"


def test_a_node_that_has_never_run_reports_not_run(client: TestClient, tmp_path: Path) -> None:
    imported(client)
    _failing_pipeline(client, tmp_path)
    state = client.get("/api/pipelines/current/state", headers=AUTH).json()
    assert state["nodes"]["narrow"]["state"] == "not_run"
    assert state["nodes"]["pca_x"]["state"] == "not_run"


def test_concurrent_reads_on_one_project_never_fail(client: TestClient) -> None:
    """A page load fires six queries at once, and every one of them opens the project.

    `open_project` used to rewrite a registry shared by every project on the
    machine on each call, so two of those six racing on the write turned a
    plain `GET` into a 500 - intermittently, which is the worst way to find it.
    """
    import concurrent.futures

    paths = [
        "/api/projects",
        "/api/schema/steps",
        "/api/pipelines/current",
        "/api/pipelines/current/state",
        "/api/experiments/current",
    ] * 6

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(lambda path: client.get(path, headers=AUTH).status_code, paths))

    assert all(code in (200, 404) for code in codes), sorted(set(codes))


# --------------------------------------------------------------------------
# the affordances are gone
# --------------------------------------------------------------------------


def test_the_1_1_only_affordances_no_longer_do_anything(client: TestClient) -> None:
    """`?empty`, `?oversize`, `?fail` and `X-Stub-Fail` die here (#89).

    Each is asserted as *ignored* rather than as absent: a query parameter the
    server does not know is not an error, and what matters is that no state can
    be reached by asking for it rather than by earning it.
    """
    project = client.get("/api/projects", headers=AUTH).json()[0]
    imported(client)

    honest = client.get(f"/api/projects/{project['project_id']}/datasets", headers=AUTH).json()
    asked = client.get(
        f"/api/projects/{project['project_id']}/datasets?empty=true&oversize=true", headers=AUTH
    ).json()
    assert asked == honest
    assert len(honest) == 1, "the project has what was imported into it, and nothing else"

    failed = client.get("/api/projects", headers=AUTH | {"X-Stub-Fail": "1"})
    assert failed.status_code == 200, "the header means nothing now"

    job = client.post("/api/experiments/current/run?fail=true", headers=AUTH).json()
    assert wait_for(client, job["job_id"])["status"] == "succeeded", "a run fails only if it fails"


def test_the_source_tree_carries_no_stub_affordances() -> None:
    """The grep #89's done-when names, as a test rather than a habit."""
    root = Path(__file__).resolve().parents[1]
    hits: list[str] = []
    for path in list((root / "src").rglob("*.py")) + list(
        (root / "frontend" / "src").rglob("*.ts*")
    ):
        text = path.read_text(encoding="utf-8")
        for needle in ("X-Stub-Fail", "?empty", "?oversize", "failrun"):
            if needle in text:
                hits.append(f"{path.relative_to(root)}: {needle}")
    assert hits == []
