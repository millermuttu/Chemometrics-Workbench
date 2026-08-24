"""The stub server returns the fixtures unchanged, checks its token, and runs jobs.

The point of these is the three things a static fixture cannot be: a request
that is refused, a request that fails, and a job that takes time. The bodies
themselves are checked for being the fixture *unmodified* - this module is
allowed to route and to delay, and nothing else.

`STUB_JOB_STEP_SECONDS` is set before the server is imported, so a job that
takes seconds in the UI takes milliseconds here.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

os.environ["STUB_JOB_STEP_SECONDS"] = "0.05"
os.environ["STUB_TOKEN"] = "test-token"
os.environ["STUB_BUNDLE"] = tempfile.mkdtemp()
Path(os.environ["STUB_BUNDLE"], "index.html").write_text("<!doctype html><title>bundle</title>")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stub"))

import server  # noqa: E402

STEP = 0.05
AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def fixture(name: str) -> Any:
    return json.loads((server.FIXTURES / f"{name}.json").read_text())


def test_a_request_with_no_token_or_a_wrong_one_is_refused(client: TestClient) -> None:
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/projects", headers={"Authorization": "Bearer wrong"}).status_code == 401
    body = client.get("/api/projects").json()
    assert body["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/projects/p", "project"),
        ("/api/projects/p/datasets", "datasets"),
        ("/api/pipelines/x", "pipeline"),
        ("/api/pipelines/x/state", "pipeline_state"),
        ("/api/experiments/e", "experiment"),
    ],
)
def test_each_endpoint_returns_its_fixture_body_unmodified(
    client: TestClient, path: str, expected: str
) -> None:
    assert client.get(path, headers=AUTH).json() == fixture(expected)


def test_the_node_keyed_endpoints_serve_the_node_and_404_the_rest(client: TestClient) -> None:
    assert client.get("/api/spectra/snv", headers=AUTH).json() == fixture("spectra")["snv"]
    assert client.get("/api/results/pca_a", headers=AUTH).json() == fixture("pca")["pca_a"]
    missing = client.get("/api/spectra/nope", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_a_failure_can_be_provoked_without_editing_code(client: TestClient) -> None:
    response = client.get("/api/projects", headers=AUTH | {"X-Stub-Fail": "1"})
    assert response.status_code == 422
    assert response.json() == fixture("error")


def test_a_job_advances_over_time_rather_than_arriving_whole(client: TestClient) -> None:
    job = client.post("/api/experiments/e/run", headers=AUTH).json()
    assert (job["status"], job["progress"]) == ("queued", 0.0)

    seen = [job["progress"]]
    deadline = time.monotonic() + 10 * STEP + 2
    while time.monotonic() < deadline:
        time.sleep(STEP / 2)
        now = client.get(f"/api/jobs/{job['job_id']}", headers=AUTH).json()
        if now["progress"] != seen[-1]:
            seen.append(now["progress"])
        if now["status"] == "succeeded":
            break
    assert seen == sorted(seen), seen
    assert len(seen) > 2, seen
    assert now["status"] == "succeeded"
    assert now["progress"] == 1.0


def test_a_run_can_be_made_to_fail_and_the_failure_is_terminal(client: TestClient) -> None:
    job = client.post("/api/experiments/e/run?fail=true", headers=AUTH).json()
    time.sleep(7 * STEP)
    final = client.get(f"/api/jobs/{job['job_id']}", headers=AUTH).json()
    assert final["status"] == "failed"
    assert final["message"] == fixture("jobs")["failed"][-1]["message"]


def test_cancelling_a_running_job_stops_it(client: TestClient) -> None:
    job = client.post("/api/experiments/e/run", headers=AUTH).json()
    time.sleep(2 * STEP)
    cancelled = client.post(f"/api/jobs/{job['job_id']}/cancel", headers=AUTH).json()
    assert cancelled["status"] == "cancelled"

    frozen = cancelled["progress"]
    time.sleep(6 * STEP)
    later = client.get(f"/api/jobs/{job['job_id']}", headers=AUTH).json()
    assert later["status"] == "cancelled"
    assert later["progress"] == frozen


def test_an_unknown_job_is_a_404_with_a_body(client: TestClient) -> None:
    response = client.get("/api/jobs/nope", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["message"]


def test_production_mode_serves_the_built_bundle(client: TestClient) -> None:
    """The mount, not StaticFiles - that the bundle is reachable without a token."""
    response = client.get("/")
    assert response.status_code == 200
    assert "bundle" in response.text


def test_development_mode_lets_the_vite_dev_server_call_the_api(client: TestClient) -> None:
    response = client.options(
        "/api/projects",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_a_deep_link_arrives_as_the_application_not_a_404(client: TestClient) -> None:
    """The frontend routes on the path, so /tokens must be served the bundle."""
    response = client.get("/tokens")
    assert response.status_code == 200
    assert "bundle" in response.text


def test_a_path_from_the_client_cannot_escape_the_bundle(client: TestClient) -> None:
    response = client.get("/../../etc/passwd")
    assert response.status_code == 200
    assert "root:" not in response.text


def test_a_project_can_be_asked_to_look_empty(client: TestClient) -> None:
    """The empty-project state is unreachable from a fixture that has a dataset."""
    assert client.get("/api/projects/p/datasets?empty=true", headers=AUTH).json() == []
    assert client.get("/api/projects/p/datasets", headers=AUTH).json() == fixture("datasets")
