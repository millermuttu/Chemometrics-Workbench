"""The HTML report: one file, nothing external, the experiment's own numbers.

`docs`-level rules are in `report.py`'s docstring. What is checked here is the
three claims that make the file worth having:

- **one file** — no link, no script, no external `src`, so it opens with the
  application shut down and the network off;
- **the experiment's own numbers** — every figure in it is one the record or
  the stored result carries, so a report cannot drift from what the
  application says;
- **the plots are drawn from the result**, and a run whose arrays are gone says
  so rather than drawing something.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from chemometrics_workbench.report import render_report, report_filename
from tests.test_server import AUTH, imported, wait_for

# isort: split
from tests.test_server import client  # noqa: F401 - the fixture

#: Anything that would make the file reach outside itself. A `src` on an
#: inline `svg` element is fine; there are none, which is the point.
EXTERNAL = re.compile(r"<link\b|<script\b|\bsrc\s*=|url\s*\(\s*['\"]?https?:|@import", re.I)


def _recipe(source: Any) -> list[dict[str, Any]]:
    """SNV, mean centre and a PLS on `fat`. Two components: the fixture is
    eight samples and `decomposition.py` refuses a limit unless `n > a + 1`."""
    return [
        source,
        {"id": "snv", "type": "preprocess", "inputs": ["source"], "step": {"kind": "snv"}},
        {"id": "centre", "type": "preprocess", "inputs": ["snv"], "step": {"kind": "mean_centre"}},
        {
            "id": "pls",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {"kind": "pls", "n_components": 2, "algorithm": "nipals", "target": "fat"},
        },
    ]


@pytest.fixture
def reported(client: TestClient) -> str:  # noqa: F811
    """A project with one succeeded run, and that run's report."""
    imported(client)
    source = client.get("/api/pipelines/current", headers=AUTH).json()["nodes"][0]
    assert (
        client.put(
            "/api/pipelines/current", json={"nodes": _recipe(source)}, headers=AUTH
        ).status_code
        == 200
    )
    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    assert wait_for(client, job["job_id"])["status"] == "succeeded"

    response = client.get("/api/experiments/current/report.html", headers=AUTH)
    assert response.status_code == 200, response.text
    document: str = response.text
    return document


def test_the_report_is_one_file_and_references_nothing_external(
    client: TestClient,  # noqa: F811
    reported: str,
) -> None:
    """The claim the whole format rests on: it opens with the application shut
    down and the network off."""
    found = EXTERNAL.findall(reported)
    assert found == [], f"the report reaches outside itself: {found}"

    # Served as a download, with a name that says which run it is.
    response = client.get("/api/experiments/current/report.html", headers=AUTH)
    assert response.headers["content-type"].startswith("text/html")
    assert "attachment" in response.headers["content-disposition"]
    assert ".html" in response.headers["content-disposition"]

    # A whole document, not a fragment.
    assert reported.startswith("<!doctype html>")
    assert "</html>" in reported


def test_the_report_carries_the_experiments_own_numbers(
    client: TestClient,  # noqa: F811
    reported: str,
) -> None:
    """Every figure is the record's. Asserted against what the API serves
    rather than against a literal, because the point is that the two agree."""
    record = client.get("/api/experiments/current", headers=AUTH).json()

    assert record["experiment_id"] in reported
    assert record["dataset_content_hash"] in reported
    assert record["pipeline_snapshot"]["pipeline_id"] in reported
    for node in record["pipeline_snapshot"]["nodes"]:
        assert node["id"] in reported

    # The metrics as the record holds them, to the digits the report prints.
    metrics = record["metrics"]
    assert f"{metrics['rmsec']:.4f}" in reported
    assert record["environment"]["python_version"] in reported


def test_the_plots_are_drawn_from_the_result(
    client: TestClient,  # noqa: F811
    reported: str,
) -> None:
    """One dot per sample, and the predicted-against-measured plot a regression
    has. The counts come from the served result, not from a literal."""
    result = client.get("/api/results/pls", headers=AUTH).json()
    n_samples = len(result["scores"])

    figures = re.findall(r"<figure>.*?</figure>", reported, re.S)
    assert len(figures) == 3, "scores, scree and predicted-against-measured"

    scores = next(figure for figure in figures if "Scores," in figure)
    assert scores.count("<circle") == n_samples

    predicted = next(figure for figure in figures if "Predicted against measured" in figure)
    assert predicted.count("<circle") == n_samples
    # The 1:1 line the cloud is read against.
    assert 'class="guide"' in predicted

    scree = next(figure for figure in figures if "Explained variance" in figure)
    assert scree.count("<rect") == len(result["explained_variance_ratio"])


def test_every_svg_in_the_report_is_well_formed(reported: str) -> None:
    """Hand-written SVG is only worth its lack of dependencies if it parses.
    A browser is forgiving; a report that renders as nothing on one machine and
    a plot on another is worse than no plot."""
    for svg in re.findall(r"<svg\b.*?</svg>", reported, re.S):
        ElementTree.fromstring(svg)


def test_a_run_whose_arrays_are_gone_keeps_its_record_and_says_why(
    client: TestClient,  # noqa: F811
    reported: str,
    tmp_path: Path,
) -> None:
    """The report renders against the run's own pipeline snapshot, so a run
    whose arrays were recomputed away gets its record and a sentence rather
    than an invented plot or a 500."""
    assert "no longer in the project's store" not in reported

    for array in (tmp_path / "project" / "arrays").glob("*.npy"):
        array.unlink()
    for stored in (tmp_path / "project" / "results").glob("*"):
        stored.unlink()

    again = client.get("/api/experiments/current/report.html", headers=AUTH)
    assert again.status_code == 200
    assert "no longer in the project's store" in again.text
    # The half that does not depend on the store is untouched.
    record = client.get("/api/experiments/current", headers=AUTH).json()
    assert record["dataset_content_hash"] in again.text
    assert f"{record['metrics']['rmsec']:.4f}" in again.text


def test_a_report_is_asked_for_by_id_as_well_as_by_current(
    client: TestClient,  # noqa: F811
    reported: str,
) -> None:
    experiment_id = client.get("/api/experiments/current", headers=AUTH).json()["experiment_id"]

    by_id = client.get(f"/api/experiments/{experiment_id}/report.html", headers=AUTH)
    assert by_id.status_code == 200
    assert by_id.text == reported

    missing = client.get("/api/experiments/nope/report.html", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_what_someone_else_chose_is_escaped(client: TestClient) -> None:  # noqa: F811
    """A project name, a dataset filename and an executor's error message are
    all strings someone else wrote, and a report is a file that gets emailed."""
    from chemometrics_workbench.models import (
        DatasetVersion,
        Experiment,
        ExperimentStatus,
        Pipeline,
        Project,
    )

    imported(client)
    project = client.get("/api/projects", headers=AUTH).json()[0]
    # The API adds the running application's version to a project; the model
    # does not carry it.
    project.pop("app_version")
    version = client.get(f"/api/projects/{project['project_id']}/datasets", headers=AUTH).json()[0][
        "versions"
    ][0]
    pipeline = client.get("/api/pipelines/current", headers=AUTH).json()

    document = render_report(
        Experiment(
            project_id=project["project_id"],
            pipeline_snapshot=Pipeline.model_validate(pipeline),
            dataset_version_id=version["version_id"],
            dataset_content_hash=version["content_hash"],
            status=ExperimentStatus.FAILED,
            error="<script>alert('x')</script>",
        ),
        DatasetVersion.model_validate(version),
        Project.model_validate({**project, "directory": "/tmp/whatever"}),
        None,
        None,
    )

    assert "<script>alert" not in document
    assert "&lt;script&gt;" in document


def test_a_run_on_a_derived_version_names_the_excluded_rows(client: TestClient) -> None:  # noqa: F811
    """#270: what the run left out is in its record, not only in the database."""
    from chemometrics_workbench.models import (
        DatasetVersion,
        Experiment,
        ExperimentStatus,
        Pipeline,
        Project,
    )

    imported(client)
    project = client.get("/api/projects", headers=AUTH).json()[0]
    project.pop("app_version")
    version = client.get(f"/api/projects/{project['project_id']}/datasets", headers=AUTH).json()[0][
        "versions"
    ][0]
    pipeline = client.get("/api/pipelines/current", headers=AUTH).json()
    parent = version["version_id"]
    derived = DatasetVersion.model_validate(
        {**version, "derived_from": parent, "excluded_samples": [4, 9]}
    )
    document = render_report(
        Experiment(
            project_id=project["project_id"],
            pipeline_snapshot=Pipeline.model_validate(pipeline),
            dataset_version_id=derived.version_id,
            dataset_content_hash=derived.content_hash,
            status=ExperimentStatus.FAILED,
            error="stopped",
        ),
        derived,
        Project.model_validate({**project, "directory": "/tmp/whatever"}),
        None,
        None,
    )
    assert f"2 rows of version {parent}: 4, 9" in document


def test_the_filename_names_the_run(client: TestClient, reported: str) -> None:  # noqa: F811
    from chemometrics_workbench.models import Experiment

    record = client.get("/api/experiments/current", headers=AUTH).json()
    name = report_filename(Experiment.model_validate(record))

    assert name.endswith(".html")
    assert record["experiment_id"][:8] in name
