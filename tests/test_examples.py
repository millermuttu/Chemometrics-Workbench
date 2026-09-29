"""Every number the worked examples quote is the number the application serves (#237).

`docs/examples/pca.md` and `docs/examples/pls.md` are teaching material, and
`PROPOSAL.md` §17 treats them as product. A kernel change that moved a quoted
number and left the page alone would teach something false. So each test here
does what its page tells the reader to do, over HTTP, on the CSV the page
offers for download. It formats each number the way the page prints it, and
asserts that the page prints exactly that. Change a kernel, and the page has to
change with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.seed_e2e import tecator_csv
from tests.test_server import AUTH, wait_for

# isort: split
from tests.test_server import client  # noqa: F401 - the fixture

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
PERMISSION = ROOT / "src" / "chemometrics_workbench" / "data" / "tecator" / "tecator.txt"

SNV: dict[str, Any] = {
    "id": "snv",
    "type": "preprocess",
    "inputs": ["source"],
    "step": {"kind": "snv"},
}
SAVGOL: dict[str, Any] = {
    "id": "savgol",
    "type": "preprocess",
    "inputs": ["snv"],
    "step": {"kind": "savgol", "window_length": 11, "polyorder": 2, "deriv": 1},
}
PCA = [
    {"id": "centre", "type": "preprocess", "inputs": ["savgol"], "step": {"kind": "mean_centre"}},
    {
        "id": "pca",
        "type": "estimator",
        "inputs": ["centre"],
        "spec": {"kind": "pca", "n_components": 5},
    },
]


def pls_branch(n_components: int) -> list[dict[str, Any]]:
    return [
        {
            "id": "kfold",
            "type": "split",
            "inputs": ["savgol"],
            "spec": {"kind": "kfold", "n_splits": 10, "shuffle": True, "seed": 42},
        },
        {
            "id": "centre_2",
            "type": "preprocess",
            "inputs": ["kfold"],
            "step": {"kind": "mean_centre"},
        },
        {
            "id": "pls",
            "type": "estimator",
            "inputs": ["centre_2"],
            "spec": {
                "kind": "pls",
                "n_components": n_components,
                "algorithm": "nipals",
                "target": "fat",
            },
        },
    ]


def run(http: TestClient, nodes: list[dict[str, Any]]) -> None:
    source = http.get("/api/pipelines/current", headers=AUTH).json()["nodes"][0]
    saved = http.put("/api/pipelines/current", json={"nodes": [source, *nodes]}, headers=AUTH)
    assert saved.status_code == 200, saved.text
    job = http.post("/api/experiments/current/run", headers=AUTH).json()
    assert wait_for(http, job["job_id"], seconds=120)["status"] == "succeeded"


def result(http: TestClient, node: str) -> Any:
    return http.get(f"/api/results/{node}", headers=AUTH).json()


def import_the_download(http: TestClient) -> None:
    csv = (EXAMPLES / "tecator.csv").read_bytes()
    files = {"file": ("tecator.csv", csv)}
    response = http.post("/api/import", files=files, data={"corrections": "{}"}, headers=AUTH)
    assert response.status_code == 200, response.text


def page(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def test_the_download_is_the_committed_tecator_data_with_its_permission_note() -> None:
    """The CSV is `tecator_csv()`, byte for byte, and the note the data's terms
    make a condition of redistribution travels with it, verbatim."""
    assert (EXAMPLES / "tecator.csv").read_bytes() == tecator_csv()
    note = (EXAMPLES / "tecator-permission.txt").read_text(encoding="utf-8")
    header = PERMISSION.read_text(encoding="latin-1")
    quoted = [line.strip() for line in note.splitlines() if line.strip()]
    body = " ".join(header.split())
    assert "redistributed as long as this permission note is attached" in body
    assert all(line in body for line in quoted if not line.startswith("#"))


def test_every_number_the_pca_example_quotes(client: TestClient) -> None:  # noqa: F811
    import_the_download(client)
    run(client, [SNV, SAVGOL, *PCA])
    pca = result(client, "pca")
    evr = pca["explained_variance_ratio"]
    diagnostics = pca["diagnostics"]
    ids = [sample["sample_id"] for sample in pca["samples"]]
    t2, spe = diagnostics["hotelling_t2"], diagnostics["spe"]
    beyond_t2 = {i for i, v in enumerate(t2) if v > diagnostics["hotelling_t2_limit"]}
    beyond_spe = {i for i, v in enumerate(spe) if v > diagnostics["spe_limit"]}
    worst = max(range(len(t2)), key=t2.__getitem__)

    quoted = [
        f"{pca['n_samples']} × {pca['n_variables']}",
        pct(evr[0]),
        pct(evr[1]),
        pct(evr[2]),
        pct(pca["cumulative_explained_variance"][-1]),
        f"T² limit of {diagnostics['hotelling_t2_limit']:.1f}",
        f"{len(beyond_t2)} samples above the T² limit",
        f"{len(beyond_spe)} above the SPE limit",
        f"{len(beyond_t2 | beyond_spe)} of {pca['n_samples']}",
        f"**{ids[worst]}**, with a T² of {t2[worst]:.1f}",
    ]
    text = page("pca.md")
    missing = [number for number in quoted if number not in text]
    assert missing == [], f"docs/examples/pca.md should quote these, and does not: {missing}"


def test_every_number_the_pls_example_quotes(client: TestClient) -> None:  # noqa: F811
    import_the_download(client)
    run(client, [SNV, SAVGOL, *PCA, *pls_branch(10)])
    ten = result(client, "pls")
    curve, spread = ten["rmsecv_curve"], ten["metrics"]["rmsecv_std"]
    lowest = min(curve)
    chosen = next(a for a, value in enumerate(curve, 1) if value <= lowest + spread)

    run(client, [SNV, SAVGOL, *PCA, *pls_branch(chosen)])
    five = result(client, "pls")
    metrics = five["metrics"]
    vip = five["regression"]["vip"]
    axis = [float(name) for name in tecator_csv().decode().splitlines()[0].split(",")[1:101]]
    peak = axis[max(range(len(vip)), key=vip.__getitem__)]

    wider = {**SAVGOL, "step": {**SAVGOL["step"], "window_length": 15}}
    run(client, [SNV, wider, *PCA, *pls_branch(chosen)])
    fifteen = result(client, "pls")["metrics"]

    quoted = [
        f"RMSECV falls from {curve[0]:.2f} with one component to {lowest:.2f} with ten",
        f"spread of {spread:.2f}",
        f"below {lowest + spread:.2f}",
        f"**{chosen} components**",
        f"RMSECV **{metrics['rmsecv']:.2f}**",
        f"Q² **{metrics['q2']:.3f}**",
        f"RMSEC **{metrics['rmsec']:.2f}**",
        f"R² **{metrics['r2']:.3f}**",
        f"near {peak:.0f} nm",
        f"RMSECV {fifteen['rmsecv']:.2f}",
    ]
    text = page("pls.md")
    missing = [number for number in quoted if number not in text]
    assert missing == [], f"docs/examples/pls.md should quote these, and does not: {missing}"
