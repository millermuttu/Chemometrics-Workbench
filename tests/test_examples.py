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


# --------------------------------------------------------------------------
# Phase 5 (#288): the classification example, on the Quadram meat set, and the
# outlier and selection how-tos, which continue the PLS example.
# --------------------------------------------------------------------------

HOW_TO = ROOT / "docs" / "how-to"
CLASSIFIERS: list[dict[str, Any]] = [
    {
        "id": "kfold",
        "type": "split",
        "inputs": ["source"],
        "spec": {
            "kind": "kfold",
            "n_splits": 10,
            "shuffle": True,
            "seed": 42,
            "stratify_by": "meat",
        },
    },
    {"id": "centre", "type": "preprocess", "inputs": ["kfold"], "step": {"kind": "mean_centre"}},
    *(
        {"id": node, "type": "estimator", "inputs": ["centre"], "spec": spec}
        for node, spec in (
            (
                "plsda",
                {"kind": "plsda", "n_components": 5, "algorithm": "nipals", "class_column": "meat"},
            ),
            ("lda", {"kind": "lda", "n_components": 5, "class_column": "meat"}),
            ("knn", {"kind": "knn", "k": 5, "n_components": 5, "class_column": "meat"}),
            ("simca", {"kind": "simca", "n_components": 3, "class_column": "meat"}),
        )
    ),
]
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
RULES = {
    "t2": "T²",
    "q": "Q",
    "leverage": "leverage",
    "residual": "residual",
    "robust": "robust distance",
}


def test_the_meat_download_is_one_row_per_sample_on_a_descending_axis() -> None:
    """`meat-source.md`'s claims about `meat.csv`: 60 samples, 20 of each meat,
    turkey from supplier E only, and the axis written high to low."""
    rows = [line.split(",") for line in (EXAMPLES / "meat.csv").read_text().splitlines()]
    header, body = rows[0], rows[1:]
    assert header[:3] == ["sample", "meat", "supplier"]
    axis = [float(value) for value in header[3:]]
    assert len(axis) == 448 and axis == sorted(axis, reverse=True)
    assert len(body) == 60 and len({row[0] for row in body}) == 60
    meats = [row[1] for row in body]
    assert {meat: meats.count(meat) for meat in set(meats)} == {
        "chicken": 20,
        "pork": 20,
        "turkey": 20,
    }
    assert {row[2] for row in body if row[1] == "turkey"} == {"E"}
    assert "E" not in {row[2] for row in body if row[1] != "turkey"}


def test_every_number_the_classification_example_quotes(client: TestClient) -> None:  # noqa: F811
    csv = (EXAMPLES / "meat.csv").read_bytes()
    response = client.post(
        "/api/import", files={"file": ("meat.csv", csv)}, data={"corrections": "{}"}, headers=AUTH
    )
    assert response.status_code == 200, response.text
    version = response.json()["versions"][0]
    run(client, CLASSIFIERS)
    plsda, simca = result(client, "plsda"), result(client, "simca")
    n = version["n_samples"]
    curve = plsda["rmsecv_curve"]
    axis = plsda["loadings"]["axis"]["values"]
    assert plsda["loadings"]["axis"]["kind"] == "wavenumber_cm-1"
    cv = plsda["classification"]["confusion"]["cross_validation"]
    accepted = simca["simca"]["sets"]["cross_validation"]
    rejected = sum(accepted["none"])

    def misassigned(node: str) -> str:
        table = result(client, node)["classification"]["confusion"]["cross_validation"]
        wrong = n - sum(table[i][i] for i in range(len(table)))
        return "none" if wrong == 0 else f"{wrong} of {n}"

    quoted = [
        f"**{n} × {version['n_variables']}**",
        f"from {axis[0]:.0f} to {axis[-1]:.0f} cm⁻¹",
        f"**lowest at A = {curve.index(min(curve)) + 1}**",
        f"**{cv[0][0]}, {cv[1][1]} and {cv[2][2]}** on the diagonal",
        f"**Accuracy (CV) {plsda['metrics']['accuracy_cv']:.3f}**",
        f"difference of {1 / n:.3f}",
        f"sensitivity is **{simca['metrics']['sensitivity_cv']:.3f}**",
        f"specificity **{simca['metrics']['specificity_cv']:.3f}**",
        f"**{rejected} of {n}** samples are accepted by no model",
        f"built from {simca['simca']['models'][0]['n_samples']} samples",
    ]
    for node, label in (("plsda", "PLS-DA 5 LV"), ("lda", "LDA 5 PC"), ("knn", "kNN k5")):
        accuracy = result(client, node)["metrics"]["accuracy_cv"]
        quoted.append(f"| {label} | **{accuracy:.3f}** | {misassigned(node)} |")
    assert {model["n_samples"] for model in simca["simca"]["models"]} == {
        simca["simca"]["models"][0]["n_samples"]
    }
    text = " ".join(page("classification.md").split())
    missing = [number for number in quoted if number not in text]
    assert missing == [], (
        f"docs/examples/classification.md should quote these, and does not: {missing}"
    )


def test_every_number_the_outliers_how_to_quotes(client: TestClient) -> None:  # noqa: F811
    import_the_download(client)
    nodes = [SNV, SAVGOL, *PCA, *pls_branch(4)]
    run(client, nodes)
    pls = result(client, "pls")
    block = client.get("/api/results/pls/outliers", headers=AUTH).json()
    flags = block["flags"]
    ids = [sample["sample_id"] for sample in pls["samples"]]
    worst = [flag for flag in flags if len(flag["rules"]) == max(len(f["rules"]) for f in flags)]
    before = pls["metrics"]["rmsecv"]

    excluded = client.post(
        f"/api/datasets/{block['dataset_id']}/versions",
        json={
            "from_version_id": block["version_id"],
            "exclude": [pls["samples"][flag["index"]]["index"] for flag in worst],
        },
        headers=AUTH,
    )
    assert excluded.status_code == 201, excluded.text
    remaining = excluded.json()["versions"][-1]["n_samples"]
    job = client.post("/api/experiments/current/run", headers=AUTH).json()
    assert wait_for(client, job["job_id"], seconds=120)["status"] == "succeeded"
    after = result(client, "pls")["metrics"]["rmsecv"]

    quoted = [
        f"**{len(flags)} of {pls['n_samples']}**",
        f"fitted on the {pls['n_samples']} samples",
        f"{sum(f['rules'] == ['robust'] for f in flags)} of the {len(flags)} are flagged only",
        f"Three break {WORDS[len(worst[0]['rules'])]} each",
        f"**Exclude {len(worst)} and rerun**",
        f"now has {remaining} samples",
        f"from {before:.2f} to **{after:.2f}**",
        *(
            f"| **{ids[flag['index']]}** | {' · '.join(RULES[rule] for rule in flag['rules'])} |"
            for flag in worst
        ),
    ]
    assert len(worst) == 3
    text = " ".join((HOW_TO / "outliers.md").read_text(encoding="utf-8").split())
    missing = [number for number in quoted if number not in text]
    assert missing == [], f"docs/how-to/outliers.md should quote these, and does not: {missing}"


def test_every_number_the_selection_how_to_quotes(client: TestClient) -> None:  # noqa: F811
    import_the_download(client)
    nodes = [SNV, SAVGOL, *PCA, *pls_branch(4)]
    run(client, nodes)
    pls = result(client, "pls")
    p = pls["n_variables"]
    vip = [i for i, value in enumerate(pls["regression"]["vip"]) if value >= 1]
    ipls = client.get("/api/results/pls/ipls", headers=AUTH).json()
    cars = client.get("/api/results/pls/cars", headers=AUTH).json()
    best = min(ipls["intervals"], key=lambda interval: interval["rmsecv"])

    def applied(indices: list[int], by: str) -> float:
        estimator = nodes[-1]
        select = {
            "id": "pls_select",
            "type": "preprocess",
            "inputs": estimator["inputs"],
            "step": {"kind": "select_variables", "indices": indices, "chosen_by": by},
        }
        n_components = min(estimator["spec"]["n_components"], len(indices))
        copy = {
            **estimator,
            "id": "pls_selected",
            "inputs": ["pls_select"],
            "spec": {**estimator["spec"], "n_components": n_components},
        }
        run(client, [*nodes, select, copy])
        return float(result(client, "pls_selected")["metrics"]["rmsecv"])

    cars_rmsecv = applied(cars["selected"], "cars")
    quoted = [
        f"RMSECV is **{pls['metrics']['rmsecv']:.2f}** on all {p} wavelengths",
        f"**{len(vip)} of {p}**",
        f"RMSECV of **{applied(vip, 'vip'):.2f}**",
        f"**{best['axis_start']:.0f} to {best['axis_end']:.0f} nm**",
        f"{WORDS[best['stop'] - best['start']]} wavelengths,",
        f"with an RMSECV of **{best['rmsecv']:.2f}**",
        f"**{len(ipls['selected'])} of {p}**",
        f"the {ipls['full']['rmsecv']:.2f} of all {p} wavelengths",
        f"**{len(cars['selected'])} of {p}**",
        f"RMSECV of **{cars_rmsecv:.2f}**",
        f"so {cars_rmsecv:.2f}",
    ]
    # The page's argument rests on these, not only on the numbers.
    assert ipls["selected"] == list(range(best["start"], best["stop"]))
    assert cars_rmsecv < pls["metrics"]["rmsecv"] < best["rmsecv"]
    text = " ".join((HOW_TO / "variable-selection.md").read_text(encoding="utf-8").split())
    missing = [number for number in quoted if number not in text]
    assert missing == [], f"docs/how-to/variable-selection.md should quote these: {missing}"


# --------------------------------------------------------------------------
# Phase 6 (#340): the validation example, on the raw meat set grouped by
# sample, and the how-tos' nested, bootstrap and class-wise sections.
# --------------------------------------------------------------------------


def grouped(group_by: str | None) -> list[dict[str, Any]]:
    spec: dict[str, Any] = {"kind": "kfold", "n_splits": 10, "shuffle": True, "seed": 42}
    if group_by:
        spec["group_by"] = group_by
    return [
        {"id": "kfold", "type": "split", "inputs": ["source"], "spec": spec},
        {
            "id": "centre",
            "type": "preprocess",
            "inputs": ["kfold"],
            "step": {"kind": "mean_centre"},
        },
        {
            "id": "plsda",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "plsda",
                "n_components": 5,
                "algorithm": "nipals",
                "class_column": "meat",
            },
        },
        {
            "id": "svm",
            "type": "estimator",
            "inputs": ["centre"],
            "spec": {
                "kind": "svm",
                "kernel": "rbf",
                "C": 1,
                "n_components": 5,
                "class_column": "meat",
            },
        },
    ]


def import_raw_meat(http: TestClient) -> dict[str, Any]:
    csv = (EXAMPLES / "meat-raw.csv").read_bytes()
    response = http.post(
        "/api/import",
        files={"file": ("meat-raw.csv", csv)},
        data={"corrections": "{}"},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    return response.json()["versions"][0]  # type: ignore[no-any-return]


def misassigned(http: TestClient, node: str, n: int) -> int:
    table = result(http, node)["classification"]["confusion"]["cross_validation"]
    return n - int(sum(table[i][i] for i in range(len(table))))


def test_the_raw_meat_download_averages_to_the_one_row_per_sample_file() -> None:
    """`meat-source.md`'s claim: `meat-raw.csv` holds both runs of each of the 60
    samples, and the mean of each pair is `meat.csv`'s row for that sample."""
    raw = [line.split(",") for line in (EXAMPLES / "meat-raw.csv").read_text().splitlines()]
    mean = [line.split(",") for line in (EXAMPLES / "meat.csv").read_text().splitlines()]
    assert raw[0][:5] == ["spectrum", "meat", "supplier", "sample", "run"]
    assert raw[0][5:] == mean[0][3:]
    assert len(raw) - 1 == 120 and len({row[0] for row in raw[1:]}) == 120
    pairs: dict[str, list[list[str]]] = {}
    for row in raw[1:]:
        pairs.setdefault(row[3], []).append(row)
    for row in mean[1:]:
        a, b = pairs[row[0]]
        assert [a[4], b[4]] == ["A", "B"] and a[1:3] == b[1:3] == row[1:3]
        averaged = [(float(x) + float(y)) / 2 for x, y in zip(a[5:], b[5:], strict=True)]
        assert all(
            abs(m - float(v)) <= 1e-8 * abs(float(v))
            for m, v in zip(averaged, row[3:], strict=True)
        ), row[0]


def test_every_number_the_validation_example_quotes(client: TestClient) -> None:  # noqa: F811
    version = import_raw_meat(client)
    n = version["n_samples"]
    run(client, grouped(None))
    loose = result(client, "plsda")["metrics"]
    run(client, grouped("sample"))
    plsda, svm = result(client, "plsda"), result(client, "svm")
    held = plsda["validation"]["samples"]
    # Grouped folds keep both runs of a sample on one side, in every fold.
    pairs = {sample["sample_id"][:-1] for sample in held}
    assert len(held) == 2 * len(pairs)

    job = client.post("/api/results/plsda/permutation?n_permutations=100", headers=AUTH).json()
    assert wait_for(client, job["job_id"], seconds=300)["status"] == "succeeded"
    permutation = client.get(f"/api/permutations/{job['job_id']}", headers=AUTH).json()
    vip = [v for v in plsda["regression"]["vip"] if v >= 1]
    nested = client.get("/api/results/plsda/nested?method=vip&cut=1", headers=AUTH).json()

    quoted = [
        f"**{n} × {version['n_variables']}**",
        f"**Accuracy (held out) {plsda['metrics']['accuracy_p']:.3f}**",
        f"{len(held)} spectra, {len(pairs)} samples",
        f"**{sum(map(sum, plsda['classification']['confusion']['calibration']))}**",
        f"**Accuracy (CV) {plsda['metrics']['accuracy_cv']:.3f}**",
        f"**{misassigned(client, 'plsda', n)} of {n}**",
        f"RMSECV is **{plsda['metrics']['rmsecv']:.3f}**",
        f"{loose['rmsecv']:.3f} without grouping",
        f"accuracy is {loose['accuracy_cv']:.3f} either way",
        f"p = {permutation['p_value']:#.3g}",
        f"1/{permutation['n_permutations'] + 1}",
        f"SVM's **Accuracy (CV)** is **{svm['metrics']['accuracy_cv']:.3f}**",
        f"{misassigned(client, 'svm', n)} of {n} misassigned",
        f"**{len(vip)} of {plsda['n_variables']}**",
        f"Nested RMSECV {nested['outer_rmsecv']:#.4g}",
        f"selected on every sample {nested['inner_rmsecv']:#.4g}",
    ]
    assert permutation["p_value"] == 1 / (permutation["n_permutations"] + 1)
    assert nested["outer_rmsecv"] > nested["inner_rmsecv"]
    text = " ".join(page("validation.md").split())
    missing = [number for number in quoted if number not in text]
    assert missing == [], f"docs/examples/validation.md should quote these, and does not: {missing}"


def test_every_number_the_selection_how_to_quotes_for_nested_and_bootstrap(
    client: TestClient,  # noqa: F811
) -> None:
    import_the_download(client)
    run(client, [SNV, SAVGOL, *PCA, *pls_branch(4)])
    pls = result(client, "pls")
    vip = pls["regression"]["vip"]
    chosen = [i for i, value in enumerate(vip) if value >= 1]
    cars = client.get("/api/results/pls/nested?method=cars&n_runs=50", headers=AUTH).json()
    by_vip = client.get("/api/results/pls/nested?method=vip&cut=1", headers=AUTH).json()
    bands = client.get("/api/results/pls/bootstrap?n_resamples=200", headers=AUTH).json()
    lower, upper = bands["vip"]["lower"], bands["vip"]["upper"]
    sure = sum(lower[i] >= 1 for i in chosen)
    straddle = sum(lower[i] < 1 <= upper[i] for i in range(len(vip)))
    per_fold = cars["selected_per_fold"]
    selection = [
        f"Nested RMSECV {cars['outer_rmsecv']:#.4g}",
        f"selected on every sample {cars['inner_rmsecv']:#.4g}",
        f"{cars['n_outer_folds']} outer × {cars['inner_splits']} inner folds",
        f"between {min(per_fold)} and {max(per_fold)} wavelengths",
        f"VIP's nested RMSECV is {by_vip['outer_rmsecv']:#.4g}",
        f"**{sure} of the {len(chosen)}**",
        f"**{straddle}** wavelengths",
    ]
    assert cars["outer_rmsecv"] > cars["inner_rmsecv"]
    text = " ".join((HOW_TO / "variable-selection.md").read_text(encoding="utf-8").split())
    missing = [number for number in selection if number not in text]
    assert missing == [], f"docs/how-to/variable-selection.md should quote these: {missing}"


def test_every_number_the_outliers_how_to_quotes_for_a_classifier(
    client: TestClient,  # noqa: F811
) -> None:
    version = import_raw_meat(client)
    run(client, grouped("sample"))
    block = client.get("/api/results/plsda/outliers", headers=AUTH).json()
    flags = block["flags"]
    ids = [sample["sample_id"] for sample in result(client, "plsda")["samples"]]
    most = max(flag["n_rules"] for flag in flags)
    worst = [ids[flag["index"]] for flag in flags if flag["n_rules"] == most]
    outliers = [
        f"**{len(flags)} of {version['n_samples']}**",
        f"{sum(flag['rules'] == ['q'] for flag in flags)} of them only on Q",
        f"{len(worst)} break {WORDS.get(most, most)}",
        ", ".join(f"**{sample}**" for sample in worst),
    ]
    assert block["classwise"]["classes"] == ["chicken", "pork", "turkey"]
    text = " ".join((HOW_TO / "outliers.md").read_text(encoding="utf-8").split())
    missing = [number for number in outliers if number not in text]
    assert missing == [], f"docs/how-to/outliers.md should quote these: {missing}"
