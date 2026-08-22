"""Render `docs/parity-report.md` from a parity run.

    uv run pytest                      # writes parity-results.json
    uv run python -m tests.parity_report

The report is the artifact `PROPOSAL.md` §10.4 calls the project's single most
valuable credibility asset, so two rules shape everything below.

**It renders, it does not compute.** Every number comes from
`parity-results.json`, which the harness writes as the comparisons happen. If
the report calculated anything itself there would be two sources of truth for
the same figure and no way to tell which one moved.

**It must not flatter.** A claim's strength is not the same as its passing.
Most PCA and PLS diagnostics here are *our* formula on scikit-learn's
decomposition, because scikit-learn reports none of them — that is an
independent decomposition, not an independent implementation, and the report
says so in the same breath as the agreement. Rows that differ by a documented
convention are rendered with their reason rather than as failures, and rows
that failed are rendered as failures rather than quietly omitted.

Determinism is a requirement, not a nicety: the report is regenerated in CI and
the build fails if it differs from the committed copy, which is what "green in
CI against published reference values" means in practice. So nothing here reads
the clock, the environment or the filesystem beyond the two JSON files, and
every table is sorted by a stable key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests import parity

REPORT = Path(__file__).parent.parent / "docs" / "parity-report.md"

# Rendered names, in the order the report presents them. A quantity's algorithm
# is what the fixture says it is; these are only labels.
ALGORITHM_TITLES = {
    "pca": "Principal component analysis",
    "pls": "PLS regression",
    "preprocess": "Preprocessing",
}
ALGORITHM_ORDER = ["preprocess", "pca", "pls"]
DATASET_ORDER = ["corn", "gasoline", "tecator"]

TIER_LABELS = {
    parity.Tier.IDENTICAL: "identical within floating point",
    parity.Tier.WITHIN_TOLERANCE: "agrees within stated tolerance",
    parity.Tier.DOCUMENTED_DIVERGENCE: "differs by documented convention",
}

# Claims where the reference is our own formula applied to somebody else's
# decomposition, rather than somebody else's implementation of the quantity.
# scikit-learn reports none of these, so there was nothing else to compare
# against — and a reader who is not told cannot tell the difference from the
# agreement column, which looks just as strong.
OUR_FORMULA_QUANTITIES = frozenset(
    {
        "hotelling_t2",
        "spe",
        "cumulative_explained_variance",
        "vip",
    }
)


def _sort_key(result: dict[str, Any]) -> tuple[int, int, str, str]:
    dataset = result["dataset"]
    algorithm = result["algorithm"]
    return (
        ALGORITHM_ORDER.index(algorithm) if algorithm in ALGORITHM_ORDER else len(ALGORITHM_ORDER),
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER),
        result["quantity"],
        result["entry_id"],
    )


def _number(value: float) -> str:
    """Enough digits to be checkable, few enough to be read."""
    if value == 0.0:
        return "0"
    if 1e-4 <= abs(value) < 1e5:
        return f"{value:.6g}"
    return f"{value:.3e}"


def _agreement(result: dict[str, Any]) -> str:
    """What the two sides actually did, in one cell."""
    if result["tier"] == str(parity.Tier.DOCUMENTED_DIVERGENCE):
        return "not compared — see below"
    if result["our_value"] is not None and result["reference_value"] is not None:
        return f"`{_number(result['our_value'])}` vs `{_number(result['reference_value'])}`"
    n_values = result["n_values"]
    worst = result["max_abs_diff"]
    # No pipe characters: this lands in a markdown table cell.
    return f"{n_values} values, worst Δ {_number(worst)}"


def _claim(result: dict[str, Any]) -> str:
    if not result["passed"]:
        return "**FAILED**"
    if result["tier"] == str(parity.Tier.IDENTICAL):
        return "identical"
    if result["tier"] == str(parity.Tier.WITHIN_TOLERANCE):
        return f"within rtol {_number(result['rtol'])}"
    return "documented divergence"


def _reference(result: dict[str, Any]) -> str:
    software = str(result["software"])
    version = str(result["software_version"])
    return software if version in ("", "unstated") else f"{software} {version}"


def _quantity(result: dict[str, Any]) -> str:
    name = f"`{result['quantity']}`"
    if result["quantity"] in OUR_FORMULA_QUANTITIES:
        return f"{name} †"
    if result["sign_aligned"]:
        return f"{name} ‡"
    return name


def _table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Dataset | Quantity | Reference | Claim | Ours vs reference |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['dataset']} | {_quantity(r)} | {_reference(r)} | {_claim(r)} | {_agreement(r)} |"
        for r in results
    ]
    return lines


def render(results: dict[str, Any], fixture: dict[str, Any]) -> str:
    """The whole report, as a string. Pure: same inputs, same bytes."""
    totals = results["totals"]
    claims = sorted(results["results"], key=_sort_key)
    failures = [r for r in claims if not r["passed"]]
    divergences = [r for r in claims if r["tier"] == str(parity.Tier.DOCUMENTED_DIVERGENCE)]

    lines: list[str] = [
        "# Parity report",
        "",
        "Generated by `uv run python -m tests.parity_report` from `parity-results.json`,",
        "which the test suite writes as the comparisons happen. **Do not edit it by hand:**",
        "CI regenerates it and fails the build if the committed copy differs, which is what",
        "`PROPOSAL.md` §16's exit criterion — *parity report green in CI against published",
        "reference values* — means in practice.",
        "",
        f"Fixture schema {results['fixture_schema_version']}, generated {fixture['generated_at']}.",
        "",
        "---",
        "",
        "## What this report claims, and what it does not",
        "",
        "Every quantity this project reports is compared against an independent",
        "implementation or a published value, and each comparison is tagged with **how",
        "strong the agreement is** rather than a bare pass or fail:",
        "",
        "| Claim | Count | Meaning |",
        "| --- | --- | --- |",
        f"| {TIER_LABELS[parity.Tier.IDENTICAL]} | {totals[str(parity.Tier.IDENTICAL)]} | "
        "The same computation reached by a different code path. Anything worse than this "
        "would be a real difference, not rounding. |",
        f"| {TIER_LABELS[parity.Tier.WITHIN_TOLERANCE]} | "
        f"{totals[str(parity.Tier.WITHIN_TOLERANCE)]} | "
        "Within a tolerance chosen per quantity class *with a reason*, and never widened "
        "to make a test pass. |",
        f"| {TIER_LABELS[parity.Tier.DOCUMENTED_DIVERGENCE]} | "
        f"{totals[str(parity.Tier.DOCUMENTED_DIVERGENCE)]} | "
        "Not compared numerically at all. The two quantities are not the same thing, and "
        "the reason is given in full below. |",
        "",
        f"**{totals['compared']} comparisons, {totals['passed']} passed, "
        f"{totals['failed']} failed.**",
        "",
        "Three things a reader should hold on to, because the agreement column cannot",
        "show them:",
        "",
        "1. **† marks a quantity where the reference is *our* formula applied to somebody",
        "   else's decomposition.** scikit-learn reports no Hotelling's T², no SPE, no",
        "   cumulative explained variance curve and no VIP, so for those the fixture value",
        "   was computed here from scikit-learn's own scores and loadings by the definition",
        "   in `docs/algorithms/`. That tests our decomposition against theirs carried",
        "   through a formula both sides agree on. It is worth having and it is not an",
        "   independent implementation of the quantity.",
        "2. **‡ marks a quantity whose component signs are arbitrary** and were aligned by",
        "   inner product with the reference before comparing (`pca.md` §5,",
        "   `pls-regression.md` §6). Comparing absolute values instead would pass a result",
        "   whose scores and loadings disagreed with each other.",
        "3. **Most references here are open implementations pinned by version, not numbers",
        "   printed in a paper.** The exceptions are the two R `pls` vignette entries and",
        "   the Tecator SEP below, and they are labelled as such. An implementation",
        "   comparison is reproducible by anyone; a published number is independent of any",
        "   code. They are different kinds of evidence and this report does not blur them.",
        "",
    ]

    if failures:
        lines += [
            "> [!WARNING]",
            f"> **{len(failures)} claim(s) failed in the run this report was generated from.**",
            "> They are listed here rather than omitted, because a report that shows only",
            "> what passed is not evidence of anything.",
            "",
        ]
        lines += [*_table(failures), ""]

    lines += ["---", "", "## The claims"]

    for algorithm in ALGORITHM_ORDER:
        for_algorithm = [r for r in claims if r["algorithm"] == algorithm]
        if not for_algorithm:
            continue
        lines += ["", f"### {ALGORITHM_TITLES[algorithm]}", ""]
        lines += _table(for_algorithm)

    if divergences:
        lines += [
            "",
            "---",
            "",
            "## Documented divergences",
            "",
            "These are **not failures**. In each case the reference quantity and ours are",
            "defined differently, the difference is recorded in the relevant specification's",
            "*known divergences* table, and comparing the two numerically would be comparing",
            "two different things.",
            "",
        ]
        for result in divergences:
            lines += [
                f"**`{result['entry_id']}`** — {_reference(result)}",
                "",
                result["reason"],
                "",
            ]

    lines += [
        "---",
        "",
        "## Tolerances, and why each one is what it is",
        "",
        "A tolerance is a claim about what two correct implementations may differ by. It is",
        "chosen per quantity class, in `tests/parity.py`, and **widening one to make a test",
        "pass would put a lie in this report**. Every class in use:",
        "",
    ]
    used = sorted({parity.QUANTITY_CLASS[r["quantity"]] for r in claims if r["passed"]})
    for name in used:
        tolerance = parity.TOLERANCES[name]
        lines += [
            f"**`{name}`** — rtol {_number(tolerance.rtol)}, atol {_number(tolerance.atol)}",
            "",
            tolerance.reason,
            "",
        ]

    compared = entries_compared(results)
    gaps = [
        e
        for e in fixture["entries"]
        if e["status"] != "sourced" or (not e["comparable"] and e["id"] not in compared)
    ]
    lines += [
        "---",
        "",
        "## Gaps",
        "",
        "What could **not** be compared, listed for the same reason the failures above are:",
        "a report that shows only its coverage overstates it. A gap written down is a task;",
        "a gap left out is a false impression.",
        "",
    ]
    for gap in sorted(gaps, key=lambda e: e["id"]):
        status = "unsourced" if gap["status"] != "sourced" else "not a parity target"
        lines += [f"**`{gap['id']}`** — {status}", "", gap["notes"], ""]

    lines += [
        "---",
        "",
        "## Reproducing this",
        "",
        "From a clean checkout, with no network access needed beyond the first dataset",
        "download:",
        "",
        "```bash",
        "uv sync",
        "CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest      # writes parity-results.json",
        "uv run python -m tests.parity_report                # rewrites this file",
        "git diff --exit-code docs/parity-report.md          # what CI asserts",
        "```",
        "",
        "The reference values themselves are regenerated by",
        "`uv run python tests/fixtures/generate_reference_values.py`, which records for every",
        "entry its preprocessing chain, algorithm variant, split, software, version and",
        "citation. The datasets are corn, gasoline and Tecator; only Tecator is committed,",
        "and the other two are downloaded on first use and verified against a pinned",
        "SHA-256.",
        "",
        "**Tecator results carry a condition of use**: publishing a result obtained with",
        "that dataset obliges you to name the instrument and the company, Tecator. That",
        "applies to this report as much as to a paper.",
        "",
    ]
    return "\n".join(lines) + "\n"


def entries_compared(results: dict[str, Any]) -> set[str]:
    return {r["entry_id"] for r in results["results"]}


def main() -> None:
    results = json.loads(parity.RESULTS.read_text(encoding="utf-8"))
    if results["not_compared"]:
        raise SystemExit(
            f"{len(results['not_compared'])} comparable fixture entries were not compared in "
            "the run that produced parity-results.json, so a report built from it would "
            "understate coverage. Run the whole suite first:\n"
            "  CHEMOMETRICS_DOWNLOAD_DATASETS=1 uv run pytest"
        )
    REPORT.write_text(render(results, parity.load_fixture()), encoding="utf-8")
    print(f"wrote {REPORT} - {results['totals']['compared']} claims")


if __name__ == "__main__":
    main()
