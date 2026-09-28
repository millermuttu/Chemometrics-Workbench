"""One experiment as a standalone HTML file (#224).

`PROPOSAL.md` §5 lists *report export (HTML/PDF) for a model or an experiment
comparison*; §16 puts it in Phase 3. This is the experiment half, in HTML.

Three rules shape everything below.

**It renders, it does not compute.** `tests/parity_report.py`'s rule, and for
its reason: every figure comes from the `Experiment` record or from the stored
`EstimatorResult`, and a report that calculated anything itself would be a
second source of truth for a number the application already publishes. The one
arithmetic this module does is turning numbers into pixel coordinates, which is
layout rather than a claim.

**One file, nothing external.** No stylesheet link, no script tag, no image
`src`, no font. The report opens with the application shut down and the
network off, which is what makes it a thing you can email or archive. That
rules out a CDN and it rules out the two obvious plotting routes: bundling
Plotly inlines about 3 MB of a dependency the backend does not have, and
matplotlib is heavier than every runtime dependency combined for three scatter
plots. So the plots are **hand-written SVG** — axes, ticks and points are
arithmetic on numbers the result already carries, and any browser draws them.

**HTML now, PDF never here.** A browser prints a PDF and does it well, so the
report carries print styles and `Cmd-P` is the PDF export. A PDF library is a
runtime dependency with real weight, which is what §7's rule is about.

## What a missing result means

The report is rendered against the experiment's **own pipeline snapshot**, so
`stored_result` is asked for that run's keys rather than the current
pipeline's. A run whose arrays have since been recomputed away still has its
record, its metrics and its provenance — and gets a sentence where its plots
would be, rather than an invented plot or a 500.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

from chemometrics_workbench.executor import EstimatorResult
from chemometrics_workbench.models import (
    DatasetVersion,
    Experiment,
    Metrics,
    PipelineNode,
    Project,
)

__all__ = ["render_report", "report_filename"]

#: The plot box, in SVG user units. Fixed rather than configurable: a report is
#: printed as often as it is read, and one width that fits a page beats a knob.
WIDTH = 520
HEIGHT = 360
PAD_LEFT = 58
PAD_BOTTOM = 44
PAD_TOP = 16
PAD_RIGHT = 16


def report_filename(experiment: Experiment) -> str:
    """A filename that says which run this is without being opened."""
    when = experiment.started_at.strftime("%Y%m%d-%H%M") if experiment.started_at else "unrun"
    return f"experiment-{when}-{str(experiment.experiment_id)[:8]}.html"


def render_report(
    experiment: Experiment,
    version: DatasetVersion,
    project: Project,
    result: EstimatorResult | None,
    node_id: str | None,
) -> str:
    """The whole report, as one HTML document.

    `result` is the last estimator's, or `None` when the store no longer holds
    this run's arrays. `node_id` names which estimator it is, so the reader is
    not left guessing which branch the plots belong to.
    """
    sections = [
        _header(experiment, project),
        _summary(experiment, version),
        _pipeline(experiment.pipeline_snapshot.nodes),
        _metrics(experiment.metrics),
        _plots(result, node_id),
        _provenance(experiment, version),
    ]
    return _document(project.name, "\n".join(sections))


# --- the document ----------------------------------------------------------


def _document(name: str, body: str) -> str:
    # Every rule inline. A link to a stylesheet would be the one thing that
    # stops this opening on a machine with no network.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_text(name)} — experiment report</title>
<style>
{_CSS}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


_CSS = """
:root {
  --ink: #16181d; --ink2: #4a5058; --ink3: #7c838d;
  --rule: #e3e6ea; --accent: #2f6f4f; --fail: #b3261e;
  --bg: #ffffff; --soft: #f6f7f9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #e9ecf1; --ink2: #b3bac4; --ink3: #858d98;
    --rule: #2b2f36; --accent: #6fbf94; --fail: #f2836f;
    --bg: #14161a; --soft: #1b1e23;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 860px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 {
  font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink3); margin: 32px 0 10px; font-weight: 600;
}
p { margin: 8px 0; color: var(--ink2); }
.mono { font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
.sub { color: var(--ink3); font-size: 12.5px; margin: 0 0 4px; }
.pill {
  display: inline-block; border: 1px solid var(--rule); border-radius: 999px;
  padding: 1px 9px; font-size: 11px; color: var(--ink2);
}
.pill.bad { color: var(--fail); border-color: var(--fail); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--rule); }
th { color: var(--ink3); font-weight: 600; font-size: 11px; letter-spacing: .04em;
     text-transform: uppercase; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; }
.note {
  border: 1px solid var(--rule); border-radius: 4px; background: var(--soft);
  padding: 10px 12px; color: var(--ink2);
}
.note.bad { border-color: var(--fail); color: var(--fail); }
figure { margin: 0; }
figcaption { font-size: 11.5px; color: var(--ink3); margin-top: 4px; }
svg { width: 100%; height: auto; display: block; }
.axis { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--ink3); font-size: 10px; }
.label { fill: var(--ink2); font-size: 11px; }
.dot { fill: var(--accent); fill-opacity: .55; }
.bar { fill: var(--accent); fill-opacity: .7; }
.guide { stroke: var(--ink3); stroke-width: 1; stroke-dasharray: 4 3; }
/* A browser prints the PDF, so the report is laid out for paper too. */
@media print {
  :root { --bg: #fff; --ink: #000; --ink2: #333; --ink3: #666; --rule: #ccc; }
  main { max-width: none; padding: 0; }
  h2 { margin-top: 18px; }
  figure, table { break-inside: avoid; }
}
"""


# --- the sections ----------------------------------------------------------


def _header(experiment: Experiment, project: Project) -> str:
    failed = experiment.status.value == "failed"
    return f"""<h1>{_text(project.name)}</h1>
<p class="sub mono">experiment {_text(str(experiment.experiment_id))}</p>
<p>
  <span class="pill{" bad" if failed else ""}">{_text(experiment.status.value)}</span>
  <span class="pill mono">{_text(_when(experiment.started_at))}</span>
  <span class="pill mono">{len(experiment.pipeline_snapshot.nodes)} nodes</span>
</p>
{_error(experiment)}"""


def _error(experiment: Experiment) -> str:
    # A failed experiment is a result, and the cause is what it holds
    # (`PROPOSAL.md` §8.2). A report that omitted it would be a report of a
    # run that never happened.
    if not experiment.error:
        return ""
    return f'<p class="note bad">{_text(experiment.error)}</p>'


def _summary(experiment: Experiment, version: DatasetVersion) -> str:
    rows = [
        ("Dataset", version.source.filename if version.source else "—"),
        ("Samples × variables", f"{version.n_samples} × {version.n_variables}"),
        ("Axis", f"{version.axis.kind.value} ({version.axis.unit})"),
        ("Content hash", version.content_hash),
        ("Pipeline hash", experiment.pipeline_hash),
    ]
    for split in experiment.resolved_splits:
        held = sum(len(fold) for fold in split.test_indices)
        folds = len(split.test_indices)
        rows.append(
            (
                "Split",
                f"{split.node_id} · {folds} fold{'' if folds == 1 else 's'} · {held} held out",
            )
        )
    return "<h2>Against what</h2>\n" + _kv_table(rows)


def _pipeline(nodes: list[PipelineNode]) -> str:
    """The recipe as the experiment froze it — by value, not as it is now."""
    body = "\n".join(
        f"<tr><td class='mono'>{_text(node.id)}</td>"
        f"<td class='mono'>{_text(node.type)}</td>"
        f"<td class='mono'>{_text(_parameters(node))}</td></tr>"
        for node in nodes
    )
    return f"""<h2>What it ran</h2>
<table>
<thead><tr><th>Node</th><th>Type</th><th>Parameters</th></tr></thead>
<tbody>
{body}
</tbody>
</table>"""


def _parameters(node: PipelineNode) -> str:
    """A node's own settings, as one line. The model serialises them; this
    prints them, and neither invents a default the node did not carry."""
    step: Any = getattr(node, "step", None) or getattr(node, "spec", None)
    if step is None:
        return "—"
    document = step.model_dump(mode="json") if hasattr(step, "model_dump") else dict(step)
    parts = [
        f"{key}={value}" for key, value in document.items() if key != "kind" and value is not None
    ]
    kind = document.get("kind", "")
    return f"{kind}{' · ' if parts else ''}{' · '.join(parts)}" if kind else " · ".join(parts)


#: `metrics-and-validation.md` §11's named table, in the order a reader of a
#: calibration reads it.
_METRIC_ROWS = (
    ("RMSEC", "rmsec", 4),
    ("RMSECV", "rmsecv", 4),
    ("RMSEP", "rmsep", 4),
    ("R²", "r2", 4),
    ("Q²", "q2", 4),
    ("Bias", "bias", 4),
    ("Accuracy", "accuracy", 3),
)


def _metrics(metrics: Metrics | None) -> str:
    if metrics is None:
        return '<h2>What it scored</h2>\n<p class="note">This run recorded no metrics.</p>'
    rows = [
        (label, _number(getattr(metrics, field), digits)) for label, field, digits in _METRIC_ROWS
    ]
    if metrics.explained_variance:
        rows.append(
            (
                "Explained variance",
                " · ".join(f"{share * 100:.1f}%" for share in metrics.explained_variance),
            )
        )
    extra = [(key, _number(value, 4)) for key, value in sorted(metrics.extra.items())]
    return "<h2>What it scored</h2>\n" + _kv_table(rows + extra)


def _provenance(experiment: Experiment, version: DatasetVersion) -> str:
    rows = [
        ("Started", _when(experiment.started_at)),
        ("Finished", _when(experiment.finished_at)),
        ("Dataset version", str(version.version_id)),
        ("Pipeline", str(experiment.pipeline_snapshot.pipeline_id)),
    ]
    environment = experiment.environment
    if environment is None:
        # `models.py` refuses a succeeded experiment with no environment, so
        # this is a cancelled or failed run rather than a gap in the record.
        rows.append(("Environment", "not recorded, which only a run that did not succeed does"))
    else:
        rows.extend(
            [
                ("Application", environment.app_version),
                ("Python", environment.python_version),
                ("Platform", environment.platform),
            ]
        )
        rows.extend(sorted(environment.packages.items()))
    return "<h2>Where</h2>\n" + _kv_table(rows)


# --- the plots -------------------------------------------------------------


def _plots(result: EstimatorResult | None, node_id: str | None) -> str:
    if result is None:
        return """<h2>What it shows</h2>
<p class="note">This run's arrays are no longer in the project's store, so its
plots cannot be drawn. Its record, its metrics and its provenance are above and
are not affected; re-running the pipeline as it is recorded here reproduces
them.</p>"""

    figures = [_scores_plot(result), _scree_plot(result)]
    if result.task in ("regression", "classification"):
        figures.append(_predicted_plot(result))
    drawn = [figure for figure in figures if figure]
    if not drawn:
        return """<h2>What it shows</h2>
<p class="note">This estimator recorded no scores to draw.</p>"""
    named = f" — {_text(node_id)}" if node_id else ""
    return f'<h2>What it shows{named}</h2>\n<div class="grid">\n' + "\n".join(drawn) + "\n</div>"


@dataclass(frozen=True)
class _Scale:
    """One axis: data range to pixel range, with the degenerate case handled.

    A constant column has `lo == hi`, and dividing by that span would be a
    division by zero. Widening it by half a unit puts the single value in the
    middle of the axis, which is what it is.
    """

    lo: float
    hi: float
    pixel_lo: float
    pixel_hi: float

    @classmethod
    def over(cls, values: list[float], pixel_lo: float, pixel_hi: float) -> _Scale:
        lo, hi = min(values), max(values)
        if not math.isfinite(lo) or not math.isfinite(hi):
            lo, hi = 0.0, 1.0
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        else:
            margin = (hi - lo) * 0.06
            lo, hi = lo - margin, hi + margin
        return cls(lo, hi, pixel_lo, pixel_hi)

    def __call__(self, value: float) -> float:
        share = (value - self.lo) / (self.hi - self.lo)
        return self.pixel_lo + share * (self.pixel_hi - self.pixel_lo)


def _axes() -> str:
    """The two axis lines every plot here draws, and the only ones."""
    bottom = HEIGHT - PAD_BOTTOM
    right = WIDTH - PAD_RIGHT
    return (
        f'<line class="axis" x1="{PAD_LEFT}" y1="{bottom}" x2="{right}" y2="{bottom}"/>'
        f'<line class="axis" x1="{PAD_LEFT}" y1="{PAD_TOP}" x2="{PAD_LEFT}" y2="{bottom}"/>'
    )


def _frame(x: _Scale, y: _Scale, x_label: str, y_label: str) -> str:
    """Two axis lines, five ticks on each, and the two labels."""
    bottom = HEIGHT - PAD_BOTTOM
    parts = [_axes()]
    for index in range(5):
        share = index / 4
        value = x.lo + share * (x.hi - x.lo)
        at = x(value)
        parts.append(
            f'<line class="axis" x1="{at:.1f}" y1="{bottom}" x2="{at:.1f}" y2="{bottom + 4}"/>'
        )
        parts.append(
            f'<text class="tick" x="{at:.1f}" y="{bottom + 15}" text-anchor="middle">'
            f"{_tick(value)}</text>"
        )
        value = y.lo + share * (y.hi - y.lo)
        at = y(value)
        parts.append(
            f'<line class="axis" x1="{PAD_LEFT - 4}" y1="{at:.1f}" x2="{PAD_LEFT}" y2="{at:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{PAD_LEFT - 7}" y="{at + 3:.1f}" text-anchor="end">'
            f"{_tick(value)}</text>"
        )
    parts.append(
        f'<text class="label" x="{(PAD_LEFT + WIDTH - PAD_RIGHT) / 2:.0f}" '
        f'y="{HEIGHT - 6}" text-anchor="middle">{_text(x_label)}</text>'
    )
    parts.append(
        f'<text class="label" x="12" y="{(PAD_TOP + bottom) / 2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90 12 {(PAD_TOP + bottom) / 2:.0f})">{_text(y_label)}</text>'
    )
    return "".join(parts)


def _figure(inner: str, caption: str) -> str:
    return (
        f'<figure><svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{_text(caption)}">{inner}</svg>'
        f"<figcaption>{_text(caption)}</figcaption></figure>"
    )


def _scores_plot(result: EstimatorResult) -> str:
    """t1 against t2. One component is a strip plot against the sample index,
    which is what a single-component model actually has to show."""
    if not result.scores:
        return ""
    first = [row[0] for row in result.scores if row]
    if not first:
        return ""
    if len(result.scores[0]) >= 2:
        second = [row[1] for row in result.scores]
        x_label, y_label = "t1", "t2"
    else:
        second = first
        first = [float(index) for index in range(len(second))]
        x_label, y_label = "sample", "t1"

    x = _Scale.over(first, PAD_LEFT, WIDTH - PAD_RIGHT)
    y = _Scale.over(second, HEIGHT - PAD_BOTTOM, PAD_TOP)
    dots = "".join(
        f'<circle class="dot" cx="{x(a):.1f}" cy="{y(b):.1f}" r="2.6"/>'
        for a, b in zip(first, second, strict=True)
    )
    return _figure(
        _frame(x, y, x_label, y_label) + dots,
        f"Scores, {len(first)} samples",
    )


def _scree_plot(result: EstimatorResult) -> str:
    """Explained variance per component, as bars. The x axis is a component
    number, so it is drawn as one bar per component rather than as a scale."""
    shares = list(result.explained_variance_ratio)
    if not shares:
        return ""
    bottom = HEIGHT - PAD_BOTTOM
    y = _Scale.over([0.0, max(shares)], bottom, PAD_TOP)
    width = (WIDTH - PAD_LEFT - PAD_RIGHT) / len(shares)
    bars = []
    for index, share in enumerate(shares):
        left = PAD_LEFT + index * width
        top = y(share)
        bars.append(
            f'<rect class="bar" x="{left + width * 0.18:.1f}" y="{top:.1f}" '
            f'width="{width * 0.64:.1f}" height="{bottom - top:.1f}"/>'
        )
        bars.append(
            f'<text class="tick" x="{left + width / 2:.1f}" y="{bottom + 15}" '
            f'text-anchor="middle">{index + 1}</text>'
        )
    axes = _axes()
    for index in range(5):
        value = y.lo + (index / 4) * (y.hi - y.lo)
        at = y(value)
        axes += (
            f'<line class="axis" x1="{PAD_LEFT - 4}" y1="{at:.1f}" x2="{PAD_LEFT}" y2="{at:.1f}"/>'
            f'<text class="tick" x="{PAD_LEFT - 7}" y="{at + 3:.1f}" text-anchor="end">'
            f"{value * 100:.0f}%</text>"
        )
    axes += (
        f'<text class="label" x="{(PAD_LEFT + WIDTH - PAD_RIGHT) / 2:.0f}" y="{HEIGHT - 6}" '
        'text-anchor="middle">component</text>'
    )
    return _figure(axes + "".join(bars), "Explained variance per component")


def _predicted_plot(result: EstimatorResult) -> str:
    """Predicted against measured, with the 1:1 line. Both arrays are the
    result's own, so this needs no dataset beside it."""
    observed, predicted = list(result.observed), list(result.predicted)
    if not observed or len(observed) != len(predicted):
        return ""
    span = observed + predicted
    x = _Scale.over(span, PAD_LEFT, WIDTH - PAD_RIGHT)
    y = _Scale.over(span, HEIGHT - PAD_BOTTOM, PAD_TOP)
    # The 1:1 line across the shared range: the thing a reader of this plot is
    # actually comparing the cloud against.
    guide = (
        f'<line class="guide" x1="{x(x.lo):.1f}" y1="{y(x.lo):.1f}" '
        f'x2="{x(x.hi):.1f}" y2="{y(x.hi):.1f}"/>'
    )
    dots = "".join(
        f'<circle class="dot" cx="{x(a):.1f}" cy="{y(b):.1f}" r="2.6"/>'
        for a, b in zip(observed, predicted, strict=True)
    )
    target = result.target or "response"
    return _figure(
        _frame(x, y, f"measured {target}", f"predicted {target}") + guide + dots,
        f"Predicted against measured, {len(observed)} calibration samples",
    )


# --- small things ----------------------------------------------------------


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(
        f"<tr><td>{_text(label)}</td><td class='mono'>{_text(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table><tbody>\n{body}\n</tbody></table>"


def _text(value: object) -> str:
    """Everything that reaches the document goes through this.

    A dataset filename, a project name and an executor's error message are all
    strings someone else chose, and a report is a file that gets emailed. `<`
    stays `<`.
    """
    return html.escape(str(value), quote=True)


def _number(value: float | None, digits: int) -> str:
    """§11: a metric that could not be computed is absent, never zero."""
    return "—" if value is None else f"{value:.{digits}f}"


def _tick(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.01:
        return f"{value:.1e}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _when(value: object) -> str:
    return "—" if value is None else str(value).replace("T", " ")[:16]
