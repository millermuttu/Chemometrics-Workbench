import type { PcaPayload } from "@/api/queries";

import type { PlotTheme } from "./theme";

/** Traces for the analysis panels. Pure, and tested.
 *
 * Nothing here computes a statistic. Scores, loadings, variances, T², SPE and
 * both limits arrive as data; the only arithmetic is turning the limit and the
 * eigenvalues the server sent into the points of an ellipse, which is drawing,
 * not deciding.
 */

export function scoresTrace(pca: PcaPayload, x: number, y: number, theme: PlotTheme) {
  return {
    type: "scattergl",
    mode: "markers",
    x: pca.scores.map((row) => row[x]),
    y: pca.scores.map((row) => row[y]),
    text: pca.samples.map((sample) => sample.sample_id),
    marker: {
      size: 5,
      color: pca.diagnostics.hotelling_t2.map((value) =>
        // Beyond the limit the server sent, a sample is drawn in --stale: an
        // outlier is a warning, not a data series.
        value > pca.diagnostics.hotelling_t2_limit ? theme.stale : theme.series[0],
      ),
    },
    hovertemplate: "%{text}<br>%{x:.3f} · %{y:.3f}<extra></extra>",
  };
}

/** The Hotelling T² ellipse, drawn from the limit and the eigenvalues in the
 * payload. `t²  = Σ tᵢ²/λᵢ ≤ limit` is an ellipse with semi-axes
 * `√(limit · λᵢ)`; every number in that comes off the wire. */
export function ellipseTrace(pca: PcaPayload, x: number, y: number, theme: PlotTheme) {
  const limit = pca.diagnostics.hotelling_t2_limit;
  const a = Math.sqrt(limit * pca.eigenvalues[x]);
  const b = Math.sqrt(limit * pca.eigenvalues[y]);
  const points = Array.from({ length: 121 }, (_, index) => (index / 120) * 2 * Math.PI);
  return {
    type: "scattergl",
    mode: "lines",
    x: points.map((angle) => a * Math.cos(angle)),
    y: points.map((angle) => b * Math.sin(angle)),
    line: { width: 1, color: theme.ink3, dash: "dot" },
    hoverinfo: "skip",
    name: "T² limit",
  };
}

export function loadingsTraces(pca: PcaPayload, components: number[], theme: PlotTheme) {
  return components.map((component) => ({
    type: "scattergl",
    mode: "lines",
    x: pca.loadings.axis.values,
    y: pca.loadings.components[component],
    line: { width: 1.3, color: theme.series[component % theme.series.length] },
    name: `PC ${component + 1}`,
    hovertemplate: `PC ${component + 1}<br>%{x:.1f} · %{y:.4f}<extra></extra>`,
  }));
}

/** Per-component variance as bars, cumulative as a line.
 *
 * The bars are layout shapes rather than a `bar` trace: the WebGL bundle this
 * project loads carries scatter and scattergl and nothing else, and a second
 * bundle for five rectangles would be a megabyte for a shape. The invisible
 * marker trace over them is what carries the hover readout.
 */
export function varianceFigure(pca: PcaPayload, theme: PlotTheme) {
  const labels = pca.explained_variance_ratio.map((_, index) => `PC${index + 1}`);
  const percent = pca.explained_variance_ratio.map((value) => value * 100);
  return {
    data: [
      {
        type: "scattergl",
        mode: "markers",
        x: labels,
        y: percent,
        marker: { size: 1, color: theme.series[0], opacity: 0 },
        hovertemplate: "%{x}<br>%{y:.2f}%<extra></extra>",
      },
      {
        type: "scattergl",
        mode: "lines+markers",
        x: labels,
        y: pca.cumulative_explained_variance.map((value) => value * 100),
        line: { width: 1.3, color: theme.ink3 },
        marker: { size: 4, color: theme.ink3 },
        yaxis: "y2",
        hovertemplate: "cumulative %{y:.2f}%<extra></extra>",
      },
    ],
    shapes: percent.map((value, index) => ({
      type: "rect",
      xref: "x",
      yref: "y",
      x0: index - 0.32,
      x1: index + 0.32,
      y0: 0,
      y1: value,
      line: { width: 0 },
      fillcolor: theme.series[0],
      layer: "below",
    })),
  };
}

/** Which samples the server's own limits put outside. Comparison, not
 * statistics: both limits arrived with the payload. */
export function outliers(pca: PcaPayload): { sample: string; t2: number; spe: number }[] {
  return pca.samples
    .map((sample, index) => ({
      sample: sample.sample_id,
      t2: pca.diagnostics.hotelling_t2[index],
      spe: pca.diagnostics.spe[index],
    }))
    .filter(
      (row) =>
        row.t2 > pca.diagnostics.hotelling_t2_limit || row.spe > pca.diagnostics.spe_limit,
    );
}

/** Predicted against measured, with the 1:1 line a reader judges it against.
 *
 * Calibration and held-out rows are separate traces rather than one coloured
 * series: they are different claims — a residual on a row the model was fitted
 * on and one on a row it never saw — and `metrics-and-validation.md` §9 keeps
 * them apart for that reason. Held-out rows take a second *series* colour and
 * an open marker, deliberately not `stale`: that token is a warning about an
 * outlier, and a validation row is not one.
 *
 * The line is drawn from the extremes of the data rather than from a
 * regression of it: it is `y = x`, not a fit, and fitting one here would be
 * deciding something rather than drawing it. */
export function predictedTraces(pca: PcaPayload, theme: PlotTheme) {
  const regression = pca.regression;
  if (!regression) return [];

  const held = pca.validation;
  const all = [
    ...regression.observed,
    ...regression.predicted,
    ...(held?.observed ?? []),
    ...(held?.predicted ?? []),
  ];
  const low = Math.min(...all);
  const high = Math.max(...all);

  const points = (x: number[], y: number[], name: string, colour: string, open: boolean) => ({
    type: "scattergl",
    mode: "markers",
    name,
    x,
    y,
    marker: { size: 5, color: open ? "transparent" : colour, line: { width: 1, color: colour } },
    hovertemplate: `${name}<br>measured %{x:.4g}<br>predicted %{y:.4g}<extra></extra>`,
  });

  return [
    {
      type: "scattergl",
      mode: "lines",
      name: "1:1",
      x: [low, high],
      y: [low, high],
      line: { width: 1, dash: "dot", color: theme.grid },
      hoverinfo: "skip",
      showlegend: false,
    },
    points(regression.observed, regression.predicted, "calibration", theme.series[0], false),
    ...(held?.observed && held.predicted
      ? [points(held.observed, held.predicted, "held out", theme.series[1], true)]
      : []),
  ];
}

/** RMSECV against component count. One trace, because §9's curve is one
 * experiment over one fold assignment rather than `A` unrelated ones. */
export function rmsecvTrace(pca: PcaPayload, theme: PlotTheme) {
  const curve = pca.rmsecv_curve ?? [];
  return {
    type: "scattergl",
    mode: "lines+markers",
    x: curve.map((_, index) => index + 1),
    y: curve,
    line: { width: 1.5, color: theme.series[0] },
    marker: { size: 5, color: theme.series[0] },
    hovertemplate: "A = %{x}<br>RMSECV %{y:.4g}<extra></extra>",
  };
}
