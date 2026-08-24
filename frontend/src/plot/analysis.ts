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
