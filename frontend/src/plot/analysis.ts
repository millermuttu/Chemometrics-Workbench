import type {
  CoefficientsPayload,
  ContributionsPayload,
  IplsPayload,
  PcaPayload,
} from "@/api/queries";

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
export function outliers(
  pca: PcaPayload,
): { index: number; sample: string; t2: number; spe: number }[] {
  return pca.samples
    .map((sample, index) => ({
      // The dataset row, which is what a contribution is asked for by (#186).
      index: sample.index,
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

/** VIP against the node's own axis (#184), with the `VIP = 1` line
 * `pls-regression.md` §8 explains: `Σ VIP² = p`, so 1 is the average and the
 * origin of the rule of thumb. The line is a shape rather than a trace so it
 * neither appears in the legend nor answers a hover. */
export function vipFigure(pca: PcaPayload, theme: PlotTheme) {
  return {
    data: [
      {
        type: "scattergl",
        mode: "lines",
        name: "VIP",
        x: pca.loadings.axis.values,
        y: pca.regression?.vip ?? [],
        line: { width: 1.3, color: theme.series[0] },
        hovertemplate: "%{x:.1f} · VIP %{y:.3f}<extra></extra>",
      },
    ],
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: 1,
        y1: 1,
        line: { width: 1, dash: "dot", color: theme.ink3 },
      },
    ],
  };
}

/** The folded coefficient vector on the dataset's raw axis (#184), or `null`
 * when the chain cannot be folded - the panel prints the reason instead. */
export function coefficientTrace(payload: CoefficientsPayload, theme: PlotTheme) {
  if (!payload.available || !payload.axis || !payload.coefficients) return null;
  return {
    type: "scattergl",
    mode: "lines",
    name: "b",
    x: payload.axis.values,
    y: payload.coefficients,
    line: { width: 1.3, color: theme.series[1] },
    hovertemplate: "%{x:.1f} · b %{y:.4g}<extra></extra>",
  };
}

/** One sample's contributions to its `T²` or its SPE against the node's axis
 * (#186), filled to zero so a signed `T²` contribution reads as a bar. The
 * numbers are the server's; their sum is the total the panel prints. */
export function contributionTrace(
  payload: ContributionsPayload,
  which: "hotelling_t2" | "spe",
  theme: PlotTheme,
) {
  const values = which === "spe" ? payload.spe.contributions : payload.hotelling_t2.contributions;
  return {
    type: "scattergl",
    mode: "lines",
    name: which === "spe" ? "SPE contribution" : "T² contribution",
    x: payload.axis.values,
    y: values,
    fill: "tozeroy",
    line: { width: 1.2, color: theme.series[which === "spe" ? 1 : 0] },
    hovertemplate: `${payload.sample.sample_id}<br>%{x:.1f} · %{y:.4g}<extra></extra>`,
  };
}

/** Calibration rows the server flagged under any rule (outliers.md section 5),
 * by their position in `samples`. */
function flagged(pca: PcaPayload): Set<number> {
  return new Set((pca.outliers?.flags ?? []).map((flag) => flag.index));
}

/** A limit drawn as a dotted line across the plot, from the server's number. */
function limitLine(x: number[], y: number[], theme: PlotTheme, name: string) {
  return {
    type: "scattergl",
    mode: "lines",
    x,
    y,
    line: { width: 1, color: theme.ink3, dash: "dot" },
    hoverinfo: "skip",
    name,
    showlegend: false,
  };
}

/** The influence plot: T² against Q, each sample named on hover, a flagged
 * one in `stale`, and both limits as the lines it is judged against. */
export function influenceTraces(pca: PcaPayload, theme: PlotTheme) {
  const { hotelling_t2: t2, spe, hotelling_t2_limit: t2Limit, spe_limit: qLimit } = pca.diagnostics;
  const marked = flagged(pca);
  const xMax = Math.max(t2Limit, ...t2) * 1.05;
  const yMax = Math.max(qLimit, ...spe) * 1.05;
  return [
    limitLine([t2Limit, t2Limit], [0, yMax], theme, "T² limit"),
    limitLine([0, xMax], [qLimit, qLimit], theme, "Q limit"),
    {
      type: "scattergl",
      mode: "markers",
      x: t2,
      y: spe,
      text: pca.samples.map((sample) => sample.sample_id),
      marker: {
        size: 5,
        color: t2.map((_, index) => (marked.has(index) ? theme.stale : theme.series[0])),
      },
      hovertemplate: "%{text}<br>T² %{x:.3f} · Q %{y:.3g}<extra></extra>",
    },
  ];
}

/** Leverage against studentised residual (outliers.md sections 2 and 3), for a
 * regression: the limit on each, and every sample named on hover. */
export function leverageTraces(pca: PcaPayload, theme: PlotTheme) {
  const block = pca.outliers;
  if (!block?.leverage || !block.studentised_residuals) return [];
  const residuals = block.studentised_residuals;
  const rows = block.leverage
    .map((h, index) => ({ h, r: residuals[index], index }))
    .filter((row): row is { h: number; r: number; index: number } => row.r !== null);
  const marked = flagged(pca);
  const hMax = Math.max(block.limits.leverage, ...block.leverage) * 1.05;
  const rMax = Math.max(block.limits.residual, ...rows.map((row) => Math.abs(row.r))) * 1.05;
  const r = block.limits.residual;
  return [
    limitLine([block.limits.leverage, block.limits.leverage], [-rMax, rMax], theme, "leverage limit"),
    limitLine([0, hMax], [r, r], theme, "residual limit"),
    limitLine([0, hMax], [-r, -r], theme, "residual limit"),
    {
      type: "scattergl",
      mode: "markers",
      x: rows.map((row) => row.h),
      y: rows.map((row) => row.r),
      text: rows.map((row) => pca.samples[row.index].sample_id),
      marker: {
        size: 5,
        color: rows.map((row) => (marked.has(row.index) ? theme.stale : theme.series[0])),
      },
      hovertemplate: "%{text}<br>leverage %{x:.3f} · residual %{y:.2f}<extra></extra>",
    },
  ];
}

/** The positions a threshold keeps (#281): VIP at or above it, or |b| at or
 * above it. Positions on the estimator's own axis, which is what a
 * `select_variables` step on its input selects from. */
export function thresholdSelection(values: number[], threshold: number, absolute: boolean): number[] {
  return values.flatMap((value, index) => ((absolute ? Math.abs(value) : value) >= threshold ? [index] : []));
}

/** The mean spectrum with the selected variables marked on it (#281). */
export function selectionTraces(pca: PcaPayload, selected: number[], theme: PlotTheme) {
  const axis = pca.loadings.axis.values;
  const mean = pca.regression?.x_mean ?? [];
  return [
    {
      type: "scattergl",
      mode: "lines",
      name: "mean spectrum",
      x: axis,
      y: mean,
      line: { width: 1.1, color: theme.ink3 },
      hovertemplate: "%{x:.1f} · %{y:.4g}<extra></extra>",
    },
    {
      type: "scattergl",
      mode: "markers",
      name: "selected",
      x: selected.map((index) => axis[index]),
      y: selected.map((index) => mean[index]),
      marker: { size: 5, color: theme.series[0] },
      hovertemplate: "%{x:.1f} · selected<extra></extra>",
    },
  ];
}

/** iPLS over the spectrum (#282): one bar per interval at its place on the
 * axis, as tall as its RMSECV, the intervals forward selection kept in the
 * series colour, and the full spectrum's RMSECV as the line they are read
 * against. */
export function iplsFigure(payload: IplsPayload, theme: PlotTheme) {
  const kept = new Set(payload.steps.map((step) => step.interval));
  const centre = (one: IplsPayload["intervals"][number]) => (one.axis_start + one.axis_end) / 2;
  return {
    data: [
      {
        type: "bar",
        x: payload.intervals.map(centre),
        y: payload.intervals.map((one) => one.rmsecv),
        width: payload.intervals.map((one) => Math.abs(one.axis_end - one.axis_start) || 1),
        marker: {
          color: payload.intervals.map((_, k) => (kept.has(k) ? theme.series[0] : theme.band)),
        },
        text: payload.intervals.map((one) => `A ${one.n_components}`),
        hovertemplate: "%{x:.1f} · RMSECV %{y:.4g} · %{text}<extra></extra>",
        name: "interval",
      },
    ],
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        y0: payload.full.rmsecv,
        y1: payload.full.rmsecv,
        line: { width: 1, dash: "dot", color: theme.ink3 },
      },
    ],
  };
}
