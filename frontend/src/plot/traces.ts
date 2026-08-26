import type { SpectraPayload } from "@/api/queries";

import type { PlotTheme } from "./theme";

/** Turning a spectra payload into Plotly traces. Pure, and tested: the rules
 * about what is drawn as a band and what is drawn at full resolution are the
 * substance of this screen, and they should not need a browser to check. */

export interface TraceOptions {
  /** Sample indices the user has selected. Drawn over the band, in series colour. */
  selected: number[];
  /** Dims a comparison layer so the primary one reads first. */
  faded?: boolean;
}

/** The envelope is two traces: the lower bound, then the upper bound filling
 * down to it. `--band` is the token for exactly this. */
export function bandTraces(
  payload: SpectraPayload,
  theme: PlotTheme,
  options: { faded?: boolean } = {},
) {
  const x = payload.axis.values;
  const opacity = options.faded ? 0.35 : 1;
  return [
    {
      type: "scattergl",
      mode: "lines",
      x,
      y: payload.band.y_lower,
      line: { width: 0, color: theme.band },
      hoverinfo: "skip",
      showlegend: false,
      opacity,
    },
    {
      type: "scattergl",
      mode: "lines",
      x,
      y: payload.band.y_upper,
      fill: "tonexty",
      fillcolor: theme.band,
      line: { width: 0, color: theme.band },
      hoverinfo: "skip",
      showlegend: false,
      opacity,
    },
    {
      type: "scattergl",
      mode: "lines",
      x,
      y: payload.band.y_median,
      line: { width: 1.2, color: theme.band, dash: "dot" },
      name: "median",
      hovertemplate: "median<br>%{x:.1f} · %{y:.4f}<extra></extra>",
      opacity,
    },
  ];
}

/** The drawn traces. Selected samples are full-strength in the series palette
 * and named on hover - clicking an outlier to find out which sample it is is
 * the workflow this screen exists for. */
export function spectraTraces(
  payload: SpectraPayload,
  theme: PlotTheme,
  { selected, faded }: TraceOptions,
) {
  const x = payload.axis.values;
  const chosen = new Set(selected);
  return payload.traces.map((trace, position) => {
    const isSelected = chosen.has(trace.index);
    return {
      type: "scattergl",
      mode: "lines",
      x,
      y: trace.y,
      name: trace.sample_id,
      customdata: trace.index,
      line: {
        width: isSelected ? 1.6 : 0.7,
        color: isSelected
          ? theme.series[position % theme.series.length]
          : theme.series[position % theme.series.length],
      },
      opacity: isSelected ? 1 : faded ? 0.12 : 0.22,
      hovertemplate: `${trace.sample_id}<br>%{x:.1f} · %{y:.4f}<extra></extra>`,
    };
  });
}
