/** Plotly draws from the design tokens, not from its own defaults.
 *
 * A plot that ignores the theme is the most likely way this screen drifts from
 * the artboards, so every colour and every font below is read out of the CSS
 * variables that `_base.css` defines - including `--band`, which exists in
 * both palettes precisely because the density envelope is a designed element.
 */

export interface PlotTheme {
  ink: string;
  ink3: string;
  surface: string;
  grid: string;
  band: string;
  /** Semantic, and deliberately off the data palette: an outlier is a warning,
   * never a series. */
  stale: string;
  series: string[];
  font: string;
  mono: string;
}

export function readTheme(element: HTMLElement): PlotTheme {
  const style = getComputedStyle(element);
  const token = (name: string) => style.getPropertyValue(`--${name}`).trim();
  return {
    ink: token("ink"),
    ink3: token("ink3"),
    surface: token("surface"),
    grid: token("grid"),
    band: token("band"),
    stale: token("stale"),
    series: ["d1", "d2", "d3", "d4", "d5", "d6"].map(token),
    font: "'IBM Plex Sans', system-ui, sans-serif",
    mono: "'IBM Plex Mono', ui-monospace, monospace",
  };
}

export function axisLayout(theme: PlotTheme, title: string) {
  return {
    title: { text: title, font: { size: 10.5, color: theme.ink3, family: theme.mono } },
    color: theme.ink3,
    gridcolor: theme.grid,
    zeroline: false,
    linecolor: theme.grid,
    tickfont: { size: 10, color: theme.ink3, family: theme.mono },
    automargin: true,
  };
}

export function baseLayout(theme: PlotTheme) {
  return {
    paper_bgcolor: theme.surface,
    plot_bgcolor: theme.surface,
    font: { family: theme.font, size: 11, color: theme.ink },
    margin: { l: 52, r: 14, t: 6, b: 40 },
    showlegend: false,
    hoverlabel: {
      bgcolor: theme.surface,
      bordercolor: theme.grid,
      font: { family: theme.mono, size: 11, color: theme.ink },
    },
    dragmode: "zoom",
  };
}

export const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  // Nothing here reaches the network, and a plot is no exception.
  showTips: false,
};
