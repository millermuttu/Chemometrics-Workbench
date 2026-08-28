/** What gets drawn as a band and what gets drawn as a line.
 *
 * This is the substance of the spectra view: 240 spectra become 60 traces
 * plus an envelope, and a selected sample is drawn over the top at full
 * strength. None of it needs a browser to check.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import type { SpectraPayload } from "@/api/queries";
import { bandTraces, spectraTraces } from "@/plot/traces";
import type { PlotTheme } from "@/plot/theme";

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const spectra = JSON.parse(
  readFileSync(path.join(FIXTURES, "spectra.json"), "utf8"),
) as Record<string, SpectraPayload>;

const theme: PlotTheme = {
  ink: "#0F1A18",
  ink3: "#6E7D79",
  surface: "#FFFFFF",
  grid: "#EDF1F0",
  band: "#CBD7D4",
  stale: "#9A6206",
  series: ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
  font: "sans",
  mono: "mono",
};

const source = spectra.source;

describe("the density band", () => {
  it("is the token's colour, not a grey chosen here", () => {
    for (const trace of bandTraces(source, theme)) {
      const line = trace.line as { color: string };
      expect([line.color, trace.fillcolor].filter(Boolean)).toContain(theme.band);
    }
  });

  it("fills between the bounds and marks the median", () => {
    const [lower, upper, median] = bandTraces(source, theme);
    expect(lower.y).toEqual(source.band.y_lower);
    expect(upper.fill).toBe("tonexty");
    expect(upper.y).toEqual(source.band.y_upper);
    expect(median.y).toEqual(source.band.y_median);
  });

  it("carries the whole set, not the drawn subset", () => {
    expect(source.band.n_spectra).toBe(source.n_spectra);
    expect(source.traces.length).toBeLessThan(source.n_spectra);
  });
});

describe("the drawn spectra", () => {
  it("are WebGL traces on the Okabe-Ito palette, never the accent", () => {
    for (const trace of spectraTraces(source, theme, { selected: [] })) {
      expect(trace.type).toBe("scattergl");
      expect(theme.series).toContain((trace.line as { color: string }).color);
    }
  });

  it("draws a selected sample at full strength over the rest", () => {
    const chosen = source.traces[3].index;
    const traces = spectraTraces(source, theme, { selected: [chosen] });
    const selected = traces.find((trace) => trace.customdata === chosen)!;
    const other = traces.find((trace) => trace.customdata !== chosen)!;
    expect(selected.opacity).toBe(1);
    expect(Number(other.opacity)).toBeLessThan(1);
    expect((selected.line as { width: number }).width).toBeGreaterThan(
      (other.line as { width: number }).width,
    );
  });

  it("names the sample on hover, because identifying an outlier is the workflow", () => {
    const [first] = spectraTraces(source, theme, { selected: [] });
    expect(first.name).toBe(source.traces[0].sample_id);
    expect(String(first.hovertemplate)).toContain(source.traces[0].sample_id);
  });

  it("keeps every processed node's payload the same shape", () => {
    for (const [node, payload] of Object.entries(spectra)) {
      expect(payload.axis.values.length, node).toBe(payload.decimation.variables_kept);
      expect(payload.traces.length, node).toBe(payload.decimation.traces_drawn);
    }
  });
});
