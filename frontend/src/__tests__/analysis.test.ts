/** The analysis panels draw what the kernel computed, and nothing else.
 *
 * The rule from #48: nothing is computed in the frontend. Scores, loadings,
 * variances, T², SPE and both limits arrive as data. The one piece of
 * arithmetic is turning the limit and the eigenvalues into the points of an
 * ellipse, which is drawing rather than deciding - and it is checked here
 * against the limit the fixture carries.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import type { PcaPayload } from "@/api/queries";
import {
  ellipseTrace,
  loadingsTraces,
  outliers,
  scoresTrace,
  varianceFigure,
} from "@/plot/analysis";
import type { PlotTheme } from "@/plot/theme";

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const pca = (JSON.parse(readFileSync(path.join(FIXTURES, "pca.json"), "utf8")) as Record<string, PcaPayload>)
  .pca_a;

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

describe("the T² ellipse", () => {
  const ellipse = ellipseTrace(pca, 0, 1, theme);

  it("has the semi-axes the served limit and eigenvalues imply", () => {
    const limit = pca.diagnostics.hotelling_t2_limit;
    expect(Math.max(...(ellipse.x as number[]))).toBeCloseTo(
      Math.sqrt(limit * pca.eigenvalues[0]),
      10,
    );
    expect(Math.max(...(ellipse.y as number[]))).toBeCloseTo(
      Math.sqrt(limit * pca.eigenvalues[1]),
      10,
    );
  });

  it("closes, so it reads as a boundary rather than an arc", () => {
    const x = ellipse.x as number[];
    const y = ellipse.y as number[];
    expect(x[0]).toBeCloseTo(x.at(-1)!, 12);
    expect(y[0]).toBeCloseTo(y.at(-1)!, 12);
  });
});

describe("the scores plot", () => {
  const trace = scoresTrace(pca, 0, 1, theme);

  it("plots the components asked for, and names every point", () => {
    expect(trace.x).toEqual(pca.scores.map((row) => row[0]));
    expect(trace.y).toEqual(pca.scores.map((row) => row[1]));
    expect((trace.text as string[])[0]).toBe(pca.samples[0].sample_id);
    expect(String(trace.hovertemplate)).toContain("%{text}");
  });

  it("marks a sample beyond the served limit in --stale, off the data palette", () => {
    const colours = (trace.marker as { color: string[] }).color;
    pca.diagnostics.hotelling_t2.forEach((value, index) => {
      const expected = value > pca.diagnostics.hotelling_t2_limit ? theme.stale : theme.series[0];
      expect(colours[index]).toBe(expected);
    });
    expect(colours).toContain(theme.stale);
  });
});

it("draws loadings against the axis in its real units", () => {
  const traces = loadingsTraces(pca, [0, 1], theme);
  expect(traces).toHaveLength(2);
  expect(traces[0].x).toEqual(pca.loadings.axis.values);
  expect(traces[0].y).toEqual(pca.loadings.components[0]);
  expect(pca.loadings.axis.unit).toBe("nm");
});

it("shows per-component variance as bars and the cumulative as a line", () => {
  const { data, shapes } = varianceFigure(pca, theme);

  // The bars are shapes, because the loaded bundle has no bar trace and a
  // second bundle for five rectangles is not worth a megabyte.
  expect(shapes).toHaveLength(pca.explained_variance_ratio.length);
  expect(shapes[0].y1).toBeCloseTo(pca.explained_variance_ratio[0] * 100, 10);
  expect(shapes[0].y0).toBe(0);
  expect(shapes[0].fillcolor).toBe(theme.series[0]);

  const cumulative = data[1];
  expect(cumulative.yaxis).toBe("y2");
  expect((cumulative.y as number[]).at(-1)).toBeCloseTo(
    pca.cumulative_explained_variance.at(-1)! * 100,
    10,
  );
});

it("lists exactly the samples the served limits put outside", () => {
  const beyond = outliers(pca);
  const expected = pca.samples.filter(
    (_, index) =>
      pca.diagnostics.hotelling_t2[index] > pca.diagnostics.hotelling_t2_limit ||
      pca.diagnostics.spe[index] > pca.diagnostics.spe_limit,
  );
  expect(beyond.map((row) => row.sample)).toEqual(expected.map((sample) => sample.sample_id));
  expect(beyond.length).toBeGreaterThan(0);
  expect(beyond.length).toBeLessThan(pca.n_samples);
});
