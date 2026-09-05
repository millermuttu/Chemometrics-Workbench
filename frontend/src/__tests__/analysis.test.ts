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
  predictedTraces,
  rmsecvTrace,
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

describe("the T² ellipse on a regression", () => {
  /* A PLS node reaches this screen through the same payload a PCA node does -
   * `task` differs and the shared half does not. The ellipse is drawn from
   * `eigenvalues`, which #142 published as an empty array for a regression, so
   * both radii were `Math.sqrt(limit * undefined)`. Nothing caught it: the
   * frozen `pca.json` fixture is a decomposition and no demo had a PLS node to
   * open. #146 publishes the score variances and this asserts the consequence
   * rather than the cause - radii that are numbers. */
  it("has finite semi-axes, which an absent eigenvalue would not give", () => {
    const regression: PcaPayload = { ...pca, task: "regression" };
    const ellipse = ellipseTrace(regression, 0, 1, theme);

    expect((ellipse.x as number[]).every(Number.isFinite)).toBe(true);
    expect((ellipse.y as number[]).every(Number.isFinite)).toBe(true);
    expect(Math.max(...(ellipse.x as number[]))).toBeGreaterThan(0);
  });
});

describe("a regression's own panels", () => {
  /* `results/{node}` serves these only when `task === "regression"`. The
   * fixture is a decomposition, so a regression payload is built from it -
   * the shared half is genuinely identical, which is the point of #142's
   * additive shape. */
  const pls: PcaPayload = {
    ...pca,
    task: "regression",
    regression: {
      target: "fat",
      observed: [10, 20, 30, 40],
      predicted: [11, 19, 31, 39],
      coefficients: [0.1, 0.2],
      vip: [1.0, 0.9],
      y_loadings: [0.5],
      y_explained_variance_ratio: [0.8],
    },
    metrics: { rmsec: 1.0, rmsecv: 1.2, r2: 0.99 },
    rmsecv_curve: [4.9, 3.2, 2.8, 2.5],
  };

  it("draws calibration against a 1:1 line, and held-out separately", () => {
    const withHeld: PcaPayload = {
      ...pls,
      validation: {
        fold: 0,
        samples: [],
        scores: [],
        hotelling_t2: [],
        spe: [],
        observed: [15, 25],
        predicted: [14, 26],
      },
    };
    const traces = predictedTraces(withHeld, theme);

    expect(traces).toHaveLength(3);
    const [line, calibration, heldOut] = traces as Record<string, unknown>[];
    // y = x across the full extent of both sets, not a fit of either.
    expect(line.x).toEqual([10, 40]);
    expect(line.y).toEqual([10, 40]);
    expect(calibration.name).toBe("calibration");
    expect(heldOut.name).toBe("held out");
  });

  it("omits the held-out trace when the node is not below a split", () => {
    expect(predictedTraces(pls, theme)).toHaveLength(2);
  });

  it("draws nothing at all for a decomposition", () => {
    expect(predictedTraces(pca, theme)).toEqual([]);
  });

  it("plots RMSECV against component count starting at one", () => {
    const trace = rmsecvTrace(pls, theme) as Record<string, unknown>;
    expect(trace.x).toEqual([1, 2, 3, 4]);
    expect(trace.y).toEqual([4.9, 3.2, 2.8, 2.5]);
  });
});
