/** #185: the confusion panel prints each set the result carries, rows observed
 * and columns assigned in the classes' order, and nothing for a set it does
 * not carry. Static markup, as `diagnostics.test.tsx` does it. */
import { readFileSync } from "node:fs";
import path from "node:path";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { PcaPayload } from "@/api/queries";

vi.mock("plotly.js-gl2d-dist-min", () => ({ default: { react: vi.fn(), purge: vi.fn() } }));

const { AcceptanceTable, ConfusionMatrix } = await import("@/screens/analysis/AnalysisResults");

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const pca = (JSON.parse(readFileSync(path.join(FIXTURES, "pca.json"), "utf8")) as Record<string, PcaPayload>)
  .pca_a;

describe("the confusion panel", () => {
  const plsda: PcaPayload = {
    ...pca,
    task: "classification",
    classification: {
      class_column: "fat_class",
      classes: ["high", "low"],
      predicted_class: [],
      confusion: { calibration: [[100, 8], [5, 103]], cross_validation: [[97, 11], [9, 99]] },
    },
  };

  it("prints the sets it has, in the classes' order", () => {
    const html = renderToStaticMarkup(<ConfusionMatrix pca={plsda} />);
    expect(html).toContain("confusion-calibration");
    expect(html).toContain("confusion-cross_validation");
    expect(html).not.toContain("confusion-held_out");
    // Row `high`, columns `→ high` then `→ low`: TN then FP.
    expect(html).toMatch(/high<\/td><td class="n">100<\/td><td class="n">8<\/td>/);
  });

  it("prints an N-by-N matrix and each set's per-class table (#269)", () => {
    const three: PcaPayload = {
      ...pca,
      task: "classification",
      classification: {
        class_column: "species",
        classes: ["chicken", "pork", "turkey"],
        predicted_class: [],
        confusion: { calibration: [[18, 1, 1], [0, 20, 0], [2, 0, 18]] },
        class_metrics: {
          calibration: [
            { n: 20, sensitivity: 0.9, specificity: 0.95, precision: 0.9 },
            { n: 20, sensitivity: 1, specificity: 0.975, precision: 0.952 },
            { n: 0, specificity: 0.9 },
          ],
        },
      },
    };
    const html = renderToStaticMarkup(<ConfusionMatrix pca={three} />);
    expect(html).toContain("→ turkey");
    expect(html).toMatch(/turkey<\/td><td class="n">2<\/td><td class="n">0<\/td><td class="n">18<\/td>/);
    expect(html).toContain("class-metrics-calibration");
    // A metric with a zero denominator is absent, and printed as an em dash.
    expect(html).toMatch(/turkey<\/td><td class="n">0<\/td><td class="n">—<\/td><td class="n">0.900<\/td><td class="n">—<\/td>/);
  });

  it("draws nothing for a decomposition", () => {
    expect(renderToStaticMarkup(<ConfusionMatrix pca={pca} />)).toBe("");
  });
});

describe("the SIMCA acceptance panel (#275)", () => {
  it("prints each set's table with the none column and per-class figures", () => {
    const simca: PcaPayload = {
      ...pca,
      task: "classification",
      simca: {
        class_column: "grade",
        classes: ["a", "b"],
        models: [],
        sets: {
          cross_validation: {
            table: [[9, 2], [1, 7]],
            none: [1, 2],
            sizes: [10, 9],
            class_metrics: [
              { n: 10, sensitivity: 0.9, specificity: 8 / 9 },
              { n: 9, sensitivity: 7 / 9 },
            ],
            samples: [],
            distances: [],
          },
        },
      },
    };
    // React separates adjacent text with comments in static markup.
    const html = renderToStaticMarkup(<AcceptanceTable pca={simca} />).replaceAll("<!-- -->", "");
    expect(html).toContain("acceptance-cross_validation");
    expect(html).not.toContain("acceptance-calibration");
    expect(html).toMatch(/a \(10\)<\/td><td class="n">9<\/td><td class="n">2<\/td><td class="n">1<\/td><td class="n">0.900<\/td>/);
    // b's specificity is absent and printed as an em dash.
    expect(html).toMatch(/0.778<\/td><td class="n">—<\/td>/);
  });
});
