/** #185: the confusion panel prints each set the result carries, rows observed
 * and columns assigned in the classes' order, and nothing for a set it does
 * not carry. Static markup, as `diagnostics.test.tsx` does it. */
import { readFileSync } from "node:fs";
import path from "node:path";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { PcaPayload } from "@/api/queries";

vi.mock("plotly.js-gl2d-dist-min", () => ({ default: { react: vi.fn(), purge: vi.fn() } }));

const { ConfusionMatrix } = await import("@/screens/analysis/AnalysisResults");

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
