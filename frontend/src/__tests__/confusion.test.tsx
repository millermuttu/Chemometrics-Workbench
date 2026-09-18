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

  it("draws nothing for a decomposition", () => {
    expect(renderToStaticMarkup(<ConfusionMatrix pca={pca} />)).toBe("");
  });
});
