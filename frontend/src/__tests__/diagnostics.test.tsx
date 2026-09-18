/** #71: the SPE limit's caveat is shown beside the limit when the kernel
 * sent one, and nothing is shown when it did not. Static markup, as
 * `overloaded.test.tsx` does it: the panel draws no plot of its own. */
import { readFileSync } from "node:fs";
import path from "node:path";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { PcaPayload } from "@/api/queries";

vi.mock("plotly.js-gl2d-dist-min", () => ({ default: { react: vi.fn(), purge: vi.fn() } }));

const { Diagnostics } = await import("@/screens/analysis/AnalysisResults");

const FIXTURES = path.resolve(import.meta.dirname, "../../../tests/fixtures/contract");
const pca = (JSON.parse(readFileSync(path.join(FIXTURES, "pca.json"), "utf8")) as Record<string, PcaPayload>)
  .pca_a;

describe("the diagnostics panel", () => {
  it("says nothing about a limit inside its domain", () => {
    const html = renderToStaticMarkup(<Diagnostics pca={pca} />);
    expect(html).toContain("SPE limit");
    expect(html).not.toContain("spe-limit-caveat");
  });

  it("prints the kernel's caveat beside a limit outside it", () => {
    const sentence = "Jackson-Mudholkar assumes h0 > 0 and this model's h0 is -0.0190";
    const flagged: PcaPayload = {
      ...pca,
      diagnostics: { ...pca.diagnostics, spe_limit_caveat: sentence },
    };
    const html = renderToStaticMarkup(<Diagnostics pca={flagged} />);
    expect(html).toContain("spe-limit-caveat");
    expect(html).toContain("h0 is -0.0190");
  });
});
